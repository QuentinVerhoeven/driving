"""
    Live loop: read frames one at a time, run YOLO + tracking, show boxes and FPS.

    Works on a video file (for testing in WSL) or a camera index (native Windows).
    Only the --source argument changes, the loop is identical.

    Usage:
    uv run python live.py --source data/raw/clip1_30s.mov --max-frames 100
    uv run python live.py --source data/raw/clip1_30s.mov --out outputs/live_test.mp4
    uv run python live.py --source 0 --show          # camera 0, native Windows

    Press q in the window (with --show) to quit.
"""

import argparse
import csv
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from lead_ttc import BAND_HALF_WIDTH, Box, LeadSelector, TTCKalman

# COCO class IDs we care about (same as track.py)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def open_source(source):
    """A string that is all digits means a camera index, anything else is a file path."""
    if source.isdigit():
        # DirectShow is the Windows camera backend (same as webcam_check.py); other OSes use the default
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        return cv2.VideoCapture(int(source), backend)
    return cv2.VideoCapture(source)


class LatestFrameGrabber:
    """Reads a camera in a background thread and keeps ONLY the newest frame + the time it arrived.

    Why: if the main loop does cap.read() itself, the camera keeps producing frames (30/s) while YOLO
    is busy (~90 ms), and the driver may queue them. The next read() then returns an OLD frame: the
    picture lags reality and the timestamps no longer match when the frame really happened.
    Here the thread drains the camera constantly, so nothing queues up, and get() hands out the newest
    frame with its true arrival time. Frames that arrive while we are busy are dropped on purpose.
    """

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()   # lock + a way to sleep until "a new frame arrived"
        self.frame = None
        self.t = None                       # time.time() when the newest frame was read
        self.seq = 0                        # counts frames read so far, so get() can tell what is new
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()     # blocks until the camera delivers; releases the GIL while waiting
            t = time.time()
            with self.cond:
                if not ok:                  # camera stopped: wake the main loop so it can exit
                    self.running = False
                else:
                    self.frame, self.t, self.seq = frame, t, self.seq + 1
                self.cond.notify_all()

    def get(self, last_seq):
        """Wait for a frame newer than last_seq. Returns (frame, capture_time, seq), or None if the camera stopped."""
        with self.cond:
            while self.running and self.seq == last_seq:
                self.cond.wait(timeout=1.0)
            if self.seq == last_seq:        # stopped without anything new
                return None
            return self.frame, self.t, self.seq

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


def to_boxes(r):
    """Turn Ultralytics' tensors for one frame into a list of our Box objects."""
    if r.boxes.id is None:  # nothing tracked in this frame
        return []
    ids = r.boxes.id.int().tolist()
    xyxy = r.boxes.xyxy.tolist()
    return [Box(tid, *corners) for tid, corners in zip(ids, xyxy)]


def ttc_label(ttc):
    """Text for the overlay. nan = box not growing (or filter just reset), so no meaningful TTC."""
    if ttc != ttc:      # nan is the only value that is not equal to itself
        return "TTC --"
    if ttc > 10:
        return "TTC >10 s"
    return f"TTC {ttc:.1f} s"


def draw_lead(img, lead, frame_w, ttc=None):
    """Lane band (thin white lines) + the chosen lead car (thick green box)."""
    h = img.shape[0]
    for x in (frame_w / 2 - BAND_HALF_WIDTH * frame_w, frame_w / 2 + BAND_HALF_WIDTH * frame_w):
        cv2.line(img, (int(x), 0), (int(x), h), (255, 255, 255), 2)
    if lead is not None:
        cv2.rectangle(img, (int(lead.x1), int(lead.y1)), (int(lead.x2), int(lead.y2)), (0, 255, 0), 8)
        # TTC goes in a fixed corner readout (under the FPS counter), not on the box: over a small distant
        # box it collided with Ultralytics' own labels and was unreadable
        if ttc is not None:
            cv2.putText(img, f"LEAD id {lead.track_id}   {ttc_label(ttc)}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")  # file path or camera index
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.4)  # same default as track.py, see NOTES.md
    parser.add_argument("--imgsz", type=int, default=1280)  # YOLO input size; the main speed vs accuracy knob on CPU
    parser.add_argument("--show", action="store_true")  # open a window (needs a display)
    parser.add_argument("--out", type=Path, default=None)  # optionally save the annotated video
    parser.add_argument("--log", type=Path, default=None)  # optionally save frame/time/width/TTC per frame as CSV
    parser.add_argument("--max-frames", type=int, default=None)  # stop early, for quick benchmarks
    args = parser.parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = reported_fps if reported_fps > 0 else 30.0  # cameras often report 0 or -1 (unknown); `or 30.0` would not catch -1
    print(f"Source: {width}x{height} at {src_fps:.1f} fps")

    writer = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    model = YOLO(args.model)
    # The very first inference is slow (lazy initialization, ~30 s on the Windows laptop). Run one on a blank
    # frame now so that cost is paid before the camera starts being timed, not on the first live frame.
    print("Warming up the model...")
    model.predict(np.zeros((height, width, 3), dtype=np.uint8), imgsz=args.imgsz, verbose=False)
    selector = LeadSelector(width, height)  # created ONCE, outside the loop: its memory must survive across frames
    kalman = TTCKalman()                    # also created once: holds [width, rate] between frames
    is_camera = args.source.isdigit()
    log_rows = []
    lead_counts = {}                        # how many frames each track id was the lead

    loop_times = deque(maxlen=30)   # timestamps of the last 30 loop iterations -> smoothed FPS
    yolo_ms = []                    # how long each YOLO call took, to see if the model is the bottleneck
    frame_idx = 0
    start = time.time()

    # Camera: background grabber that always gives the newest frame. File: plain read(), we want every frame in order.
    grabber = LatestFrameGrabber(cap) if is_camera else None
    last_seq = 0
    skipped = 0        # camera frames that arrived while we were busy and were dropped
    age_ms = []        # how old each frame was when we started processing it (lag), camera only

    while True:
        if grabber:
            got = grabber.get(last_seq)
            if got is None:
                break  # camera stopped giving frames
            frame, t_capture, seq = got
            skipped += seq - last_seq - 1 if last_seq else 0
            last_seq = seq
            age_ms.append((time.time() - t_capture) * 1000)
        else:
            ok, frame = cap.read()
            if not ok:
                break  # end of file
            t_capture = None

        t0 = time.time()
        # One frame in, tracker state kept between calls thanks to persist=True
        r = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=list(VEHICLE_CLASSES),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]  # track() returns a list with one Results per input image
        yolo_ms.append((time.time() - t0) * 1000)

        # Timestamp of this frame: when the camera delivered it (NOT when we finished processing), or its
        # position in the video for a file. This is the time the Kalman filter's dt is built from.
        t = (t_capture - start) if is_camera else frame_idx / src_fps

        lead = selector.update(to_boxes(r))
        ttc = None
        if lead is not None:
            lead_counts[lead.track_id] = lead_counts.get(lead.track_id, 0) + 1
            _, _, ttc = kalman.update(lead.w, t, lead.track_id)
            log_rows.append((frame_idx, round(t, 3), lead.track_id, round(lead.w, 1), ttc))

        annotated = r.plot()  # frame with boxes + IDs drawn
        draw_lead(annotated, lead, width, ttc)

        loop_times.append(time.time())
        if len(loop_times) > 1:
            fps = (len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
            cv2.putText(annotated, f"{fps:.1f} fps", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("live", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx}: {fps:.1f} fps overall, YOLO {sum(yolo_ms[-20:]) / 20:.0f} ms/frame")
        if args.max_frames and frame_idx >= args.max_frames:
            break

    if grabber:
        grabber.stop()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t", "track_id", "width", "ttc"])
            w.writerows(log_rows)
        print(f"Saved {args.log}  ({len(log_rows)} rows)")

    elapsed = time.time() - start
    if frame_idx:
        print(f"\n{frame_idx} frames in {elapsed:.1f}s = {frame_idx / elapsed:.1f} fps end to end")
        print(f"YOLO + tracker: {statistics.median(yolo_ms):.0f} ms/frame (median)")
        if age_ms:
            print(f"Frame age when processing started: median {statistics.median(age_ms):.0f} ms, "
                  f"max {max(age_ms):.0f} ms; camera frames dropped: {skipped} (processed {frame_idx})")
        print(f"Lead picks (track id -> frames): {lead_counts}")


if __name__ == "__main__":
    main()
