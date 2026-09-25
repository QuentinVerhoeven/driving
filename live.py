"""
    Live loop: YOLO + tracking + lead-vehicle selection + Kalman TTC, drawn on the video.

    RESEARCH PROTOTYPE, NOT A SAFETY SYSTEM. The driver must never look at or operate the laptop while driving:
    have a passenger run it, or record with --save-raw and analyse later.

    Works on a video file (for testing in WSL) or a camera index (native Windows).

    Camera mode uses two threads so the picture stays smooth even though YOLO is slow:
      - display loop (main thread): newest camera frame -> draw the latest results on it -> show (~30 fps)
      - inference worker (background thread): YOLO + lead + TTC on the newest frame whenever it is free (~10 fps)
    The picture is live; only the boxes/TTC trail it by about one YOLO call.
    File mode processes frames in order (every --stride-th one goes through the model).

    Usage:
    uv run python live.py --source 1 --show --save-raw outputs/drive1      # in the car (Windows): record + live view + log
    uv run python live.py --source outputs/drive1.mp4 --times outputs/drive1_times.csv --stride 3 --out outputs/drive1_annotated.mp4
    uv run python live.py --source data/raw/clip1_30s.mov --max-frames 100  # test on a recorded clip

    Keys (with --show):  q quit | a / d move the lane band left / right | w / s widen / narrow the lane band
"""

import argparse
import csv
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from lead_ttc import Box, LeadSelector, TTCKalman

# COCO class IDs we care about (same as track.py)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
STALE_OVERLAY_S = 1.0     # if the newest model result is older than this, stop drawing it (an old TTC on a new picture misleads)
NO_FRAME_TIMEOUT_S = 0.5  # if the camera delivers nothing for this long, show a banner instead of a frozen picture
STALE = object()          # returned by LatestFrameGrabber.get() when the wait timed out (camera alive but silent)


def open_source(source):
    """A string that is all digits means a camera index, anything else is a file path."""
    if source.isdigit():
        # DirectShow is the Windows camera backend (same as webcam_check.py); other OSes use the default
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        return cv2.VideoCapture(int(source), backend)
    return cv2.VideoCapture(source)


def check_camera(cap, seconds=4.0):
    """Fail fast if the camera gives no frames or only black ones (the wrong index, or a stalled virtual camera)."""
    n, best, start = 0, 0.0, time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if ok:
            n += 1
            best = max(best, float(frame.mean()))   # 0 = pure black
            if n >= 3 and best > 5:
                break
    if n == 0:
        raise SystemExit("Camera opened but no frames arrived. Close other apps using it, restart Iriun (phone app first, "
                         "then the Windows client), and try again.")
    if best < 5:
        raise SystemExit(f"Camera frames are black (brightness {best:.1f}). Probably the wrong --source index or a stalled "
                         "virtual camera: run camera_scan.py to find the index that shows a real picture.")
    print(f"Camera check OK: {n} frames, brightness {best:.0f}")


def upright(frame, code):
    """Rotate a frame if the phone is mounted rotated (code from ROTATE_CODES, None = leave alone)."""
    return frame if code is None else cv2.rotate(frame, code)


class LatestFrameGrabber:
    """Reads a camera in a background thread and keeps ONLY the newest frame + the time it arrived.

    Why: if a loop does cap.read() itself while YOLO is busy (~90 ms), the camera keeps producing frames
    (30/s) and the driver may queue them, so the next read() returns an OLD frame. Here the thread drains
    the camera constantly, so nothing queues up, and get() hands out the newest frame with its true
    arrival time. Several threads can call get() at once (each with its own last_seq).
    """

    def __init__(self, cap, rotate_code=None):
        self.cap = cap
        self.rotate_code = rotate_code
        self.cond = threading.Condition()   # lock + a way to sleep until "a new frame arrived"
        self.frame = None
        self.t = None                       # time.time() when the newest frame was read
        self.seq = 0                        # counts frames read so far, so get() can tell what is new
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                ok, frame = self.cap.read()     # blocks until the camera delivers; releases the GIL while waiting
            except Exception:
                ok, frame = False, None
            t = time.time()
            if ok:
                frame = upright(frame, self.rotate_code)
            with self.cond:
                if not ok:                  # camera stopped: wake everyone waiting so they can exit
                    self.running = False
                else:
                    self.frame, self.t, self.seq = frame, t, self.seq + 1
                self.cond.notify_all()

    def get(self, last_seq, timeout=None):
        """Wait for a frame newer than last_seq.

        Returns (frame, capture_time, seq); None if the camera stopped; STALE if `timeout` seconds passed with nothing new.
        """
        deadline = None if timeout is None else time.time() + timeout
        with self.cond:
            while self.running and self.seq == last_seq:
                remaining = 1.0 if deadline is None else deadline - time.time()
                if remaining <= 0:
                    return STALE
                self.cond.wait(timeout=min(remaining, 1.0))
            if self.seq == last_seq:        # stopped without anything new
                return None
            return self.frame, self.t, self.seq

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


@dataclass
class FrameResult:
    """Everything the display needs to draw for one processed frame."""
    boxes: list     # all tracked Box objects
    lead: object    # the chosen lead Box, or None
    ttc: float      # Kalman TTC in seconds (nan = no meaningful value), or None if there is no lead
    t: float        # timestamp of the frame this was computed from


class Pipeline:
    """Frame in -> FrameResult out: YOLO tracking, lead selection, Kalman TTC. No threads, no drawing."""

    def __init__(self, model, args, frame_w, frame_h):
        self.model = model
        self.args = args
        # created ONCE: their memory must survive across frames
        self.selector = LeadSelector(frame_w, frame_h, args.band, args.lane_offset)
        self.kalman = TTCKalman(frame_w)
        self.n = 0              # frames processed so far
        self.yolo_ms = []       # how long each YOLO call took
        self.lead_counts = {}   # how many frames each track id was the lead
        self.log_rows = []

    def process(self, frame, t, frame_id):
        t0 = time.time()
        # One frame in; tracker state is kept between calls thanks to persist=True
        r = self.model.track(
            frame,
            persist=True,
            tracker=self.args.tracker,
            classes=list(VEHICLE_CLASSES),
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            verbose=False,
        )[0]  # track() returns a list with one Results per input image
        self.yolo_ms.append((time.time() - t0) * 1000)

        boxes = to_boxes(r)
        lead = self.selector.update(boxes, t)
        ttc = None
        if lead is not None:
            self.lead_counts[lead.track_id] = self.lead_counts.get(lead.track_id, 0) + 1
            _, _, ttc = self.kalman.update(lead.w, t, lead.track_id)
            self.log_rows.append((frame_id, round(t, 3), lead.track_id, round(lead.w, 1), ttc))
        self.n += 1
        return FrameResult(boxes, lead, ttc, t)


class InferenceWorker:
    """Background thread: runs the Pipeline on the newest frame whenever it is free, publishes the latest result.

    If anything inside raises, the error is stored in .error and .done is set, so the display loop can stop
    with a message instead of silently showing an ever-older overlay.
    """

    def __init__(self, grabber, pipeline, t_origin, max_frames=None):
        self.grabber = grabber
        self.pipeline = pipeline
        self.t_origin = t_origin            # camera timestamps are seconds since this moment
        self.max_frames = max_frames
        self.lock = threading.Lock()
        self.latest = None
        self.model_fps = 0.0
        self.age_ms = []                    # how old each frame was when processing started
        self.error = None
        self.done = False
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            last_seq = 0
            times = deque(maxlen=30)
            while not self.stop_flag:
                got = self.grabber.get(last_seq, timeout=1.0)
                if got is None:
                    break  # camera stopped
                if got is STALE:
                    continue  # nothing new yet: check stop_flag and wait again
                frame, t_capture, last_seq = got
                self.age_ms.append((time.time() - t_capture) * 1000)

                # Timestamp = when the camera delivered the frame (NOT when we finish processing): the Kalman dt uses it
                result = self.pipeline.process(frame, t_capture - self.t_origin, last_seq)
                with self.lock:
                    self.latest = result

                times.append(time.time())
                if len(times) > 1:
                    self.model_fps = (len(times) - 1) / (times[-1] - times[0])
                n = self.pipeline.n
                if n % 20 == 0:
                    print(f"  processed {n}: {self.model_fps:.1f} fps, YOLO {statistics.median(self.pipeline.yolo_ms[-20:]):.0f} ms/frame")
                if self.max_frames and n >= self.max_frames:
                    break
        except Exception as e:  # noqa: BLE001 - we want ANY failure surfaced, not a silently dead thread
            self.error = e
        finally:
            self.done = True

    def get_latest(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=5.0)


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


def draw_banner(img, text):
    """A big red warning line across the middle of the frame."""
    h, w = img.shape[:2]
    s = max(w / 1280, 1.0)
    thick = max(2, round(3 * s))
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.4 * s, thick)
    x, y = max((w - tw) // 2, 5), h // 2
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, 1.4 * s, (0, 0, 255), thick)


def draw_overlay(img, res, info=None, band=None):
    """Draw a FrameResult onto img (in place): lane band, every box + id, the lead in green, TTC readout.

    Sizes scale with the frame width so it looks the same at 640 px and at 4K. `band` = (left, right) as fractions of the width.
    """
    h, w = img.shape[:2]
    s = w / 1280
    thin = max(1, round(s))
    thick = max(2, round(2.7 * s))
    hs = max(s, 1.0)                         # HUD text never gets smaller than readable, even on small frames
    hud_thick = max(2, round(2.5 * hs))

    if info:
        cv2.putText(img, info, (int(10 * hs), int(30 * hs)), FONT, 0.6 * hs, (0, 255, 0), hud_thick)
    if band:
        for frac in band:                    # the "in my lane" band
            x = int(frac * w)
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), thin)
    if res is None:
        return

    for b in res.boxes:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 255, 255), thin)
        cv2.putText(img, f"id {b.track_id}", (int(b.x1), max(int(b.y1) - 5, 15)), FONT, max(0.4, 0.5 * s), (255, 255, 255), thin)
    if res.lead is not None:
        b = res.lead
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), thick)
        # TTC goes in a fixed corner readout, not on the box: over a small distant box it was unreadable
        cv2.putText(img, f"LEAD id {b.track_id}   {ttc_label(res.ttc)}", (int(10 * hs), int(75 * hs)), FONT, 1.0 * hs, (0, 255, 0), hud_thick)


def lane_info(selector):
    return f"lane {selector.center_offset:+.2f} +-{selector.band_half:.2f}"


def handle_key(key, selector):
    """Returns True to quit. a/d move the lane band, w/s widen/narrow it (a passenger can align it while parked)."""
    if key == ord("q"):
        return True
    if key == ord("a"):
        selector.center_offset -= 0.01
    elif key == ord("d"):
        selector.center_offset += 0.01
    elif key == ord("w"):
        selector.band_half += 0.01
    elif key == ord("s"):
        selector.band_half = max(0.02, selector.band_half - 0.01)
    return False


def run_file(cap, pipeline, writer, args, src_fps, rotate_code):
    """Frames in order. Every `--stride`-th frame goes through the model (stride 3 ~ what the live loop manages
    on the laptop CPU); the frames in between are still shown/saved, with the latest results drawn on them,
    exactly like camera mode. Timestamps come from the saved capture times (--times) or the video container, so
    variable-frame-rate phone videos (where frame_index / fps is wrong) still give correct dt for the Kalman filter."""
    loop_times = deque(maxlen=30)
    frame_idx = 0
    res = None
    fps = 0.0
    last_t = -1.0
    saved_times = None
    if args.times:  # times recorded live by --save-raw: the truth, since the saved video has a fixed nominal frame rate
        with open(args.times) as f:
            saved_times = [float(row["t"]) for row in csv.DictReader(f)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of file
        frame = upright(frame, rotate_code)
        # Time of this frame in seconds: from the saved capture times if given, else from the video container.
        # If that gives nothing usable (not increasing), fall back to one nominal frame after the previous one.
        if saved_times is not None:
            t = saved_times[frame_idx] if frame_idx < len(saved_times) else last_t + 1.0 / src_fps
        else:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= last_t:
            t = last_t + 1.0 / src_fps
        last_t = t

        if frame_idx % args.stride == 0:
            res = pipeline.process(frame, t, frame_idx)
            loop_times.append(time.time())
            if len(loop_times) > 1:
                fps = (len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
            if pipeline.n % 20 == 0:
                print(f"  processed {pipeline.n} (frame {frame_idx}): {fps:.1f} fps, YOLO {statistics.median(pipeline.yolo_ms[-20:]):.0f} ms/frame")
        draw_overlay(frame, res, f"model {fps:.1f} fps (stride {args.stride}) | {lane_info(pipeline.selector)}", pipeline.selector.band())

        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("live", frame)
            if handle_key(cv2.waitKey(1) & 0xFF, pipeline.selector):
                break
        frame_idx += 1
        if args.max_frames and pipeline.n >= args.max_frames:
            break
    return {}


def run_camera(cap, pipeline, writer, args, rotate_code):
    """Display loop at camera speed + inference worker at model speed. Returns stats for the summary."""
    grabber = LatestFrameGrabber(cap, rotate_code)
    t_origin = time.time()      # camera timestamps count from here (after the warm-up)
    worker = InferenceWorker(grabber, pipeline, t_origin, args.max_frames)

    disp_times = deque(maxlen=30)
    last_seq = 0
    shown = 0
    img = None
    raw_writer, raw_times = None, []
    try:
        while not worker.done:
            got = grabber.get(last_seq, timeout=NO_FRAME_TIMEOUT_S)
            if got is None:
                break  # camera stopped
            if got is STALE:
                # Camera alive but silent: never leave a frozen picture on screen without saying so
                if img is not None and args.show:
                    frozen = img.copy()
                    draw_banner(frozen, "NO CAMERA FRAMES")
                    cv2.imshow("live", frozen)
                    if handle_key(cv2.waitKey(1) & 0xFF, pipeline.selector):
                        break
                continue
            frame, t_capture, last_seq = got

            if args.save_raw:
                if raw_writer is None:  # created on the first frame so we know the real size
                    args.save_raw.parent.mkdir(parents=True, exist_ok=True)
                    raw_writer = cv2.VideoWriter(str(args.save_raw.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 30.0, (frame.shape[1], frame.shape[0]))
                raw_writer.write(frame)  # the untouched camera frame, before any drawing
                raw_times.append(t_capture - t_origin)  # same clock the live TTC uses, so a replay matches

            img = frame.copy()  # the worker may still be reading `frame` inside YOLO: never draw on it in place

            res = worker.get_latest()
            disp_times.append(time.time())
            disp_fps = (len(disp_times) - 1) / (disp_times[-1] - disp_times[0]) if len(disp_times) > 1 else 0.0
            info = f"display {disp_fps:.0f} fps | model {worker.model_fps:.1f} fps | {lane_info(pipeline.selector)}"
            stalled = res is not None and (t_capture - t_origin) - res.t > STALE_OVERLAY_S
            if stalled:
                res = None      # the model has not produced anything recent: do not show an old TTC on a new picture
            draw_overlay(img, res, info, pipeline.selector.band())
            if stalled:
                draw_banner(img, "MODEL STALLED")

            if writer:
                writer.write(img)
            if args.show:
                cv2.imshow("live", img)
                if handle_key(cv2.waitKey(1) & 0xFF, pipeline.selector):
                    break
                if shown > 5 and cv2.getWindowProperty("live", cv2.WND_PROP_VISIBLE) < 1:
                    break  # the window was closed with the X button
            shown += 1
    finally:
        worker.stop()
        grabber.stop()
        if raw_writer:
            raw_writer.release()
            times_path = args.save_raw.with_name(args.save_raw.stem + "_times.csv")
            with open(times_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t"])
                w.writerows(enumerate(raw_times))
            print(f"Saved raw feed: {args.save_raw.with_suffix('.mp4')} + {times_path} ({len(raw_times)} frames)")
    if worker.error:
        print(f"\nThe model thread stopped with an error: {worker.error!r}")
    return {"shown": shown, "read": grabber.seq, "age_ms": worker.age_ms, "error": worker.error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")  # file path or camera index
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.4)  # same default as track.py, see NOTES.md
    parser.add_argument("--imgsz", type=int, default=None)  # YOLO input size, the main speed vs accuracy knob on CPU. Default: 640 for cameras, 1280 for files
    parser.add_argument("--stride", type=int, default=1)  # files only: run the model on every Nth frame (3 ~ live speed on CPU)
    parser.add_argument("--width", type=int, default=None)   # cameras only: capture size to request (default 1280x720)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])  # rotate frames if the phone is mounted rotated
    parser.add_argument("--lane-offset", dest="lane_offset", type=float, default=0.0)  # move the lane band left(-)/right(+), fraction of width; also a/d keys
    parser.add_argument("--band", type=float, default=0.10)  # lane band half-width, fraction of frame width; also w/s keys
    parser.add_argument("--save-raw", type=Path, default=None)  # cameras only: save the unannotated feed as NAME.mp4 + NAME_times.csv
    parser.add_argument("--times", type=Path, default=None)     # files only: per-frame capture times saved by --save-raw
    parser.add_argument("--show", action="store_true")  # open a window (needs a display)
    parser.add_argument("--out", type=Path, default=None)  # optionally save the annotated video
    parser.add_argument("--log", type=Path, default=None)  # optionally save time/width/TTC of the lead per processed frame as CSV
    parser.add_argument("--max-frames", type=int, default=None)  # stop early after this many PROCESSED frames, for quick benchmarks
    args = parser.parse_args()

    is_camera = args.source.isdigit()
    if args.imgsz is None:
        args.imgsz = 640 if is_camera else 1280
    if args.save_raw and not args.log:
        args.log = args.save_raw.with_name(args.save_raw.stem + "_live_log.csv")  # a raw drive should always come with its live log
    rotate_code = ROTATE_CODES[args.rotate]

    print("Research prototype, not a safety system. The driver must not use the laptop while driving.")
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source}")
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width or 1280)    # a request, not a promise: the driver may pick another size
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height or 720)
        check_camera(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))               # always read back what we really got
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.rotate in (90, 270):
        width, height = height, width                            # the frames the pipeline sees are rotated
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = reported_fps if reported_fps > 0 else 30.0  # cameras often report 0 or -1 (unknown); `or 30.0` would not catch -1
    print(f"Source: {width}x{height} at {src_fps:.1f} fps, imgsz {args.imgsz}")

    writer = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    model = YOLO(args.model)
    # The very first inference is slow (lazy initialization, ~30 s on the Windows laptop). Run one on a blank
    # frame now so that cost is paid before the camera starts being timed, not on the first live frame.
    print("Warming up the model...")
    model.predict(np.zeros((height, width, 3), dtype=np.uint8), imgsz=args.imgsz, verbose=False)
    pipeline = Pipeline(model, args, width, height)
    if args.show:
        print("Ready. Keys: q quit | a/d move lane band | w/s widen/narrow lane band")

    start = time.time()
    error = None
    stats = {}
    try:
        if is_camera:
            stats = run_camera(cap, pipeline, writer, args, rotate_code)
            error = stats.get("error")
        else:
            stats = run_file(cap, pipeline, writer, args, src_fps, rotate_code)
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t", "track_id", "width", "ttc"])
                w.writerows(pipeline.log_rows)
            print(f"Saved {args.log}  ({len(pipeline.log_rows)} rows)")
    elapsed = time.time() - start

    if pipeline.n:
        print(f"\n{pipeline.n} frames run through the model in {elapsed:.1f}s = {pipeline.n / elapsed:.1f} fps")
        print(f"YOLO + tracker: {statistics.median(pipeline.yolo_ms):.0f} ms/frame (median)")
        if stats.get("age_ms"):
            print(f"Camera: {stats['read']} frames read, {stats['shown']} shown ({stats['shown'] / elapsed:.0f} fps display), "
                  f"{pipeline.n} processed by the model")
            print(f"Frame age when processing started: median {statistics.median(stats['age_ms']):.0f} ms, "
                  f"max {max(stats['age_ms']):.0f} ms")
        print(f"Lead picks (track id -> frames): {pipeline.lead_counts}")
        sel = pipeline.selector
        print(f"Lane band at exit: --lane-offset {sel.center_offset:.2f} --band {sel.band_half:.2f}  (pass these next time to keep it)")
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
