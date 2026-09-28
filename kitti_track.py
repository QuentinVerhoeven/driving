"""
Step 1 of the real KITTI comparison: run our actual YOLO + tracker on a KITTI
image sequence and save a tracks CSV, same format as track.py, plus an
annotated video to check by eye that it's actually seeing the car we care
about before anything downstream trusts it.

track.py assumes a single video file (it reads fps/width/height with
cv2.VideoCapture). KITTI gives us a folder of numbered PNGs instead, so this
is a separate small script rather than bending track.py to a case it wasn't
built for.

Frames are read and fed to the model ONE AT A TIME (model.track(frame,
persist=True, ...) on a single image array), the same pattern live.py already
uses successfully, rather than handing Ultralytics the whole list of image
paths as `source`. That first approach (source=[list of 260+ paths],
stream=True) reliably OOM-killed this 7.4 GB machine on the very first
result -- RSS jumped from ~290 MB straight to 6.6+ GB (confirmed in dmesg)
before a single frame was even processed, so something in Ultralytics
eagerly allocates relative to the *length of the list*, not per-frame. Rather
than chase that down, feeding one frame at a time avoids it entirely and
matches code we already know works.

Usage:
    uv run python kitti_track.py 0019
"""

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FPS = 10.0   # KITTI tracking sequences are 10 Hz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")  # e.g. "0019"
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.4)   # same default as track.py, see NOTES.md
    parser.add_argument("--imgsz", type=int, default=1280)   # same default as track.py
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    # Restrict to a frame window if useful (e.g. just the stretch around one labeled track); not needed
    # for the OOM avoidance anymore (see docstring), just handy for a quick/targeted run.
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    img_dir = Path(f"data/kitti/training/image_02/{args.sequence}")
    frames = sorted(img_dir.glob("*.png"))[args.start:args.end]
    if not frames:
        raise SystemExit(f"No frames found in {img_dir}")

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    print(f"Sequence {args.sequence}: {len(frames)} frames, {width}x{height} at {FPS:.0f} fps (assumed)", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_video = args.outdir / f"kitti_{args.sequence}_tracked.mp4"
    out_csv = args.outdir / f"kitti_{args.sequence}_tracks.csv"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

    model = YOLO(args.model)

    rows = []
    start = time.time()
    for i, path in enumerate(frames):
        frame_idx = args.start + i   # real KITTI frame number, so this lines up with the label files
        frame = cv2.imread(str(path))

        r = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=list(VEHICLE_CLASSES),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]   # track() returns a list with one Results per input image

        boxes = r.boxes
        if boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.int().tolist()
            for tid, (x1, y1, x2, y2), conf, cls in zip(ids, xyxy, confs, classes):
                rows.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / FPS, 3),
                    "track_id": tid,
                    "cls": VEHICLE_CLASSES.get(cls, str(cls)),
                    "conf": round(conf, 3),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                })
        writer.write(r.plot())

        if i % 50 == 0:
            elapsed = time.time() - start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  frame {i}/{len(frames)} (KITTI frame {frame_idx})  ({speed:.1f} fps processing)", flush=True)

    writer.release()
    elapsed = time.time() - start

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Saved {out_video}")
    print(f"Saved {out_csv}  ({len(df)} rows)")

    lengths = df.groupby("track_id").size()
    print(f"\nUnique track IDs:    {len(lengths)}")
    print(f"Longest track:       {lengths.max()} frames (ID {lengths.idxmax()})")


if __name__ == "__main__":
    main()
