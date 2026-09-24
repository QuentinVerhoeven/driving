"""
    Step 1: detect and track vehicles in a dashcam video

    Outputs:
    <clip>_tracked.mp4 - annotated video with boxes and IDs
    <clip>_tracks.csv - one row per frame and tracked vehicle

    Usage:
    uv run python track.py data/raw/clip1.mov
    uv run python track.py data/raw/clip1.mov --tracker botsort.yaml

"""



import argparse
import time
from pathlib import Path
 
import cv2
import pandas as pd
from ultralytics import YOLO
 
# COCO class IDs we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    #parse arguments from command
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)  # required positional: path to the input clip, e.g. data/raw/clip1.mov
    parser.add_argument("--model", default="yolo26n.pt")  # pretrained YOLO weights; "n" = nano, smallest/fastest, needed for CPU-only inference
    parser.add_argument("--tracker", default="bytetrack.yaml")  # ByteTrack vs BoT-SORT (--tracker botsort.yaml); tested both, near-identical ID-flicker results (see NOTES.md), so kept the faster default
    parser.add_argument("--conf", type=float, default=0.4)  # min detection confidence to keep a box, applied BEFORE the tracker (this filters what ByteTrack even sees). 0.5 caused real gaps in the lead car's own track (up to 26 frames) because it discarded low-confidence detections ByteTrack's own track_low_thresh=0.1 is designed to recover through occlusion with. 0.4 fixed the lead car's continuity (0 gaps > 5 frames on clip1_30s) without reopening the original flicker problem that motivated raising it from 0.3 in the first place (see NOTES.md)
    parser.add_argument("--imgsz", type=int, default=1280)  # resolution YOLO resizes frames to before detecting; source is 4K, so the old default of 640 shrank distant cars to ~20px wide, causing detection dropout/flicker. Raised to 1280 after testing: detection rate on real tracks went 76% -> 85% (see NOTES.md), at the cost of slower CPU inference.
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))  # where the annotated video + CSV get written
    args = parser.parse_args()

    # Read video properties so the output video matches the input
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}, check the path and format")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Input: {width}x{height} at {fps:.1f} fps, {total_frames} total frames")

    #create output files
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    out_video = args.outdir / f"{stem}_tracked.mp4"
    out_csv = args.outdir / f"{stem}_tracks.csv"

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))





    #Now the actual detection and tracking of vehicles
    model = YOLO(args.model) #use the pretrained weights of YOLO
    results = model.track(source = str(args.video), 
        stream = True, #yield one frame at at a time instead of loading everything into memory
        persist=True, # keep ID's consistent accross frames
        tracker=args.tracker, #feeds detections through ByteTrack/BoT-SORT to assign and maintaing track ids
        classes=list(VEHICLE_CLASSES),
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False, #supresses cosole output printing per frame
    )



    # Write to plot
    rows = []
    start = time.time()
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        # boxes.id is None when nothing is being tracked in this frame
        if boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.int().tolist()
            for tid, (x1, y1, x2, y2), conf, cls in zip(ids, xyxy, confs, classes):
                rows.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 3),
                    "track_id": tid,
                    "cls": VEHICLE_CLASSES.get(cls, str(cls)),
                    "conf": round(conf, 3),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                })

        writer.write(r.plot())  # frame with boxes + IDs drawn

        if frame_idx % 50 == 0:
            elapsed = time.time() - start
            speed = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  frame {frame_idx}/{total_frames}  ({speed:.1f} fps processing)")

    writer.release()
    elapsed = time.time() - start


    # save teh track table
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
 
    # quick sanity summary
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Saved {out_video}")
    print(f"Saved {out_csv}  ({len(df)} rows)")

    lengths = df.groupby("track_id").size()
    print(f"\nUnique track IDs:        {len(lengths)}")
    print(f"Median track length:     {lengths.median():.0f} frames ({lengths.median() / fps:.1f}s)")
    print(f"Longest track:           {lengths.max()} frames (ID {lengths.idxmax()})")
    print(f"Tracks shorter than 10 frames: {(lengths < 10).sum()}  "
          "(lots of these = flicker or ID switches)")


if __name__ == "__main__":
    main()