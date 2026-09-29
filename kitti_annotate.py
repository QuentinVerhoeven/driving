"""
Annotated-video deliverable: overlay boxes, TTC, distance, headway, and event
flags on a real KITTI drive sequence, combining every piece built this week
(kitti_track.py's tracked boxes, kitti_headway_events.py's per-frame
distance/TTC/headway/events, kitti_speeding.py's per-frame speed/limit).

Why KITTI and not an arbitrary downloaded dashcam clip: the ground-plane
distance model (kitti_distance.py) was fit specifically to KITTI's camera
intrinsics and mounting height. Applying those fitted (v0, C) constants to a
random YouTube dashcam with a different field of view/mounting would produce
distance numbers with no real meaning -- silently wrong, not just imprecise.
Detection/tracking/box-width TTC don't have this problem (they're camera-
independent), but distance/headway/speeding do, so the full-pipeline
annotated video uses KITTI, a real public driving dataset, honestly labeled
as such rather than presented as "my dashcam."

Usage:
    uv run python kitti_annotate.py 0009
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

FPS = 10.0

LEAD_COLOR = (0, 140, 255)     # orange (BGR) -- the selected lead vehicle
OTHER_COLOR = (0, 200, 0)      # green -- every other tracked vehicle
EVENT_COLOR = (0, 0, 255)      # red -- event banner


def draw_text_block(frame, lines, x, y, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    sequence = args.sequence

    tracks = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")
    events = pd.read_csv(f"outputs/kitti_{sequence}_headway_events.csv").set_index("frame")
    speeding = pd.read_csv(f"outputs/kitti_{sequence}_speeding.csv").set_index("frame")

    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    frames = sorted(img_dir.glob("*.png"))
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    out_path = args.outdir / f"kitti_{sequence}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    boxes_by_frame = {frame: g for frame, g in tracks.groupby("frame")}

    for frame_idx, path in enumerate(frames):
        img = cv2.imread(str(path))
        ev = events.loc[frame_idx] if frame_idx in events.index else None
        sp = speeding.loc[frame_idx] if frame_idx in speeding.index else None
        lead_id = ev["lead_track_id"] if ev is not None and pd.notna(ev["lead_track_id"]) else None

        if frame_idx in boxes_by_frame:
            for r in boxes_by_frame[frame_idx].itertuples():
                is_lead = lead_id is not None and r.track_id == int(lead_id)
                color = LEAD_COLOR if is_lead else OTHER_COLOR
                thickness = 3 if is_lead else 1
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
                label = f"LEAD id{r.track_id}" if is_lead else f"id{r.track_id}"
                cv2.putText(img, label, (int(r.x1), int(r.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # top-left status block
        lines = []
        if sp is not None and pd.notna(sp["speed_kph"]):
            limit_str = f"{sp['speed_limit_kph']:.0f}" if pd.notna(sp["speed_limit_kph"]) else "?"
            lines.append(f"ego speed: {sp['speed_kph']:.1f} km/h (limit {limit_str})")
        if ev is not None and pd.notna(ev["distance_m"]):
            lines.append(f"distance to lead: {ev['distance_m']:.1f} m")
        if ev is not None and pd.notna(ev["headway_s"]):
            lines.append(f"headway: {ev['headway_s']:.1f} s")
        if ev is not None and pd.notna(ev["ttc_s"]):
            lines.append(f"TTC: {ev['ttc_s']:.1f} s")
        draw_text_block(img, lines, 10, 25)

        # event banner -- only drawn when something is actually active this frame
        active = []
        if ev is not None:
            if ev.get("tailgating"):
                active.append("TAILGATING")
            if ev.get("rapid_closing"):
                active.append("RAPID CLOSING")
            if ev.get("hard_braking"):
                active.append("HARD BRAKING")
        if sp is not None and sp.get("speeding"):
            active.append("SPEEDING")
        if active:
            banner = " | ".join(active)
            cv2.rectangle(img, (0, h - 35), (w, h), EVENT_COLOR, -1)
            cv2.putText(img, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
