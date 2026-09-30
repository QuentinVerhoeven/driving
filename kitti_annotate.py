"""
Annotated-video deliverable: overlay boxes, the selected lead, TTC (for the
lead AND every other nearby tracked vehicle), and the rapid-closing event
flag on a real KITTI drive sequence. Combines kitti_track.py's tracked boxes
with kitti_headway_events.py's per-frame lead/TTC/rapid-closing output.

Trimmed 2026-09-29 (see NOTES.md): this used to also overlay distance,
headway, ego speed/speed-limit, tailgating, and hard braking -- all of which
needed KITTI's oxts/calib (GPS+IMU), dropped in that day's scope cut. What's
left (boxes, lead, TTC, rapid-closing) is camera-independent, so this same
script's approach (and live.py's file mode) now works on any video, KITTI
or not.

Per-vehicle TTC (2026-09-29, cont.): NearbyVehicleTTC (lead_ttc.py) was built
earlier for exactly this ("situational awareness" -- every visible vehicle's
box-width TTC, not just the lead) but had never actually been wired into a
rendered video. Reused here unchanged, not rebuilt.

Lane-band guide lines removed (2026-09-29, cont.): useful while diagnosing
the band-width bug, but it's diagnostic clutter for a finished demo, not
something a viewer needs to see.

Usage:
    uv run python kitti_annotate.py 0020
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from lead_ttc import Box, NearbyVehicleTTC

FPS = 10.0

LEAD_COLOR = (0, 140, 255)     # orange (BGR) -- the selected lead vehicle
OTHER_COLOR = (0, 200, 0)      # green -- every other tracked vehicle
EVENT_COLOR = (0, 0, 255)      # red -- event banner


def ttc_label(ttc):
    if ttc != ttc:   # nan
        return ""
    return " >10s" if ttc > 10 else f" {ttc:.1f}s"

# How long to keep showing the last known TTC/event flags through a brief
# lead-loss gap, purely a rendering choice (see module docstring) -- found
# 2026-09-28 that ~90% of KITTI 0020's lead-loss gaps are only 1-4 frames
# (0.1-0.4s), and blanking the text for each one made the video look like it
# was constantly flickering even though the underlying detection is fine.
# Doesn't change any CSV/algorithm output, only what's drawn on screen.
GRACE_FRAMES = 5


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

    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    frames = sorted(img_dir.glob("*.png"))
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    out_path = args.outdir / f"kitti_{sequence}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    boxes_by_frame = {frame: g for frame, g in tracks.groupby("frame")}
    nearby_ttc = NearbyVehicleTTC(w)

    last_good_lines, last_good_active, frames_since_good = [], [], GRACE_FRAMES + 1
    last_good_lead_id = None

    for frame_idx, path in enumerate(frames):
        img = cv2.imread(str(path))
        ev = events.loc[frame_idx] if frame_idx in events.index else None
        lead_id = ev["lead_track_id"] if ev is not None and pd.notna(ev["lead_track_id"]) else None

        # Same grace period as the text below: if the real lead is missing for
        # only a few frames, keep treating its last known track_id as the lead
        # for HIGHLIGHTING purposes too, so the box and the text agree -- a
        # box drawn orange with no TTC text (or vice versa) looks like a bug
        # even when each half is individually correct.
        display_lead_id = lead_id if lead_id is not None else (
            last_good_lead_id if frames_since_good < GRACE_FRAMES else None)

        if frame_idx in boxes_by_frame:
            frame_boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2)
                           for r in boxes_by_frame[frame_idx].itertuples()]
            nearby = nearby_ttc.update(frame_boxes, frame_idx / FPS)   # {track_id: (width, rate, ttc)}
            for r in boxes_by_frame[frame_idx].itertuples():
                is_lead = display_lead_id is not None and r.track_id == int(display_lead_id)
                color = LEAD_COLOR if is_lead else OTHER_COLOR
                thickness = 3 if is_lead else 1
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
                _, _, ttc = nearby.get(r.track_id, (None, None, float("nan")))
                label = ("LEAD" if is_lead else f"id{r.track_id}") + ttc_label(ttc)
                cv2.putText(img, label, (int(r.x1), int(r.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        lead_lines = []
        if ev is not None and pd.notna(ev["ttc_s"]):
            lead_lines.append(f"TTC: {ev['ttc_s']:.1f} s")

        active = []
        if ev is not None and ev.get("rapid_closing"):
            active.append("RAPID CLOSING")

        # Grace period: a lead lost for only a few frames (median gap on real
        # KITTI data is ~3 frames = 0.3s -- see NOTES.md, 2026-09-28) keeps
        # showing its last reading instead of blanking, so the text doesn't
        # visibly flicker on/off for a loss too brief to matter. A gap longer
        # than GRACE_FRAMES genuinely blanks -- this never fabricates a
        # reading beyond a short, honest hold-over.
        if lead_lines:
            last_good_lines, last_good_active, frames_since_good = lead_lines, active, 0
            last_good_lead_id = int(lead_id)
        else:
            frames_since_good += 1
            if frames_since_good <= GRACE_FRAMES:
                lead_lines, active = last_good_lines, last_good_active

        draw_text_block(img, lead_lines, 10, 25)

        if active:
            banner = " | ".join(active)
            cv2.rectangle(img, (0, h - 35), (w, h), EVENT_COLOR, -1)
            cv2.putText(img, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
