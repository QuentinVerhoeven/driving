"""
Annotated-video deliverable: overlay boxes, the selected lead, TTC, and the
rapid-closing event flag on a real KITTI drive sequence. Combines
kitti_track.py's tracked boxes with kitti_headway_events.py's per-frame
lead/TTC/rapid-closing output.

Trimmed 2026-09-29 (see NOTES.md): this used to also overlay distance,
headway, ego speed/speed-limit, tailgating, and hard braking -- all of which
needed KITTI's oxts/calib (GPS+IMU), dropped in that day's scope cut. What's
left (boxes, lead, TTC, rapid-closing) is camera-independent, so this same
script's approach (and live.py's file mode) now works on any video, KITTI
or not.

A per-vehicle TTC label (NearbyVehicleTTC, lead_ttc.py) was tried on every
box, not just the lead, then reverted the same day at the user's request
(preferred the plainer look) -- back to just the lead's TTC. The class
itself stays in lead_ttc.py, documented, just not wired into this renderer.

TTC urgency tiers (2026-09-30): the lead box and TTC readout change color
and grow as TTC drops (green/safe, yellow/caution under RAPID_CLOSING_TTC_S,
red/urgent under TTC_URGENT_S) -- see lead_ttc.ttc_urgency, shared with
live.py so both renderers escalate at the same thresholds. Purely a
rendering choice, not a new measurement.

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

from lead_ttc import ttc_urgency

FPS = 10.0

LEAD_COLOR = (0, 140, 255)     # orange (BGR) -- the selected lead vehicle
OTHER_COLOR = (0, 200, 0)      # green -- every other tracked vehicle
EVENT_COLOR = (0, 0, 255)      # red -- event banner
# TTC urgency colors (BGR), keyed by lead_ttc.ttc_urgency()'s tier names -- overrides LEAD_COLOR
# once a TTC reading exists, so the box/text visibly escalate as TTC drops.
URGENCY_COLOR = {"safe": (0, 200, 0), "caution": (0, 220, 255), "urgent": (0, 0, 255)}

# How long to keep showing the last known TTC/event flags through a brief
# lead-loss gap, purely a rendering choice (see module docstring) -- found
# 2026-09-28 that ~90% of KITTI 0020's lead-loss gaps are only 1-4 frames
# (0.1-0.4s), and blanking the text for each one made the video look like it
# was constantly flickering even though the underlying detection is fine.
# Doesn't change any CSV/algorithm output, only what's drawn on screen.
GRACE_FRAMES = 5


def draw_text_block(frame, lines, x, y, color=(255, 255, 255), scale=0.6):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


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

    last_good_lines, last_good_active, frames_since_good = [], [], GRACE_FRAMES + 1
    last_good_lead_id = None
    last_good_urgency = "safe"
    last_good_color, last_good_scale = (255, 255, 255), 0.6

    for frame_idx, path in enumerate(frames):
        img = cv2.imread(str(path))
        ev = events.loc[frame_idx] if frame_idx in events.index else None
        lead_id = ev["lead_track_id"] if ev is not None and pd.notna(ev["lead_track_id"]) else None
        lead_ttc = ev["ttc_s"] if ev is not None and pd.notna(ev["ttc_s"]) else None

        # Same grace period as the text below: if the real lead is missing for
        # only a few frames, keep treating its last known track_id as the lead
        # for HIGHLIGHTING purposes too, so the box and the text agree -- a
        # box drawn orange with no TTC text (or vice versa) looks like a bug
        # even when each half is individually correct.
        display_lead_id = lead_id if lead_id is not None else (
            last_good_lead_id if frames_since_good < GRACE_FRAMES else None)
        # Urgency for the CURRENT frame's TTC, if any -- used below to decide whether to update the
        # held-over "last good" urgency, and to color the box even before the grace-period text logic runs.
        current_urgency = ttc_urgency(lead_ttc) if lead_ttc is not None else None
        display_urgency = current_urgency if current_urgency is not None else (
            last_good_urgency if frames_since_good < GRACE_FRAMES else "safe")

        if frame_idx in boxes_by_frame:
            for r in boxes_by_frame[frame_idx].itertuples():
                is_lead = display_lead_id is not None and r.track_id == int(display_lead_id)
                color = URGENCY_COLOR[display_urgency] if is_lead else OTHER_COLOR
                thickness = (3 if display_urgency == "safe" else (4 if display_urgency == "caution" else 5)) if is_lead else 1
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
                label = f"LEAD id{r.track_id}" if is_lead else f"id{r.track_id}"
                cv2.putText(img, label, (int(r.x1), int(r.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        lead_lines = []
        lead_color = (255, 255, 255)
        text_scale = 0.6
        if lead_ttc is not None:
            lead_lines.append(f"TTC: {lead_ttc:.1f} s")
            lead_color = URGENCY_COLOR[current_urgency]
            text_scale = {"safe": 0.6, "caution": 0.7, "urgent": 0.85}[current_urgency]

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
            last_good_urgency = current_urgency
            last_good_color, last_good_scale = lead_color, text_scale
        else:
            frames_since_good += 1
            if frames_since_good <= GRACE_FRAMES:
                lead_lines, active = last_good_lines, last_good_active
                lead_color, text_scale = last_good_color, last_good_scale

        draw_text_block(img, lead_lines, 10, 25, color=lead_color, scale=text_scale)

        if active:
            banner = " | ".join(active)
            cv2.rectangle(img, (0, h - 35), (w, h), EVENT_COLOR, -1)
            cv2.putText(img, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
