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

from lead_ttc import BAND_HALF_WIDTH

FPS = 10.0

LEAD_COLOR = (0, 140, 255)     # orange (BGR) -- the selected lead vehicle
OTHER_COLOR = (0, 200, 0)      # green -- every other tracked vehicle
EVENT_COLOR = (0, 0, 255)      # red -- event banner
BAND_COLOR = (0, 255, 255)     # yellow -- lane-band guide lines

# How long to keep showing the last known status text/event flags through a
# brief lead-loss gap, purely a rendering choice (see module docstring) --
# found 2026-09-28 that ~90% of KITTI 0020's lead-loss gaps are only 1-4
# frames (0.1-0.4s), and blanking the text for each one made the video look
# like it was constantly flickering even though the underlying detection is
# fine. Doesn't change any CSV/algorithm output, only what's drawn on screen.
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
    speeding = pd.read_csv(f"outputs/kitti_{sequence}_speeding.csv").set_index("frame")

    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    frames = sorted(img_dir.glob("*.png"))
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    out_path = args.outdir / f"kitti_{sequence}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    boxes_by_frame = {frame: g for frame, g in tracks.groupby("frame")}

    # Lane-band guide lines, in pixels -- same band LeadSelector actually uses
    # (BAND_HALF_WIDTH, center_offset=0.0, matching kitti_headway_events.py's
    # LeadSelector(frame_w, frame_h) call). Drawn so a viewer can see WHY a
    # given car is or isn't eligible to be the lead, instead of it looking
    # arbitrary -- see NOTES.md, 2026-09-28 (a car that looks like "the real
    # lead" to a human can be in an adjacent lane, just outside this band).
    band_lo = int((0.5 - BAND_HALF_WIDTH) * w)
    band_hi = int((0.5 + BAND_HALF_WIDTH) * w)

    last_good_lines, last_good_active, frames_since_good = [], [], GRACE_FRAMES + 1
    last_good_lead_id = None

    for frame_idx, path in enumerate(frames):
        img = cv2.imread(str(path))
        ev = events.loc[frame_idx] if frame_idx in events.index else None
        sp = speeding.loc[frame_idx] if frame_idx in speeding.index else None
        lead_id = ev["lead_track_id"] if ev is not None and pd.notna(ev["lead_track_id"]) else None

        # Same grace period as the text below: if the real lead is missing for
        # only a few frames, keep treating its last known track_id as the lead
        # for HIGHLIGHTING purposes too, so the box and the text agree -- a
        # box drawn orange with no "distance to lead" text (or vice versa)
        # looks like a bug even when each half is individually correct.
        display_lead_id = lead_id if lead_id is not None else (
            last_good_lead_id if frames_since_good < GRACE_FRAMES else None)

        overlay = img.copy()
        cv2.rectangle(overlay, (band_lo, 0), (band_hi, h), BAND_COLOR, -1)
        cv2.addWeighted(overlay, 0.06, img, 0.94, 0, img)
        cv2.line(img, (band_lo, 0), (band_lo, h), BAND_COLOR, 1)
        cv2.line(img, (band_hi, 0), (band_hi, h), BAND_COLOR, 1)

        if frame_idx in boxes_by_frame:
            for r in boxes_by_frame[frame_idx].itertuples():
                is_lead = display_lead_id is not None and r.track_id == int(display_lead_id)
                color = LEAD_COLOR if is_lead else OTHER_COLOR
                thickness = 3 if is_lead else 1
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
                label = f"LEAD id{r.track_id}" if is_lead else f"id{r.track_id}"
                cv2.putText(img, label, (int(r.x1), int(r.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # top-left status block. The ego-speed line is independent of whether
        # a lead is currently tracked, so it's kept separate from the
        # lead-derived lines below -- otherwise it would always count as "a
        # reading present" and defeat the grace-period hold below (found
        # while checking a real gap frame: the speed line alone was silently
        # replacing the held-over distance/headway/TTC instead of preserving
        # them).
        speed_line = []
        if sp is not None and pd.notna(sp["speed_kph"]):
            limit_str = f"{sp['speed_limit_kph']:.0f}" if pd.notna(sp["speed_limit_kph"]) else "?"
            speed_line = [f"ego speed: {sp['speed_kph']:.1f} km/h (limit {limit_str})"]

        lead_lines = []
        if ev is not None and pd.notna(ev["distance_m"]):
            lead_lines.append(f"distance to lead: {ev['distance_m']:.1f} m")
        if ev is not None and pd.notna(ev["headway_s"]):
            lead_lines.append(f"headway: {ev['headway_s']:.1f} s")
        if ev is not None and pd.notna(ev["ttc_s"]):
            lead_lines.append(f"TTC: {ev['ttc_s']:.1f} s")

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

        draw_text_block(img, speed_line + lead_lines, 10, 25)

        if active:
            banner = " | ".join(active)
            cv2.rectangle(img, (0, h - 35), (w, h), EVENT_COLOR, -1)
            cv2.putText(img, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
