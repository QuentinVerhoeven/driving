"""
Screen all 21 KITTI tracking sequences for good candidate clips, using ONLY
the already-downloaded label_02/oxts/calib data (no new image downloads) --
cheap to run, so it's worth doing before picking which sequences' images to
fetch and build the demo/evaluation around.

Two things this looks for, since they need different things from a clip:
  1. A genuine, clean, sustained lead-vehicle scenario (for a compelling
     annotated-video demo): a single track that stays roughly centered
     laterally (|x| small -- same lane) and heading the same way as the ego
     car (rotation_y near 0 or pi, not perpendicular/parked), for a real
     stretch of time (many seconds), not just a couple of frames.
  2. Real ego-motion "interest" (for event-detection variety): meaningful
     speed variation and/or genuine hard-braking in the oxts log, so the
     evaluation set isn't all quiet residential creeping like sequence 0019.

Usage:
    uv run python kitti_scout.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kitti_gt_ttc import parse_label_file, parse_oxts_file
from kitti_ego_motion import HARD_BRAKE_THRESHOLD

LATERAL_MAX_M = 2.5          # |x| under this = roughly centered in our lane
HEADING_TOL_RAD = 0.35       # rotation_y within this of 0 or pi = aligned with ego's direction of travel
MIN_TRACK_S = 5.0            # only care about tracks visible for at least this long


def best_lead_candidate(labels):
    """Return the best same-lane, aligned-heading, long-lived Car track in one
    sequence's labels, or None. 'Best' = longest duration meeting the filters."""
    cars = labels[(labels["type"] == "Car") & (labels["truncated"] < 0.01) & (labels["occluded"] <= 1)]
    best = None
    for track_id, g in cars.groupby("track_id"):
        aligned = (g["rotation_y"].abs() < HEADING_TOL_RAD) | ((g["rotation_y"].abs() - np.pi).abs() < HEADING_TOL_RAD)
        centered = g["x"].abs() < LATERAL_MAX_M
        good = g[aligned & centered]
        if len(good) == 0:
            continue
        duration_s = (good["frame"].max() - good["frame"].min()) * 0.1
        if duration_s < MIN_TRACK_S:
            continue
        if best is None or duration_s > best["duration_s"]:
            best = {
                "track_id": track_id, "duration_s": duration_s,
                "n_frames": len(good), "z_min": good["z"].min(), "z_max": good["z"].max(),
                "frame_start": int(good["frame"].min()), "frame_end": int(good["frame"].max()),
            }
    return best


def main():
    rows = []
    for seq_dir in sorted(Path("data/kitti/training/label_02").glob("*.txt")):
        seq = seq_dir.stem
        labels = parse_label_file(seq_dir)
        oxts = parse_oxts_file(Path(f"data/kitti/training/oxts/{seq}.txt"))

        lead = best_lead_candidate(labels)
        speed_kph = oxts["vf"] * 3.6
        n_hard_brake_frames = (oxts["af"] < HARD_BRAKE_THRESHOLD).sum()

        rows.append({
            "sequence": seq, "n_frames": len(oxts), "duration_s": len(oxts) * 0.1,
            "speed_min_kph": speed_kph.min(), "speed_max_kph": speed_kph.max(),
            "hard_brake_frames": n_hard_brake_frames,
            "lead_track_id": lead["track_id"] if lead else None,
            "lead_duration_s": lead["duration_s"] if lead else 0.0,
            "lead_z_range_m": f"{lead['z_min']:.0f}-{lead['z_max']:.0f}" if lead else "",
        })

    df = pd.DataFrame(rows).sort_values("lead_duration_s", ascending=False)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    df.to_csv("outputs/kitti_scout.csv", index=False)
    print("\nsaved outputs/kitti_scout.csv")


if __name__ == "__main__":
    main()
