"""
FROZEN 2026-09-29: dropped in that day's scope cut (GPS/calibration-dependent
features + quantitative evaluation removed as deliverables -- see NOTES.md).
Kept in the repo as real, completed, honestly-documented work, not deleted.
Not run or extended further; nothing downstream depends on it anymore.

Week 3, step 1: ego speed and hard-braking from KITTI's oxts log.

No more phone GPS/accelerometer logging (see CLAUDE.md, NOTES.md 2026-09-28)
-- this uses KITTI's oxts file instead, which is a real GPS+IMU log, one row
per frame, already parsed by kitti_gt_ttc.parse_oxts_file(). Two columns do
all the work here:
  - vf: forward velocity in the vehicle's own frame (m/s) -- this IS ego
    speed, no combining vn/ve or converting from lat/lon needed.
  - af: forward acceleration (m/s^2) -- negative means braking. This is a
    real IMU reading, not a derivative of a noisy GPS speed, so it's a
    cleaner braking signal than differencing vf would be.

Hard-braking event: af below HARD_BRAKE_THRESHOLD for at least
HARD_BRAKE_MIN_S seconds -- a time threshold (not a frame count), same
reasoning as lead_ttc.py's SWITCH_TIME/LOST_TIME: a single noisy frame
shouldn't count as an event.

Usage:
    uv run python kitti_ego_motion.py 0019
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from kitti_gt_ttc import parse_oxts_file, FRAME_DT_S

# Braking harder than -3 m/s^2 (~0.3 g) sustained is a commonly used threshold
# for a real "hard braking" event in driving-behavior literature -- gentle
# everyday slowing is usually under -2 m/s^2.
HARD_BRAKE_THRESHOLD = -3.0
HARD_BRAKE_MIN_S = 0.3


def find_events(df, threshold=HARD_BRAKE_THRESHOLD, min_s=HARD_BRAKE_MIN_S):
    """Return a boolean column: True for frames inside a hard-braking event
    that lasted at least min_s seconds. Built the same way as lead_ttc.py's
    streak logic: mark frames under threshold, then only keep streaks long
    enough, so one noisy frame under threshold doesn't count on its own.
    """
    below = df["af"] < threshold
    # group consecutive True/False runs
    group = (below != below.shift()).cumsum()
    event = df.groupby(group)["frame"].transform(lambda f: len(f) * FRAME_DT_S >= min_s) & below
    return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    args = parser.parse_args()

    oxts_path = Path(f"data/kitti/training/oxts/{args.sequence}.txt")
    df = parse_oxts_file(oxts_path)
    df["time_s"] = df["frame"] * FRAME_DT_S
    df["speed_kph"] = df["vf"] * 3.6
    df["hard_braking"] = find_events(df)

    out_csv = Path(f"outputs/kitti_{args.sequence}_ego_motion.csv")
    df[["frame", "time_s", "vf", "speed_kph", "af", "hard_braking"]].to_csv(out_csv, index=False)
    n_events = (df["hard_braking"] & ~df["hard_braking"].shift(1, fill_value=False)).sum()
    print(f"{len(df)} frames, {df['time_s'].iloc[-1]:.1f} s total")
    print(f"speed: {df['speed_kph'].min():.1f}-{df['speed_kph'].max():.1f} km/h")
    print(f"{n_events} hard-braking event(s) (af < {HARD_BRAKE_THRESHOLD} m/s^2 for >= {HARD_BRAKE_MIN_S}s)")
    print(f"saved {out_csv}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(df["time_s"], df["speed_kph"])
    ax1.set_ylabel("ego speed (km/h)")
    ax1.set_title(f"KITTI {args.sequence}: ego speed & forward acceleration (oxts)")

    ax2.plot(df["time_s"], df["af"], color="tab:orange")
    ax2.axhline(HARD_BRAKE_THRESHOLD, color="gray", linestyle="--", linewidth=1,
                label=f"hard-brake threshold ({HARD_BRAKE_THRESHOLD} m/s²)")
    ax2.fill_between(df["time_s"], ax2.get_ylim()[0], ax2.get_ylim()[1],
                      where=df["hard_braking"], color="red", alpha=0.2, label="hard braking")
    ax2.set_ylabel("forward accel (m/s²)")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_png = out_csv.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
