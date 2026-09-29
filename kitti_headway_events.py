"""
Week 3, step 3: headway and core events, combining every piece built so far
on a real KITTI sequence run through our actual detector+tracker.

    headway (s)   = distance to lead / ego speed
    distance      = kitti_distance.py's fitted ground-plane model, applied to
                    OUR detector's chosen lead box (not KITTI's ground-truth
                    box) -- reuses the (v0, C) already fit and saved by
                    kitti_distance.py, does NOT refit here.
    lead selection = lead_ttc.LeadSelector, same class used for TTC all along.
    TTC             = lead_ttc.TTCKalman, same as kitti_pipeline_compare.py.
    ego speed       = kitti_ego_motion.py's oxts vf (forward ego speed).

Events (all use a TIME-based sustained-streak rule, not a frame count -- same
reasoning as lead_ttc.py's SWITCH_TIME/LOST_TIME and kitti_ego_motion.py's
hard-braking rule: a single noisy frame shouldn't count as an event):
  - tailgating:     headway < TAILGATING_HEADWAY_S for >= MIN_EVENT_S
  - rapid closing:  ttc < RAPID_CLOSING_TTC_S (and TTC is defined, i.e. lead is
                     actually closing) for >= MIN_EVENT_S
  - hard braking:   reused directly from kitti_ego_motion.find_events (af
                     threshold), no lead vehicle needed for this one.

Threshold values (TAILGATING_HEADWAY_S = 2.0s, RAPID_CLOSING_TTC_S = 4.0s) are
commonly cited driving-safety rules of thumb (the "2-second rule" for
following distance; ~4s is a typical forward-collision-warning caution
threshold), not values tuned against labeled ground truth -- that tuning is
Week 6-8's job (event detection precision/recall on hand-labeled clips, per
CLAUDE.md's evaluation plan). Treat these as reasonable defaults to look at
results with, not a validated threshold yet.

Important honesty note: this uses OUR detector's own lead pick each frame,
not a verified "this actually is a real lead car" ground truth -- earlier
today's KITTI TTC check found candidates in these same sequences that turned
out to be parked/merging cars, not steady lead vehicles (see NOTES.md,
2026-09-28, "Scope" decision). This script is testing whether the headway/
event MATH is correctly wired end to end, not re-litigating lead-selection
accuracy, which is tracked separately.

Usage:
    uv run python kitti_headway_events.py 0019
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kitti_ego_motion import find_events as find_hard_braking
from kitti_gt_ttc import FRAME_DT_S, parse_oxts_file
from lead_ttc import Box, LeadSelector, TTCKalman

TAILGATING_HEADWAY_S = 2.0
RAPID_CLOSING_TTC_S = 4.0
MIN_EVENT_S = 0.3   # same value used throughout the project for "sustained", see lead_ttc.py
EDGE_MARGIN_PX = 2.0
FPS = 10.0


def sustained_below(is_below: pd.Series, min_s=MIN_EVENT_S, dt_s=FRAME_DT_S):
    """True for frames inside a run of is_below==True that lasts >= min_s
    seconds. Same streak-grouping approach as kitti_ego_motion.find_events,
    generalized so both headway and TTC events can reuse it."""
    group = (is_below != is_below.shift()).cumsum()
    long_enough = is_below.groupby(group).transform(lambda g: len(g) * dt_s >= min_s)
    return long_enough & is_below


def event_spans(flag: pd.Series, time_s: pd.Series):
    """Collapse a per-frame boolean flag into a list of (start_s, end_s) spans,
    for a short human-readable summary instead of a wall of per-frame rows."""
    spans = []
    start = None
    for on, t in zip(flag, time_s):
        if on and start is None:
            start = t
        elif not on and start is not None:
            spans.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None:
        spans.append((start, prev_t))
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    args = parser.parse_args()
    sequence = args.sequence

    # --- geometry: reuse the already-fit ground-plane parameters, don't refit ---
    fit_params = pd.read_csv("outputs/kitti_distance_fit_params.csv").iloc[0]
    v0, C = fit_params["v0_px"], fit_params["C_px_m"]
    print(f"Using ground-plane fit from {fit_params['fit_sequence']}: v0={v0:.1f}px, C={C:.1f}px*m")

    # --- ego motion timeline: the FULL sequence, every frame, independent of
    # whether a lead vehicle was even visible that frame. Hard braking is a
    # property of the ego car, not of what our detector saw -- computing it
    # only over frames where a lead happened to be detected would silently
    # corrupt event durations whenever a frame's detection was missed. ---
    oxts = parse_oxts_file(Path(f"data/kitti/training/oxts/{sequence}.txt"))
    ego = oxts[["frame", "vf", "af"]].rename(columns={"vf": "ego_speed_mps", "af": "af_mps2"}).copy()
    ego["time_s"] = ego["frame"] * FRAME_DT_S
    ego["hard_braking"] = find_hard_braking(ego.rename(columns={"af_mps2": "af"}))

    # --- our detector's real tracked boxes: only covers frames where SOMETHING
    # was detected, so this is joined onto the full ego timeline below rather
    # than driving the timeline itself. ---
    tracks = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")
    # frame_w/frame_h only needed by LeadSelector's lane-band logic; read actual image size.
    import cv2
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample_img = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    frame_h, frame_w = sample_img.shape[:2]

    selector = LeadSelector(frame_w, frame_h)
    kalman = TTCKalman(frame_w)

    lead_rows = []
    for frame, g in tracks.groupby("frame"):
        t = frame / FPS
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2) for r in g.itertuples()]
        lead = selector.update(boxes, t)
        if lead is None:
            continue

        edge_touch = lead.x1 <= EDGE_MARGIN_PX or lead.x2 >= frame_w - EDGE_MARGIN_PX
        _, _, ttc_s = kalman.update(lead.w, t, lead.track_id, truncated=edge_touch)
        distance_m = C / (lead.y2 - v0) if lead.y2 > v0 else float("nan")
        # guard: lead.y2 > v0 required by the model (a box bottom AT or ABOVE
        # the fitted horizon row isn't geometrically valid -- it would imply
        # the object is at or beyond infinite distance).

        lead_rows.append({"frame": frame, "lead_track_id": lead.track_id,
                           "distance_m": distance_m, "ttc_s": ttc_s})

    lead_df = pd.DataFrame(lead_rows)

    # Left join onto the full ego timeline: frames with no lead simply get NaN
    # distance/TTC (honestly "no reading," not "safe" or zero).
    out = ego.merge(lead_df, on="frame", how="left")
    out["headway_s"] = out["distance_m"] / out["ego_speed_mps"]
    out.loc[out["ego_speed_mps"] <= 0.5, "headway_s"] = float("nan")
    # ego_speed <= 0.5 m/s guard: dividing by near-zero ego speed (stopped/crawling) would blow
    # headway up to meaningless huge numbers, not a real "safe" reading.

    out["tailgating"] = sustained_below(out["headway_s"] < TAILGATING_HEADWAY_S)
    out["rapid_closing"] = sustained_below((out["ttc_s"] < RAPID_CLOSING_TTC_S) & out["ttc_s"].notna())

    out_csv = f"outputs/kitti_{sequence}_headway_events.csv"
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv} ({len(out)} frames)")

    for name, col in [("tailgating", "tailgating"), ("rapid closing", "rapid_closing"), ("hard braking", "hard_braking")]:
        spans = event_spans(out[col], out["time_s"])
        print(f"\n{name}: {len(spans)} event(s)")
        for s, e in spans:
            print(f"  {s:.1f}s - {e:.1f}s  ({e - s:.1f}s)")

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(out["time_s"], out["headway_s"], marker=".", ms=3, linewidth=1)
    axes[0].axhline(TAILGATING_HEADWAY_S, color="gray", linestyle="--", linewidth=1,
                     label=f"{TAILGATING_HEADWAY_S}s tailgating threshold")
    axes[0].fill_between(out["time_s"], 0, out["headway_s"].max(skipna=True) or 1,
                          where=out["tailgating"], color="red", alpha=0.2)
    axes[0].set_ylabel("headway (s)")
    axes[0].set_ylim(0, 10)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"KITTI {sequence}: headway, TTC, hard braking (real detector + tracker output)")

    axes[1].plot(out["time_s"], out["ttc_s"], marker=".", ms=3, linewidth=1, color="tab:green")
    axes[1].axhline(RAPID_CLOSING_TTC_S, color="gray", linestyle="--", linewidth=1,
                     label=f"{RAPID_CLOSING_TTC_S}s rapid-closing threshold")
    axes[1].fill_between(out["time_s"], 0, 30, where=out["rapid_closing"], color="red", alpha=0.2)
    axes[1].set_ylabel("TTC (s)")
    axes[1].set_ylim(0, 30)
    axes[1].legend(fontsize=8)

    axes[2].plot(out["time_s"], out["af_mps2"], color="tab:orange")
    axes[2].fill_between(out["time_s"], out["af_mps2"].min(), out["af_mps2"].max(),
                          where=out["hard_braking"], color="red", alpha=0.2)
    axes[2].set_ylabel("forward accel (m/s²)")
    axes[2].set_xlabel("time (s)")

    fig.tight_layout()
    out_png = f"outputs/kitti_{sequence}_headway_events.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
