"""
Lead-vehicle TTC + rapid-closing events on a real KITTI sequence run through
our actual detector+tracker.

    lead selection = lead_ttc.LeadSelector
    TTC             = lead_ttc.TTCKalman

Trimmed 2026-09-29 (see NOTES.md): this used to also compute distance,
headway, tailgating, and hard braking, all of which needed KITTI's
oxts/calib (GPS+IMU) -- dropped along with the rest of the GPS-dependent
work in that day's scope cut. What's left (lead-selection + TTC +
rapid-closing) is camera-independent and needs nothing but the video itself.
The dropped code isn't gone: `kitti_ego_motion.py`, `kitti_distance.py`, and
git history still have it, frozen.

Rapid closing (a TIME-based sustained-streak rule, not a frame count -- same
reasoning as lead_ttc.py's SWITCH_TIME/LOST_TIME): ttc < RAPID_CLOSING_TTC_S
for >= MIN_EVENT_S. RAPID_CLOSING_TTC_S = 4.0s is a commonly cited
forward-collision-warning caution threshold, not tuned against labeled
ground truth -- treat it as a reasonable default, not a validated one.

Important honesty note: this uses OUR detector's own lead pick each frame,
not a verified "this actually is a real lead car" ground truth -- KITTI TTC
checks earlier found candidates in these same sequences that turned out to
be parked/merging cars, not steady lead vehicles (see NOTES.md, 2026-09-28,
"Scope" decision).

Usage:
    uv run python kitti_headway_events.py 0020
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kitti_gt_ttc import FRAME_DT_S
from lead_ttc import Box, LeadSelector, TTCKalman

RAPID_CLOSING_TTC_S = 4.0
MIN_EVENT_S = 0.3   # same value used throughout the project for "sustained", see lead_ttc.py
EDGE_MARGIN_PX = 2.0
FPS = 10.0

# Found 2026-09-28 running this on the FULL (not truncated) 0009 sequence: in a busy scene with
# many simultaneously-visible candidate vehicles, LeadSelector's lead can legitimately switch to a
# genuinely different real vehicle every few hundred ms, and EVERY such switch resets TTCKalman's
# state by design (a new track_id might be a different car). For a few frames after any switch, the
# filter is converging from a cold start on immature width measurements, and can report a spuriously
# fast-dropping TTC that has nothing to do with real closing behaviour.
MIN_TRUST_AFTER_SWITCH_S = 1.0


def sustained_below(is_below: pd.Series, min_s=MIN_EVENT_S, dt_s=FRAME_DT_S):
    """True for frames inside a run of is_below==True that lasts >= min_s
    seconds. A single noisy frame shouldn't count as an event."""
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

    tracks = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")
    # frame_w/frame_h only needed by LeadSelector's lane-band logic; read actual image size.
    import cv2
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample_img = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    frame_h, frame_w = sample_img.shape[:2]

    selector = LeadSelector(frame_w, frame_h)
    kalman = TTCKalman(frame_w)

    rows = []
    last_lead_id, lead_switch_time = None, None
    for frame, g in tracks.groupby("frame"):
        t = frame / FPS
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2) for r in g.itertuples()]
        lead = selector.update(boxes, t)

        lead_track_id, ttc_s = None, float("nan")
        if lead is not None:
            if lead.track_id != last_lead_id:
                lead_switch_time = t
                last_lead_id = lead.track_id
            time_since_switch = t - lead_switch_time

            edge_touch = lead.x1 <= EDGE_MARGIN_PX or lead.x2 >= frame_w - EDGE_MARGIN_PX
            _, _, ttc_s = kalman.update(lead.w, t, lead.track_id, truncated=edge_touch)
            lead_track_id = lead.track_id
        else:
            time_since_switch = float("nan")

        rows.append({"frame": frame, "time_s": t, "lead_track_id": lead_track_id,
                     "ttc_s": ttc_s, "time_since_lead_switch_s": time_since_switch})

    out = pd.DataFrame(rows)
    ttc_trustworthy = (out["ttc_s"] < RAPID_CLOSING_TTC_S) & out["ttc_s"].notna() \
        & (out["time_since_lead_switch_s"] >= MIN_TRUST_AFTER_SWITCH_S)
    out["rapid_closing"] = sustained_below(ttc_trustworthy)

    out_csv = f"outputs/kitti_{sequence}_headway_events.csv"
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv} ({len(out)} frames)")

    spans = event_spans(out["rapid_closing"], out["time_s"])
    print(f"\nrapid closing: {len(spans)} event(s)")
    for s, e in spans:
        print(f"  {s:.1f}s - {e:.1f}s  ({e - s:.1f}s)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(out["time_s"], out["ttc_s"], marker=".", ms=3, linewidth=1, color="tab:green")
    ax.axhline(RAPID_CLOSING_TTC_S, color="gray", linestyle="--", linewidth=1,
               label=f"{RAPID_CLOSING_TTC_S}s rapid-closing threshold")
    ax.fill_between(out["time_s"], 0, 30, where=out["rapid_closing"], color="red", alpha=0.2)
    ax.set_ylabel("TTC (s)")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, 30)
    ax.set_title(f"KITTI {sequence}: lead TTC (real detector + tracker output)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_png = f"outputs/kitti_{sequence}_headway_events.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
