"""
Situational awareness: TTC for every tracked vehicle in a clip, not just the
selected lead (see CLAUDE.md, "Pipeline" -- Situational awareness; "Scope").

Takes track.py's output CSV (one row per vehicle per frame) and runs
NearbyVehicleTTC (lead_ttc.py) over it -- the exact same box-width-expansion
method as the lead's TTCKalman, just one filter per track id instead of one
singleton for the chosen lead. This does NOT choose a lead or decide which
vehicle matters most; it just answers "how is each visible vehicle's box
growing," for every vehicle track.py already found.

Usage:
    uv run python nearby_ttc.py outputs/clip1_30s_tracks.csv --frame-w 3840
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lead_ttc import Box, NearbyVehicleTTC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_csv", type=Path)
    parser.add_argument("--frame-w", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=15,
                         help="skip tracks shorter than this many frames when plotting (declutters flicker/noise)")
    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks_csv)
    ttc_calc = NearbyVehicleTTC(args.frame_w)

    rows = []
    for frame, g in tracks.groupby("frame"):
        t = g["time_s"].iloc[0]
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2) for r in g.itertuples()]
        results = ttc_calc.update(boxes, t)
        for track_id, (w_est, rate, ttc) in results.items():
            rows.append({"frame": frame, "time_s": t, "track_id": track_id,
                         "width_est_px": round(w_est, 1), "rate_px_s": round(rate, 1), "ttc_s": ttc})

    out = pd.DataFrame(rows)
    out_csv = args.tracks_csv.with_name(args.tracks_csv.stem.replace("_tracks", "") + "_nearby_ttc.csv")
    out.to_csv(out_csv, index=False)
    print(f"{len(out)} rows across {out['track_id'].nunique()} tracked vehicles")
    print(f"saved {out_csv}")

    fig, ax = plt.subplots(figsize=(10, 5))
    lengths = out.groupby("track_id").size()
    plotted = 0
    for track_id, g in out.groupby("track_id"):
        if lengths[track_id] < args.min_frames:
            continue
        ax.plot(g["time_s"], g["ttc_s"], marker=".", ms=3, linewidth=1, label=f"id {track_id}")
        plotted += 1
    ax.axhline(8, color="gray", linestyle="--", linewidth=1, label="8 s (illustrative)")
    ax.set_ylim(0, 30)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("TTC (s)")
    ax.set_title(f"{args.tracks_csv.stem}: TTC for every tracked vehicle ({plotted} shown, "
                 f"tracks < {args.min_frames} frames hidden)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_png = out_csv.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
