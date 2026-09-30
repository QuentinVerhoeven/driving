"""
FROZEN 2026-09-29: dropped in that day's scope cut (GPS/calibration-dependent
features + quantitative evaluation removed as deliverables -- see NOTES.md).
Kept in the repo as real, completed, honestly-documented work, not deleted.
Not run or extended further; nothing downstream depends on it anymore.

Week 4: speeding detection -- ego GPS speed vs. OpenStreetMap speed-limit tags.

Only runs on KITTI: it's the only source in this project with real GPS
coordinates (see CLAUDE.md, "Not doing" -- no more phone GPS logging, and a
downloaded/YouTube-style dashcam clip has no GPS at all to check against a
speed limit in the first place).

Method:
  1. Download the real road network around the sequence's GPS trace once
     (a single Overpass API query via osmnx), not per-point -- OSM data is
     the same regardless of which frame is asking.
  2. Match every frame's (lat, lon) to its nearest road edge
     (ox.distance.nearest_edges -- one vectorized call for the whole trace).
  3. Read that edge's `maxspeed` tag, if it has one. Many minor roads are
     untagged in OSM, and some tags aren't a plain number (e.g. a list of
     directional limits, or a non-numeric zone code) -- report coverage
     honestly (see CLAUDE.md: "many minor roads are untagged, skip them
     rather than guess") instead of assuming a default limit.
  4. Compare ego speed (oxts vf, converted to km/h) against the matched
     limit. A "speeding" event is SUSTAINED excess -- above the limit by
     more than SPEEDING_MARGIN_KPH for at least MIN_EVENT_S -- not a single
     noisy GPS sample, same time-based streak reasoning used everywhere
     else in this project (lead_ttc.py, kitti_ego_motion.py,
     kitti_headway_events.py).

Usage:
    uv run python kitti_speeding.py 0019
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

from kitti_gt_ttc import FRAME_DT_S, parse_oxts_file

SPEEDING_MARGIN_KPH = 5.0   # allow some GPS/limit noise before calling it "speeding"
MIN_EVENT_S = 0.3           # same "sustained, not single-frame" rule used throughout this project
BBOX_MARGIN_DEG = 0.003     # ~300m padding around the trace, so nearest-edge matching works at the path's endpoints too


def parse_maxspeed(tag):
    """OSM's maxspeed tag is sometimes a plain number, sometimes a list
    (different directions of a road can carry different tags), and
    sometimes not a number at all (e.g. a country zone code). Returns an
    int km/h, or None if it can't be confidently parsed -- callers must
    treat None as "no reading," not "no limit" (see module docstring)."""
    if tag is None:
        return None
    if isinstance(tag, list):
        # Take the lowest of multiple directional tags -- the conservative reading.
        parsed = [parse_maxspeed(t) for t in tag]
        parsed = [p for p in parsed if p is not None]
        return min(parsed) if parsed else None
    match = re.match(r"^(\d+)", str(tag))
    return int(match.group(1)) if match else None


def sustained_above(is_above: pd.Series, min_s=MIN_EVENT_S, dt_s=FRAME_DT_S):
    """Same streak-grouping rule as kitti_headway_events.sustained_below,
    just for 'above threshold' instead of 'below' -- kept local since this
    is the only script needing the 'above' direction."""
    group = (is_above != is_above.shift()).cumsum()
    long_enough = is_above.groupby(group).transform(lambda g: len(g) * dt_s >= min_s)
    return long_enough & is_above


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    args = parser.parse_args()
    sequence = args.sequence

    oxts = parse_oxts_file(Path(f"data/kitti/training/oxts/{sequence}.txt"))
    oxts["time_s"] = oxts["frame"] * FRAME_DT_S
    oxts["speed_kph"] = oxts["vf"] * 3.6

    lat_min, lat_max = oxts["lat"].min() - BBOX_MARGIN_DEG, oxts["lat"].max() + BBOX_MARGIN_DEG
    lon_min, lon_max = oxts["lon"].min() - BBOX_MARGIN_DEG, oxts["lon"].max() + BBOX_MARGIN_DEG
    print(f"Downloading OSM road network for sequence {sequence}'s area "
          f"(lat {lat_min:.4f}-{lat_max:.4f}, lon {lon_min:.4f}-{lon_max:.4f})...")
    G = ox.graph_from_bbox((lon_min, lat_min, lon_max, lat_max), network_type="drive")
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    edges = ox.distance.nearest_edges(G, oxts["lon"].values, oxts["lat"].values)
    limits = []
    for u, v, key in edges:
        tag = G.get_edge_data(u, v, key).get("maxspeed")
        limits.append(parse_maxspeed(tag))
    oxts["speed_limit_kph"] = limits

    n_total = len(oxts)
    n_tagged = oxts["speed_limit_kph"].notna().sum()
    print(f"\nSpeed-limit tag coverage: {n_tagged}/{n_total} frames matched to a road with a "
          f"parseable maxspeed tag ({100 * n_tagged / n_total:.0f}%)")
    print("Frames without a usable tag are excluded from speeding detection below, not assumed compliant.")

    oxts["is_speeding_sample"] = (
        oxts["speed_limit_kph"].notna()
        & (oxts["speed_kph"] > oxts["speed_limit_kph"] + SPEEDING_MARGIN_KPH)
    )
    oxts["speeding"] = sustained_above(oxts["is_speeding_sample"])

    out_csv = f"outputs/kitti_{sequence}_speeding.csv"
    oxts[["frame", "time_s", "lat", "lon", "speed_kph", "speed_limit_kph", "speeding"]].to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    n_events = (oxts["speeding"] & ~oxts["speeding"].shift(1, fill_value=False)).sum()
    print(f"{n_events} speeding event(s) (>{SPEEDING_MARGIN_KPH} km/h over the matched limit, sustained >= {MIN_EVENT_S}s)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(oxts["time_s"], oxts["speed_kph"], label="ego speed (GPS)")
    ax.plot(oxts["time_s"], oxts["speed_limit_kph"], color="gray", linestyle="--", drawstyle="steps-post",
            label="matched speed limit (OSM)")
    ax.fill_between(oxts["time_s"], 0, oxts["speed_kph"].max() * 1.1, where=oxts["speeding"],
                     color="red", alpha=0.2, label="speeding (sustained)")
    untagged = oxts["speed_limit_kph"].isna()
    ax.fill_between(oxts["time_s"], 0, oxts["speed_kph"].max() * 1.1, where=untagged,
                     color="gray", alpha=0.08, label="no speed-limit tag")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title(f"KITTI {sequence}: ego speed vs. OSM speed limit "
                 f"({100 * n_tagged / n_total:.0f}% tag coverage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = f"outputs/kitti_{sequence}_speeding.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
