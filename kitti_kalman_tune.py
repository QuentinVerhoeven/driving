"""
Tune TTCKalman's q (process noise / reactivity) against KITTI ground truth,
instead of leaving it at its original, never-validated value.

--- Why this came up ---
Watching the autobahn demo clip, a real near-collision's TTC never dropped
below 1.27s in the rendered (Kalman-smoothed) video -- but the RAW,
unfiltered box-width signal for the same car genuinely touched 0.72-0.82s
several times in the same stretch (checked directly, not assumed). The
filter was smoothing away a real fast event, not just noise. TTCKalman's own
docstring already said q/r_meas were "NOT tuned against ground truth yet" --
this is the first time that untuned state actually cost real accuracy on a
genuine fast-closing event, not just a theoretical gap.

--- Method ---
For a grid of candidate q values, run TTCKalman (unchanged class, just a
different q) on every real "Car" track across all 3 available KITTI
sequences (0019, 0009, 0020), using labeled ground-truth 2D boxes (perfect
detection, so this isolates the FILTER's behavior from detector noise --
same reasoning as kitti_label_ttc_check.py) and compare against the exact
3D-derived ground-truth TTC (kitti_gt_ttc.compute_gt_ttc). Two things are
measured, since a more reactive filter is not free:
  - fast-closing accuracy: median |error| on frames where gt_ttc < 3s (the
    regime the autobahn near-miss falls into) -- this is what we want a
    higher q to improve.
  - overall accuracy: median |error| on ALL frames, including slow/stable
    ones -- this is what a too-high q would hurt (more sensitive to
    ordinary per-frame width jitter, not just real fast changes).
Truncated boxes (touching the frame edge) are excluded from error scoring,
same reasoning as kitti_label_ttc_check.py -- box-width TTC is known to
break there regardless of q, see NOTES.md.

Usage:
    uv run python kitti_kalman_tune.py
"""

from pathlib import Path

import cv2
import pandas as pd

from kitti_gt_ttc import compute_gt_ttc, parse_label_file
from lead_ttc import TTCKalman

SEQUENCES = ["0019", "0009", "0020"]
CANDIDATE_Q = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
EDGE_MARGIN_PX = 2.0
FAST_CLOSING_GT_TTC_S = 3.0   # the regime the autobahn near-miss falls into


def frame_size(sequence):
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    return sample.shape[1]   # width only -- TTCKalman only needs frame_w


def run_one_sequence(sequence, q):
    frame_w = frame_size(sequence)
    labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    cars = labels[labels["type"] == "Car"].copy()
    gt = compute_gt_ttc(cars)

    rows = []
    for track_id, g in gt.groupby("track_id"):
        kalman = TTCKalman(frame_w, q=q)
        for row in g.sort_values("frame").itertuples():
            bbox_w = row.bbox_right - row.bbox_left
            t = row.frame * 0.1
            edge = row.bbox_left <= EDGE_MARGIN_PX or row.bbox_right >= frame_w - EDGE_MARGIN_PX
            _, _, ttc_kalman = kalman.update(bbox_w, t, track_id, truncated=edge)
            rows.append({"track_id": track_id, "frame": row.frame, "gt_ttc_s": row.gt_ttc_s,
                         "kalman_ttc_s": ttc_kalman, "truncated": edge})
    return pd.DataFrame(rows)


def main():
    results = []
    for q in CANDIDATE_Q:
        all_rows = pd.concat([run_one_sequence(seq, q) for seq in SEQUENCES], ignore_index=True)
        clean = all_rows[~all_rows["truncated"]].dropna(subset=["gt_ttc_s", "kalman_ttc_s"])

        err_all = (clean["kalman_ttc_s"] - clean["gt_ttc_s"]).abs()
        fast = clean[clean["gt_ttc_s"] < FAST_CLOSING_GT_TTC_S]
        err_fast = (fast["kalman_ttc_s"] - fast["gt_ttc_s"]).abs()

        results.append({
            "q": q,
            "n_all": len(clean), "median_err_all": err_all.median(),
            "n_fast": len(fast), "median_err_fast": err_fast.median(),
        })

    out = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    print(out.to_string(index=False))
    out.to_csv("outputs/kitti_kalman_tune.csv", index=False)
    print("\nsaved outputs/kitti_kalman_tune.csv")


if __name__ == "__main__":
    main()
