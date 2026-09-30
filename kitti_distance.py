"""
FROZEN 2026-09-29: dropped in that day's scope cut (GPS/calibration-dependent
features + quantitative evaluation removed as deliverables -- see NOTES.md).
Kept in the repo as real, completed, honestly-documented work, not deleted --
same treatment live.py's camera mode got earlier in the project. Not run or
extended further; nothing downstream depends on it anymore.

Week 3, step 2: distance from ground-plane geometry, fit and validated against
KITTI's LiDAR-derived ground truth (see CLAUDE.md's evaluation plan: "distance
accuracy on KITTI (LiDAR ground truth), reported by range").

--- The model ---
For a camera at height H above a flat ground plane, with its optical axis
roughly parallel to the ground, similar triangles relate an object's
ground-contact row in the image (v = the bottom edge of its bounding box) to
its forward distance Z:

    Z = (fy * H) / (v - v0)

where fy is the focal length in pixels and v0 is the image row where the
horizon sits. Rearranged, v is LINEAR in 1/Z:

    v = v0 + C / Z            (C := fy * H)

--- Why fit v0 and C instead of assuming a camera height + horizon row ---
It's tempting to just look up "KITTI's camera is ~1.65 m high" and assume
v0 = the calibrated principal point (zero pitch). But that assumes the
camera's optical axis has exactly zero pitch relative to the ground, which
is unlikely to be exactly true. Instead: v0 and C are fit by LINEAR
REGRESSION of real ground-truth objects' (1/z, bbox_bottom) pairs -- this is
a real, checkable calibration step against real data, not a trivia fact
copied from a spec sheet.

--- Train/test split, so the evaluation is honest ---
Fit (v0, C) on one sequence (FIT_SEQUENCE), then evaluate accuracy by range
bucket on a DIFFERENT, held-out sequence (EVAL_SEQUENCE). Fitting and
evaluating on the same sequence would let the model memorize that specific
recording's camera quirks rather than test whether the geometry generalizes.

--- Filtering training/eval points ---
Only type == "Car", truncated < 0.01, occluded <= 1: a truncated or heavily
occluded box's bottom edge is not the true ground-contact point (the same
truncation lesson as TTC's box-width -- see NOTES.md, 2026-09-28), so
including those points would fit the line to the wrong measurement.

--- Perfect boxes first ---
This script uses KITTI's own labeled 2D boxes, not our detector's noisy
ones -- same layering as the TTC check (kitti_label_ttc_check.py before
kitti_pipeline_compare.py): validate the geometry itself is sound before
asking whether detector noise degrades it further.

Usage:
    uv run python kitti_distance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kitti_gt_ttc import parse_label_file

FIT_SEQUENCE = "0019"
EVAL_SEQUENCE = "0009"

# Range buckets (meters) for reporting accuracy -- matches the granularity
# typically used in KITTI 3D-detection papers (near/mid/far).
RANGE_BUCKETS = [(0, 10), (10, 20), (20, 40), (40, np.inf)]


def load_clean_cars(sequence):
    """Ground-truth Car boxes from one sequence, filtered to trustworthy ones
    (not truncated, not heavily occluded -- see module docstring)."""
    labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    clean = labels[
        (labels["type"] == "Car")
        & (labels["truncated"] < 0.01)
        & (labels["occluded"] <= 1)
        & (labels["z"] > 0)
    ].copy()
    return clean


def fit_ground_plane(df):
    """Fit v = v0 + C/z by linear regression of bbox_bottom against 1/z.
    Returns (v0, C)."""
    inv_z = 1.0 / df["z"].values
    v = df["bbox_bottom"].values
    C, v0 = np.polyfit(inv_z, v, 1)   # polyfit returns [slope, intercept]
    return v0, C


def read_fy(sequence):
    """Read the rectified left-camera focal length (pixels) from calib, just
    for an honesty check: does the fitted C = fy*H imply a plausible camera
    height H, not some geometrically meaningless number?"""
    calib_path = Path(f"data/kitti/training/calib/{sequence}.txt")
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                values = [float(x) for x in line.split(":")[1].split()]
                return values[5]   # P2 row-major 3x4: index 5 = fy
    raise ValueError(f"P2 not found in {calib_path}")


def main():
    fit_df = load_clean_cars(FIT_SEQUENCE)
    v0, C = fit_ground_plane(fit_df)
    fy = read_fy(FIT_SEQUENCE)
    implied_H = C / fy

    print(f"Fit on sequence {FIT_SEQUENCE}: {len(fit_df)} clean Car boxes")
    print(f"  v0 (fitted horizon row) = {v0:.2f} px")
    print(f"  C  (= fy * H)           = {C:.1f} px*m")
    print(f"  fy (from calib)         = {fy:.1f} px")
    print(f"  implied camera height H = {implied_H:.2f} m  "
          f"(sanity check: real dashcams/KITTI's rig are typically ~1.5-1.7 m -- "
          f"{'plausible' if 1.3 < implied_H < 2.0 else 'IMPLAUSIBLE, check the fit'})")

    # Save the fitted parameters explicitly -- this IS the calibration result,
    # worth keeping as its own artifact, not just printed and lost.
    fit_summary = pd.DataFrame([{
        "fit_sequence": FIT_SEQUENCE, "n_points": len(fit_df),
        "v0_px": v0, "C_px_m": C, "fy_px": fy, "implied_H_m": implied_H,
    }])
    fit_summary.to_csv("outputs/kitti_distance_fit_params.csv", index=False)
    print("saved outputs/kitti_distance_fit_params.csv")

    # --- Evaluate on the held-out sequence ---
    eval_df = load_clean_cars(EVAL_SEQUENCE)
    eval_df["z_pred"] = C / (eval_df["bbox_bottom"] - v0)
    eval_df["error_m"] = eval_df["z_pred"] - eval_df["z"]
    eval_df["abs_error_m"] = eval_df["error_m"].abs()

    print(f"\nEvaluated on held-out sequence {EVAL_SEQUENCE}: {len(eval_df)} clean Car boxes")
    print(f"{'range (m)':<12}{'n':>6}{'median |err| (m)':>20}{'mean |err| (m)':>18}")
    rows = []
    for lo, hi in RANGE_BUCKETS:
        bucket = eval_df[(eval_df["z"] >= lo) & (eval_df["z"] < hi)]
        label = f"{lo}-{hi if np.isfinite(hi) else '+'}"
        if len(bucket) == 0:
            print(f"{label:<12}{0:>6}{'--':>20}{'--':>18}")
            continue
        med = bucket["abs_error_m"].median()
        mean = bucket["abs_error_m"].mean()
        print(f"{label:<12}{len(bucket):>6}{med:>20.2f}{mean:>18.2f}")
        rows.append({"range": label, "n": len(bucket), "median_abs_err_m": med, "mean_abs_err_m": mean})

    pd.DataFrame(rows).to_csv(f"outputs/kitti_{EVAL_SEQUENCE}_distance_by_range.csv", index=False)
    eval_df.to_csv(f"outputs/kitti_{EVAL_SEQUENCE}_distance_eval.csv", index=False)
    print(f"\nsaved outputs/kitti_{EVAL_SEQUENCE}_distance_by_range.csv")
    print(f"saved outputs/kitti_{EVAL_SEQUENCE}_distance_eval.csv")

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    inv_z_fit = 1.0 / fit_df["z"].values
    ax1.scatter(inv_z_fit, fit_df["bbox_bottom"], s=8, alpha=0.4, label=f"seq {FIT_SEQUENCE} (fit data)")
    xs = np.linspace(inv_z_fit.min(), inv_z_fit.max(), 50)
    ax1.plot(xs, v0 + C * xs, color="red", linewidth=2, label=f"fit: v = {v0:.1f} + {C:.0f}/z")
    ax1.set_xlabel("1/z (1/m)")
    ax1.set_ylabel("bbox bottom row (px)")
    ax1.set_title(f"Ground-plane fit on seq {FIT_SEQUENCE}")
    ax1.legend(fontsize=8)

    ax2.scatter(eval_df["z"], eval_df["z_pred"], s=10, alpha=0.5)
    lims = [0, max(eval_df["z"].max(), eval_df["z_pred"].max()) * 1.05]
    ax2.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="perfect prediction")
    ax2.set_xlabel("ground-truth z (m)")
    ax2.set_ylabel("predicted z (m)")
    ax2.set_title(f"Held-out eval on seq {EVAL_SEQUENCE}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_png = "outputs/kitti_distance_fit_and_eval.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
