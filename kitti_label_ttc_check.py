"""
Sanity check: run our real TTCKalman on PERFECT (hand-labeled) boxes, before any
detector/tracker noise enters the picture, and compare it to the exact
3D-derived ground truth from kitti_gt_ttc.py.

Why this check, and why now (before the KITTI images have even finished
downloading): KITTI's label files include the 2D bounding box (bbox_left/top/
right/bottom) as well as the 3D location used for ground-truth TTC. That means
we can feed the *labeled* box width straight into our own TTCKalman and ask
"if detection and tracking were perfect, would the box-width-expansion idea
give the right answer?" -- separating the filter's own error from whatever
error YOLO/ByteTrack will add later, once the images are available.

Frame width: confirmed at 1238 px against a real image
(data/kitti/training/image_02/0019/000420.png). Before the images landed this
was estimated at 1242 (the well-known KITTI figure) from the calib file's
principal point (P2[0,2] ~ 609.6 px for sequence 0000) -- close, but the real
value is 4 px narrower; the principal point isn't exactly half-width for a
real lens, which is why that estimate was flagged as approximate rather than
exact.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from kitti_gt_ttc import parse_label_file, compute_gt_ttc
from lead_ttc import TTCKalman

FRAME_W_ASSUMED = 1238   # confirmed against a real image (data/kitti/training/image_02/0019/000420.png), 2026-09-28
EDGE_MARGIN_PX = 2.0     # box within this many px of x=0 or x=frame_w counts as "touching the edge"
SEQUENCE = "0019"
TRACK_ID = 42


def main():
    labels = parse_label_file(Path(f"data/kitti/training/label_02/{SEQUENCE}.txt"))
    labels = labels[labels["track_id"] == TRACK_ID].sort_values("frame")
    gt = compute_gt_ttc(labels)   # exact 3D-derived ground truth for this track

    # Our own truncation heuristic, from box coordinates alone -- no access to KITTI's `truncated`
    # label, since a real YOLO detection won't have one either. Compared against that label below,
    # as a check that the heuristic would generalize.
    touches_edge = (gt["bbox_left"] <= EDGE_MARGIN_PX) | (gt["bbox_right"] >= FRAME_W_ASSUMED - EDGE_MARGIN_PX)
    agree = (touches_edge == (gt["truncated"] > 0))
    print(f"edge-touch heuristic vs KITTI's `truncated` label: agree on {agree.mean():.0%} of frames "
          f"({(touches_edge & ~(gt['truncated'] > 0)).sum()} frames flagged extra by the heuristic, "
          f"{(~touches_edge & (gt['truncated'] > 0)).sum()} missed)")

    kalman = TTCKalman(FRAME_W_ASSUMED)
    rows = []
    for (_, row), edge in zip(gt.iterrows(), touches_edge):
        bbox_w = row["bbox_right"] - row["bbox_left"]
        t = row["frame"] * 0.1   # KITTI tracking is 10 Hz
        _, _, ttc_kalman = kalman.update(bbox_w, t, TRACK_ID, truncated=edge)
        rows.append({"frame": row["frame"], "gt_ttc_s": row["gt_ttc_s"], "kalman_ttc_s": ttc_kalman})

    import pandas as pd
    out = pd.DataFrame(rows)

    both_defined = out.dropna(subset=["gt_ttc_s", "kalman_ttc_s"])
    err = (both_defined["kalman_ttc_s"] - both_defined["gt_ttc_s"]).abs()
    print(f"sequence {SEQUENCE}, track {TRACK_ID}: {len(out)} frames, "
          f"{len(both_defined)} with both TTCs defined")
    print(f"median |error|: {err.median():.2f} s, max |error|: {err.max():.2f} s")

    # Event agreement: does "gt_ttc < 8 s" line up with "kalman_ttc < 8 s"?
    gt_event = both_defined["gt_ttc_s"] < 8
    kalman_event = both_defined["kalman_ttc_s"] < 8
    tp = (gt_event & kalman_event).sum()
    fn = (gt_event & ~kalman_event).sum()
    fp = (~gt_event & kalman_event).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"TTC<8s event: {gt_event.sum()} ground-truth frames, "
          f"precision={precision:.2f}, recall={recall:.2f}")

    # Shade frames where KITTI marks this box as truncated (clipped by the image edge) -- this is
    # where the box-width-expansion idea breaks down, see NOTES.md.
    truncated = gt.set_index("frame")["truncated"].reindex(out["frame"]).to_numpy() > 0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out["frame"], out["gt_ttc_s"], label="ground truth (3D depth)", marker="o", ms=3)
    ax.plot(out["frame"], out["kalman_ttc_s"], label="our TTCKalman (labeled 2D box)", marker="o", ms=3)
    ax.axhline(8, color="gray", linestyle="--", linewidth=1, label="8 s event threshold")
    if truncated.any():
        ax.fill_between(out["frame"], 0, 30, where=truncated, color="red", alpha=0.08,
                         label="box truncated (clipped by frame edge)")
    ax.set_ylim(0, 30)
    ax.set_xlabel("frame")
    ax.set_ylabel("TTC (s)")
    ax.set_title(f"KITTI {SEQUENCE} track {TRACK_ID}: TTCKalman on labeled boxes vs. ground truth")
    ax.legend()
    fig.tight_layout()
    out_path = Path("outputs/kitti_label_ttc_check.png")
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
