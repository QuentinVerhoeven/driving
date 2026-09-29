"""
Event detection precision/recall: compare our FULL pipeline's detected
tailgating/rapid-closing events (real detector + tracker + LeadSelector +
TTCKalman + fitted ground-plane distance) against an objective ground-truth
event timeline derived from KITTI's own real 3D labels and real ego motion --
not hand-labeled, since a portfolio project doesn't have access to a real
labeling team, but genuinely independent of our detector.

--- How ground truth is built ---
Same lead-selection ALGORITHM (lead_ttc.LeadSelector, completely unchanged),
just fed KITTI's real ground-truth 2D boxes (label_02) instead of our
detector's noisy tracked ones. Whichever object it picks as "the lead" each
frame gets:
  - distance = real LiDAR-derived depth (label z), not our fitted
    ground-plane estimate
  - TTC = computed directly from real dz/dt (kitti_gt_ttc.compute_gt_ttc,
    already built and self-tested)
  - headway = distance / real ego speed (oxts vf)
The exact same thresholds (TAILGATING_HEADWAY_S, RAPID_CLOSING_TTC_S,
MIN_EVENT_S) used throughout this project are applied to this clean signal,
giving a ground-truth event timeline built from real sensor data, not from
our own noisy pipeline circularly checking itself.

--- Why hard braking is excluded ---
Both "predicted" and "ground truth" hard-braking come from the exact same
raw oxts `af` reading -- there's no detector step in between to test.
Comparing them would report a tautological 100/100, not a real evaluation,
so it's left out rather than padding the results with a meaningless number.

Usage:
    uv run python kitti_event_eval.py
"""

from pathlib import Path

import pandas as pd

from kitti_gt_ttc import FRAME_DT_S, compute_gt_ttc, parse_label_file, parse_oxts_file
from kitti_headway_events import TAILGATING_HEADWAY_S, RAPID_CLOSING_TTC_S, sustained_below
from lead_ttc import Box, LeadSelector

SEQUENCES = ["0019", "0009", "0020"]


def ground_truth_events(sequence):
    """Build the ground-truth tailgating/rapid-closing timeline for one
    sequence, from real labels + real oxts, independent of our detector."""
    import cv2
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    frame_h, frame_w = sample.shape[:2]

    labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    cars = labels[labels["type"] == "Car"].copy()
    gt_ttc = compute_gt_ttc(cars).set_index(["track_id", "frame"])["gt_ttc_s"]

    oxts = parse_oxts_file(Path(f"data/kitti/training/oxts/{sequence}.txt"))
    ego_speed_by_frame = dict(zip(oxts["frame"], oxts["vf"]))

    selector = LeadSelector(frame_w, frame_h)
    rows = []
    for frame, g in cars.groupby("frame"):
        t = frame * FRAME_DT_S
        boxes = [Box(track_id=int(r.track_id), x1=r.bbox_left, y1=r.bbox_top,
                      x2=r.bbox_right, y2=r.bbox_bottom) for r in g.itertuples()]
        lead = selector.update(boxes, t)

        distance_m, ttc_s = float("nan"), float("nan")
        if lead is not None:
            row = g[g["track_id"] == lead.track_id].iloc[0]
            distance_m = row["z"]
            ttc_s = gt_ttc.get((lead.track_id, frame), float("nan"))

        ego_speed = ego_speed_by_frame.get(frame, float("nan"))
        headway_s = distance_m / ego_speed if ego_speed and ego_speed > 0.5 else float("nan")
        rows.append({"frame": frame, "distance_m": distance_m, "ttc_s": ttc_s, "headway_s": headway_s})

    out = pd.DataFrame(rows)
    # left-join onto the FULL frame range, same reasoning as kitti_headway_events.py:
    # a frame with no ground-truth lead is "no reading," not "safe."
    full = pd.DataFrame({"frame": range(len(oxts))})
    out = full.merge(out, on="frame", how="left")

    out["gt_tailgating"] = sustained_below(out["headway_s"] < TAILGATING_HEADWAY_S)
    out["gt_rapid_closing"] = sustained_below((out["ttc_s"] < RAPID_CLOSING_TTC_S) & out["ttc_s"].notna())
    return out[["frame", "gt_tailgating", "gt_rapid_closing"]]


def confusion(pred: pd.Series, gt: pd.Series):
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def main():
    per_seq = {}
    pooled_pred = {"tailgating": [], "rapid_closing": []}
    pooled_gt = {"tailgating": [], "rapid_closing": []}

    for seq in SEQUENCES:
        pred = pd.read_csv(f"outputs/kitti_{seq}_headway_events.csv")[["frame", "tailgating", "rapid_closing"]]
        gt = ground_truth_events(seq)
        m = pred.merge(gt, on="frame", how="left").fillna(False)

        results = {
            "tailgating": confusion(m["tailgating"], m["gt_tailgating"]),
            "rapid_closing": confusion(m["rapid_closing"], m["gt_rapid_closing"]),
        }
        per_seq[seq] = results
        for kind in pooled_pred:
            pooled_pred[kind].append(m[kind if kind == "tailgating" else "rapid_closing"])
            pooled_gt[kind].append(m[f"gt_{kind}"])

        m.to_csv(f"outputs/kitti_{seq}_event_eval.csv", index=False)
        print(f"\n=== {seq} ===")
        for kind, r in results.items():
            print(f"  {kind}: precision={r['precision']:.2f} recall={r['recall']:.2f} "
                  f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")

    print("\n=== POOLED across all sequences ===")
    summary_rows = []
    for kind in pooled_pred:
        pred_all = pd.concat(pooled_pred[kind], ignore_index=True)
        gt_all = pd.concat(pooled_gt[kind], ignore_index=True)
        r = confusion(pred_all, gt_all)
        print(f"  {kind}: precision={r['precision']:.2f} recall={r['recall']:.2f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']}, n={len(pred_all)} frames)")
        summary_rows.append({"event": kind, **r})

    pd.DataFrame(summary_rows).to_csv("outputs/kitti_event_eval_summary.csv", index=False)
    print("\nsaved outputs/kitti_event_eval_summary.csv")
    print("(hard braking excluded -- both predicted and ground truth read the same raw oxts af, no detector step to test)")


if __name__ == "__main__":
    main()
