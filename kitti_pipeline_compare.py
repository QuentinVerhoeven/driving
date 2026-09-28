"""
Step 2 of the real KITTI comparison: feed our ACTUAL detector's tracked boxes
(from kitti_track.py's output, real YOLO+ByteTrack noise and all -- ID
switches, missed detections during occlusion, everything) through the same
LeadSelector + TTCKalman pipeline live.py uses, and compare the resulting TTC
against the exact ground truth from kitti_gt_ttc.py.

This is the point of the whole KITTI exercise: kitti_label_ttc_check.py
already showed the filter is basically sound on PERFECT boxes (plus one real
failure mode, frame-edge truncation, since fixed). This script instead asks
"what does the full pipeline actually see," detector noise included.

Since our own tracker doesn't know KITTI's ground-truth track IDs, whichever
car LeadSelector picks as the lead is matched back to KITTI's labeled track
42 by box overlap (IoU) each frame -- the same matching used to investigate
the tracker's ID switches in NOTES.md. Frames where our pipeline's lead
picks a DIFFERENT car (IoU < 0.3) are marked as a lead-selection error rather
than silently compared as if they were the same car.

Usage:
    uv run python kitti_pipeline_compare.py 0009 1
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from kitti_gt_ttc import parse_label_file, compute_gt_ttc
from lead_ttc import Box, LeadSelector, TTCKalman

# Image resolution varies slightly per KITTI sequence (e.g. 1238x374 for 0019, 1242x375 for 0009) --
# read it from an actual frame rather than assuming a fixed size, unlike the earlier single-sequence version.
EDGE_MARGIN_PX = 2.0
FPS = 10.0
IOU_MATCH_THRESHOLD = 0.3   # below this, our lead is a different car, not the GT track we're comparing to


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("track_id", type=int, help="KITTI ground-truth track_id to compare against")
    args = parser.parse_args()
    sequence, gt_track_id = args.sequence, args.track_id

    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    frame_h, frame_w = sample.shape[:2]
    print(f"sequence {sequence}: frame size {frame_w}x{frame_h}")

    ours = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")
    gt_labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    gt_track = gt_labels[gt_labels["track_id"] == gt_track_id]
    gt = compute_gt_ttc(gt_track).set_index("frame")

    frame_range = range(ours["frame"].min(), ours["frame"].max() + 1)
    gt_box_by_frame = {row.frame: (row.bbox_left, row.bbox_top, row.bbox_right, row.bbox_bottom)
                        for row in gt_track.itertuples()}

    selector = LeadSelector(frame_w, frame_h)
    kalman = TTCKalman(frame_w)

    rows = []
    for frame in frame_range:
        t = frame / FPS
        frame_boxes = ours[ours["frame"] == frame]
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2)
                 for r in frame_boxes.itertuples()]

        lead = selector.update(boxes, t)

        our_ttc = float("nan")
        matches_gt = False
        lead_track_id = None
        if lead is not None:
            lead_track_id = lead.track_id
            edge_touch = lead.x1 <= EDGE_MARGIN_PX or lead.x2 >= frame_w - EDGE_MARGIN_PX
            _, _, our_ttc = kalman.update(lead.w, t, lead.track_id, truncated=edge_touch)

            gt_box = gt_box_by_frame.get(frame)
            if gt_box is not None:
                matches_gt = iou((lead.x1, lead.y1, lead.x2, lead.y2), gt_box) >= IOU_MATCH_THRESHOLD

        rows.append({
            "frame": frame,
            "our_lead_track_id": lead_track_id,
            "our_ttc_s": our_ttc,
            "matches_gt_car": matches_gt,
            "gt_ttc_s": gt.loc[frame, "gt_ttc_s"] if frame in gt.index else float("nan"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"outputs/kitti_{sequence}_pipeline_vs_gt.csv", index=False)

    n_lead_present = out["our_lead_track_id"].notna().sum()
    n_matches = out["matches_gt_car"].sum()
    print(f"{len(out)} frames total; our pipeline had a lead in {n_lead_present}; "
          f"of those, {n_matches} were confirmed (IoU>={IOU_MATCH_THRESHOLD}) to be GT track {gt_track_id}")
    print(f"lead-selection errors (had a lead, but wrong car): {n_lead_present - n_matches}")

    # Only compare TTC on frames where we know we were actually looking at the right car --
    # otherwise a TTC "error" could just be that we measured a different vehicle entirely.
    comparable = out[out["matches_gt_car"] & out["gt_ttc_s"].notna() & out["our_ttc_s"].notna()]
    err = (comparable["our_ttc_s"] - comparable["gt_ttc_s"]).abs()
    print(f"\nOf those correctly-matched frames, {len(comparable)} have both a defined GT and our TTC")
    if len(comparable):
        print(f"median |error|: {err.median():.2f} s, max |error|: {err.max():.2f} s")

        gt_event = comparable["gt_ttc_s"] < 8
        our_event = comparable["our_ttc_s"] < 8
        tp = (gt_event & our_event).sum()
        fn = (gt_event & ~our_event).sum()
        fp = (~gt_event & our_event).sum()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"TTC<8s event (on comparable frames only): {gt_event.sum()} GT frames, "
              f"precision={precision:.2f}, recall={recall:.2f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out["frame"], out["gt_ttc_s"], label="ground truth (3D depth)", marker="o", ms=3)
    ax.plot(out["frame"], out["our_ttc_s"], label="our full pipeline (YOLO+ByteTrack+Kalman)",
             marker="o", ms=3)
    not_matched = ~out["matches_gt_car"]
    ax.fill_between(out["frame"], 0, 30, where=not_matched, color="gray", alpha=0.15,
                     label="no lead / wrong car matched")
    ax.axhline(8, color="gray", linestyle="--", linewidth=1, label="8 s event threshold")
    ax.set_ylim(0, 30)
    ax.set_xlabel("frame")
    ax.set_ylabel("TTC (s)")
    ax.set_title(f"KITTI {sequence}: full real pipeline vs. ground truth (GT track {gt_track_id})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = f"outputs/kitti_{sequence}_pipeline_compare.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
