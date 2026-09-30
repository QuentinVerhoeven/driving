"""
Tune LeadSelector's BAND_HALF_WIDTH against real ground truth, instead of
guessing a number.

--- Why this is needed ---
The default (0.10) was tuned and validated on clip1_30s (a single-lane-each-
way street, Week 2). On a wide multi-lane highway (KITTI 0020), a car in a
genuinely different lane can still fall inside that same fixed central band,
especially when it's close (large box, near the bottom of the frame) while
the true same-lane traffic is far away (small, easy to land just outside a
rigid band). Found and confirmed visually on 2026-09-29 (see NOTES.md):
frame 0 of outputs/kitti_0020_annotated.mp4 picked a car that IS inside the
band as "lead," while a closer, more obviously-in-front cluster of cars sat
just outside it.

--- The ground-truth reference ---
Same idea as kitti_event_eval.py's ground_truth_events, but narrower: this
only needs the ground-truth LEAD PICK per frame (which car LeadSelector would
choose if given perfect boxes), not the oxts-dependent headway/tailgating
math -- lead-pick and TTC/rapid-closing don't need GPS at all, so this
survived the 2026-09-29 scope cut untouched.

--- The method ---
For each candidate band_half, run LeadSelector on OUR REAL detector's tracked
boxes (outputs/kitti_{seq}_tracks.csv) and compare its pick each frame to the
ground-truth pick via IoU (same iou()/threshold pattern as
kitti_pipeline_compare.py). Score = fraction of frames where both sides have
a lead AND their boxes overlap enough to call them the same car. Pooled
across all 3 sequences (0019, 0009, 0020), plus reported per-sequence so a
sequence-specific quirk doesn't get hidden by averaging.

center_offset is NOT tuned here: the calib check (P2's cx vs frame center,
~1% off) already ruled out a meaningful principal-point offset, so there's
no evidence-based case for touching it, and tuning an unverified parameter
alongside a verified one would muddy which change caused what.

Usage:
    uv run python kitti_band_tune.py
"""

from pathlib import Path

import cv2
import pandas as pd

from kitti_gt_ttc import FRAME_DT_S, parse_label_file
from lead_ttc import Box, LeadSelector

SEQUENCES = ["0019", "0009", "0020"]
CANDIDATE_WIDTHS = [0.06, 0.07, 0.075, 0.08, 0.09, 0.10, 0.12, 0.15]
IOU_MATCH_THRESHOLD = 0.3   # same threshold used in kitti_pipeline_compare.py


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def frame_size(sequence):
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    return sample.shape[1], sample.shape[0]   # w, h


def ground_truth_lead_boxes(sequence, frame_w, frame_h, band_half):
    """Ground-truth lead box per frame: LeadSelector run on real KITTI 2D
    labels (perfect boxes), for one band_half. Returns {frame: (x1,y1,x2,y2)}."""
    labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    cars = labels[labels["type"] == "Car"].copy()

    selector = LeadSelector(frame_w, frame_h, band_half=band_half)
    out = {}
    for frame, g in cars.groupby("frame"):
        t = frame * FRAME_DT_S
        boxes = [Box(track_id=int(r.track_id), x1=r.bbox_left, y1=r.bbox_top,
                      x2=r.bbox_right, y2=r.bbox_bottom) for r in g.itertuples()]
        lead = selector.update(boxes, t)
        if lead is not None:
            out[frame] = (lead.x1, lead.y1, lead.x2, lead.y2)
    return out


def our_lead_boxes(sequence, frame_w, frame_h, band_half):
    """Our real detector's lead box per frame, for one band_half."""
    tracks = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")
    selector = LeadSelector(frame_w, frame_h, band_half=band_half)
    out = {}
    for frame, g in tracks.groupby("frame"):
        t = frame * FRAME_DT_S
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2)
                  for r in g.itertuples()]
        lead = selector.update(boxes, t)
        if lead is not None:
            out[frame] = (lead.x1, lead.y1, lead.x2, lead.y2)
    return out


def main():
    rows = []
    for band_half in CANDIDATE_WIDTHS:
        pooled_agree, pooled_total = 0, 0
        per_seq = {}
        for seq in SEQUENCES:
            frame_w, frame_h = frame_size(seq)
            gt = ground_truth_lead_boxes(seq, frame_w, frame_h, band_half)
            ours = our_lead_boxes(seq, frame_w, frame_h, band_half)

            # "total" = frames where OUR pipeline had a lead at all -- that's the
            # population this band_half actually affects; frames neither side
            # has a lead aren't a disagreement, they're just both empty.
            total = len(ours)
            agree = sum(1 for f, box in ours.items()
                        if f in gt and iou(box, gt[f]) >= IOU_MATCH_THRESHOLD)
            per_seq[seq] = (agree, total)
            pooled_agree += agree
            pooled_total += total

        rate = pooled_agree / pooled_total if pooled_total else float("nan")
        row = {"band_half": band_half, "pooled_agree": pooled_agree,
               "pooled_total": pooled_total, "pooled_rate": rate}
        for seq, (a, t) in per_seq.items():
            row[f"{seq}_rate"] = a / t if t else float("nan")
            row[f"{seq}_n"] = t
        rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    out.to_csv("outputs/kitti_band_tune.csv", index=False)
    print("\nsaved outputs/kitti_band_tune.csv")

    best = out.loc[out["pooled_rate"].idxmax()]
    print(f"\nBest pooled agreement: band_half={best['band_half']} "
          f"(rate={best['pooled_rate']:.3f}, n={int(best['pooled_total'])} frames)")
    print("Note: 'total' counts frames where OUR pipeline had a lead at all -- a narrower")
    print("band can also change how many frames that is (more/fewer drop-outs), so read")
    print("the *_n columns alongside the rate, not the rate alone.")


if __name__ == "__main__":
    main()
