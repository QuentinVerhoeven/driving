"""
Tune RapidClosingDetector's MIN_TRUST_AFTER_SWITCH_S against KITTI ground truth.

--- Why this came up ---
RapidClosingDetector (lead_ttc.py, added 2026-09-30 so live.py's streaming demo
path shares the same rapid-closing logic as the offline KITTI batch script) never
fires on outputs/autobahn_braking_annotated.mp4 despite two real fast-closing
windows in the log (track 160, track 104) -- because in that clip the ego car is
continuously overtaking, so every lead track is short-lived (0.27s-0.97s), always
SHORTER than the 1.0s MIN_TRUST_AFTER_SWITCH_S trust window, so the guard never
clears before the track ends. MIN_TRUST_AFTER_SWITCH_S = 1.0s was never itself
validated against ground truth -- it was picked to solve KITTI sequence 0009's
noise problem (busy, many short-lived spurious lead switches causing the Kalman
filter to cold-start and briefly report a fake fast TTC drop), see NOTES.md,
2026-09-28.

--- Method ---
For a grid of candidate values, run RapidClosingDetector (unchanged class, just
different min_trust_after_switch_s) over OUR REAL detector+tracker's output on
KITTI sequences 0009, 0019, 0020 (outputs/kitti_<seq>_tracks.csv, the same input
kitti_headway_events.py uses), through the same LeadSelector + TTCKalman used
everywhere else. Compare the resulting rapid_closing flag against a ground truth
built from KITTI's real 3D box labels (kitti_gt_ttc.compute_gt_ttc gives an exact
dz/dt-based TTC; LeadSelector run on the real labeled boxes instead of our noisy
detector output picks the ground-truth lead; RAPID_CLOSING_TTC_S + sustained_below
gives a ground-truth rapid_closing timeline) -- same idea as
kitti_event_eval.py's ground_truth_events / confusion(), for the rapid_closing
event only (that half needs no oxts/GPS; tailgating/headway does, and stays
frozen, see NOTES.md 2026-09-29). Re-implemented here (not imported) because
kitti_event_eval.py currently fails to import -- it still imports
TAILGATING_HEADWAY_S from kitti_headway_events.py, a name removed in the
2026-09-29 trim that dropped tailgating from that file. That's a pre-existing
bitrot in a frozen file, out of scope for this task to fix.

0009 is the sequence the 1.0s value was originally created to fix (busy,
multi-candidate, many short-lived spurious lead switches) -- its false-positive
count at each candidate value is the thing this script most needs to check before
lowering the default, not just pooled/0020 agreement.

Usage:
    uv run python kitti_trust_tune.py
"""

from pathlib import Path

import cv2
import pandas as pd

from kitti_gt_ttc import FRAME_DT_S, compute_gt_ttc, parse_label_file
from kitti_headway_events import sustained_below
from lead_ttc import Box, LeadSelector, TTCKalman, RapidClosingDetector, RAPID_CLOSING_TTC_S, MIN_EVENT_S

SEQUENCES = ["0009", "0019", "0020"]
CANDIDATE_TRUST_S = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
EDGE_MARGIN_PX = 2.0
FPS = 10.0


def frame_size(sequence):
    img_dir = Path(f"data/kitti/training/image_02/{sequence}")
    sample = cv2.imread(str(sorted(img_dir.glob("*.png"))[0]))
    return sample.shape[:2][::-1]   # (w, h)


def ground_truth_rapid_closing(sequence):
    """Ground-truth rapid_closing timeline built ONLY from real 3D labels (no
    oxts/GPS needed -- confirmed by reading kitti_gt_ttc.compute_gt_ttc and
    kitti_event_eval.ground_truth_events: TTC there comes from dz/dt on the
    label z field, GPS is only used for the separate tailgating/headway metric).
    Same method as kitti_event_eval.py's ground_truth_events, trimmed to just
    the rapid_closing half (re-implemented here, see module docstring, because
    kitti_event_eval.py currently fails to import due to unrelated bitrot)."""
    frame_w, frame_h = frame_size(sequence)
    labels = parse_label_file(Path(f"data/kitti/training/label_02/{sequence}.txt"))
    cars = labels[labels["type"] == "Car"].copy()
    gt_ttc = compute_gt_ttc(cars).set_index(["track_id", "frame"])["gt_ttc_s"]

    selector = LeadSelector(frame_w, frame_h)
    rows = []
    for frame, g in cars.groupby("frame"):
        t = frame * FRAME_DT_S
        boxes = [Box(track_id=int(r.track_id), x1=r.bbox_left, y1=r.bbox_top,
                      x2=r.bbox_right, y2=r.bbox_bottom) for r in g.itertuples()]
        lead = selector.update(boxes, t)
        ttc_s = gt_ttc.get((lead.track_id, frame), float("nan")) if lead is not None else float("nan")
        rows.append({"frame": frame, "ttc_s": ttc_s})

    out = pd.DataFrame(rows)
    out["gt_rapid_closing"] = sustained_below((out["ttc_s"] < RAPID_CLOSING_TTC_S) & out["ttc_s"].notna())
    return out[["frame", "gt_rapid_closing"]]


def confusion(pred: pd.Series, gt: pd.Series):
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def predicted_rapid_closing(sequence, trust_s):
    """Run our REAL detector+tracker's output through LeadSelector + TTCKalman +
    RapidClosingDetector(min_trust_after_switch_s=trust_s), exactly the streaming
    path live.py uses, over one KITTI sequence's tracked boxes."""
    frame_w, frame_h = frame_size(sequence)
    tracks = pd.read_csv(f"outputs/kitti_{sequence}_tracks.csv")

    selector = LeadSelector(frame_w, frame_h)
    kalman = TTCKalman(frame_w)
    detector = RapidClosingDetector(min_trust_after_switch_s=trust_s)

    rows = []
    for frame, g in tracks.groupby("frame"):
        t = frame / FPS
        boxes = [Box(track_id=int(r.track_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2) for r in g.itertuples()]
        lead = selector.update(boxes, t)

        lead_track_id, ttc_s = None, float("nan")
        if lead is not None:
            edge_touch = lead.x1 <= EDGE_MARGIN_PX or lead.x2 >= frame_w - EDGE_MARGIN_PX
            _, _, ttc_s = kalman.update(lead.w, t, lead.track_id, truncated=edge_touch)
            lead_track_id = lead.track_id

        flag = detector.update(ttc_s if ttc_s == ttc_s else None, t, lead_track_id)
        rows.append({"frame": frame, "rapid_closing": flag})

    return pd.DataFrame(rows)


def main():
    gt_cache = {seq: ground_truth_rapid_closing(seq) for seq in SEQUENCES}

    all_rows = []
    for trust_s in CANDIDATE_TRUST_S:
        pooled_pred, pooled_gt = [], []
        for seq in SEQUENCES:
            pred = predicted_rapid_closing(seq, trust_s)
            m = pred.merge(gt_cache[seq], on="frame", how="left").fillna(False)
            r = confusion(m["rapid_closing"], m["gt_rapid_closing"])
            all_rows.append({"trust_s": trust_s, "sequence": seq, **r})
            pooled_pred.append(m["rapid_closing"])
            pooled_gt.append(m["gt_rapid_closing"])

        pooled_r = confusion(pd.concat(pooled_pred, ignore_index=True), pd.concat(pooled_gt, ignore_index=True))
        all_rows.append({"trust_s": trust_s, "sequence": "POOLED", **pooled_r})

    out = pd.DataFrame(all_rows)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    out.to_csv("outputs/kitti_trust_tune.csv", index=False)
    print("\nsaved outputs/kitti_trust_tune.csv")


if __name__ == "__main__":
    main()
