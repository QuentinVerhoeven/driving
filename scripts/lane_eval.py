"""Fixed lane band vs. YOLOP lane gating for lead selection, checked against an INDEPENDENT ground truth.

Why a new ground truth: kitti_band_tune.py's reference is LeadSelector itself run on KITTI's label boxes
with the SAME band rule -- fine for tuning a band width, but it would grade the lane-gated selector by the
band's own rule. Here the reference comes from KITTI's 3D labels instead: each labeled object has a
position (x lateral, z forward, meters, camera frame). Ground-truth lead = nearest labeled Car/Van/Truck
ahead (z > 0) with |x| < LANE_HALF_M. Approximation, stated honestly: it assumes the ego car drives
roughly centered in its lane, and on a curve a far car's lateral offset drifts even when it's in-lane.

Both methods run on OUR real detector + tracker output (outputs/kitti_<seq>_tracks.csv), the same
LeadSelector logic, differing ONLY in the in-lane test (band on box center vs. lane bounds at the box's
bottom-center). Where no lane is available on a frame, the lane method falls back to the band.

  precision = frames our lead matches the ground-truth lead (IoU >= 0.3) / frames we show a lead
  recall    = same matches / frames where a ground-truth lead exists

Then the autobahn clip (no 3D labels): run the full LeadSelector -> TTCKalman -> RapidClosingDetector chain
with each method and report who was lead when, and whether/when the rapid-closing flag fires.

Usage:
  uv run python lane_eval.py
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scripts", _ROOT / "legacy"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from pathlib import Path

import numpy as np
import pandas as pd

from ego_lane import load_egolane
from kitti_band_tune import IOU_MATCH_THRESHOLD, frame_size, iou
from kitti_gt_ttc import FRAME_DT_S, parse_label_file
from lead_ttc import Box, LeadSelector, RapidClosingDetector, TTCKalman

SEQUENCES = ["0019", "0009", "0020"]
LANE_HALF_M = 1.8          # half of a ~3.5-3.75 m lane
GT_TYPES = {"Car", "Van", "Truck"}


def gt_leads(seq):
    """{frame: (x1, y1, x2, y2)} of the nearest labeled vehicle ahead within the lane's lateral extent."""
    lab = parse_label_file(Path(f"data/kitti/training/label_02/{seq}.txt"))
    lab = lab[lab["type"].isin(GT_TYPES) & (lab["z"] > 0) & (lab["x"].abs() < LANE_HALF_M)]
    nearest = lab.loc[lab.groupby("frame")["z"].idxmin()]
    return {int(r.frame): (r.bbox_left, r.bbox_top, r.bbox_right, r.bbox_bottom) for r in nearest.itertuples()}


def load_lanes(name):
    """{frame: EgoLaneBounds or None} from ego_lane.py's CSV."""
    return load_egolane(f"outputs/{name}_egolane.csv")


def run_selector(tracks, frame_w, frame_h, frame_dt, lanes=None, frame_col="frame"):
    """Yield (frame, t, lead Box or None) for every frame with detections, band or lane gating."""
    sel = LeadSelector(frame_w, frame_h)
    for frame, g in tracks.groupby(frame_col):
        t = frame * frame_dt
        boxes = [Box(int(r.track_id), r.x1, r.y1, r.x2, r.y2) for r in g.itertuples()]
        lane = lanes.get(int(frame)) if lanes is not None else None
        yield int(frame), t, sel.update(boxes, t, lane)


def score(ours, gt):
    matched = sum(1 for f, b in ours.items() if f in gt and iou(b, gt[f]) >= IOU_MATCH_THRESHOLD)
    return matched, len(ours), len(gt)


def kitti_table():
    rows = []
    for method in ("band", "lane"):
        tot = np.zeros(3, int)
        row = {"method": method}
        for seq in SEQUENCES:
            fw, fh = frame_size(seq)
            tracks = pd.read_csv(f"outputs/kitti_{seq}_tracks.csv")
            lanes = load_lanes(f"kitti_{seq}") if method == "lane" else None
            ours = {f: (b.x1, b.y1, b.x2, b.y2) for f, _, b in run_selector(tracks, fw, fh, FRAME_DT_S, lanes)
                    if b is not None}
            m, n_ours, n_gt = score(ours, gt_leads(seq))
            tot += (m, n_ours, n_gt)
            row[f"{seq}_prec"], row[f"{seq}_rec"] = m / n_ours, m / n_gt
        row["pooled_prec"], row["pooled_rec"] = tot[0] / tot[1], tot[0] / tot[2]
        row["frames_with_lead"] = tot[1]
        rows.append(row)
    return pd.DataFrame(rows)


def autobahn_report(method):
    tracks = pd.read_csv("outputs/autobahn_braking_tracks.csv")
    fps = 29.97002997002997
    lanes = load_lanes("autobahn_braking") if method == "lane" else None
    kal, rc = TTCKalman(1920), RapidClosingDetector()
    log = []
    for frame, t, lead in run_selector(tracks, 1920, 1080, 1 / fps, lanes):
        if lead is None:
            rc.update(None, t, None)
            log.append((t, None, np.nan, False))
            continue
        _, _, ttc = kal.update(lead.w, t, lead.track_id)
        log.append((t, lead.track_id, ttc, rc.update(ttc, t, lead.track_id)))
    df = pd.DataFrame(log, columns=["t", "lead", "ttc", "rapid"])
    # contiguous lead stints
    df["stint"] = (df["lead"] != df["lead"].shift()).cumsum()
    stints = df.dropna(subset=["lead"]).groupby("stint").agg(lead=("lead", "first"), t0=("t", "min"),
                                                             t1=("t", "max"), min_ttc=("ttc", "min"))
    stints = stints[stints.t1 - stints.t0 >= 0.5]
    print(f"\n--- autobahn, {method} gating: lead stints >= 0.5 s ---")
    print(stints.round(2).to_string(index=False))
    fired = df[df.rapid]
    if len(fired):
        print(f"RAPID CLOSING fires: {len(fired)} frames, t={fired.t.min():.2f}-{fired.t.max():.2f}s, "
              f"lead id(s) {sorted(fired.lead.unique().astype(int))}, min TTC {fired.ttc.min():.2f}s")
    else:
        print("RAPID CLOSING never fires")


def main():
    pd.set_option("display.width", 160)
    table = kitti_table()
    print("KITTI lead pick vs 3D-label ground truth (nearest vehicle ahead with |x| < %.1f m)" % LANE_HALF_M)
    print(table.round(3).to_string(index=False))
    table.to_csv("outputs/lane_eval_kitti.csv", index=False)
    for method in ("band", "lane"):
        autobahn_report(method)


if __name__ == "__main__":
    main()
