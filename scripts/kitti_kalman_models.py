"""
Does a different Kalman MODEL fix TTCKalman's lag on fast-closing cars? (experiment, 2026-10-01)

--- Why ---
On the autobahn clip the filtered TTC of the closing Golf runs ~0.25 s above the raw width-based TTC and
bottoms out ~0.2 s late (1.03 s vs raw ~0.80 s, NOTES.md 2026-10-01). TTCKalman is a constant-velocity
model on box width w, but for a car closing at constant speed w ~ 1/distance, so dw/dt keeps ACCELERATING
as TTC drops -- the model is systematically behind exactly when it matters. Loosening q was already ruled
out against ground truth (kitti_kalman_tune.py). So try changing the model instead:

  A. InvWidthKalman: constant-velocity Kalman on u = 1/w. Width is inversely proportional to distance
     (w = f*W/Z), so u is proportional to Z, and at constant closing speed Z -- hence u -- changes
     LINEARLY in time: a constant-velocity model on u is exactly right, not an approximation.
     TTC = u / (-du/dt). Measurement noise is per-measurement: a fixed width error sigma_w becomes
     sigma_u = sigma_w / w^2 (delta method), so tiny far-away boxes are trusted less than big close ones.
  B. CAWidthKalman: constant-ACCELERATION Kalman on w (state [w, w', w'']), the generic fix for a signal
     whose rate keeps changing. TTC = w / w'.

Same scoring as kitti_kalman_tune.py: every labeled KITTI Car track in 0019/0009/0020, perfect 2D boxes
(isolates the filter from detector noise), against the exact 3D-derived ground-truth TTC; median |error|
overall and on fast-closing frames (gt TTC < 3 s), plus the signed median error on fast frames (positive =
filter reads TOO HIGH, i.e. lagging). Then the autobahn Golf (no ground truth): minimum TTC and when.

Usage:
    uv run python kitti_kalman_models.py
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scripts", _ROOT / "legacy"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

from kitti_gt_ttc import compute_gt_ttc, parse_label_file
from kitti_kalman_tune import EDGE_MARGIN_PX, FAST_CLOSING_GT_TTC_S, SEQUENCES, frame_size
from lead_ttc import TTCKalman
from pathlib import Path


def cv_Q(q, dt):
    return q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])


class InvWidthKalman:
    """Constant-velocity Kalman on u = frame_w / width (proportional to distance). Same interface as TTCKalman."""

    def __init__(self, frame_w, q=1.0, r_meas=2e-5):
        self.frame_w, self.q, self.Rw = float(frame_w), q, r_meas   # Rw: width variance, relative-width units (same as TTCKalman)
        self.x = self.P = self.last_t = self.last_id = None

    def update(self, width, t, track_id, truncated=False):
        w = width / self.frame_w
        u = 1.0 / w
        R = self.Rw / w ** 4                           # var(1/w) ~ var(w) / w^4
        if self.x is None or track_id != self.last_id:
            self.last_id, self.last_t = track_id, t
            self.x = np.array([u, 0.0])
            self.P = np.diag([R, (u / 2) ** 2])        # rate unknown: allow up to "TTC ~ 2 s" either way
            return width, 0.0, np.nan
        dt = t - self.last_t if t > self.last_t else 1 / 30
        self.last_t = t
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + cv_Q(self.q, dt)
        if not truncated:
            S = self.P[0, 0] + R
            K = self.P[:, 0] / S
            self.x = self.x + K * (u - self.x[0])
            self.P = self.P - np.outer(K, self.P[0, :])
        u_est, du = self.x
        ttc = u_est / -du if du < 0 else np.nan        # distance shrinking -> u shrinking
        return self.frame_w / u_est, -du, ttc


class CAWidthKalman:
    """Constant-acceleration Kalman on width: state [w, w', w'']. Same interface as TTCKalman."""

    def __init__(self, frame_w, q=1e-2, r_meas=2e-5):
        self.frame_w, self.q, self.R = float(frame_w), q, r_meas
        self.x = self.P = self.last_t = self.last_id = None

    def update(self, width, t, track_id, truncated=False):
        w = width / self.frame_w
        if self.x is None or track_id != self.last_id:
            self.last_id, self.last_t = track_id, t
            self.x = np.array([w, 0.0, 0.0])
            self.P = np.diag([(10 / 3840) ** 2, 1e-3, 1e-2])
            return width, 0.0, np.nan
        dt = t - self.last_t if t > self.last_t else 1 / 30
        self.last_t = t
        F = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1.0]])
        Q = self.q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                               [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                               [dt ** 3 / 6, dt ** 2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if not truncated:
            S = self.P[0, 0] + self.R
            K = self.P[:, 0] / S
            self.x = self.x + K * (w - self.x[0])
            self.P = self.P - np.outer(K, self.P[0, :])
        w_est, rate, _ = self.x
        ttc = w_est / rate if rate > 0 else np.nan
        return w_est * self.frame_w, rate * self.frame_w, ttc


CANDIDATES = [("CV width (current)", TTCKalman, [1e-3])] + \
    [("CV 1/width", InvWidthKalman, [0.1, 0.3, 1, 3, 10, 30, 100])] + \
    [("CA width", CAWidthKalman, [1e-4, 1e-3, 1e-2, 1e-1, 1])]


def kitti_score(cls, q):
    rows = []
    for seq in SEQUENCES:
        fw = frame_size(seq)
        labels = parse_label_file(Path(f"data/kitti/training/label_02/{seq}.txt"))
        gt = compute_gt_ttc(labels[labels["type"] == "Car"].copy())
        for tid, g in gt.groupby("track_id"):
            f = cls(fw, q=q)
            for r in g.sort_values("frame").itertuples():
                edge = r.bbox_left <= EDGE_MARGIN_PX or r.bbox_right >= fw - EDGE_MARGIN_PX
                _, _, ttc = f.update(r.bbox_right - r.bbox_left, r.frame * 0.1, tid, truncated=edge)
                rows.append((r.gt_ttc_s, ttc, edge))
    d = pd.DataFrame(rows, columns=["truth", "est", "trunc"])
    d = d[~d.trunc].dropna()
    fast = d[d.truth < FAST_CLOSING_GT_TTC_S]
    return (len(d), (d.est - d.truth).abs().median(), len(fast), (fast.est - fast.truth).abs().median(),
            (fast.est - fast.truth).median())


def autobahn_golf(cls, q):
    """Min TTC on the Golf (track 160) and when, from the real detector boxes."""
    tr = pd.read_csv("outputs/autobahn_braking_tracks.csv")
    g = tr[tr.track_id == 160].sort_values("time_s")
    f = cls(1920, q=q)
    out = [(r.time_s, f.update(r.w, r.time_s, 160, truncated=r.x1 <= 2 or r.x2 >= 1918)[2]) for r in g.itertuples()]
    d = pd.DataFrame(out, columns=["t", "ttc"]).dropna()
    d = d[(d.t > 13.5) & (d.t < 16.5)]
    i = d.ttc.idxmin()
    return d.ttc[i], d.t[i], d[(d.t >= 15) & (d.t <= 16.2)].ttc.median()


def main():
    rows = []
    for name, cls, qs in CANDIDATES:
        for q in qs:
            n, e_all, nf, e_fast, bias_fast = kitti_score(cls, q)
            mn, tmn, med = autobahn_golf(cls, q)
            rows.append({"model": name, "q": q, "n": n, "err_all": e_all, "n_fast": nf, "err_fast": e_fast,
                         "bias_fast": bias_fast, "golf_min": mn, "golf_min_t": tmn, "golf_med_15_16.2": med})
            print(rows[-1])
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.round(3).to_string(index=False))
    out.to_csv("outputs/kitti_kalman_models.csv", index=False)


if __name__ == "__main__":
    main()
