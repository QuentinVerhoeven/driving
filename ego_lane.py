"""Ego-lane boundaries from cached YOLOP lane-line masks (see lane_yolop.py --no-video).

YOLOP marks every lane-line pixel in the frame; this turns that into the LEFT and RIGHT boundary of MY
lane as curves x = a*y^2 + b*y + c in frame pixels (x as a function of y, because lane lines are
near-vertical in the image), one pair per frame, smoothed over time.

Per frame:
  1. Row scan, bottom to top: on each row find the separate blobs of lane pixels. Relative to a
     reference x (the middle of my lane: the image center at the bottom row, then the midpoint of the
     two lines found as the scan moves up), the nearest blob on each side is a candidate boundary point.
  2. RANSAC: where a dash gap leaves my own line empty on a row, the scan grabs the NEXT lane's line
     instead. Those points don't line up with the majority, so the largest set of points consistent with
     one straight line is kept and the rest dropped. (A first version chained each point to the one
     below and to last frame's fit instead; it locked onto whichever line it saw first and then rejected
     the real one -- see NOTES.md, 2026-10-01.)
  3. Fit per side: straight line if the visible paint is short, quadratic (curve) only when it spans
     enough height to actually show curvature. One outlier-rejection pass, then refit.
  4. Over time (EgoLaneTracker): a fit that jumps implausibly far from the current estimate is treated as
     not seen (unless the jump persists, then it's real and gets taken); moving average on the
     coefficients; a side that disappears is held for
     HOLD_S; a side missing while the other is seen is inferred from the last known lane width
     (right - left, itself a polynomial, so it narrows correctly toward the horizon) -- the common case
     when the lead car covers one line.

Outputs outputs/<name>_egolane.csv (one row per frame: both boundaries' coefficients + where each came
from: seen / held / inferred / none), and optionally an overlay video.

Usage:
  uv run python ego_lane.py autobahn_braking --horizon 0.47 --bottom 0.75 --mask 0.585,0.6,1,1 --video data/raw/autobahn_braking.mp4
  uv run python ego_lane.py kitti_0020 --horizon 0.48 --video data/kitti/training/image_02/0020
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lane_hough import frames
from lead_ttc import EgoLaneBounds

CACHE = Path("cache/yolop")
ROW_STEP = 2            # scan every 2nd mask row
MIN_POINTS = 6          # fewer accepted points than this on a side = no fit
MIN_SPAN = 0.06         # ...or if they cover less than this fraction of frame height
QUAD_SPAN = 0.15        # need at least this much height before allowing a curved (quadratic) fit
RANSAC_TOL = 0.015      # a point within this fraction of frame width of the RANSAC line counts as on it
EMA_ALPHA = 0.3         # weight of the new fit in the moving average
HOLD_S = 1.0            # keep a vanished side this long before dropping it
GATE = 0.06             # a new fit further than this (fraction of frame width) from the current estimate = outlier
REACQUIRE_S = 0.5       # ...unless it keeps landing there this long: then it's real (curve, lane shift), take it.
                        # 1.0 and 1.5 s were tried: they also PROTECT a wrong first estimate (the autobahn clip's
                        # first right-edge fit lands on the next lane's line) long enough to cause a false
                        # rapid-closing event at t=1.5 s. 0.5 s is the longest that avoids it (NOTES.md, 2026-10-01).
MIN_WIDTH, MAX_WIDTH = 0.12, 1.3   # plausible lane width at the bottom row, fraction of frame width


def load_masks(name):
    d = np.load(CACHE / f"{name}_lanemask.npz")
    masks = np.unpackbits(d["packed"], axis=2)[:, :, :int(d["width"])]
    return masks, float(d["fps"]), int(d["frame_w"]), int(d["frame_h"])


def run_centers(row):
    """x centers of each contiguous run of lane pixels in one mask row."""
    xs = np.flatnonzero(row)
    if len(xs) == 0:
        return []
    runs = np.split(xs, np.where(np.diff(xs) > 2)[0] + 1)
    return [r.mean() for r in runs]


def scan(mask, scale, y_top, y_bot, frame_w):
    """Collect (x, y) boundary points in FRAME pixels for the left and right side of my lane.

    Every frame starts from the image center at the bottom row -- the camera sits in the middle of the
    car, so that point is always in my lane -- and follows the lane's center upward. Deliberately NOT
    seeded from last frame's lane: a first version did that, drifted into the neighboring lane on KITTI
    0020, and then kept itself there, since every later frame searched around the wrong center.
    """
    pts = {"l": [], "r": []}
    ref = frame_w / 2
    for my in range(int(y_bot / scale) - 1, int(y_top / scale), -ROW_STEP):
        y = my * scale
        centers = [c * scale for c in run_centers(mask[my])]
        found = {"l": max((c for c in centers if c < ref), default=None),
                 "r": min((c for c in centers if c > ref), default=None)}
        for side, c in found.items():
            if c is not None:
                pts[side].append((c, y))
        if found["l"] is not None and found["r"] is not None:
            ref = (found["l"] + found["r"]) / 2   # follow the lane center upward as it shifts with curves/perspective
    return pts["l"], pts["r"]


def ransac_inliers(xs, ys, tol, iters=100):
    """Boolean mask of the largest set of points lying within tol (px) of one straight line through 2 of them.
    Rows where my own line has a dash gap contribute a point on the NEXT lane's line instead; those don't
    line up with the majority, so they end up outside the winning set."""
    rng = np.random.default_rng(0)   # fixed seed: same input -> same output, run to run
    best = np.zeros(len(xs), bool)
    for _ in range(iters):
        i, j = rng.choice(len(xs), 2, replace=False)
        if ys[i] == ys[j]:
            continue
        pred = xs[i] + (xs[j] - xs[i]) / (ys[j] - ys[i]) * (ys - ys[i])
        inl = np.abs(xs - pred) < tol
        if inl.sum() > best.sum():
            best = inl
    return best


def fit(points, frame_w, frame_h):
    """x = a*y^2 + b*y + c through the points (a = 0 for a straight fit), or None."""
    if len(points) < MIN_POINTS:
        return None
    xs, ys = np.array(points).T
    keep = ransac_inliers(xs, ys, RANSAC_TOL * frame_w)
    if keep.sum() < MIN_POINTS:
        return None
    xs, ys = xs[keep], ys[keep]
    span = ys.max() - ys.min()
    if span < MIN_SPAN * frame_h:
        return None
    deg = 2 if span >= QUAD_SPAN * frame_h and len(xs) >= 12 else 1
    for _ in range(2):   # fit, drop outliers, refit
        c = np.polyfit(ys, xs, deg)
        resid = np.abs(xs - np.polyval(c, ys))
        keep = resid < max(3 * np.median(resid), 0.01 * frame_w)
        if keep.all() or keep.sum() < MIN_POINTS:
            break
        xs, ys = xs[keep], ys[keep]
    return np.concatenate([[0.0] * (2 - deg), c])


class EgoLaneTracker:
    """Smooths per-frame boundary fits and fills short gaps. update() returns (left, right, sources)."""

    def __init__(self, frame_w, y_top, y_bot):
        self.frame_w, self.y_top, self.y_bot = frame_w, y_top, y_bot
        self.fit = {"l": None, "r": None}
        self.last_seen = {"l": None, "r": None}
        self.width = None   # right - left coefficients, from the last frame where both sides were seen
        self.gated_since = {"l": None, "r": None}   # start of the current run of gated-out fits, per side

    def plausible(self, l, r):
        width_bottom = np.polyval(r - l, self.y_bot)
        return MIN_WIDTH * self.frame_w < width_bottom < MAX_WIDTH * self.frame_w and \
            np.polyval(r - l, self.y_top) > 0

    def blend(self, old, new):
        return new if old is None else EMA_ALPHA * new + (1 - EMA_ALPHA) * old

    def gate(self, side, raw, t):
        """Reject a fit that jumps implausibly far from the current estimate (returns None = "not seen this
        frame", so the old edge is held). On the autobahn clip's dashed right line, a dash gap sometimes
        leaves more points on the NEXT lane's line than on mine, RANSAC's majority then picks that line, and
        the edge jumped ~450 px for single frames -- visible flicker even through the moving average
        (NOTES.md, 2026-10-01). Compared mid-way down the fitted range, inside the visible paint. Escape
        hatch against getting stuck on a stale estimate: a jump that persists for REACQUIRE_S is accepted."""
        old = self.fit[side]
        if raw is None or old is None:
            return raw
        y = (self.y_top + self.y_bot) / 2
        if abs(np.polyval(raw, y) - np.polyval(old, y)) <= GATE * self.frame_w:
            self.gated_since[side] = None
            return raw
        if self.gated_since[side] is None:
            self.gated_since[side] = t
        if t - self.gated_since[side] >= REACQUIRE_S:
            self.gated_since[side] = None
            self.fit[side] = None    # start over from the new position instead of averaging into it
            return raw
        return None

    def crossing(self, l, r):
        """Both sides fitted to the SAME line: it runs right under the camera, i.e. my car is straddling it
        mid lane change. Then there is no single "my lane" to report."""
        return abs(np.polyval(r - l, self.y_bot)) < MIN_WIDTH * self.frame_w

    def update(self, raw_l, raw_r, t):
        if raw_l is not None and raw_r is not None and self.crossing(raw_l, raw_r):
            # Lane change in progress: drop the lane NOW instead of holding the old one (the caller falls back
            # to the band). Holding it for HOLD_S kept a car in the lane I was leaving as "lead" through a whole
            # lane change on KITTI 0020 (~17.7 s) and produced a false rapid-closing event (NOTES.md, 2026-10-01).
            self.fit = {"l": None, "r": None}
            return None, None, {"l": "none", "r": "none"}
        raw_l, raw_r = self.gate("l", raw_l, t), self.gate("r", raw_r, t)
        if raw_l is not None and raw_r is not None and not self.plausible(raw_l, raw_r):
            raw_l = raw_r = None   # an impossible pair (crossed lines, absurd width): trust neither
        if raw_l is not None and raw_r is not None:
            new_w = raw_r - raw_l
            self.width = self.blend(self.width, new_w)
        sources = {}
        for side, raw in (("l", raw_l), ("r", raw_r)):
            if raw is not None:
                old = self.fit[side]
                self.fit[side] = self.blend(old, raw)
                self.last_seen[side] = t
                sources[side] = "seen"
            elif self.fit[side] is not None and t - self.last_seen[side] <= HOLD_S:
                sources[side] = "held"
            else:
                self.fit[side] = None
                sources[side] = "none"
        # one side gone for good but the other still there: rebuild it from the last known lane width
        if self.width is not None:
            if sources["l"] == "none" and sources["r"] != "none":
                self.fit["l"], sources["l"] = self.fit["r"] - self.width, "inferred"
            elif sources["r"] == "none" and sources["l"] != "none":
                self.fit["r"], sources["r"] = self.fit["l"] + self.width, "inferred"
        return self.fit["l"], self.fit["r"], sources


def load_egolane(path):
    """{frame index: EgoLaneBounds or None} from an <name>_egolane.csv written by main() below."""
    df = pd.read_csv(path)
    return {int(r.frame): EgoLaneBounds.from_row(r) for r in df.itertuples()}


# Lane "carpet" colors (BGR) for the demo renderers, keyed like lead_ttc.ttc_urgency(); None = no TTC yet
CARPET_COLOR = {None: (230, 120, 20), "safe": (230, 120, 20), "caution": (0, 170, 255), "urgent": (30, 30, 230),
                "critical": (0, 0, 230)}   # same red as urgent, but drawn much more opaque (CRITICAL_ALPHA)
CRITICAL_ALPHA = 0.65   # the critical carpet is much more opaque than the others
CARPET_ALPHA = 0.45     # fill opacity at the bottom of the frame (0.30 was too faint to see well, per user)
EDGE_ALPHA = 0.85       # boundary line opacity at the bottom of the frame
FAR_FADE = 0.35         # opacity multiplier at the far end, so the carpet fades into the distance


def draw_ego_lane(img, lane, urgency=None, y_end=None, inset=None):
    """Draw my lane as a translucent "carpet" (in place), like an ADAS display: filled between the two
    boundaries, fading toward the horizon, stopping at y_end (the lead car's bumper, if there is one) so it
    reads as "the road between me and the car I'm tracking". Colored by TTC urgency. inset: x1,y1,x2,y2
    fractions of a burned-in overlay (the autobahn clip's speedometer) to keep clear.
    Draw this BEFORE the boxes, so it sits under them. Critical tier: a steady, much more opaque red
    (a flashing black/red version was tried and dropped at the user's request, 2026-10-01)."""
    if lane is None:
        return
    h, w = img.shape[:2]
    y0 = int(max(lane.y_top, y_end if y_end is not None else lane.y_top))
    y1 = int(min(lane.y_bot, h - 1))
    if y1 - y0 < 4:
        return
    ys = np.arange(y0, y1 + 1, 3, dtype=float)
    lx, rx = np.polyval(lane.left, ys), np.polyval(lane.right, ys)
    left_pts = np.stack([lx, ys], 1).astype(np.int32)
    right_pts = np.stack([rx, ys], 1).astype(np.int32)
    fill = np.zeros((h, w), np.uint8)
    cv2.fillPoly(fill, [np.vstack([left_pts, right_pts[::-1]])], 255)
    edge = np.zeros((h, w), np.uint8)
    thick = max(2, round(w / 640))
    for pts in (left_pts, right_pts):
        cv2.polylines(edge, [pts], False, 255, thick, cv2.LINE_AA)
    if inset:
        x1, iy1, x2, iy2 = inset
        fill[int(iy1 * h):int(iy2 * h), int(x1 * w):int(x2 * w)] = 0
        edge[int(iy1 * h):int(iy2 * h), int(x1 * w):int(x2 * w)] = 0
    rows = slice(y0, y1 + 1)
    fade = np.linspace(FAR_FADE, 1.0, y1 - y0 + 1, dtype=np.float32)[:, None]   # far end faint, near end strong
    fill_alpha = CRITICAL_ALPHA if urgency == "critical" else CARPET_ALPHA
    color = CARPET_COLOR[urgency]
    alpha = np.maximum(fill[rows] / 255 * fill_alpha, edge[rows] / 255 * EDGE_ALPHA) * fade
    alpha = alpha[..., None]
    roi = img[rows].astype(np.float32)
    img[rows] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")                                   # cache name, e.g. autobahn_braking or kitti_0020
    ap.add_argument("--horizon", type=float, default=0.48)    # top of the scanned region, fraction of height
    ap.add_argument("--bottom", type=float, default=1.0)      # bottom of the scanned region (above the hood)
    ap.add_argument("--mask", default=None)                   # x1,y1,x2,y2 fractions of a burned-in inset to ignore
    ap.add_argument("--video", default=None)                  # source video / image folder: also render an overlay
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--gate", type=float, default=GATE)            # for comparing settings; 0 disables the gate
    ap.add_argument("--reacquire", type=float, default=REACQUIRE_S)
    args = ap.parse_args()
    globals().update(GATE=args.gate if args.gate > 0 else 1e9, REACQUIRE_S=args.reacquire)

    masks, fps, fw, fh = load_masks(args.name)
    scale = fw / masks.shape[2]   # mask px -> frame px (letterbox kept the aspect ratio, so one factor)
    if args.mask:
        x1, y1, x2, y2 = [float(v) for v in args.mask.split(",")]
        mh, mw = masks.shape[1:]
        masks[:, int(y1 * mh):int(y2 * mh), int(x1 * mw):int(x2 * mw)] = 0
    y_top, y_bot = args.horizon * fh, args.bottom * fh - 1

    tracker = EgoLaneTracker(fw, y_top, y_bot)
    rows = []
    for i, mask in enumerate(masks):
        t = i / fps
        pl, pr = scan(mask, scale, y_top, y_bot, fw)
        left, right, src = tracker.update(fit(pl, fw, fh), fit(pr, fw, fh), t)
        row = {"frame": i, "t": round(t, 4), "y_top": y_top, "y_bot": y_bot, "l_src": src["l"], "r_src": src["r"]}
        for side, c in (("l", left), ("r", right)):
            for k, v in zip("abc", c if c is not None else [np.nan] * 3):
                row[f"{side}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = args.outdir / f"{args.name}_egolane.csv"
    df.to_csv(out_csv, index=False)

    both = ((df.l_src != "none") & (df.r_src != "none")).mean()
    seen = ((df.l_src == "seen") & (df.r_src == "seen")).mean()
    print(f"{len(df)} frames: lane available {both:.1%} (both sides directly seen {seen:.1%}) -> {out_csv}")

    if args.video:
        writer = None
        for (frame, _), r in zip(frames(args.video), df.itertuples()):
            if writer is None:
                writer = cv2.VideoWriter(str(args.outdir / f"{args.name}_egolane.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
            lane = EgoLaneBounds.from_row(r)
            if lane is not None:
                ys = np.linspace(y_top, y_bot, 30)
                lx, rx = np.polyval(lane.left, ys), np.polyval(lane.right, ys)
                poly = np.array(list(zip(lx, ys)) + list(zip(rx, ys))[::-1], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], (0, 200, 0))
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                for xs, src in ((lx, r.l_src), (rx, r.r_src)):
                    color = {"seen": (0, 255, 255), "held": (0, 165, 255), "inferred": (255, 0, 255)}[src]
                    cv2.polylines(frame, [np.array(list(zip(xs, ys)), np.int32)], False, color, 2)
            cv2.putText(frame, f"L:{r.l_src} R:{r.r_src}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()


if __name__ == "__main__":
    main()
