"""Classical lane detection (OpenCV: color mask + Canny + Hough) on a video or a KITTI image folder.

Experiment, not wired into the pipeline yet: the question this answers is "can a classical detector
find MY lane's two boundaries reliably on the real demo clips?" before anything depends on it.

Per frame:
  1. ROI: keep only the road between the horizon and the hood (minus any burned-in dashcam inset).
  2. Lane-paint mask: white/yellow pixels (HLS threshold) AND Canny edges, so guardrails, shadows and
     car outlines (strong edges, but not lane-colored) mostly drop out.
  3. HoughLinesP: straight segments. Near-horizontal ones are dropped, the rest split by slope sign
     into left boundary candidates (x shrinks going down the image) and right ones.
  4. Ego-lane pick: on each side keep only the segments whose bottom end is closest to the image
     center (the next lane's line and the guardrail are further out), then fit x = m*y + b
     (x as a function of y, because lane lines are near-vertical in the image).
  5. Smoothing: exponential moving average on (m, b) per side; a side that vanishes (dash gap, a car
     covering the paint) keeps its last fit for up to HOLD_FRAMES frames before being dropped.

Usage:
  uv run python lane_hough.py data/raw/autobahn_braking.mp4 --horizon 0.47 --bottom 0.75 --mask 0.585,0.6,1,1
  uv run python lane_hough.py data/kitti/training/image_02/0020 --horizon 0.52
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

MIN_ABS_SLOPE = 0.4     # |dy/dx| below this = too close to horizontal to be a lane line in front of us
NEAREST_BAND = 0.06     # keep segments whose bottom-x is within this fraction of width of the innermost one
EMA_ALPHA = 0.3         # weight of the new fit in the moving average (higher = reacts faster, jitters more)
HOLD_FRAMES = 10        # keep a side's last fit this many frames after it stops being detected


def frames(source):
    """Yield (frame, fps) from either a video file or a folder of KITTI pngs (KITTI is 10 fps)."""
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.glob("*.png")):
            yield cv2.imread(str(f)), 10.0
    else:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f, fps


def roi_mask(h, w, horizon, bottom, inset):
    """Trapezoid from the bottom corners up to a narrow strip at the horizon, minus the inset rectangle."""
    m = np.zeros((h, w), np.uint8)
    yt, yb = int(horizon * h), int(bottom * h)
    pts = np.array([[0, yb], [int(0.42 * w), yt], [int(0.58 * w), yt], [w, yb]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    if inset:
        x1, y1, x2, y2 = inset
        m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
    return m


def paint_edges(frame, roi):
    """Canny edges restricted to white/yellow pixels inside the ROI."""
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 170, 0), (180, 255, 255))      # high lightness, any hue
    yellow = cv2.inRange(hls, (15, 80, 90), (40, 255, 255))     # yellow hue, fairly saturated
    paint = cv2.dilate(white | yellow, np.ones((5, 5), np.uint8))  # widen so edges at paint borders survive
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges & paint & roi


def fit_side(segs, y_bottom, w, side):
    """Fit x = m*y + b to the innermost segments on one side. segs: list of (x1, y1, x2, y2)."""
    if not segs:
        return None
    def x_at(s, y):
        x1, y1, x2, y2 = s
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    bx = np.array([x_at(s, y_bottom) for s in segs])
    inner = bx.max() if side == "left" else bx.min()   # left line: innermost = largest x
    keep = [s for s, x in zip(segs, bx) if abs(x - inner) < NEAREST_BAND * w]
    ys = np.array([v for s in keep for v in (s[1], s[3])], float)
    xs = np.array([v for s in keep for v in (s[0], s[2])], float)
    lengths = np.repeat([np.hypot(s[2] - s[0], s[3] - s[1]) for s in keep], 2)
    m, b = np.polyfit(ys, xs, 1, w=lengths)   # longer segments count more
    return m, b


def detect(frame, roi, y_bottom):
    """One frame -> (left_fit, right_fit, raw_segments); a fit is (m, b) or None."""
    h, w = frame.shape[:2]
    lines = cv2.HoughLinesP(paint_edges(frame, roi), rho=1, theta=np.pi / 180, threshold=20,
                            minLineLength=int(0.02 * w), maxLineGap=int(0.01 * w))
    left, right = [], []
    for x1, y1, x2, y2 in (lines.reshape(-1, 4).tolist() if lines is not None else []):
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            continue
        mid = (x1 + x2) / 2
        if slope < 0 and mid < 0.6 * w:     # left boundary: rises to the right, on the left half
            left.append((x1, y1, x2, y2))
        elif slope > 0 and mid > 0.4 * w:   # right boundary: rises to the left, on the right half
            right.append((x1, y1, x2, y2))
    return fit_side(left, y_bottom, w, "left"), fit_side(right, y_bottom, w, "right"), left + right


class SideSmoother:
    """EMA over (m, b) for one lane boundary, with a short hold when the detection drops out."""
    def __init__(self):
        self.fit, self.missing = None, 0

    def update(self, fit):
        if fit is None:
            self.missing += 1
            if self.missing > HOLD_FRAMES:
                self.fit = None
        else:
            self.missing = 0
            new = np.array(fit)
            self.fit = new if self.fit is None else EMA_ALPHA * new + (1 - EMA_ALPHA) * self.fit
        return self.fit


def draw(frame, left, right, segs, y_top, y_bottom, roi):
    out = frame.copy()
    # faint ROI outline so it's visible what the detector is allowed to look at
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 1)          # raw Hough segments in red
    pts = {}
    for name, fit in (("L", left), ("R", right)):
        if fit is not None:
            m, b = fit
            pts[name] = [(int(m * y_bottom + b), y_bottom), (int(m * y_top + b), y_top)]
            cv2.line(out, *pts[name], (0, 255, 255), 3)                # fitted boundary in yellow
    if len(pts) == 2:   # both sides -> shade the ego lane
        poly = np.array([pts["L"][0], pts["L"][1], pts["R"][1], pts["R"][0]], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--horizon", type=float, default=0.5)   # top of the ROI, fraction of frame height
    ap.add_argument("--bottom", type=float, default=1.0)    # bottom of the ROI (above the hood), fraction of height
    ap.add_argument("--mask", default=None)                 # x1,y1,x2,y2 fractions of a burned-in inset to ignore
    ap.add_argument("--start", type=float, default=0.0)     # seconds to skip
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    inset = [float(v) for v in args.mask.split(",")] if args.mask else None

    name = Path(args.source).stem if Path(args.source).is_file() else "kitti_" + Path(args.source).name
    args.outdir.mkdir(exist_ok=True)
    writer, roi = None, None
    sm_l, sm_r = SideSmoother(), SideSmoother()
    rows = []
    for i, (frame, fps) in enumerate(frames(args.source)):
        t = i / fps
        if t < args.start:
            continue
        if args.duration is not None and t > args.start + args.duration:
            break
        h, w = frame.shape[:2]
        if roi is None:
            roi = roi_mask(h, w, args.horizon, args.bottom, inset)
            writer = cv2.VideoWriter(str(args.outdir / f"{name}_lanes_hough.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        y_top, y_bottom = int(args.horizon * h), int(args.bottom * h) - 1
        raw_l, raw_r, segs = detect(frame, roi, y_bottom)
        left, right = sm_l.update(raw_l), sm_r.update(raw_r)
        writer.write(draw(frame, left, right, segs, y_top, y_bottom, roi))
        rows.append({"t": round(t, 3), "left_raw": raw_l is not None, "right_raw": raw_r is not None,
                     "left": left is not None, "right": right is not None})
    writer.release()
    with open(args.outdir / f"{name}_lanes_hough.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    n = len(rows)
    both_raw = sum(r["left_raw"] and r["right_raw"] for r in rows)
    both = sum(r["left"] and r["right"] for r in rows)
    print(f"{n} frames: both sides detected raw {both_raw / n:.1%}, after hold/smoothing {both / n:.1%}")


if __name__ == "__main__":
    main()
