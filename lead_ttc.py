"""
    Streaming lead-vehicle selection + Kalman-filtered TTC.

    Same logic as explore.ipynb, but written so it can be fed ONE FRAME AT A TIME
    (which is all a live camera gives us). Each class keeps its own state between
    calls instead of looking at the whole clip at once.

    Usage (per frame):
        lead = selector.update(boxes, t)       # boxes: list of Box, t: frame time in seconds
        if lead is not None:
            w_est, rate, ttc = kalman.update(lead.w, t, lead.track_id)
"""

from dataclasses import dataclass

import numpy as np

# --- lead-selection settings (values tuned on clip1_30s, see NOTES.md) ---
# BAND_HALF_WIDTH retuned 2026-09-29 from 0.10 to 0.07: the original value was tuned on clip1_30s
# (single-lane-each-way) and never checked against a multi-lane highway. On KITTI 0020 (a real
# Autobahn scene) it let an adjacent-lane car win because it was numerically "in the band" while
# being bigger/closer than the true same-lane traffic. Retuned by grid search against ground-truth
# lead picks (LeadSelector run on real KITTI labels) across 3 sequences (kitti_band_tune.py) --
# 0.07 gave the best pooled IoU-agreement (0.798 vs 0.713 at 0.10), and specifically fixed 0020
# (0.83 -> 0.98 agreement) without hurting 0019 (stays >0.93) or clip1_30s (same lead identity
# picked throughout at every width tested). See NOTES.md, 2026-09-29.
BAND_HALF_WIDTH = 0.07   # "in my lane" = box center within +-7% of frame width from the (adjustable) center
MAX_ASPECT_RATIO = 2.5   # w/h above this = self-detection of our own hood, not a car
BOTTOM_MARGIN = 0.95     # boxes whose bottom edge is in the bottom 5% of the frame = our hood
# Streaks are in SECONDS (and at least MIN_FRAMES frames), not frames: the live loop only sees ~10 frames/s, so a
# frame count would mean 3x more real time than the 30 fps clips these were tuned on (10 frames = 9 gaps = 0.30 s,
# 5 frames = 4 gaps = 0.13 s at 30 fps). LOST_TIME was then raised to 0.30 s: at ~10 fps the direct conversion (0.13 s)
# picked a wrong car once on a clip where the lead never changes; 0.30 s gave 100% at both rates (see NOTES.md).
SWITCH_TIME = 0.30       # s a challenger must stay the biggest candidate before we switch to it
LOST_TIME = 0.30         # s the current lead must be absent before we give up on it
MIN_FRAMES = 2           # ...but never on a single frame, however slow the frame rate


@dataclass
class Box:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.w * self.h

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2


class LeadSelector:
    def __init__(self, frame_w, frame_h, band_half=BAND_HALF_WIDTH, center_offset=0.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Lane band settings, adjustable at run time because a phone on a windshield is never perfectly centered:
        # the band is centered at (0.5 + center_offset) of the frame width and extends +-band_half either side.
        self.band_half = band_half
        self.center_offset = center_offset
        # state that persists between frames
        self.current_lead = None       # track_id of the car we currently call "the lead"
        self.challenger_id = None      # a different car that is currently the biggest candidate
        self.challenger_since = None   # time it first became the biggest candidate
        self.challenger_n = 0          # number of frames it has been
        self.missing_since = None      # time the current lead was first not seen
        self.missing_n = 0             # number of frames it has been absent

    def band(self):
        """Left and right edge of the lane band as fractions of the frame width."""
        center = 0.5 + self.center_offset
        return center - self.band_half, center + self.band_half

    def _candidates(self, boxes):
        """Boxes in our lane band that are not our own hood.

        The self-detection check (aspect ratio, bottom margin) is skipped for whichever box is
        ALREADY our established lead: a real lead car closing to near-collision range ends up filling
        the bottom of the frame almost identically to our own hood (occupies most of the frame width,
        bottom edge near the frame edge), so applying the same check to it would reject the real lead
        at exactly the range TTC matters most (see NOTES.md, 2026-09-28). New candidates still get the
        full check, so a stray hood-like box can never be picked as a fresh lead in the first place --
        only a car already confirmed as the lead is grandfathered in as it grows.
        """
        lo, hi = self.band()
        out = []
        for b in boxes:
            in_lane = lo * self.frame_w < b.cx < hi * self.frame_w
            if not in_lane:
                continue
            if b.track_id == self.current_lead:
                out.append(b)
                continue
            not_self = b.w / b.h < MAX_ASPECT_RATIO and b.y2 < BOTTOM_MARGIN * self.frame_h
            if not_self:
                out.append(b)
        return out

    def _switch_to(self, box):
        self.current_lead = box.track_id
        self.missing_since, self.missing_n = None, 0
        self.challenger_id, self.challenger_since, self.challenger_n = None, None, 0
        return box

    def _lead_missing(self, t):
        if self.missing_since is None:
            self.missing_since = t
        self.missing_n += 1

    def update(self, boxes, t):
        """Feed one frame's tracked boxes and its timestamp t (s). Returns the lead Box, or None if there is no pick."""
        cands = self._candidates(boxes)
        if not cands:
            self._lead_missing(t)
            return None

        top = max(cands, key=lambda b: b.area)

        if self.current_lead is None:
            return self._switch_to(top)

        lead = next((b for b in cands if b.track_id == self.current_lead), None)

        if lead is None:
            # current lead not seen this frame; give up only after it has been gone long enough
            self._lead_missing(t)
            if self.missing_n >= MIN_FRAMES and t - self.missing_since >= LOST_TIME:
                return self._switch_to(top)
            return None

        self.missing_since, self.missing_n = None, 0

        if top.track_id == self.current_lead:
            self.challenger_id, self.challenger_since, self.challenger_n = None, None, 0
            return lead

        # a different car is biggest: measure how long it has been
        if top.track_id == self.challenger_id:
            self.challenger_n += 1
        else:
            self.challenger_id, self.challenger_since, self.challenger_n = top.track_id, t, 1

        if self.challenger_n >= MIN_FRAMES and t - self.challenger_since >= SWITCH_TIME:
            return self._switch_to(top)
        return lead


class TTCKalman:
    """Kalman filter on box width. State x = [width, rate dw/dt]. TTC = width / rate.

    Two design choices so the behaviour does not depend on the camera setup (see NOTES.md):
    - Width is filtered in units of FRAME WIDTH, not pixels, so the noise constants mean the same thing at 640 px,
      1080p or 4K. TTC is a ratio, so it is unaffected; the returned width/rate are converted back to pixels.
    - Process noise Q is the textbook continuous-white-noise form q * [[dt^3/3, dt^2/2], [dt^2/2, dt]], so the filter
      responds to the same real-time changes whatever the time between measurements (30 fps or ~10 fps).
      (The original diag(q*dt) form was tuned at 30 fps and lost most closing events at ~10 fps.)
    q and r_meas were tuned so it agrees with the original filter at every frame and stays consistent when only every
    3rd measurement is used. They are NOT tuned against ground truth yet.
    """

    def __init__(self, frame_w, q=5e-5, r_meas=2e-5):
        self.frame_w = float(frame_w)
        self.q = q            # how fast the rate may change unexplained (relative-width units)
        self.R = r_meas       # variance of one width measurement (relative-width units): sigma ~ 0.0045 of the frame width
        self.x = None         # None until the first measurement arrives
        self.P = None
        self.last_t = None
        self.last_id = None   # which track the state belongs to

    def update(self, width, t, track_id, truncated=False):
        """Fold in one width measurement (pixels) taken at time t (s) for car `track_id`.

        `truncated=True` means the box is clipped by the frame edge (car exiting frame): its width no
        longer reflects the car's true extent and can even shrink while the car keeps closing (see
        NOTES.md, 2026-09-28). We still predict forward so the estimate keeps moving, but skip folding
        in that measurement -- coasting on the last trusted rate is better than trusting a lying one.

        Returns (width_px, rate_px_per_s, ttc).
        """
        w = width / self.frame_w
        if self.x is None or track_id != self.last_id:
            # first measurement, or the lead is a DIFFERENT car: the old [width, rate] describes another vehicle,
            # and folding the new width in would look like a huge fake closing speed
            self.last_id = track_id
            self.x = np.array([w, 0.0])                 # trust the width, know nothing about the rate
            sigma0 = 10 / 3840                          # ~10 px on a 4K frame, as a fraction of frame width
            self.P = np.eye(2) * sigma0 ** 2
            self.last_t = t
            return self.x[0] * self.frame_w, 0.0, np.nan

        dt = t - self.last_t
        if dt <= 0:
            dt = 1 / 30   # guard against duplicate/out-of-order timestamps
        self.last_t = t

        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])   # bigger gap -> trust prediction less

        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        if not truncated:
            # update (H = [1, 0]: we only measure width)
            y = w - self.x[0]
            S = self.P[0, 0] + self.R
            K = self.P[:, 0] / S
            self.x = self.x + K * y
            self.P = self.P - np.outer(K, self.P[0, :])   # (I - K H) P with H = [1, 0]
        # else: predicted state stands uncorrected; P keeps growing via Q, so the filter is honestly
        # less certain for as long as the box stays truncated.

        w_est, rate = self.x
        ttc = w_est / rate if rate > 0 else np.nan   # only meaningful while the box is growing
        return w_est * self.frame_w, rate * self.frame_w, ttc


EDGE_MARGIN_PX = 2.0     # box within this many px of x=0 or x=frame_w counts as truncated (see TTCKalman.update)
STALE_AFTER_S = 1.0      # drop a vehicle's filter if it hasn't been seen for this long


class NearbyVehicleTTC:
    """TTC for EVERY tracked vehicle, not just the selected lead -- situational awareness, not a second
    lead-selector. Same box-width-expansion method as TTCKalman, just one filter per track id instead of
    one singleton, and each gets the same edge-truncation handling (see NOTES.md, 2026-09-28).

    Deliberately does NOT try to figure out which of these vehicles might cut in, merge, or cross our
    path -- that's real trajectory/intent prediction, out of scope (see CLAUDE.md, "Scope"). This only
    answers "how is each currently-visible vehicle's box growing," the same question TTCKalman already
    answers for the lead, applied to everything else in view too.
    """

    def __init__(self, frame_w):
        self.frame_w = frame_w
        self.filters = {}      # track_id -> TTCKalman
        self.last_seen = {}    # track_id -> t

    def update(self, boxes, t):
        """Feed one frame's tracked boxes (list of Box) and its timestamp t (s).

        Returns {track_id: (width_px, rate_px_per_s, ttc)} for every box this frame.
        """
        results = {}
        for b in boxes:
            if b.track_id not in self.filters:
                self.filters[b.track_id] = TTCKalman(self.frame_w)
            truncated = b.x1 <= EDGE_MARGIN_PX or b.x2 >= self.frame_w - EDGE_MARGIN_PX
            results[b.track_id] = self.filters[b.track_id].update(b.w, t, b.track_id, truncated=truncated)
            self.last_seen[b.track_id] = t

        # Drop filters for vehicles not seen in a while, so memory doesn't grow unbounded over a long
        # clip -- every track this pipeline has ever seen would otherwise be kept forever.
        stale_ids = [tid for tid, last_t in self.last_seen.items() if t - last_t > STALE_AFTER_S]
        for tid in stale_ids:
            del self.filters[tid]
            del self.last_seen[tid]

        return results
