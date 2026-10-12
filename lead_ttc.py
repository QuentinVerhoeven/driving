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
# BAND_HALF_WIDTH retuned 2026-09-30 from 0.07 to 0.08: investigating why the rapid-closing flag never
# fired on the autobahn clip (see NOTES.md) found the true root cause wasn't the trust-window guard --
# it was the band excluding a real, continuously-closing same-lane car (track 160) for most of its
# approach because its box center drifted to 0.071-0.081 deviation, just past the 0.07 cutoff, while it
# grew from 18px to 180px wide. Grid-searched BAND_HALF_WIDTH again against all 3 KITTI sequences
# (kitti_band_tune.py) plus a direct LeadSelector+TTCKalman+RapidClosingDetector simulation against the
# autobahn clip's real detector output (not just a raw-position check -- an earlier simulation wrongly
# suggested 0.075 was enough; re-simulating the actual pipeline end to end showed 0.075 still leaves a gap
# where other real cars grab the lead, splitting track 160's stint in two, and only 0.08 keeps it
# continuously in band long enough to clear MIN_TRUST_AFTER_SWITCH_S + MIN_EVENT_S). This is a real,
# small tradeoff, not a free win: KITTI pooled agreement drops slightly at 0.08 vs 0.07 (0.790 vs 0.798;
# 0009 actually improves, 0.295 vs 0.286, but 0020 dips a little, 0.968 vs 0.984). Accepted because the
# regression is small and the autobahn clip's rapid-closing flag firing on a genuine near-miss (see
# NOTES.md, 2026-09-30) was judged worth it. A hysteresis-based approach (don't drop a car from
# contention for one brief excursion outside the band) would likely fix this without any KITTI cost at
# all, but is a bigger design change -- noted as a possible follow-up, not built today.
BAND_HALF_WIDTH = 0.08   # "in my lane" = box center within +-8% of frame width from the (adjustable) center
LANE_OVERLAP_FRAC = 0.5  # with a detected lane: share of the box's bottom edge that must be inside it (EgoLaneBounds)
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


@dataclass
class EgoLaneBounds:
    """My lane's left/right boundary as x = a*y^2 + b*y + c in frame pixels (from ego_lane.py), valid
    between rows y_top and y_bot. Replaces the fixed vertical band when available (see NOTES.md, 2026-10-01):
    a lane is a trapezoid/curve in the image, which a constant-width vertical strip can't follow."""
    left: np.ndarray
    right: np.ndarray
    y_top: float
    y_bot: float

    def bounds_at(self, y):
        """(left x, right x) of the lane at row y. y is clamped into the fitted range, since the curves
        are only trustworthy where there was paint to fit them to."""
        y = min(max(y, self.y_top), self.y_bot)
        return np.polyval(self.left, y), np.polyval(self.right, y)

    def contains_box(self, box):
        """Is most of the box's bottom edge (where the tyres touch the road) inside my lane at that row?
        Overlap of [x1, x2] with [left, right] must be at least LANE_OVERLAP_FRAC of the narrower of the
        two. A first version tested only the bottom-CENTER point, which was too brittle far away: near the
        horizon the lane is a few pixels wide, so a few pixels of fit error excluded the real lead (KITTI
        0009). Measuring overlap scales the tolerance with distance, since a far car is about as wide as
        the lane there. A car cutting in counts once more than half of it has crossed the line."""
        left, right = self.bounds_at(box.y2)
        overlap = min(box.x2, right) - max(box.x1, left)
        return overlap >= LANE_OVERLAP_FRAC * min(box.w, right - left)

    @classmethod
    def from_row(cls, r):
        """Build from one row of an <name>_egolane.csv (as a namedtuple); None if either side is missing."""
        if r.l_src == "none" or r.r_src == "none":
            return None
        return cls(np.array([r.l_a, r.l_b, r.l_c]), np.array([r.r_a, r.r_b, r.r_c]), r.y_top, r.y_bot)


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

    def _candidates(self, boxes, lane=None):
        """Boxes in our lane band that are not our own hood.

        With a detected lane (EgoLaneBounds), "in our lane" means most of the box's bottom edge -- where the
        tyres touch the road -- lies between the lane boundaries evaluated at that box's own row. Without
        one (lane lines not visible / not available) it falls back to the fixed vertical band on box center.

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
            if lane is not None:
                in_lane = lane.contains_box(b)
            else:
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

    def update(self, boxes, t, lane=None):
        """Feed one frame's tracked boxes and its timestamp t (s), plus optionally this frame's EgoLaneBounds.
        Returns the lead Box, or None if there is no pick."""
        cands = self._candidates(boxes, lane)
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
    q was retuned 2026-09-29 (5e-5 -> 1e-3) against real KITTI ground truth (kitti_kalman_tune.py),
    after watching a real near-collision on a downloaded dashcam clip never show TTC below 1.27s in
    the rendered video, despite the raw box-width signal genuinely touching 0.72-0.82s several times
    in the same stretch -- the old q was smoothing away a real fast event, not just noise. Grid-search
    across every real Car track in 3 KITTI sequences (6300+ frames) showed this was not a tradeoff:
    q=1e-3 improved BOTH fast-closing accuracy (median error 0.59s -> 0.21s on frames where ground
    truth TTC < 3s) AND overall accuracy (1.18s -> 0.74s) versus the original, never-validated value.
    r_meas was left alone -- only q was in question here, and changing both at once would muddy which
    change did what. See NOTES.md, 2026-09-29.
    """

    def __init__(self, frame_w, q=1e-3, r_meas=2e-5):
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


class TTCKalmanCA:
    """Constant-ACCELERATION Kalman filter on box width: state [w, dw/dt, d2w/dt2]. TTC = w / (dw/dt).
    Used for the LEAD's TTC since 2026-10-01; same interface as TTCKalman (which NearbyVehicleTTC still uses).

    Why: a car closing at constant speed has w ~ 1/distance, so dw/dt keeps ACCELERATING as TTC drops.
    TTCKalman's constant-rate model can only follow that from behind. Against KITTI ground truth on cars
    roughly straight ahead (|lateral| < 2 m, the lead-car case; 227 frames from 14 cars), it reads
    +0.35 s too high (median error 0.52 s for TTC < 6 s); this model reads -0.14 s (median error 0.39 s),
    and on the TTC < 3 s frames 0.21 s vs 0.35 s. Pooled over ALL cars it looks worse, but most of those
    are cars off to the side, where raw width-TTC reads low for a different reason (the visible side
    widens the box), and the old model's lag happened to cancel that. Same rapid-closing events on every
    clip (start/end times unchanged); the autobahn Golf's minimum TTC is now 0.86 s instead of 1.03 s.
    q = 1e-2 from a grid (1e-4..1) in kitti_kalman_models.py. See NOTES.md, 2026-10-01.
    """

    def __init__(self, frame_w, q=1e-2, r_meas=2e-5):
        self.frame_w = float(frame_w)
        self.q = q            # how fast the width's ACCELERATION may change unexplained (relative-width units)
        self.R = r_meas       # same width measurement noise as TTCKalman
        self.x = self.P = self.last_t = self.last_id = None

    def update(self, width, t, track_id, truncated=False):
        """Same contract as TTCKalman.update: returns (width_px, rate_px_per_s, ttc)."""
        w = width / self.frame_w
        if self.x is None or track_id != self.last_id:
            self.last_id, self.last_t = track_id, t
            self.x = np.array([w, 0.0, 0.0])
            self.P = np.diag([(10 / 3840) ** 2, 1e-3, 1e-2])   # width known; rate and acceleration not
            return width, 0.0, np.nan
        dt = t - self.last_t if t > self.last_t else 1 / 30
        self.last_t = t
        F = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1.0]])
        Q = self.q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],    # continuous white-noise jerk,
                               [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],     # same reasoning as TTCKalman's Q
                               [dt ** 3 / 6, dt ** 2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if not truncated:   # see TTCKalman: coast on the prediction while the box is clipped by the frame edge
            S = self.P[0, 0] + self.R
            K = self.P[:, 0] / S
            self.x = self.x + K * (w - self.x[0])
            self.P = self.P - np.outer(K, self.P[0, :])
        w_est, rate, _ = self.x
        ttc = w_est / rate if rate > 0 else np.nan
        return w_est * self.frame_w, rate * self.frame_w, ttc


EDGE_MARGIN_PX = 2.0     # box within this many px of x=0 or x=frame_w counts as truncated (see TTCKalman.update)
STALE_AFTER_S = 1.0      # drop a vehicle's filter if it hasn't been seen for this long

# --- rapid-closing event settings ---
# These were originally defined only in kitti_headway_events.py (the offline KITTI batch script);
# moved here so the same detector logic can run BOTH on a batch of KITTI frames and streaming, one
# frame at a time, from live.py -- a single shared source of truth instead of two copies that could
# drift apart. Values themselves are unchanged from the original (see kitti_headway_events.py's own
# history for how they were chosen).
RAPID_CLOSING_TTC_S = 4.0        # a commonly cited forward-collision-warning caution threshold, not tuned against ground truth
MIN_EVENT_S = 0.3                # a closing streak must last this long (not a single noisy frame) to count, same "sustained" pattern as SWITCH_TIME/LOST_TIME above
MIN_TRUST_AFTER_SWITCH_S = 1.0   # suppress the flag for this long after any lead switch: a fresh Kalman filter can report a spuriously fast-dropping TTC while it's still converging (see NOTES.md, 2026-09-28)

# --- TTC urgency tiers, for making the overlay visually prominent as TTC drops ---
# Purely a rendering choice (color/size), not a new measurement or claim -- TTC itself is unchanged.
# TTC_URGENT_S reuses the same "how close is dangerous" judgment as RAPID_CLOSING_TTC_S's caution
# tier, just adding one more, closer tier so the overlay escalates before the sustained-event banner
# would even fire. Added 2026-09-30 at the user's request to make TTC more visually prominent.
TTC_URGENT_S = 2.0   # below this, TTC readout turns red/urgent (also RapidClosingDetector's typical range once sustained)
# One more, most extreme tier (added 2026-10-01 at the user's request for something "crazier" at very harsh TTC):
# roughly where real automatic emergency braking systems start to intervene -- a judgment call, not tuned
# to any clip. Rendering only, like the tiers above.
TTC_CRITICAL_S = 1.5


def ttc_urgency(ttc):
    """Classify a TTC value into a tier name for display styling: 'safe' (no lead, nan, or TTC not
    shrinking), 'caution' (< RAPID_CLOSING_TTC_S), 'urgent' (< TTC_URGENT_S) or 'critical' (< TTC_CRITICAL_S). Pure classification,
    no drawing -- kept here so live.py and kitti_annotate.py render the exact same thresholds instead
    of two copies that could drift.
    """
    if ttc is None or ttc != ttc or ttc <= 0:   # ttc != ttc guards nan
        return "safe"
    if ttc < TTC_CRITICAL_S:
        return "critical"
    if ttc < TTC_URGENT_S:
        return "urgent"
    if ttc < RAPID_CLOSING_TTC_S:
        return "caution"
    return "safe"


class UrgencyHysteresis:
    """ttc_urgency with hysteresis, for display: escalate to a more severe tier immediately, but step back
    down only once TTC has risen HYSTERESIS_S above the current tier's threshold. Without it, a TTC hovering
    around 1.5 s flipped the whole frame in and out of the critical styling 4 times in 0.6 s on the autobahn
    clip (2026-10-01). Same idea as a thermostat. Rendering only."""

    HYSTERESIS_S = 0.2
    ORDER = ["safe", "caution", "urgent", "critical"]
    LIMIT = {"caution": RAPID_CLOSING_TTC_S, "urgent": TTC_URGENT_S, "critical": TTC_CRITICAL_S}

    def __init__(self):
        self.tier = "safe"

    def update(self, ttc):
        raw = ttc_urgency(ttc)
        if self.ORDER.index(raw) >= self.ORDER.index(self.tier):
            self.tier = raw                                    # same or worse: follow immediately
        elif ttc is None or ttc != ttc or ttc <= 0 or ttc >= self.LIMIT[self.tier] + self.HYSTERESIS_S:
            self.tier = raw                                    # clearly better (or no reading): step down
        return self.tier

    def reset(self):
        self.tier = "safe"


class RapidClosingDetector:
    """Streaming version of kitti_headway_events.py's sustained-TTC-drop rule: flags a frame as part of a
    rapid-closing event when TTC has been below RAPID_CLOSING_TTC_S for at least MIN_EVENT_S seconds (and
    MIN_FRAMES frames), while trusting a fresh lead pick only after MIN_TRUST_AFTER_SWITCH_S seconds.

    Same design as LeadSelector's own switch/missing streak tracking: a single noisy dip shouldn't flip
    the flag on, and a brand new lead (fresh Kalman state, not yet converged) shouldn't either.
    """

    def __init__(self, ttc_threshold=RAPID_CLOSING_TTC_S, min_event_s=MIN_EVENT_S,
                 min_trust_after_switch_s=MIN_TRUST_AFTER_SWITCH_S):
        self.ttc_threshold = ttc_threshold
        self.min_event_s = min_event_s
        self.min_trust_after_switch_s = min_trust_after_switch_s
        self.last_lead_id = None
        self.lead_switch_time = None
        self.below_since = None
        self.below_n = 0

    def update(self, ttc, t, lead_track_id):
        """Feed one frame's TTC (may be nan/None), timestamp t (s), and current lead track_id (or None
        if there is no lead this frame). Returns True if this frame is part of a sustained rapid-closing event.
        """
        if lead_track_id is None:
            self.last_lead_id, self.lead_switch_time = None, None
            self.below_since, self.below_n = None, 0
            return False

        if lead_track_id != self.last_lead_id:
            self.last_lead_id, self.lead_switch_time = lead_track_id, t
            self.below_since, self.below_n = None, 0

        trustworthy = (t - self.lead_switch_time) >= self.min_trust_after_switch_s
        is_below = trustworthy and ttc is not None and ttc == ttc and ttc < self.ttc_threshold   # ttc == ttc guards nan
        if not is_below:
            self.below_since, self.below_n = None, 0
            return False

        if self.below_since is None:
            self.below_since = t
        self.below_n += 1
        return self.below_n >= MIN_FRAMES and (t - self.below_since) >= self.min_event_s


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
