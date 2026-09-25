"""
    Streaming lead-vehicle selection + Kalman-filtered TTC.

    Same logic as explore.ipynb, but written so it can be fed ONE FRAME AT A TIME
    (which is all a live camera gives us). Each class keeps its own state between
    calls instead of looking at the whole clip at once.

    Usage (per frame):
        lead = selector.update(boxes)          # boxes: list of (track_id, x1, y1, x2, y2)
        if lead is not None:
            w_est, rate, ttc = kalman.update(lead.w, t, lead.track_id)
"""

from dataclasses import dataclass

import numpy as np

# --- lead-selection settings (values tuned on clip1_30s, see NOTES.md) ---
BAND_HALF_WIDTH = 0.10   # "in my lane" = box center within +-10% of frame width from center
MAX_ASPECT_RATIO = 2.5   # w/h above this = self-detection of our own hood, not a car
BOTTOM_MARGIN = 0.95     # boxes whose bottom edge is in the bottom 5% of the frame = our hood
SWITCH_STREAK = 10       # frames a challenger must be the biggest candidate before we switch
LOST_STREAK = 5          # frames the current lead must be absent before we give up on it


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
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # state that persists between frames (this is what the notebook loop kept in variables)
        self.current_lead = None       # track_id of the car we currently call "the lead"
        self.challenger_id = None      # a different car that is currently the biggest candidate
        self.challenger_streak = 0     # for how many frames in a row it has been
        self.missing_streak = 0        # for how many frames the current lead has been absent

    def _candidates(self, boxes):
        """Stateless filter: boxes in our lane band that are not our own hood."""
        center = self.frame_w / 2
        out = []
        for b in boxes:
            in_lane = abs(b.cx - center) < BAND_HALF_WIDTH * self.frame_w
            not_self = b.w / b.h < MAX_ASPECT_RATIO and b.y2 < BOTTOM_MARGIN * self.frame_h
            if in_lane and not_self:
                out.append(b)
        return out

    def _switch_to(self, box):
        self.current_lead = box.track_id
        self.missing_streak = 0
        self.challenger_id, self.challenger_streak = None, 0
        return box

    def update(self, boxes):
        """Feed one frame's tracked boxes. Returns the lead Box, or None if there is no pick this frame."""
        cands = self._candidates(boxes)
        if not cands:
            self.missing_streak += 1
            return None

        top = max(cands, key=lambda b: b.area)

        if self.current_lead is None:
            return self._switch_to(top)

        lead = next((b for b in cands if b.track_id == self.current_lead), None)

        if lead is None:
            # current lead not seen this frame; give up only after LOST_STREAK frames in a row
            self.missing_streak += 1
            if self.missing_streak >= LOST_STREAK:
                return self._switch_to(top)
            return None

        self.missing_streak = 0

        if top.track_id == self.current_lead:
            self.challenger_id, self.challenger_streak = None, 0
            return lead

        # a different car is biggest: count how long it has been
        if top.track_id == self.challenger_id:
            self.challenger_streak += 1
        else:
            self.challenger_id, self.challenger_streak = top.track_id, 1

        if self.challenger_streak >= SWITCH_STREAK:
            return self._switch_to(top)
        return lead


class TTCKalman:
    """Kalman filter on box width. State x = [width (px), rate dw/dt (px/s)]. TTC = width / rate."""

    def __init__(self, q_width=1.0, q_rate=50.0, r_meas=25.0):
        self.q_width = q_width
        self.q_rate = q_rate
        self.R = r_meas
        self.x = None          # None until the first measurement arrives
        self.P = None
        self.last_t = None
        self.last_id = None    # which track the state belongs to

    def update(self, width, t, track_id):
        """Fold in one width measurement taken at time t (s) for car `track_id`. Returns (width_est, rate_est, ttc)."""
        if self.x is None or track_id != self.last_id:
            # first measurement, or the lead is a DIFFERENT car: the old [width, rate] describes
            # another vehicle, and folding the new width in would look like a huge fake closing speed
            self.last_id = track_id
            # first measurement: trust the width, know nothing about the rate, be unsure
            self.x = np.array([width, 0.0])
            self.P = np.eye(2) * 100.0
            self.last_t = t
            return self.x[0], self.x[1], np.nan

        dt = t - self.last_t
        if dt <= 0:
            dt = 1 / 30   # guard against duplicate/out-of-order timestamps
        self.last_t = t

        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[self.q_width * dt, 0.0], [0.0, self.q_rate * dt]])  # bigger gap -> trust prediction less

        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # update (H = [1, 0]: we only measure width)
        y = width - self.x[0]
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.P[0, :])   # (I - K H) P with H = [1, 0]

        w_est, rate = self.x
        ttc = w_est / rate if rate > 0 else np.nan   # only meaningful while the box is growing
        return w_est, rate, ttc
