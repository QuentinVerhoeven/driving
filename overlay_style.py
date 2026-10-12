"""Shared HUD drawing for the two demo renderers (live.py file mode, kitti_annotate.py), so both videos
look the same. Rendering only: no measurement or threshold logic lives here.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
PANEL_ALPHA = 0.78   # dark enough that green text still reads over bright sky
HUD_EMPHASIS = {"safe": 1.0, "caution": 1.15, "urgent": 1.35, "critical": 1.5}   # value text grows as TTC gets more urgent
CRITICAL_RED = (0, 0, 230)


def draw_hud(img, title, value, color, emphasis=1.0, critical=False):
    """Top-left readout on a dark translucent panel: a small grey title ("LEAD id 160") above a big value
    ("TTC 1.2 s") in the urgency color. emphasis > 1 grows the value text as TTC gets more urgent.
    critical=True: solid red panel with white text instead, the loudest version."""
    h, w = img.shape[:2]
    s = max(w / 1280, 0.75)                      # scale with frame width, never unreadably small
    pad, x0, y0 = int(12 * s), int(14 * s), int(14 * s)
    t_scale, v_scale = 0.55 * s, 1.05 * s * emphasis
    t_thick, v_thick = max(1, round(1.2 * s)), max(2, round(2 * s * emphasis))
    (tw, th), _ = cv2.getTextSize(title, FONT, t_scale, t_thick)
    (vw, vh), vb = cv2.getTextSize(value, FONT, v_scale, v_thick)
    pw, ph = max(tw, vw) + 2 * pad, th + vh + vb + 3 * pad
    x1, y1 = min(x0 + pw, w), min(y0 + ph, h)
    roi = img[y0:y1, x0:x1]
    if critical:
        roi[:] = CRITICAL_RED
        color = (255, 255, 255)
    else:
        roi[:] = (roi * (1 - PANEL_ALPHA)).astype(np.uint8)          # darken = black at PANEL_ALPHA opacity
    cv2.rectangle(img, (x0, y0), (x0 + max(3, int(4 * s)), y1), color, -1)   # urgency-colored accent bar
    tx = x0 + pad + int(4 * s)
    cv2.putText(img, title, (tx, y0 + pad + th), FONT, t_scale, (200, 200, 200), t_thick, cv2.LINE_AA)
    cv2.putText(img, value, (tx, y0 + 2 * pad + th + vh), FONT, v_scale, color, v_thick, cv2.LINE_AA)


def draw_critical_border(img):
    """Thick steady red frame around the whole image, for the critical TTC tier."""
    h, w = img.shape[:2]
    b = max(6, int(14 * w / 1280))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), CRITICAL_RED, b)


def draw_banner(img, text, color=(0, 0, 220), text_color=(255, 255, 255)):
    """Full-width warning strip along the bottom edge."""
    h, w = img.shape[:2]
    s = max(w / 1280, 0.75)
    bh = int(40 * s)
    cv2.rectangle(img, (0, h - bh), (w, h), color, -1)
    scale, thick = 0.85 * s, max(2, round(2 * s))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, h - (bh - th) // 2), FONT, scale, text_color, thick, cv2.LINE_AA)
