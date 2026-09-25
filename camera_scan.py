"""
    Find which camera index (and which OpenCV backend) gives real video.

    For every index 0-5 and both Windows backends it opens the camera, reads frames for ~2 seconds,
    and prints the resolution, the FPS, and the average brightness. A black/dead camera has
    brightness near 0. A real picture is usually 40+. The first frame of each working camera
    is saved as scan_<index>_<backend>.jpg so you can look at it.

    Usage (native Windows PowerShell, with the phone connected and the Iriun app open):
    uv run python camera_scan.py
"""

import sys
import time

import cv2

# DSHOW = DirectShow, MSMF = Media Foundation (the default on modern Windows). Some virtual
# cameras like Iriun only work with one of them, so we try both.
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF} if sys.platform == "win32" else {"default": cv2.CAP_ANY}


def probe(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames, first, brightness = 0, None, 0.0
    start = time.time()
    while time.time() - start < 2.0:
        ok, frame = cap.read()
        if not ok:
            continue
        if first is None:
            first = frame
        frames += 1
        brightness = float(frame.mean())  # 0 = pure black, 255 = pure white
    elapsed = time.time() - start
    cap.release()
    return width, height, frames / elapsed, brightness, first


def main():
    for index in range(6):
        for name, backend in BACKENDS.items():
            result = probe(index, backend)
            if result is None:
                print(f"index {index} [{name}]: could not open")
                continue
            width, height, fps, brightness, first = result
            print(f"index {index} [{name}]: {width}x{height}, {fps:.1f} fps, brightness {brightness:.0f}")
            if first is not None:
                cv2.imwrite(f"scan_{index}_{name}.jpg", first)


if __name__ == "__main__":
    main()
