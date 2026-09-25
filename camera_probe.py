"""
    Try many settings on ONE camera index to find what makes it deliver a real picture.

    Some virtual cameras (like a phone streamed by Iriun) show fine in the Windows Camera app but give
    OpenCV black frames, because OpenCV asks for a default video format the camera doesn't handle.
    This tries every combination of backend x resolution x pixel format (FOURCC) and reports the
    resolution actually received, the FPS, and the brightness (0 = black). Any combination that gives
    a real picture is saved as probe_<index>_<backend>_<size>_<fourcc>.jpg.

    Usage (native Windows PowerShell, phone connected, Iriun client showing the phone's picture):
    uv run python camera_probe.py --index 1
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide OpenCV's per-frame warning spam (set before importing cv2)

import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF} if sys.platform == "win32" else {"default": cv2.CAP_ANY}
SIZES = [None, (1280, 720), (1920, 1080)]   # None = leave OpenCV's default
FOURCCS = [None, "MJPG", "YUY2"]            # None = leave the default pixel format


def try_settings(index, backend, size, fourcc):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    # the order matters for many drivers: pixel format first, then size
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    frames, brightness, last = 0, 0.0, None
    start = time.time()
    while time.time() - start < 2.0:
        ok, frame = cap.read()
        if ok:
            frames += 1
            brightness = float(frame.mean())
            last = frame
    elapsed = time.time() - start
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return got_w, got_h, frames / elapsed, brightness, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1)
    args = parser.parse_args()

    for name, backend in BACKENDS.items():
        for size in SIZES:
            for fourcc in FOURCCS:
                label = f"[{name}] asked {size or 'default'} {fourcc or 'default'}"
                result = try_settings(args.index, backend, size, fourcc)
                if result is None:
                    print(f"{label}: could not open")
                    continue
                w, h, fps, brightness, frame = result
                verdict = "REAL PICTURE?" if brightness > 5 and fps > 5 else ""
                print(f"{label}: got {w}x{h}, {fps:.1f} fps, brightness {brightness:.0f} {verdict}")
                if brightness > 5 and frame is not None:
                    tag = f"{size[0]}x{size[1]}" if size else "default"
                    cv2.imwrite(f"probe_{args.index}_{name}_{tag}_{fourcc or 'default'}.jpg", frame)


if __name__ == "__main__":
    main()
