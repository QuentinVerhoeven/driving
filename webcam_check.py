"""
    Week 3, step 1: can we open the webcam and how fast do raw frames arrive?
    (No YOLO yet -- this isolates camera speed from model speed.)

    Usage (native Windows PowerShell, WSL can't see the webcam):
    uv run python webcam_check.py
    uv run python webcam_check.py --camera 1     # if index 0 is the wrong camera

    Press q in the video window to quit.
"""

import argparse
import time
from collections import deque

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)  # webcam index; 0 = first camera Windows finds
    args = parser.parse_args()

    # CAP_DSHOW = DirectShow backend, usually opens faster/more reliably than the default on Windows
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1, or check that no other app is using it.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera} opened: {width}x{height}. Press q in the window to quit.")

    frame_times = deque(maxlen=30)  # timestamps of the last 30 frames, for a smoothed FPS
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame, stopping.")
            break

        frame_times.append(time.time())
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            cv2.putText(frame, f"{fps:.1f} fps", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("webcam check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
