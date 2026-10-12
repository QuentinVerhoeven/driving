"""Pretrained YOLOP (ONNX, CPU) lane-line + drivable-area segmentation on a video or KITTI image folder.

Experiment, not wired into the pipeline yet: draws YOLOP's raw masks over each frame so we can judge,
on the real demo clips, whether the lane lines it finds are good enough to build an ego-lane from.

YOLOP (Wu et al. 2022, github.com/hustvl/YOLOP) is one network trained on BDD100K with three heads:
vehicle boxes, drivable area, lane lines. We only use the two segmentation heads -- detection and
tracking already come from YOLO26 + ByteTrack. Weights: the repo's official ONNX export, run with
onnxruntime, so there's no PyTorch-version or CUDA dependency.

Usage:
  uv run python lane_yolop.py data/raw/autobahn_braking.mp4 --mask 0.585,0.6,1,1
  uv run python lane_yolop.py data/kitti/training/image_02/0020
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from lane_hough import frames

MODEL = Path("cache/yolop/yolop-640-640.onnx")
SIZE = 640
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalization, what YOLOP was trained with
STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(frame):
    """Resize keeping aspect ratio so the long side is SIZE, pad the rest with gray (114).
    Returns the padded image plus (scale, pad_x, pad_y) to undo it on the output masks."""
    h, w = frame.shape[:2]
    s = SIZE / max(h, w)
    nw, nh = round(w * s), round(h * s)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (nw, nh, px, py)


class YOLOP:
    def __init__(self, path=MODEL):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, frame, full_res=True):
        """BGR frame -> (drivable_mask, lane_mask), uint8 0/1. full_res=True resizes them to the frame's
        resolution; False keeps the model's own (letterbox-cropped, e.g. 640x194 for KITTI) resolution,
        which is what gets cached -- much smaller, and plenty for fitting lane curves."""
        h, w = frame.shape[:2]
        img, (nw, nh, px, py) = letterbox(frame)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        x = x.transpose(2, 0, 1)[None]                          # HWC -> 1xCxHxW
        _, da, ll = self.sess.run(None, {"images": x})
        masks = []
        for seg in (da, ll):
            m = seg[0].argmax(0).astype(np.uint8)               # per pixel: class 1 = drivable / lane line
            m = m[py:py + nh, px:px + nw]                       # strip the letterbox padding
            masks.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if full_res else m)
        return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--mask", default=None)   # x1,y1,x2,y2 fractions of a burned-in inset to blank out of the masks
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--no-video", action="store_true")   # only cache the lane masks (for ego_lane.py)
    args = ap.parse_args()
    inset = [float(v) for v in args.mask.split(",")] if args.mask else None

    name = Path(args.source).stem if Path(args.source).is_file() else "kitti_" + Path(args.source).name
    args.outdir.mkdir(exist_ok=True)
    model = YOLOP()
    writer, n, t0 = None, 0, time.time()
    cached = []   # lane-line masks at model resolution, saved for ego_lane.py so it never reruns the network
    for frame, fps in frames(args.source):
        h, w = frame.shape[:2]
        if writer is None and not args.no_video:
            writer = cv2.VideoWriter(str(args.outdir / f"{name}_lanes_yolop.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        da_small, ll_small = model(frame, full_res=False)
        cached.append(np.packbits(ll_small, axis=1))   # 1 bit per pixel
        n += 1
        if args.no_video:
            continue
        da = cv2.resize(da_small, (w, h), interpolation=cv2.INTER_NEAREST)
        ll = cv2.resize(ll_small, (w, h), interpolation=cv2.INTER_NEAREST)
        if inset:
            x1, y1, x2, y2 = inset
            da[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
            ll[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 0
        out = frame.copy()
        out[da == 1] = (0.6 * out[da == 1] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)   # drivable: green tint
        out[ll == 1] = (0, 0, 255)                                                            # lane lines: red
        writer.write(out)
    if writer is not None:
        writer.release()
    cache = MODEL.parent / f"{name}_lanemask.npz"
    np.savez_compressed(cache, packed=np.stack(cached), width=ll_small.shape[1], fps=fps, frame_w=w, frame_h=h)
    print(f"saved {cache}")
    print(f"{n} frames, {(time.time() - t0) / n * 1000:.0f} ms/frame on CPU")


if __name__ == "__main__":
    main()
