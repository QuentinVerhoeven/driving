"""
Ground-truth TTC from KITTI tracking labels.

KITTI's tracking-label format gives every object's 3D location (x, y, z) in
the CAMERA's own coordinate frame, and that frame moves with the car each
frame. So `z` (depth, i.e. distance straight ahead of the camera) at a given
frame is already the distance relative to the moving ego vehicle -- the car's
own motion is already baked in. That means the closing rate is just dz/dt;
no separate correction using the OXTS ego-speed log is needed (adding one
would double-count ego motion). OXTS is parsed here anyway, for a sanity
check later (does implied ego speed roughly match vf?) and because we'll
want it for calibration work regardless.

Ground-truth TTC for a tracked object, at each frame where we have a next
frame for the same track:
    dz = z[t+1] - z[t]
    dt = time[t+1] - time[t]           (KITTI tracking is 10 Hz -> dt = 0.1 s)
    closing_speed = -dz / dt           (positive = getting closer)
    ttc = z / closing_speed            (only defined while closing_speed > 0)

This file only parses labels/oxts and computes that ground truth -- it does
not run our detector. That comparison comes once the KITTI images are
downloaded and this has been checked against real sequences.
"""

from pathlib import Path

import pandas as pd

# KITTI tracking runs at 10 Hz.
FRAME_DT_S = 0.1

# label_02/<seq>.txt column order (space-separated, no header).
LABEL_COLUMNS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
    "dim_h", "dim_w", "dim_l",
    "x", "y", "z",
    "rotation_y",
]

# oxts/<seq>.txt column order (space-separated, no header, one line per frame).
OXTS_COLUMNS = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu",
    "pos_accuracy", "vel_accuracy",
    "navstat", "numsats", "posmode", "velmode", "orimode",
]


def parse_label_file(path: Path) -> pd.DataFrame:
    """Read a label_02/<seq>.txt file into a DataFrame, one row per object per frame."""
    df = pd.read_csv(path, sep=" ", header=None, names=LABEL_COLUMNS)
    return df


def parse_oxts_file(path: Path) -> pd.DataFrame:
    """Read an oxts/<seq>.txt file into a DataFrame, one row per frame.

    Adds a `frame` column (0-indexed, matching the label file's frame numbers)
    since the oxts file itself has no frame index.
    """
    df = pd.read_csv(path, sep=" ", header=None, names=OXTS_COLUMNS)
    df.insert(0, "frame", range(len(df)))
    return df


def compute_gt_ttc(labels: pd.DataFrame) -> pd.DataFrame:
    """For every tracked object, compute per-frame ground-truth TTC from depth (z).

    Returns a copy of `labels` with three extra columns:
      - closing_speed_mps: -dz/dt to the next frame for this track (NaN on the
        track's last frame, since there is no next frame to difference against)
      - gt_ttc_s: z / closing_speed_mps where closing_speed_mps > 0, else NaN
        (not closing, so TTC is undefined/infinite)
      - dt_s: time to the next frame actually used (lets a caller spot gaps,
        e.g. missed detections, where dt is a multiple of FRAME_DT_S)
    """
    out = labels.sort_values(["track_id", "frame"]).copy()

    dz = out.groupby("track_id")["z"].diff().shift(-1)
    dframe = out.groupby("track_id")["frame"].diff().shift(-1)
    dt = dframe * FRAME_DT_S

    closing_speed = -dz / dt
    ttc = out["z"] / closing_speed
    ttc[closing_speed <= 0] = float("nan")

    out["dt_s"] = dt
    out["closing_speed_mps"] = closing_speed
    out["gt_ttc_s"] = ttc
    return out


if __name__ == "__main__":
    # Self-test on a synthetic fixture: one track closing at a known constant
    # rate, so the expected TTC at each frame is exact and easy to check by
    # hand. This does not touch real KITTI data -- it only checks the parsing
    # + math above are correct before they're pointed at the real dataset.
    import io

    synthetic_label_text = "\n".join(
        # frame track_id type trunc occ alpha bbox(4) dim(3) x y z rot_y
        f"{frame} 1 Car 0 0 0 0 0 0 0 1.5 1.6 4.0 0 1.5 {z:.2f} 0"
        for frame, z in enumerate([40.0, 39.0, 38.0, 37.0, 36.0])
    )
    labels = pd.read_csv(io.StringIO(synthetic_label_text), sep=" ", header=None, names=LABEL_COLUMNS)

    result = compute_gt_ttc(labels)
    print(result[["frame", "track_id", "z", "dt_s", "closing_speed_mps", "gt_ttc_s"]])

    # z closes by 1.0 m every 0.1 s -> closing_speed = 10 m/s exactly, so
    # gt_ttc_s should equal z / 10 at every frame except the last (no next
    # frame to difference against).
    expected_ttc = labels["z"][:-1] / 10.0
    actual_ttc = result["gt_ttc_s"][:-1]
    n_compared = actual_ttc.notna().sum()
    assert n_compared == len(expected_ttc), f"expected {len(expected_ttc)} defined values, got {n_compared}"
    assert (actual_ttc.reset_index(drop=True) - expected_ttc.reset_index(drop=True)).abs().max() < 1e-9
    assert pd.isna(result["gt_ttc_s"].iloc[-1]), "last frame of a track should have no TTC (no next frame)"
    print(f"\nSelf-test passed: {n_compared} TTC values matched the known closing rate exactly.")
