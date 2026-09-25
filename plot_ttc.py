"""
    Plot the lead car's box width and Kalman TTC over time from one or more live.py --log CSVs.
    Several logs on the same axes lets you compare settings (e.g. --stride 1 vs --stride 3, or two imgsz values).

    Usage:
    uv run python plot_ttc.py outputs/stride1.csv outputs/stride3.csv --labels "every frame" "every 3rd frame" --out outputs/ttc_compare.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a PNG, no window needed (works in WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colors: slots 1 and 2 of the reference categorical palette (light mode), in fixed order
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
TTC_CLIP = 10  # seconds: above this "not closing" and the curve just blows up


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/ttc_plot.png"))
    args = parser.parse_args()
    labels = args.labels or [p.stem for p in args.logs]
    if len(args.logs) > len(SERIES):
        raise SystemExit(f"At most {len(SERIES)} logs per plot")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=INK_2, alpha=0.15, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2)
        ax.yaxis.label.set_color(INK_2)

    for path, label, color in zip(args.logs, labels, SERIES):
        df = pd.read_csv(path).sort_values("t")
        ttc = df["ttc"].clip(upper=TTC_CLIP)  # nan (not closing / filter just reset) stays a gap, not a zero
        axes[0].plot(df["t"], df["width"], color=color, linewidth=1.5, label=label)
        axes[1].plot(df["t"], ttc, color=color, linewidth=1.5, label=label)
        n_low = int((df["ttc"] < 8).sum())
        print(f"{label}: {len(df)} rows with a lead, TTC defined on {df['ttc'].notna().mean():.0%}, {n_low} rows with TTC < 8 s")

    axes[0].set_ylabel("lead box width (px)")
    axes[1].set_ylabel(f"Kalman TTC (s, clipped at {TTC_CLIP})")
    axes[1].set_xlabel("time (s)", color=INK_2)
    axes[0].set_title("Lead vehicle: box width and time-to-collision", color=INK, loc="left")
    if len(args.logs) > 1:
        axes[0].legend(frameon=False, labelcolor=INK)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
