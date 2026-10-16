#!/usr/bin/env python3
"""Compact FaceNet RSA learning curve for the paper."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("FER_ROOT", "."))
SOURCE = ROOT / "analysis/facenet_low_size_repeats/max_r_facenet_all_sizes.csv"
OUTPUT = ROOT / "analysis/facenet_low_size_repeats/facenet_vggface2_size_curve.pdf"
ORDER = ["Random", "VGGFace2 10k", "VGGFace2 50k", "VGGFace2 100k",
         "VGGFace2 200k", "VGGFace2 300k", "VGGFace2 500k",
         "VGGFace2 800k", "Complete VGGFace2"]
LABELS = ["Random", "10k", "50k", "100k", "200k", "300k", "500k",
          "800k", "Complete\n(3.31M)"]


def main():
    data = pd.read_csv(SOURCE)
    data = data[data.scope == "ROI-avg"]
    panels = [("all", "All stimuli"), ("fear", "Fear"),
              ("happy", "Happiness"), ("neutral", "Neutral")]
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (emotion, title) in zip(axes.flat, panels):
        panel = data[data.emotion == emotion].set_index("condition").loc[ORDER]
        y = panel.max_r.to_numpy(float)
        error = panel.standard_deviation.to_numpy(float)
        ax.errorbar(x, y, yerr=error, color="#0072B2", marker="o", markersize=4.8,
                    linewidth=1.8, capsize=2.5, elinewidth=1)
        ax.fill_between(x, y - error, y + error, color="#56B4E9", alpha=.16)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, LABELS, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(r"ROI $\rho_{\max}$")
        ax.grid(axis="y", color="#e8e8e8")
        ax.set_axisbelow(True)
    fig.supxlabel("Number of VGGFace2 training images", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {OUTPUT}")


if __name__ == "__main__":
    main()
