#!/usr/bin/env python3
"""Compact in-sample accuracy summary for the 54-stimulus paper figure."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(os.environ.get("FER_ROOT", "."))
INPUT = ROOT / "analysis/model_performance_54/complete_54/performance_54_same_models.csv"
OUTPUT = ROOT / "analysis/model_performance_54/complete_54/model_accuracy_summary_54.pdf"

COLORS = {
    "Random": "#0072B2",
    "ImageNet": "#E69F00",
    "VGGFace2": "#009E73",
    "VGGFace": "#009E73",
    "AffectNet FT": "#D55E00",
    "Kinetics": "#CC79A7",
    "DFEW FT": "#882255",
    "Pretrained": "#56B4E9",
}

def regime(name):
    for key in ["AffectNet FT", "DFEW FT", "ImageNet", "VGGFace2",
                "VGGFace", "Kinetics", "Pretrained", "Random"]:
        if key in name:
            return key
    return "Other"


def family(name):
    return name.split()[0]


def main():
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        cm_text = row["cm"]
        if not cm_text or not cm_text.startswith("[["):
            cm_text = row.get(None, [cm_text])[-1]
        cm = np.asarray(json.loads(cm_text), dtype=int)
        fam, state = family(row["display_name"]), regime(row["display_name"])
        value = float(np.trace(cm) / cm.sum())
        records.append((fam, state, value))

    families = ["ResNet18", "ResNet50", "FaceNet", "VGG16", "M3DFEL"]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    width = .18
    centers = np.arange(len(families))
    for fi, fam in enumerate(families):
        group = [(state, value) for rec_fam, state, value in records if rec_fam == fam]
        offsets = (np.arange(len(group)) - (len(group) - 1) / 2) * width
        for offset, (state, value) in zip(offsets, group):
            ax.bar(centers[fi] + offset, value, width=width * .92,
                   color=COLORS.get(state, "#777777"), edgecolor="white", linewidth=.5)
    ax.set_ylim(0, 1.07)
    ax.set_ylabel("In-sample accuracy")
    ax.set_xticks(centers, families, fontsize=9)
    ax.set_yticks(np.arange(0, 1.01, .2), [f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLORS[key], label=key) for key in
               ["Random", "ImageNet", "VGGFace2", "AffectNet FT", "Kinetics",
                "DFEW FT"]]
    ax.legend(handles=handles, ncol=7, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.18), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {OUTPUT}")


if __name__ == "__main__":
    main()
