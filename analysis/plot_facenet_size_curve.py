#!/usr/bin/env python3
"""Fig. 3: FaceNet ROI-averaged rho_max by VGGFace2-HQ training-set size.

Reads analysis/paper_stats/rho_max.csv (maxr_table.py --stimuli 18): random
initialisation, the size curve (nested subsets, 40 epochs each, ending at the full
715k-image pool) and, next to it, the step-matched check (10k-300k trained for the
full pool's number of steps). Points: mean over ten training runs (random: ten
seeds); bars and band: SD.

    python analysis/plot_facenet_size_curve.py --out analysis/paper_stats/fig3_facenet_size_curve
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rsa_core
import variants

POINTS = ["Random"] + list(variants.SIZES) + [variants.FULL_POOL]
LABELS = ["Random"] + list(variants.SIZES) + ["715k\n(full pool)"]
PANELS = [("all", "All stimuli"), ("fear", "Fear"), ("happy", "Happiness"), ("neutral", "Neutral")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rsa_core.ROOT / "analysis/paper_stats/fig3_facenet_size_curve"),
                    help="output path without extension")
    args = ap.parse_args()
    with (rsa_core.ROOT / "analysis/paper_stats/rho_max.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["family"] == "FaceNet" and r["roi"] == rsa_core.ROI_AVG]
    get = {(r["group"], r["variant"], r["emotion"]): r for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    x = np.arange(len(POINTS))
    for ax, (emotion, title) in zip(axes.flat, PANELS):
        size = [get["main", "Random", emotion]] + [get["size", p, emotion] for p in POINTS[1:]]
        y = np.array([float(r["max_r"]) for r in size]); e = np.array([float(r["sd"]) for r in size])
        ax.errorbar(x, y, yerr=e, color="#0072B2", marker="o", markersize=4.8, linewidth=1.8,
                    capsize=2.5, elinewidth=1, label="40 epochs per subset")
        ax.fill_between(x, y - e, y + e, color="#56B4E9", alpha=.16)
        steps = [get["step", p, emotion] for p in variants.SIZES[:-1]]
        xs = x[1:len(steps) + 1]
        ys = np.array([float(r["max_r"]) for r in steps]); es = np.array([float(r["sd"]) for r in steps])
        ax.errorbar(xs, ys, yerr=es, color="#D55E00", marker="s", markersize=4.2, linewidth=1.4,
                    linestyle="--", capsize=2.5, elinewidth=1, label="step-matched to the full pool")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, LABELS, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(r"ROI $\rho_{\max}$")
        ax.grid(axis="y", color="#e8e8e8")
        ax.set_axisbelow(True)
    axes.flat[0].legend(fontsize=7.5, frameon=False)
    fig.supxlabel("Number of VGGFace2-HQ training images", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
