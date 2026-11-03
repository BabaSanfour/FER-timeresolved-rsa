#!/usr/bin/env python3
"""Fig. 2: ROI-averaged rho_max of every variant, by emotion, one panel per family.

Reads analysis/paper_stats/rho_max.csv (maxr_table.py --stimuli 18). Bars are the
mean over a variant's ten seeds or training runs (error bar: SD; a released model
has none); a bar is grey when it lies below the lower noise ceiling.

    python analysis/plot_model_comparison.py --out analysis/paper_stats/fig2_model_comparison
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import rsa_core
import variants

EMOS = ["all", "fear", "happy", "neutral", rsa_core.EMO_AVG]
COLORS = {"Random": "#0072B2", "ImageNet": "#E69F00", "VGGFace2": "#009E73", "VGG-Face": "#009E73",
          "Kinetics-400": "#CC79A7", "Pretrained": "#56B4E9",
          "AffectNet FT": "#D55E00", "DFEW FT": "#882255"}
GREY = "#B9B9B9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rsa_core.ROOT / "analysis/paper_stats/fig2_model_comparison"),
                    help="output path without extension")
    args = ap.parse_args()
    with (rsa_core.ROOT / "analysis/paper_stats/rho_max.csv").open() as fh:
        table = {(r["family"], r["variant"], r["emotion"]): r for r in csv.DictReader(fh)
                 if r["group"] == "main" and r["roi"] == rsa_core.ROI_AVG}

    families = list(variants.MAIN)
    ncol = 3; nrow = int(np.ceil(len(families) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, family in zip(axes, families):
        rows = variants.MAIN[family]
        x = np.arange(len(EMOS)); w = 0.8 / len(rows)
        for vi, (variant, tags) in enumerate(rows):
            cells = [table[family, variant, e] for e in EMOS]
            m = np.array([float(c["max_r"]) for c in cells])
            lo = np.array([float(c["nc_lower"]) for c in cells])
            xpos = x + (vi - (len(rows) - 1) / 2) * w
            ax.bar(xpos, m, width=w * 0.92, color=[GREY if v < l else COLORS[variant] for v, l in zip(m, lo)],
                   edgecolor="white", linewidth=0.5, zorder=3)
            if len(tags) > 1:
                sd = np.array([float(c["sd"]) for c in cells])
                ax.errorbar(xpos, m, yerr=sd, fmt="none", ecolor="#222", elinewidth=0.9,
                            capsize=2.5, capthick=0.9, zorder=5)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(EMOS, rotation=20, ha="right", fontsize=8)
        ax.set_title(family, fontsize=11)
        ax.grid(axis="y", color="#eee", zorder=0)
        ax.set_ylabel(r"$\rho_{\max}$ (ROI-avg)", fontsize=8)
    for ax in axes[len(families):]:
        ax.axis("off")

    used = list(dict.fromkeys(v for rows in variants.MAIN.values() for v, _ in rows))
    labels = {v: (f"{v} (mean ± SD, n = 10)" if v in ("Random", "AffectNet FT") else v) for v in used}
    labels["VGGFace2"] = "VGGFace2 (mean ± SD, n = 10; VGG16: VGG-Face)"
    handles = [Patch(fc=COLORS[v], ec="white", label=labels[v]) for v in used if v != "VGG-Face"]
    handles += [Patch(fc=GREY, ec="white", label="below the lower noise ceiling")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
