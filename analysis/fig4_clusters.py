#!/usr/bin/env python3
"""Fig. 4: alignment that survives the study-wide test (54 videos).

  a  for every variant (rows, by family) and scope (columns), the number of the four stimulus
     conditions (all, fear, happiness, neutral) with a cluster surviving the study-wide maximum
     statistic across all grids of all variants (cluster_permutation.py, p_study < .05)
  b  three examples, the best occipito-temporal grid of pretrained Qwen3-VL, ImageNet VGG16 and Kinetics M3DFEL:
     layer x time map (participant mean of the grid averaged over the variant's RDMs, as tested)
     with the study-wide clusters outlined, and the time course at the layer with the highest
     mean inside them, against the family's random initialisation (mean +/- SEM over participants;
     bar = the cluster's windows at that layer, triangle = peak)

Reads analysis/rsa_grid_54/ and analysis/cluster_permutation_54/; writes
analysis/paper_stats_54/fig4_clusters.{pdf,png}.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_fill_holes, label

import figure_style as st
import rsa_core
import variants

RESULTS = rsa_core.ROOT / "analysis/cluster_permutation_54"
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])        # four-neighbour, as in the test
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
VMAX = .1


def outer(mask):
    """Cluster filled in, so only its outer boundary is drawn."""
    padded = np.pad(mask, 1)
    padded[0, :], padded[-1, :] = mask[0].any(), mask[-1].any()
    padded[:, 0], padded[:, -1] = mask[:, 0].any(), mask[:, -1].any()
    return binary_fill_holes(padded)[1:-1, 1:-1]


EXAMPLES = [("Qwen3-VL", "Pretrained"), ("VGG16", "ImageNet"), ("M3DFEL", "Kinetics-400")]
REGIONS = {"Frontal": "F", "Central": "C", "CentroParietal": "CP", "OccipitoTemporal": "OT"}


def scope_order(scopes):
    return [f"{r}-{x}" for r in REGIONS for x in "LMR" if f"{r}-{x}" in scopes] + ["wholescalp"]


def matrix(ax, summary):
    rows = [(f, v) for f, vs in variants.MAIN.items() for v, _ in vs]
    scopes = scope_order(set(summary.scope))
    count = summary.groupby(["family", "variant", "scope"]).survives_study.sum()
    for i, (f, v) in enumerate(rows):
        color = st.REGIME_COLORS[st.REGIME_OF[v]]
        for j, sc in enumerate(scopes):
            n = int(count.get((f, v, sc), 0))
            if n:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, color=color, alpha=0.25 + 0.75 * n / 4, lw=0))
                ax.text(j, i, str(n), ha="center", va="center", fontsize=5.2, color="white" if n >= 3 else st.INK)
            else:
                ax.plot(j, i, ".", ms=1.5, color=st.RULE)
    ax.set_xlim(-0.6, len(scopes) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(scopes)), ["whole\nscalp" if s == "wholescalp" else
                                       f"{REGIONS[s.split('-')[0]]}-{s.split('-')[1]}" for s in scopes], fontsize=5.4)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"{f}  {v.lower()}" if v != rows[i - 1][1] or f != rows[i - 1][0] else ""
                                     for i, (f, v) in enumerate(rows)], fontsize=5.4)
    for i in range(1, len(rows)):
        if rows[i][0] != rows[i - 1][0]:
            ax.axhline(i - 0.5, color=st.RULE, lw=0.6)
    ax.axvline(len(scopes) - 1.5, color=st.RULE, lw=0.6)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rsa_core.ROOT / "analysis/paper_stats_54/fig4_clusters"),
                    help="output path without extension")
    args = ap.parse_args()
    st.setup()
    summary = pd.read_csv(RESULTS / "grid_summary.csv")
    tags = {(f, v): t for g, f, v, t in variants.variants("54") if g == "main"}
    fig = plt.figure(figsize=(5.5, 6.4))
    matrix(fig.add_axes([0.2, 0.555, 0.74, 0.39]), summary)
    fig.text(0.02, 0.975, "a", fontsize=8, fontweight="bold")
    fig.text(0.05, 0.975, "Conditions with a study-wide cluster (of 4), by variant and region", fontsize=6.8, fontweight="bold")
    gs = fig.add_gridspec(2, 3, left=0.1, right=0.86, top=0.44, bottom=0.07, height_ratios=[1.15, 1], hspace=0.45, wspace=0.3)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
    fig.text(0.02, 0.475, "b", fontsize=8, fontweight="bold")
    fig.text(0.05, 0.475, "Examples: layer x time maps and time courses against random initialization", fontsize=6.8, fontweight="bold")
    examples = []
    for f, v in EXAMPLES:
        sel = summary[(summary.family == f) & (summary.variant == v) & (summary.survives_study == 1)]
        ot = sel[sel.scope.str.startswith("OccipitoTemporal")]
        sel = ot if len(ot) else sel
        if len(sel):
            examples.append(sel.sort_values("p_study").iloc[0])
    grids = {}
    for col, r in enumerate(examples):
        for variant in (r.variant, "Random"):
            if (r.family, variant) not in grids:
                grids[r.family, variant] = rsa_core.variant_grid("54", tags[r.family, variant])
        rsa, z = grids[r.family, r.variant]
        rnd, _ = grids[r.family, "Random"]
        si = [str(s) for s in z["scopes"]].index(r.scope)
        ei = [str(e) for e in z["emotions"]].index(r.emotion)
        bounds = np.asarray(z["window_bounds"], float)
        centres = bounds.mean(1)
        sig = np.load(RESULTS / f"{r.model}_cluster_masks.npz")["significant_study"][si, :, ei, :]
        mean = np.nanmean(rsa[:, si, :, ei, :], axis=0)
        n_layers, n_windows = mean.shape
        ax = axes[0, col]
        im = ax.imshow(mean, aspect="auto", origin="lower", cmap=DIVERGING, vmin=-VMAX, vmax=VMAX,
                       extent=(bounds[0, 0], bounds[-1, 1], 0, 1), interpolation="nearest")
        labels, count = label(sig, structure=STRUCTURE)
        y = (np.arange(n_layers) + 0.5) / n_layers
        for k in range(1, count + 1):
            ax.contour(centres, y, outer(labels == k).astype(float), levels=[.5], colors=st.INK, linewidths=0.8)
        best = np.unravel_index(np.nanargmax(np.where(sig, mean, -np.inf)), mean.shape)[0]
        ax.axhline(y[best], color=st.INK, lw=0.5, ls=(0, (2, 2)))
        short = "whole scalp" if r.scope == "wholescalp" else f"{REGIONS[r.scope.split('-')[0]]}-{r.scope.split('-')[1]}"
        ax.set_title(f"{r.family} {r.variant.lower()}\n{short}" + ("" if r.emotion == "all" else f", {r.emotion}"),
                     fontsize=5.8, pad=2)
        ax.set_ylabel("relative layer depth" if col == 0 else "")
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_xticklabels([])
        ax = axes[1, col]
        for grid, color, name in ((rnd, st.REGIME_COLORS["Random"], "random"),
                                  (rsa, st.REGIME_COLORS[st.REGIME_OF[r.variant]], "trained")):
            trace = grid[:, si, best, ei, :]
            m = np.nanmean(trace, 0)
            sem = np.nanstd(trace, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(trace), 0))
            ax.fill_between(centres, m - sem, m + sem, color=color, alpha=0.25, lw=0)
            ax.plot(centres, m, color=color, lw=1.1, label=name)
        m = np.nanmean(rsa[:, si, best, ei, :], 0)
        top = np.nanmax(m) + 0.02
        on = sig[best]
        for start, stop in zip(*[np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]) == v) for v in (1, -1)]):
            ax.plot([bounds[start, 0], bounds[stop - 1, 1]], [top, top], color=st.INK, lw=1.6, solid_capstyle="butt")
        if on.any():
            peak = int(np.nanargmax(np.where(on, m, -np.inf)))
            ax.plot(centres[peak], top + 0.012, "v", ms=3.5, color=st.INK)
            ax.text(centres[peak], top + 0.02, f"{centres[peak]:.0f} ms", ha="center", va="bottom", fontsize=5.6)
        ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(bounds[0, 0], bounds[-1, 1])
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_xlabel("time from onset (ms)", fontsize=5.8)
        ax.set_ylabel(r"$\rho$ at the marked layer" if col == 0 else "")
        ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
        ax.set_ylim(min(-0.05, ax.get_ylim()[0]), top + 0.05)
    cax = fig.add_axes([0.88, axes[0, -1].get_position().y0, 0.012, axes[0, -1].get_position().height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"participant-mean $\rho$", fontsize=5.6)
    cbar.ax.tick_params(labelsize=5.2, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    handles, names = axes[1, 0].get_legend_handles_labels()
    pos = axes[1, -1].get_position()
    fig.legend(handles[::-1], names[::-1], loc="upper left", bbox_to_anchor=(pos.x1 + 0.01, pos.y1),
               handlelength=1.2, borderaxespad=0, fontsize=5.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
