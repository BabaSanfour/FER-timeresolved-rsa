#!/usr/bin/env python3
"""Fig. 3: where and when model-EEG alignment survives the study-wide test (54 videos).

One column per grid that survives Holm across all grids of all variants
(cluster_permutation.py, p_study < .05).
  top     layer x time map of the variant (participant mean; grid averaged over its
          RDMs, as tested), significant clusters outlined
  bottom  time course at the layer with the highest participant mean inside the
          clusters: the variant and its family's random initialisation (same layer),
          mean +/- SEM over participants; bar = the cluster's windows at that layer,
          triangle = peak

Reads analysis/rsa_grid_54/ and analysis/cluster_permutation_54/; writes
analysis/paper_stats_54/fig3_time.{pdf,png}.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(rsa_core.ROOT / "analysis/paper_stats_54/fig3_time"),
                    help="output path without extension")
    args = ap.parse_args()
    st.setup()
    summary = pd.read_csv(RESULTS / "grid_summary.csv")
    survivors = summary[summary.survives_study == 1].sort_values(["family", "variant", "scope", "emotion"])
    if survivors.empty:
        raise SystemExit("no grid survives the study-wide Holm correction: nothing to plot")
    tags = {(f, v): t for g, f, v, t in variants.variants("54") if g == "main"}
    ncol = len(survivors)
    fig, axes = plt.subplots(2, ncol, figsize=(2.75 * ncol, 3.3), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.42, wspace=0.3,
                                              left=0.1, right=0.85, top=0.92, bottom=0.13))
    grids = {}
    for col, r in enumerate(survivors.itertuples()):
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
        mean = np.nanmean(rsa[:, si, :, ei, :], axis=0)                       # layers x windows
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
        ax.set_title(f"{r.family} {r.variant.lower()} · {r.scope}" +
                     ("" if r.emotion == "all" else f" · {r.emotion}") +
                     rf"  ($p$ = {st.point_format(round(r.p_study, 3))}, study-wide)", fontsize=6.5)
        ax.set_ylabel("relative layer depth" if col == 0 else "")
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xticklabels([])
        st.panel_label(ax, "ab"[col] if ncol == 2 else chr(97 + col), x=-0.2 if col == 0 else -0.14)

        ax = axes[1, col]
        for grid, color, name in ((rnd, st.REGIME_COLORS["Random"], "random\n(10 seeds)"),
                                  (rsa, st.REGIME_COLORS[st.REGIME_OF[r.variant]], r.variant.lower())):
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
        peak = int(np.nanargmax(np.where(on, m, -np.inf)))
        ax.plot(centres[peak], top + 0.012, "v", ms=3.5, color=st.INK)
        ax.text(centres[peak], top + 0.02, f"{centres[peak]:.0f} ms", ha="center", va="bottom", fontsize=5.8)
        ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(bounds[0, 0], bounds[-1, 1])
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlabel("time from video onset (ms)")
        ax.set_ylabel(r"$\rho$ at the marked layer" if col == 0 else "")
        ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
        ax.set_ylim(min(-0.05, ax.get_ylim()[0]), top + 0.05)
        st.panel_label(ax, chr(97 + ncol + col), x=-0.2 if col == 0 else -0.14)
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    cax = fig.add_axes([0.87, axes[0, -1].get_position().y0, 0.012, axes[0, -1].get_position().height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"participant-mean $\rho$", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    handles, names = axes[1, 0].get_legend_handles_labels()
    pos = axes[1, -1].get_position()
    fig.legend(handles[::-1], names[::-1], loc="upper left", bbox_to_anchor=(pos.x1 + 0.01, pos.y1),
               handlelength=1.4, borderaxespad=0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
