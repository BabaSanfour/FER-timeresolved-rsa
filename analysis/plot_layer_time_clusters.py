#!/usr/bin/env python3
"""Fig. 4: layer x time RSA maps of variants with study-wide significant clusters (54 stimuli).

Participant-mean Spearman rho of each variant's grid averaged over its RDMs
(rsa_core.variant_grid, as tested), with the outline of every cluster that is
significant within its grid (grid FWER < .05) in a grid that survives Holm
across all grids of all variants (cluster_permutation.py, p_holm_study). Layers
are plotted by relative depth. One column per surviving grid (variant, scope,
condition): the variant on top, its family's random initialisation (same scope
and condition) below.

Reads analysis/rsa_grid_54/ and analysis/cluster_permutation_54/ and writes
analysis/cluster_permutation_54/plots/layer_time_clusters.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, label

import rsa_core
import variants

RESULTS = rsa_core.ROOT / "analysis/cluster_permutation_54"
OUTPUT = RESULTS / "plots/layer_time_clusters.pdf"
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])   # four-neighbour, as in the test
VMAX = .1


def outer(mask):
    """Cluster filled in, so only its outer boundary is drawn (gaps opening onto an
    edge the cluster touches count as enclosed)."""
    padded = np.pad(mask, 1)
    padded[0, :], padded[-1, :] = mask[0].any(), mask[-1].any()
    padded[:, 0], padded[:, -1] = mask[:, 0].any(), mask[:, -1].any()
    return binary_fill_holes(padded)[1:-1, 1:-1]


def main():
    summary = pd.read_csv(RESULTS / "grid_summary.csv")
    survivors = summary[summary.survives_holm_study == 1].sort_values(["family", "variant", "scope", "emotion"])
    if survivors.empty:
        raise SystemExit("no grid survives the study-wide Holm correction: nothing to plot")
    tags = {(f, v): t for g, f, v, t in variants.variants("54") if g == "main"}
    panels = [[(r.family, r.variant, r.scope, r.emotion) for r in survivors.itertuples()],
              [(r.family, "Random", r.scope, r.emotion) for r in survivors.itertuples()]]
    ncol = len(survivors)
    fig, axes = plt.subplots(2, ncol, figsize=(3.5 * ncol, 3.4), sharex=True, sharey=True, squeeze=False)
    for ax, (family, variant, scope, emotion) in zip(axes.flat, panels[0] + panels[1]):
        model = rsa_core.slug(family, variant)
        rsa, z = rsa_core.variant_grid("54", tags[family, variant])
        si = [str(s) for s in z["scopes"]].index(scope)
        ei = [str(e) for e in z["emotions"]].index(emotion)
        grid = np.nanmean(rsa[:, si, :, ei, :], axis=0)                     # layers x windows
        n_layers, n_windows = grid.shape
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                       extent=(0, 1000, 0, 1), interpolation="nearest")
        row = summary[(summary.model == model) & (summary.scope == scope) & (summary.emotion == emotion)].iloc[0]
        title = f"{family}, {variant.lower() if variant == 'Random' else variant}, {scope}"
        title += "" if emotion == "all" else f", {emotion}"
        if row.survives_holm_study:
            sig = np.load(RESULTS / f"{model}_cluster_masks.npz")["significant"][si, :, ei, :]
            labels, count = label(sig, structure=STRUCTURE)
            x = np.linspace(0, 1000, n_windows + 1); y = np.linspace(0, 1, n_layers + 1)
            for k in range(1, count + 1):
                ax.contour((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, outer(labels == k).astype(float),
                           levels=[.5], colors="black", linewidths=1)
            title += rf" ($p_{{\mathrm{{Holm}}}}={row.p_holm_study:.3f}$)".replace("=0.", "=.")
        ax.set_title(title, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("EEG time (ms)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative layer depth", fontsize=8)
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes, fraction=.03, pad=.02)
    cbar.set_label(r"Participant-mean $\rho$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"[ok] -> {OUTPUT}")


if __name__ == "__main__":
    main()
