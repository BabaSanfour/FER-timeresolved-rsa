#!/usr/bin/env python3
"""Layer x time RSA maps for the clusters that survive Holm correction (54 stimuli).

Participant-mean Spearman rho for FaceNet Complete VGGFace2 and pretrained
Qwen3-VL (fear, right posterior ROI) next to their random-init counterparts
(seed 42), with the outline of every cluster surviving Holm correction across
the model's 28 grids. Layers are plotted by relative depth.

Reads analysis/rsa_grid_54/ and analysis/cluster_permutation_54/ (written by
cluster_permutation_rsa_54.py --study 54) and writes
analysis/cluster_permutation_54/plots/layer_time_clusters.pdf.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, label

ROOT = Path(os.environ.get("FER_ROOT", "."))
GRIDS = ROOT / "analysis/rsa_grid_54"
RESULTS = ROOT / "analysis/cluster_permutation_54"
OUTPUT = RESULTS / "plots/layer_time_clusters.pdf"
SCOPE, EMOTION = "Post-Right", "fear"
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])   # four-neighbour, as in the test
PANELS = [("vggface_complete_16f_pretrained", "FaceNet, complete VGGFace2"),
          ("facenet_random_16f_weights", "FaceNet, random init"),
          ("qwen3vl_pretrained_16f_weights", "Qwen3-VL, pretrained"),
          ("qwen3vl_random_16f_weights", "Qwen3-VL, random init")]
VMAX = .1


def outer(mask):
    """Cluster filled in, so only its outer boundary is drawn.

    Gaps that open onto an edge of the grid the cluster touches count as
    enclosed (that edge is treated as closed).
    """
    padded = np.pad(mask, 1)
    padded[0, :], padded[-1, :] = mask[0].any(), mask[-1].any()
    padded[:, 0], padded[:, -1] = mask[:, 0].any(), mask[:, -1].any()
    return binary_fill_holes(padded)[1:-1, 1:-1]


def holm_clusters(tag, si, ei, summary, clusters):
    """Masks of the clusters in this grid that survive Holm across the 28 grids."""
    row = summary[(summary.model == tag) & (summary.scope == SCOPE) &
                  (summary.emotion == EMOTION)].iloc[0]
    if not row.survives_holm_28_grids:
        return [], None
    # Holm adjusts the grid's minimum p; apply the same factor to each cluster.
    factor = row.p_holm_28_grids / row.min_cluster_p
    sig = np.load(RESULTS / f"{tag}_cluster_masks.npz")["significant"][si, :, ei, :]
    labels, count = label(sig, structure=STRUCTURE)
    rows = clusters[(clusters.model == tag) & (clusters.scope == SCOPE) &
                    (clusters.emotion == EMOTION)]
    keep = []
    for k in range(1, count + 1):
        mask = labels == k
        layers, windows = np.where(mask)
        match = rows[(rows["size"] == mask.sum()) &
                     (rows.layer_min == layers.min()) & (rows.layer_max == layers.max()) &
                     (rows.window_min == windows.min()) & (rows.window_max == windows.max())]
        if len(match) == 1 and match.p_grid_fwer.iloc[0] * factor < .05:
            keep.append(outer(mask))
    return keep, row.p_holm_28_grids


def main():
    summary = pd.read_csv(RESULTS / "grid_summary.csv")
    clusters = pd.read_csv(RESULTS / "clusters.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    for ax, (tag, title) in zip(axes.flat, PANELS):
        z = np.load(GRIDS / f"{tag}_rsa_grid.npz", allow_pickle=True)
        si = [str(s) for s in z["scopes"]].index(SCOPE)
        ei = [str(e) for e in z["emotions"]].index(EMOTION)
        grid = np.nanmean(np.asarray(z["rsa"], float)[:, si, :, ei, :], axis=0)   # layers x windows
        n_layers, n_windows = grid.shape
        extent = (0, 1000, 0, 1)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-VMAX, vmax=VMAX, extent=extent, interpolation="nearest")
        masks, p_holm = holm_clusters(tag, si, ei, summary, clusters)
        for mask in masks:
            x = np.linspace(0, 1000, n_windows + 1)
            y = np.linspace(0, 1, n_layers + 1)
            ax.contour((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, mask.astype(float),
                       levels=[.5], colors="black", linewidths=1)
        if p_holm is not None:
            title += rf" ($p_{{\mathrm{{Holm}}}}={p_holm:.3f}$)".replace("=0.", "=.")
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
