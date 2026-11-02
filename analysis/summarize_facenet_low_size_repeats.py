#!/usr/bin/env python3
"""Aggregate and plot ten-repeat FaceNet RSA for 10k/50k/100k/200k subsets."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import rsa_core

ROOT = Path(os.environ.get("FER_ROOT", "."))
GRID = ROOT / "analysis/rsa_grid_facenet_size_repeats"
OUT = ROOT / "analysis/facenet_low_size_repeats"
OUT.mkdir(parents=True, exist_ok=True)
ROIS = ["Ant-Left", "Ant-Right", "Post-Left", "Post-Right", "Temp-Left", "Temp-Right"]
SCOPES = ROIS + ["wholescalp", "ROI-avg"]
EMOS = list(rsa_core.EMOTIONS["18"]) + ["emo-avg"]
SIZES = [
    ("10k", "10k"), ("50k", "50k"), ("100k", "100k"),
    ("200k", "200k"), ("300k", "300k"), ("500k", "500k"),
    ("800k", "800k_bootstrap"),
]
BELOW = "#C7C7C7"


def summarize(path, lo, up, statistic):
    """-> {(scope, emotion): (value, nc_lower, nc_upper)}; max_r is rho_max
    (rsa_core.cell_maxima), mean_r the mean over layer x window, with the
    window-averaged ceilings."""
    rsa = np.load(path, allow_pickle=True)["rsa"]
    if statistic == "max_r":
        cells = rsa_core.cell_maxima(rsa, lo, up)
    else:
        gm = np.nanmean(rsa, axis=0)
        cells = {(oi, ei): (float(np.nanmean(gm[oi, :, ei, :])), float(lo[oi, ei].mean()),
                            float(up[oi, ei].mean()), -1, -1)
                 for oi in range(7) for ei in range(4)}
    cells = rsa_core.averaged_cells(cells)
    return {(scope, emotion): cells[oi, ei][:3]
            for oi, scope in enumerate(SCOPES) for ei, emotion in enumerate(EMOS)}


def fixed(tag, lo, up, statistic):
    values = summarize(rsa_core.grid_path("18", tag), lo, up, statistic)
    return {key: (*cell, 0.0) for key, cell in values.items()}


def repeated(condition_tag, lo, up, statistic):
    runs = []
    for repeat in range(1, 11):
        tag = f"facenet_vggface2_repeat{repeat:02d}_{condition_tag}_16f_weights"
        runs.append(summarize(GRID / f"{tag}_rsa_grid.npz", lo, up, statistic))
    result = {}
    for key in runs[0]:
        means = tuple(float(np.mean([run[key][j] for run in runs])) for j in range(3))
        sd = float(np.std([run[key][0] for run in runs], ddof=1))
        result[key] = (*means, sd)
    return result


def make(statistic, lo, up):
    palette = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5",
               "#08519C", "#6A51A3", "#CC79A7"]
    conditions = [("Random", "#7F7F7F", fixed("facenet_random_16f_weights", lo, up, statistic))]
    conditions += [(f"VGGFace2 {label}", color, repeated(tag, lo, up, statistic))
                   for (label, tag), color in zip(SIZES, palette)]
    conditions += [
        ("Complete VGGFace2", "#009E73",
         fixed("vggface_affectnet_16f_pretrained", lo, up, statistic)),
    ]
    rows = []
    for label, _, values in conditions:
        for scope in SCOPES:
            for emotion in EMOS:
                value, lower, upper, sd = values[scope, emotion]
                rows.append({"condition": label, "scope": scope, "emotion": emotion,
                             statistic: value, "standard_deviation": sd,
                             "noise_ceiling_lower": lower,
                             "noise_ceiling_upper": upper})
    pd.DataFrame(rows).to_csv(OUT / f"{statistic}_facenet_all_sizes.csv",
                              index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 5, figsize=(23, 4.8), sharey=True)
    x = np.arange(len(SCOPES)); width = .86 / len(conditions)
    for ci, (label, color, values) in enumerate(conditions):
        xpos = x + (ci - (len(conditions) - 1) / 2) * width
        for ax, emotion in zip(axes, EMOS):
            heights, colors, ceilings, errors = [], [], [], []
            for scope in SCOPES:
                value, lower, upper, sd = values[scope, emotion]
                heights.append(value); colors.append(BELOW if value < lower else color)
                ceilings.append(upper)
                errors.append(sd)
            ax.bar(xpos, heights, width=width * .92, color=colors,
                   edgecolor="white", linewidth=.4, yerr=errors, capsize=1.8,
                   error_kw={"elinewidth": .65, "capthick": .65}, zorder=3)
            for xp, ceiling in zip(xpos, ceilings):
                ax.plot([xp-width*.4, xp+width*.4], [ceiling, ceiling],
                        color="#777", lw=.6, alpha=.5, zorder=4)
    for ax, emotion in zip(axes, EMOS):
        ax.axhline(0, color="#888", lw=.6); ax.set_title(emotion)
        ax.set_xticks(x); ax.set_xticklabels(SCOPES, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", color="#eee", zorder=0)
    axes[0].set_ylabel(f"FaceNet {statistic}")
    handles = [Patch(fc=c, ec="white", label=l) for l, c, _ in conditions]
    handles += [Patch(fc=BELOW, ec="white", label="below lower NC"),
                plt.Line2D([0], [0], color="#777", lw=.8, label="upper NC")]
    fig.legend(handles=handles, loc="upper center", ncol=9, frameon=False,
               fontsize=8.5, bbox_to_anchor=(.5, .985))
    fig.suptitle(f"FaceNet: time-locked {statistic} by VGGFace2 training-set size",
                 y=1.04, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(OUT / f"{statistic}_bars_facenet_all_sizes.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT / f"{statistic}_bars_facenet_all_sizes.pdf",
                bbox_inches="tight")
    plt.close(fig)


eeg = rsa_core.load_eeg("18")[0]
lo, up = rsa_core.noise_ceilings(eeg, "18")
make("max_r", lo, up)
make("mean_r", lo, up)
print(f"[ok] all-size max_r/mean_r CSVs and plots -> {OUT}")
