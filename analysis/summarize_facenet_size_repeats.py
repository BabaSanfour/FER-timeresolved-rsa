#!/usr/bin/env python3
"""Aggregate max-RSA across ten FaceNet subset repeats and plot mean variability."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FER_ROOT", "."))
GRID = ROOT / "analysis/rsa_grid_facenet_size_repeats"
OUT = ROOT / "analysis/facenet_size_repeats"
OUT.mkdir(parents=True, exist_ok=True)
ROIS = ["Ant-Left", "Ant-Right", "Post-Left", "Post-Right", "Temp-Left", "Temp-Right"]
SCOPES = ROIS + ["wholescalp", "ROI-avg"]
EMOTIONS = ["all", "fear", "happy", "neutral", "emo-avg"]
CONDITIONS = [("300k", 300_000), ("500k", 500_000), ("800k bootstrap", 800_000)]


def summarize_grid(path):
    data = np.load(path, allow_pickle=True)
    rsa = data["rsa"]
    scopes = [str(x) for x in data["scopes"]]
    emotions = [str(x) for x in data["emotions"]]
    gm = np.nanmean(rsa, axis=0)  # scope, layer, emotion, window
    base = {}
    for oi, scope in enumerate(scopes):
        for ei, emotion in enumerate(emotions):
            base[scope, emotion] = float(np.nanmax(gm[oi, :, ei, :]))
    result = dict(base)
    for emotion in emotions:
        result["ROI-avg", emotion] = np.mean([base[roi, emotion] for roi in ROIS])
    for scope in scopes + ["ROI-avg"]:
        result[scope, "emo-avg"] = np.mean([result[scope, e]
                                             for e in ("fear", "happy", "neutral")])
    return result


rows = []
for repeat in range(1, 11):
    for condition, draws in CONDITIONS:
        token = condition.replace(" ", "_")
        tag = f"facenet_vggface2_repeat{repeat:02d}_{token}_16f_weights"
        path = GRID / f"{tag}_rsa_grid.npz"
        values = summarize_grid(path)
        for scope in SCOPES:
            for emotion in EMOTIONS:
                rows.append({"repeat": repeat, "seed": 1000 + repeat,
                             "condition": condition, "training_draws": draws,
                             "scope": scope, "emotion": emotion,
                             "max_r": values[scope, emotion]})

run = pd.DataFrame(rows)
run.to_csv(OUT / "facenet_size_repeat_maxr_runs.csv", index=False, float_format="%.6f")
summary = (run.groupby(["condition", "training_draws", "scope", "emotion"], sort=False)
           .max_r.agg(["count", "mean", "std", "sem"]).reset_index())
summary["ci95"] = 2.262 * summary["sem"]  # t_0.975, df=9
summary.to_csv(OUT / "facenet_size_repeat_maxr_summary.csv", index=False, float_format="%.6f")

# Fixed complete-corpus reference from the original pretrained FaceNet.
complete_candidates = [
    ROOT / "analysis/rsa_grid_facenet_original_vggface2_affectnet_16win/vggface_affectnet_16f_pretrained_rsa_grid.npz",
    ROOT / "analysis/rsa_grid/vggface_affectnet_16f_pretrained_rsa_grid.npz",
]
complete_path = next((p for p in complete_candidates if p.exists()), None)
complete = summarize_grid(complete_path) if complete_path else None

colors = {"all": "#333333", "fear": "#D55E00", "happy": "#009E73",
          "neutral": "#0072B2", "emo-avg": "#CC79A7"}
fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
x = np.array([300_000, 500_000, 800_000]) / 1000
for ax, scope in zip(axes.flat, SCOPES):
    for emotion in EMOTIONS:
        q = summary[(summary.scope == scope) & (summary.emotion == emotion)].sort_values("training_draws")
        ax.errorbar(x, q["mean"], yerr=q.ci95, marker="o", lw=1.6, capsize=3,
                    color=colors[emotion], label=emotion)
        if complete is not None:
            ax.scatter([1100], [complete[scope, emotion]], marker="D", s=28,
                       color=colors[emotion], edgecolor="white", linewidth=.5)
    ax.axvline(715.81, color="#888", ls=":", lw=.8)
    ax.set_title(scope)
    ax.grid(alpha=.2)
    ax.set_xticks([300, 500, 800, 1100], ["300k", "500k", "800k*", "complete"])
for ax in axes[:, 0]:
    ax.set_ylabel("mean max RSA")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5,
           frameon=False, bbox_to_anchor=(.5, 1.015))
fig.suptitle("FaceNet RSA by pretraining exposure: mean and 95% CI across 10 repeats\n"
             "*800k is bootstrap sampling from 715,810 unique training images; diamonds are the fixed complete-model reference",
             y=1.09)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT / "facenet_size_repeat_mean_rsa.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[ok] {len(run)} run-level values and {len(summary)} summaries -> {OUT}")
