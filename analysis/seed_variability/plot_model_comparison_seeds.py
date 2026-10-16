#!/usr/bin/env python3
"""ROI-averaged max_r bar plot (x = emotion) for all models, 25-subject cohort,
with the random-init bars replaced by the seed ensemble (mean +/- 1 SD).

Redraws panel 3 of analysis/plot_maxr_barplots_allmodels.py (the paper's
figures/new_model_comparison.pdf) on the same data as the regenerated Table 1:

  random   per-seed grids in seed_variability/rsa_no-ceg; rho_max is taken per
           seed and only then averaged (see aggregate_seeds.py), error bar = SD
  trained  seed_variability/rsa_noceg_pretrained, except the two Complete
           VGGFace2 FaceNet rows, read from the layer-matched grids with the
           random 3-unit head dropped (as in layer_matched/build_table1.py)

Noise ceilings are recomputed on the same 25 subjects, at the window of each
per-cell maximum, and averaged exactly as build_maxr_table.py does. A random
bar is grey when its seed-mean lies below the seed-mean lower ceiling.
"""
import argparse
import csv
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(os.environ.get("FER_ROOT", "."))
HERE = ROOT / "analysis" / "seed_variability"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))
import build_maxr_table as B          # noqa: E402  (noise_ceilings)
from rsa_grid_seeds import load_eeg   # noqa: E402
from aggregate_seeds import FAMILIES as SEED_FAMILIES  # noqa: E402

SEED_DIR = HERE / "rsa_no-ceg"
GRID_DIR = HERE / "rsa_noceg_pretrained"
MATCHED_DIR = ROOT / "analysis" / "layer_matched" / "rsa_no-ceg"

EMOS = ["all", "fear", "happy", "neutral", "emo-avg"]
N_ROI = 6

# family -> [(state, tag, source)]; same variants as the published figure
MODELS = {
    "ResNet18": [("ImageNet", "resnet18_imagenet_16f_weights", "grid"),
                 ("VGGFace2", "resnet18_vggface2_16f_weights", "grid"),
                 ("AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights", "grid")],
    "ResNet50": [("ImageNet", "resnet50_imagenet_16f_weights", "grid"),
                 ("VGGFace2", "resnet50_vggface2_16f_weights", "grid"),
                 ("AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights", "grid")],
    "FaceNet":  [("ImageNet", "facenet_imagenet_16f_weights", "grid"),
                 ("VGGFace2", "vggface_affectnet_16f_pretrained_weights", "matched"),
                 ("AffectNet FT", "vggface_affectnet_16f_finetuned_weights", "matched")],
    "VGG16":    [("ImageNet", "vgg16bn_imagenet_16f_weights", "grid"),
                 ("VGGFace2", "vgg16bn_vggface_1D_16f_weights", "grid"),
                 ("AffectNet FT", "vgg16_vggface_affectnet_3class_16f_weights", "grid")],
    "Qwen3-VL": [("pretrained", "qwen3vl_pretrained_16f_weights", "grid")],
    "M3DFEL":   [("Kinetics", "m3dfel_pretrained_16f_weights", "grid"),
                 ("DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights", "grid")],
}
DROP_LAST = {"vggface_affectnet_16f_pretrained_weights",
             "vggface_affectnet_16f_finetuned_weights"}

SCOL = {"random": "#0072B2", "ImageNet": "#E69F00", "VGGFace2": "#009E73",
        "Kinetics": "#CC79A7", "pretrained": "#56B4E9",
        "AffectNet FT": "#D55E00", "DFEW FT": "#882255"}
SLABEL = {"random": "random (mean ± 1 SD over seeds)", "ImageNet": "ImageNet",
          "VGGFace2": "VGGFace2 (3.31M images)", "Kinetics": "Kinetics-400",
          "pretrained": "pretrained", "AffectNet FT": "AffectNet fine-tuned",
          "DFEW FT": "DFEW fine-tuned"}
STATE_ORDER = list(SCOL)
GREY = "#B9B9B9"


def roiavg_by_emotion(rsa, lo):
    """-> {emo: (max_r, nc_lower)} for the ROI-avg scope, following
    build_maxr_table.py: per-cell max over layer x window first, then average."""
    gm = np.nanmean(rsa, axis=0)                       # (O, L, E, W)
    v = np.zeros((N_ROI, 4)); l = np.zeros((N_ROI, 4))
    for oi in range(N_ROI):
        for ei in range(4):
            block = gm[oi, :, ei, :]
            li, ki = np.unravel_index(np.nanargmax(block), block.shape)
            v[oi, ei] = block[li, ki]
            l[oi, ei] = lo[oi, ei, ki]
    out = {e: (v[:, ei].mean(), l[:, ei].mean()) for ei, e in enumerate(EMOS[:4])}
    out["emo-avg"] = (v[:, 1:].mean(), l[:, 1:].mean())
    return out


def load(path, drop_last=False):
    rsa = np.load(path, allow_pickle=True)["rsa"]
    return rsa[:, :, :-1] if drop_last else rsa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--csv", default=str(HERE / "results" / "model_comparison_no-ceg.csv"))
    args = ap.parse_args()

    eeg, subjects, _ = load_eeg("no-ceg")
    print(f"[eeg] {len(subjects)} subjects; computing noise ceilings ...")
    lo, _up = B.noise_ceilings(eeg)                    # (7, 4, W)

    data, rows = {}, []                                # data[model][state][emo] = (m, sd, nc)
    for model, trained in MODELS.items():
        prefix, seeds, _ = SEED_FAMILIES[model]
        per_seed = [roiavg_by_emotion(load(SEED_DIR / f"{prefix}_seed{s}_16f_weights_rsa_grid.npz"), lo)
                    for s in seeds]
        d = {"random": {}}
        for e in EMOS:
            vs = np.array([p[e][0] for p in per_seed]); ls = np.array([p[e][1] for p in per_seed])
            d["random"][e] = (vs.mean(), vs.std(ddof=1), ls.mean())
            rows.append({"model": model, "state": "random", "emotion": e, "n": len(seeds),
                         "max_r": round(vs.mean(), 5), "sd": round(vs.std(ddof=1), 5),
                         "nc_lower": round(ls.mean(), 5)})
        for state, tag, src in trained:
            f = (MATCHED_DIR if src == "matched" else GRID_DIR) / f"{tag}_rsa_grid.npz"
            r = roiavg_by_emotion(load(f, tag in DROP_LAST), lo)
            d[state] = {e: (r[e][0], np.nan, r[e][1]) for e in EMOS}
            for e in EMOS:
                rows.append({"model": model, "state": state, "emotion": e, "n": 1,
                             "max_r": round(r[e][0], 5), "sd": "", "nc_lower": round(r[e][1], 5)})
        data[model] = d
        print(f"  {model:9s} random emo-avg = {d['random']['emo-avg'][0]:.4f} "
              f"± {d['random']['emo-avg'][1]:.4f} (n={len(seeds)})")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[write] {args.csv}")

    # ---- plot: same layout as plot_maxr_barplots_allmodels.py panel 3 ----
    names = list(MODELS)
    ncol = 3; nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, names):
        states = [s for s in STATE_ORDER if s in data[model]]
        nS = len(states); x = np.arange(len(EMOS)); w = 0.8 / nS
        for si, st in enumerate(states):
            m = np.array([data[model][st][e][0] for e in EMOS])
            sd = np.array([data[model][st][e][1] for e in EMOS])
            nc = np.array([data[model][st][e][2] for e in EMOS])
            cols = [GREY if v < l else SCOL[st] for v, l in zip(m, nc)]
            xpos = x + (si - (nS - 1) / 2) * w
            ax.bar(xpos, m, width=w * 0.92, color=cols, edgecolor="white",
                   linewidth=0.5, zorder=3)
            if st == "random":
                ax.errorbar(xpos, m, yerr=sd, fmt="none", ecolor="#222", elinewidth=0.9,
                            capsize=2.5, capthick=0.9, zorder=5)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(EMOS, rotation=20, ha="right", fontsize=8)
        ax.set_title(model, fontsize=11)
        ax.grid(axis="y", color="#eee", zorder=0)
        ax.set_ylabel("max_r (ROI-avg)", fontsize=8)
    for ax in axes[len(names):]:
        ax.axis("off")

    used = [s for s in STATE_ORDER if any(s in data[m] for m in names)]
    handles = [Patch(fc=SCOL[s], ec="white", label=SLABEL[s]) for s in used]
    handles += [Patch(fc=GREY, ec="white", label="below lower NC")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Time-locked max_r averaged over the 6 ROIs — all models",
                 y=0.93, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
