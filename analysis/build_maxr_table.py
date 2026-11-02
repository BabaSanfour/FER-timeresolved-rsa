#!/usr/bin/env python3
"""Collapse the full RSA grid into the max_r table the bar plots consume.

max_r = max over (layer x 16 windows) of the SUBJECT-MEAN time-locked Spearman.

Scopes   : 6 ROIs + wholescalp + ROI-avg (mean of 6 per-ROI max_r) -> 8
Emotions : all / fear / happy / neutral + emo-avg (mean of the 3
           per-emotion max_r values)                                -> 5

Noise ceilings are computed per (scope, emotion, window) from the EEG alone:
  upper = subject vs group mean INCLUDING itself
  lower = subject vs leave-one-out group mean
and reported at the window where max_r was found. Averaged scopes/emotions get
correspondingly averaged ceilings.

Output: analysis/paper_stats/timelocked_maxr_over_layers.csv
"""
import csv
import os
from pathlib import Path

import numpy as np

import rsa_core

ROOT = Path(os.environ.get("FER_ROOT", "."))
GRID_DIRS = [
    ROOT / "analysis/rsa_grid",
    ROOT / "analysis/rsa_grid_today",
    ROOT / "analysis/rsa_grid_facenet_original_vggface2_affectnet_16win",
]
OUT = ROOT / "analysis/paper_stats"

EMOTIONS = rsa_core.EMOTIONS["18"]
ROI_AVG, EMO_AVG = "ROI-avg", "emo-avg"

# family -> [(state, rdm tag)]. vgg16bn_random doubles as the vgg16 control:
# randomised BatchNorm is an identity transform (measured r = 0.94).
FAMILIES = {
    "ResNet18": [("random", "resnet18_random_16f_weights"),
                 ("ImageNet", "resnet18_imagenet_16f_weights"),
                 ("VGGFace2", "resnet18_vggface2_16f_weights"),
                 ("AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights")],
    "ResNet50": [("random", "resnet50_random_16f_weights"),
                 ("ImageNet", "resnet50_imagenet_16f_weights"),
                 ("VGGFace2", "resnet50_vggface2_16f_weights"),
                 ("AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights")],
    "FaceNet":  [("random", "facenet_random_16f_weights"),
                 ("ImageNet", "facenet_imagenet_16f_weights"),
                 ("VGGFace2 300k", "facenet_vggface2_300k_16f_weights"),
                 ("VGGFace2 500k", "facenet_vggface2_500k_16f_weights"),
                 ("VGGFace2 800k", "facenet_vggface3_16f"),
                 ("VGGFace2", "vggface_affectnet_16f_pretrained"),
                 ("AffectNet FT", "vggface_affectnet_16f_finetuned")],
    "VGG16-BN": [("random", "vgg16bn_random_16f_weights"),
                 ("ImageNet", "vgg16bn_imagenet_16f_weights"),
                 ("VGGFace", "vgg16bn_vggface_1D_16f_weights")],
    "VGG16":    [("random", "vgg16bn_random_16f_weights"),
                 ("VGGFace", "vgg16_vggface_16f_weights"),
                 ("AffectNet FT", "vgg16_vggface_affectnet_3class_16f_weights")],
    # video-input models
    "Qwen3-VL": [("random", "qwen3vl_random_16f_weights"),
                 ("pretrained", "qwen3vl_pretrained_16f_weights")],
    "M3DFEL":   [("random", "m3dfel_random_16f_weights"),
                 ("Kinetics", "m3dfel_pretrained_16f_weights"),
                 ("DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights")],
}


def grid_path(tag):
    for directory in GRID_DIRS:
        path = directory / f"{tag}_rsa_grid.npz"
        if path.exists():
            return path
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eeg, _, scopes = rsa_core.load_eeg("18")
    wb = rsa_core.window_bounds("18")
    print(f"[eeg] {eeg.shape}  scopes={scopes}")

    print("[nc] computing noise ceilings ...")
    lo, up = rsa_core.noise_ceilings(eeg, "18")              # (7, 4, W)
    scopes_x = scopes + [ROI_AVG]
    emos_x = list(EMOTIONS) + [EMO_AVG]

    rows = []
    for family, states in FAMILIES.items():
        for state, tag in states:
            f = grid_path(tag)
            if f is None:
                print(f"  [skip] {tag}: no grid"); continue
            g = np.load(f, allow_pickle=True)
            rsa = g["rsa"]                                      # (S, O, L, E, W)
            gm = np.nanmean(rsa, axis=0)                        # (O, L, E, W)

            # Select each base scope x emotion maximum FIRST.
            base = {}
            for oi, sc in enumerate(scopes):
                for ei, emo in enumerate(EMOTIONS):
                    block = gm[oi, :, ei, :]
                    if np.all(np.isnan(block)):
                        continue
                    li, ki = np.unravel_index(np.nanargmax(block), block.shape)
                    base[(oi, ei)] = {
                        "v": float(block[li, ki]), "li": int(li), "ki": int(ki),
                        "lo": float(lo[oi, ei, ki]), "up": float(up[oi, ei, ki]),
                    }

            for oi, sc in enumerate(scopes_x):
                for ei, emo in enumerate(emos_x):
                    if oi < len(scopes) and ei < len(EMOTIONS):
                        cells = [base[(oi, ei)]]
                    elif oi == len(scopes) and ei < len(EMOTIONS):
                        cells = [base[(ri, ei)] for ri in range(6)]
                    elif oi < len(scopes) and ei == len(EMOTIONS):
                        cells = [base[(oi, xi)] for xi in (1, 2, 3)]
                    else:
                        cells = [base[(ri, xi)] for ri in range(6) for xi in (1, 2, 3)]

                    v = float(np.mean([c["v"] for c in cells]))
                    ncl = float(np.mean([c["lo"] for c in cells]))
                    ncu = float(np.mean([c["up"] for c in cells]))
                    averaged = len(cells) > 1
                    li = -1 if averaged else cells[0]["li"]
                    ki = -1 if averaged else cells[0]["ki"]
                    rows.append({
                        "model": family, "state": state, "tag": tag, "emotion": emo,
                        "roi": sc, "max_r": round(v, 5), "max_layer": int(li),
                        "max_window": int(ki),
                        "win_start_ms": -1 if averaged else int(wb[ki][0]),
                        "win_end_ms": -1 if averaged else int(wb[ki][1]),
                        "nc_lower": round(ncl, 5), "nc_upper": round(ncu, 5),
                        "within_nc": "ABOVE" if v > ncu else ("below" if v < ncl else "within"),
                    })
            print(f"  {family:9s} {state:9s} {tag:34s} L={rsa.shape[2]:3d}")

    cols = ["model", "state", "tag", "emotion", "roi", "max_r", "max_layer", "max_window",
            "win_start_ms", "win_end_ms", "nc_lower", "nc_upper", "within_nc"]
    with (OUT / "timelocked_maxr_over_layers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_below = sum(r["within_nc"] == "below" for r in rows)
    print(f"\n[ok] {len(rows)} rows -> {OUT / 'timelocked_maxr_over_layers.csv'}")
    print(f"     {n_below} below lower NC, {sum(r['within_nc'] == 'ABOVE' for r in rows)} above upper NC")


if __name__ == "__main__":
    main()
