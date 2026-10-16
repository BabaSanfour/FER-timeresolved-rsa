#!/usr/bin/env python3
"""Build corrected max-r summary table for the 54-stimulus study."""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("FER_ROOT", "."))
GRID = ROOT / "analysis/rsa_grid_54"
EEG = ROOT / "eeg_rdms_complete"
OUT = ROOT / "analysis/paper_stats_54"
EMOS = ("all", "fear", "happy", "neutral")
FAMILIES = {
    "ResNet18": [("random", "resnet18_random_16f_weights"),
                 ("ImageNet", "resnet18_imagenet_16f_weights"),
                 ("VGGFace2", "resnet18_vggface2_16f_weights"),
                 ("AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights")],
    "ResNet50": [("random", "resnet50_random_16f_weights"),
                 ("ImageNet", "resnet50_imagenet_16f_weights"),
                 ("VGGFace2", "resnet50_vggface2_16f_weights"),
                 ("AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights")],
    "FaceNet": [("random", "facenet_random_16f_weights"),
                ("ImageNet", "facenet_imagenet_16f_weights"),
                ("VGGFace2 300k", "facenet_vggface2_300k_16f_weights"),
                ("VGGFace2 500k", "facenet_vggface2_500k_16f_weights"),
                ("VGGFace2 800k", "facenet_vggface3_16f"),
                ("Complete VGGFace2", "vggface_complete_16f_pretrained"),
                ("AffectNet FT", "facenet_vggface2_affectnet_3class_16f_weights")],
    "VGG16": [("random", "vgg16_random_16f_weights"),
              ("VGGFace", "vgg16_vggface_16f_weights"),
              ("AffectNet FT", "vgg16_vggface_affectnet_3class_16f_weights")],
    "M3DFEL": [("random", "m3dfel_random_16f_weights"),
               ("Kinetics", "m3dfel_pretrained_16f_weights"),
               ("DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights")],
    "Qwen3-VL": [("random", "qwen3vl_random_16f_weights"),
                 ("pretrained", "qwen3vl_pretrained_16f_weights")],
}


def upper(x):
    iu = np.triu_indices(x.shape[-1], 1)
    return x[..., iu[0], iu[1]]


def rho(x, y):
    x, y = rankdata(x), rankdata(y)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def row_rho(x, y):
    """Vectorised row-wise Spearman; x/y are (subjects,pairs)."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    denom = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry, axis=1)
    return np.divide(np.sum(rx * ry, axis=1), denom,
                     out=np.full(len(x), np.nan), where=denom > 0)


def load_eeg(emotion, subjects):
    roi = np.stack([np.load(EEG / "perroi" / f"{s}_{emotion}_rdms.npy") for s in subjects])
    ws = np.stack([np.load(EEG / "wholescalp" / f"{s}_{emotion}_rdms.npy") for s in subjects])
    return np.concatenate([roi, ws[:, None]], 1)


def ceilings(subjects):
    result = {}
    for ei, emotion in enumerate(EMOS):
        eeg = load_eeg(emotion, subjects)
        S, O, W = eeg.shape[:3]
        lo, up = np.full((O, W), np.nan), np.full((O, W), np.nan)
        vec = upper(eeg)
        for oi in range(O):
            for wi in range(W):
                group = vec[:, oi, wi]
                total = group.sum(0)
                gm = group.mean(0)
                up[oi, wi] = np.nanmean(row_rho(
                    group, np.broadcast_to(gm, group.shape)))
                lo[oi, wi] = np.nanmean(row_rho(
                    group, (total[None] - group) / (S - 1)))
        result[ei] = (lo, up)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = np.load(next(GRID.glob("*_rsa_grid.npz")), allow_pickle=True)
    subjects = [str(x) for x in sample["subjects"]]
    scopes = [str(x) for x in sample["scopes"]]
    nc = ceilings(subjects)
    rows = []
    for model, states in FAMILIES.items():
        for state, tag in states:
            path = GRID / f"{tag}_rsa_grid.npz"
            if not path.exists():
                print(f"[skip] {tag}"); continue
            grid = np.load(path, allow_pickle=True)["rsa"]
            gm = np.nanmean(grid, axis=0)
            base = {}
            for oi in range(7):
                for ei in range(4):
                    block = gm[oi, :, ei, :]
                    li, wi = np.unravel_index(np.nanargmax(block), block.shape)
                    lo, up = nc[ei]
                    base[oi, ei] = (float(block[li, wi]), float(lo[oi, wi]),
                                    float(up[oi, wi]), int(li), int(wi))
            for oi, scope in enumerate(scopes + ["ROI-avg"]):
                for ei, emotion in enumerate(list(EMOS) + ["emo-avg"]):
                    if oi < 7 and ei < 4:
                        cells = [base[oi, ei]]
                    elif oi == 7 and ei < 4:
                        cells = [base[r, ei] for r in range(6)]
                    elif oi < 7 and ei == 4:
                        cells = [base[oi, e] for e in (1, 2, 3)]
                    else:
                        cells = [base[r, e] for r in range(6) for e in (1, 2, 3)]
                    val, lower, upper = [float(np.mean([c[j] for c in cells]))
                                         for j in range(3)]
                    rows.append({"model": model, "state": state, "tag": tag,
                                 "emotion": emotion, "roi": scope, "max_r": val,
                                 "nc_lower": lower, "nc_upper": upper,
                                 "within_nc": "below" if val < lower else
                                 ("ABOVE" if val > upper else "within"),
                                 "max_layer": cells[0][3] if len(cells) == 1 else -1,
                                 "max_window": cells[0][4] if len(cells) == 1 else -1})
            print(f"[ok] {model} {state}: L={grid.shape[2]}")
    path = OUT / "timelocked_maxr_54.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
