#!/usr/bin/env python3
"""rho_max table of every model variant, with noise ceilings (--stimuli 18 | 54).

rho_max = maximum over (layer x window) of the participant-mean time-locked
Spearman, taken separately in each scope x emotion cell (rsa_core.cell_maxima).
Scopes: 6 ROIs, whole scalp, ROI-avg (mean of the six ROI values); emotions:
all, fear, happy, neutral, emo-avg (mean of the three emotion values). The noise
ceilings are read at the window of each maximum and averaged like the values.

Output: analysis/paper_stats/timelocked_maxr_over_layers.csv (18 stimuli)
        analysis/paper_stats_54/timelocked_maxr_54.csv       (54 stimuli)
"""
import argparse
import csv

import numpy as np

import rsa_core

ROOT = rsa_core.ROOT
OUT = {"18": ROOT / "analysis/paper_stats/timelocked_maxr_over_layers.csv",
       "54": ROOT / "analysis/paper_stats_54/timelocked_maxr_54.csv"}

# family -> [(state, rdm tag)]. vgg16bn_random doubles as the vgg16 control:
# randomised BatchNorm is an identity transform (measured r = 0.94).
FAMILIES_18 = {
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

FAMILIES_54 = {
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
FAMILIES = {"18": FAMILIES_18, "54": FAMILIES_54}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    args = ap.parse_args()
    out = OUT[args.stimuli]
    out.parent.mkdir(parents=True, exist_ok=True)
    eeg, _, scopes = rsa_core.load_eeg(args.stimuli)
    wb = rsa_core.window_bounds(args.stimuli)
    print(f"[eeg] {eeg.shape}; computing noise ceilings ...")
    lo, up = rsa_core.noise_ceilings(eeg, args.stimuli)              # (7, 4, W)
    scopes_x = scopes + [rsa_core.ROI_AVG]
    emos_x = list(rsa_core.EMOTIONS[args.stimuli]) + [rsa_core.EMO_AVG]

    rows = []
    for family, states in FAMILIES[args.stimuli].items():
        for state, tag in states:
            path = rsa_core.grid_path(args.stimuli, tag)
            if path is None:
                print(f"  [skip] {tag}: no grid"); continue
            rsa = np.load(path, allow_pickle=True)["rsa"]              # (S, O, L, E, W)
            cells = rsa_core.averaged_cells(rsa_core.cell_maxima(rsa, lo, up))
            for oi, sc in enumerate(scopes_x):
                for ei, emo in enumerate(emos_x):
                    v, ncl, ncu, li, ki = cells[oi, ei]
                    rows.append({
                        "model": family, "state": state, "tag": tag, "emotion": emo,
                        "roi": sc, "max_r": round(v, 5), "max_layer": li, "max_window": ki,
                        "win_start_ms": -1 if ki < 0 else int(wb[ki][0]),
                        "win_end_ms": -1 if ki < 0 else int(wb[ki][1]),
                        "nc_lower": round(ncl, 5), "nc_upper": round(ncu, 5),
                        "within_nc": "ABOVE" if v > ncu else ("below" if v < ncl else "within"),
                    })
            print(f"  {family:9s} {state:13s} {tag:46s} L={rsa.shape[2]:3d}")

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_below = sum(r["within_nc"] == "below" for r in rows)
    print(f"\n[ok] {len(rows)} rows -> {out}")
    print(f"     {n_below} below lower NC, {sum(r['within_nc'] == 'ABOVE' for r in rows)} above upper NC")


if __name__ == "__main__":
    main()
