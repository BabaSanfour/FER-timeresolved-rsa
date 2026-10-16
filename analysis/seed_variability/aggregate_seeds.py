#!/usr/bin/env python3
"""Aggregate rho_max over random-initialization seeds for the 18-stimulus Table 1.

rho_max is computed SEPARATELY FOR EACH SEED and only then averaged. Averaging
the grids first and taking the maximum afterwards would cancel noise before the
max-selection step and drive the random baseline artificially low; a single
pretrained model goes through the full selection, so each random seed must too.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rho_max import cell_maxima, N_ROI, EMO_SPECIFIC  # noqa: E402

ANALYSIS = HERE.parent
CNN_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
QWEN_SEEDS = [42, 0, 1, 2, 3]

# family -> (seed tag prefix, seeds, [(label, pretrained tag), ...])
FAMILIES = {
    "ResNet18": ("resnet18", CNN_SEEDS, [
        ("ImageNet", "resnet18_imagenet_16f_weights"),
        ("VGGFace2", "resnet18_vggface2_16f_weights"),
        ("AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights")]),
    "ResNet50": ("resnet50", CNN_SEEDS, [
        ("ImageNet", "resnet50_imagenet_16f_weights"),
        ("VGGFace2", "resnet50_vggface2_16f_weights"),
        ("AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights")]),
    "FaceNet": ("facenet", CNN_SEEDS, [
        ("ImageNet", "facenet_imagenet_16f_weights"),
        ("VGGFace2 300k", "facenet_vggface2_300k_16f_weights"),
        ("VGGFace2 500k", "facenet_vggface2_500k_16f_weights"),
        ("VGGFace2 800k", "facenet_vggface3_16f"),
        ("Complete VGGFace2", "vggface_affectnet_16f_pretrained"),
        ("AffectNet FT", "vggface_affectnet_16f_finetuned")]),
    "VGG16": ("vgg16bn", CNN_SEEDS, [
        ("ImageNet (BN)", "vgg16bn_imagenet_16f_weights"),
        ("VGGFace (BN)", "vgg16bn_vggface_1D_16f_weights"),
        ("VGGFace (non-BN)", "vgg16_vggface_16f_weights"),
        ("AffectNet FT (non-BN)", "vgg16_vggface_affectnet_3class_16f_weights")]),
    "M3DFEL": ("m3dfel", CNN_SEEDS, [
        ("Kinetics", "m3dfel_pretrained_16f_weights"),
        ("DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights")]),
    "Qwen3-VL": ("qwen3vl", QWEN_SEEDS, [
        ("Pretrained", "qwen3vl_pretrained_16f_weights")]),
}
COLS = ["Ant-L", "Ant-R", "Post-L", "Post-R", "Temp-L", "Temp-R", "Whole", "ROI avg."]


def eight_columns(rsa):
    """The eight Table 1 numbers: 6 ROIs, wholescalp, ROI-average -- each the
    mean over the three emotion-specific per-cell maxima."""
    c = cell_maxima(rsa)
    per_roi = [float(np.mean([c[(oi, ei)] for ei in EMO_SPECIFIC])) for oi in range(N_ROI)]
    whole = float(np.mean([c[(6, ei)] for ei in EMO_SPECIFIC]))
    return per_roi + [whole, float(np.mean(per_roi))]


def load(grid_dir, tag):
    p = Path(grid_dir) / f"{tag}_rsa_grid.npz"
    if not p.exists():
        return None
    return eight_columns(np.load(p, allow_pickle=True)["rsa"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-grids", required=True)
    ap.add_argument("--pretrained-grids", required=True)
    ap.add_argument("--cohort-label", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    per_seed_rows, summary_rows, cmp_rows = [], [], []
    for fam, (prefix, seeds, pretrained) in FAMILIES.items():
        vals, used = [], []
        for s in seeds:
            v = load(args.seed_grids, f"{prefix}_seed{s}_16f_weights")
            if v is None:
                print(f"  [missing] {prefix}_seed{s}"); continue
            vals.append(v); used.append(s)
            per_seed_rows.append({"family": fam, "seed": s,
                                  **{c: round(x, 5) for c, x in zip(COLS, v)}})
        if not vals:
            continue
        a = np.array(vals)                       # (n_seeds, 8)
        mean, sd = a.mean(0), a.std(0, ddof=1)
        summary_rows.append({"family": fam, "n_seeds": len(used),
                             "seeds": " ".join(map(str, used)),
                             **{f"{c}_mean": round(m, 5) for c, m in zip(COLS, mean)},
                             **{f"{c}_sd": round(x, 5) for c, x in zip(COLS, sd)}})
        roi = a[:, -1]
        print(f"\n{fam}: rho_max (ROI avg.) over {len(used)} seeds = "
              f"{roi.mean():.4f} +/- {roi.std(ddof=1):.4f}  "
              f"[min {roi.min():.4f}, max {roi.max():.4f}]")
        for label, tag in pretrained:
            pv = load(args.pretrained_grids, tag)
            if pv is None:
                print(f"    [missing pretrained] {tag}"); continue
            p = pv[-1]
            n_above = int((roi < p).sum())
            verdict = ("exceeds all seeds" if n_above == len(roi)
                       else f"WITHIN seed range (beats {n_above}/{len(roi)})")
            z = (p - roi.mean()) / roi.std(ddof=1) if roi.std(ddof=1) > 0 else np.nan
            print(f"    {label:24s} {p:.4f}   {verdict}   z = {z:+.1f}")
            cmp_rows.append({"family": fam, "variant": label, "tag": tag,
                             "pretrained_roi_avg": round(p, 5),
                             "seed_mean": round(roi.mean(), 5),
                             "seed_sd": round(roi.std(ddof=1), 5),
                             "seed_min": round(roi.min(), 5),
                             "seed_max": round(roi.max(), 5),
                             "n_seeds_below_pretrained": n_above,
                             "n_seeds": len(roi),
                             "exceeds_all_seeds": int(n_above == len(roi)),
                             "z_vs_seed_dist": round(float(z), 3)})

    def write(name, rows):
        if not rows:
            return
        p = Path(f"{args.out_prefix}_{name}.csv")
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"[write] {p}")

    write("per_seed", per_seed_rows)
    write("summary", summary_rows)
    write("vs_pretrained", cmp_rows)

    n_fam = len({r["family"] for r in cmp_rows})
    all_exceed = [f for f in {r["family"] for r in cmp_rows}
                  if all(r["exceeds_all_seeds"] for r in cmp_rows if r["family"] == f)]
    print(f"\n[{args.cohort_label}] families where EVERY trained variant exceeds every "
          f"random seed: {len(all_exceed)}/{n_fam} -> {sorted(all_exceed)}")
    inside = sorted({r["family"] for r in cmp_rows if not r["exceeds_all_seeds"]})
    if inside:
        print(f"[{args.cohort_label}] families with at least one trained variant inside "
              f"the seed range: {inside}")


if __name__ == "__main__":
    main()
