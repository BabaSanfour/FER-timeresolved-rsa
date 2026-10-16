#!/usr/bin/env python3
"""Rebuild Table 1 (tab:main-results, 18-stimulus) for the 25-subject cohort.

Three corrections relative to the published table:
  1. cohort      -- CEG dropped, so the table matches the stated 25 participants
  2. Random rows -- mean +/- SD over random-init seeds instead of seed 42 alone
  3. FaceNet     -- the two Complete VGGFace2 rows are read from grids
                    re-extracted at the `weights` layer selection, so every
                    FaceNet row is a max over the same 133-layer axis
Row order and labels are copied from the published table verbatim.
"""
import argparse
import csv
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FER_ROOT", "."))
sys.path.insert(0, str(ROOT / "analysis" / "seed_variability"))
import rho_max as RM  # noqa: E402

# (Model, Training, tag, source)   source: grid | matched | seeds
ROWS = [
    ("ResNet18", "Random", "ResNet18", "seeds"),
    ("ResNet18", "ImageNet", "resnet18_imagenet_16f_weights", "grid"),
    ("ResNet18", "VGGFace2", "resnet18_vggface2_16f_weights", "grid"),
    ("ResNet18", "AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights", "grid"),
    ("ResNet50", "Random", "ResNet50", "seeds"),
    ("ResNet50", "ImageNet", "resnet50_imagenet_16f_weights", "grid"),
    ("ResNet50", "VGGFace2", "resnet50_vggface2_16f_weights", "grid"),
    ("ResNet50", "AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights", "grid"),
    ("FaceNet", "Random", "FaceNet", "seeds"),
    ("FaceNet", "ImageNet", "facenet_imagenet_16f_weights", "grid"),
    ("FaceNet", "VGGFace2 (300k)", "facenet_vggface2_300k_16f_weights", "grid"),
    ("FaceNet", "Complete VGGFace2", "vggface_affectnet_16f_pretrained_weights", "matched"),
    ("FaceNet", "AffectNet FT", "vggface_affectnet_16f_finetuned_weights", "matched"),
    ("VGG16", "Random (BN)", "VGG16", "seeds"),
    ("VGG16", "ImageNet (BN)", "vgg16bn_imagenet_16f_weights", "grid"),
    ("VGG16", "VGGFace (BN)", "vgg16bn_vggface_1D_16f_weights", "grid"),
    ("VGG16", "VGGFace (non-BN)", "vgg16_vggface_16f_weights", "grid"),
    ("VGG16", "AffectNet FT (non-BN)", "vgg16_vggface_affectnet_3class_16f_weights", "grid"),
    ("Qwen3-VL", "Random", "Qwen3-VL", "seeds"),
    ("Qwen3-VL", "Pretrained", "qwen3vl_pretrained_16f_weights", "grid"),
    ("M3DFEL", "Random", "M3DFEL", "seeds"),
    ("M3DFEL", "Kinetics", "m3dfel_pretrained_16f_weights", "grid"),
    ("M3DFEL", "DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights", "grid"),
]
COLS = ["Ant-L", "Ant-R", "Post-L", "Post-R", "Temp-L", "Temp-R", "Whole", "ROI avg."]
# the Complete VGGFace2 model's last hook is its randomly initialised 3-unit
# head, excluded from every contrast and excluded here
DROP_LAST = {"vggface_affectnet_16f_pretrained_weights",
             "vggface_affectnet_16f_finetuned_weights"}


def three(x):
    """.083 rather than 0.083, matching the table's existing style."""
    return f"{float(x):.3f}".lstrip("0")


def grid_values(path, drop_last):
    rsa = np.load(path, allow_pickle=True)["rsa"]
    if drop_last:
        rsa = rsa[:, :, :-1]
    vals = RM.per_roi_emo_avg(rsa) + [RM.wholescalp_emo_avg(rsa), RM.roi_avg_emo_avg(rsa)]
    return vals, int(rsa.shape[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--matched-dir", required=True)
    ap.add_argument("--summary", required=True, help="seed summary CSV for the same cohort")
    ap.add_argument("--out-tex", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    summ = {r["family"]: r for r in csv.DictReader(open(args.summary))}
    body, recs, missing = [], [], []
    for model, training, tag, src in ROWS:
        if src == "seeds":
            r = summ.get(tag)
            if r is None:
                missing.append(f"{model}/{training} (seed summary)"); continue
            cells = " & ".join(f"{three(r[f'{c}_mean'])}\\,$\\pm$\\,{three(r[f'{c}_sd'])}"
                               for c in COLS)
            body.append(f"{model} & {training} & {cells}\\\\")
            recs.append({"model": model, "training": training, "source": f"{r['n_seeds']} seeds",
                         "n_layers": "", **{c: three(r[f"{c}_mean"]) for c in COLS}})
            continue
        d = Path(args.matched_dir if src == "matched" else args.grid_dir)
        f = d / f"{tag}_rsa_grid.npz"
        if not f.exists():
            missing.append(f"{model}/{training} -> {f}"); continue
        vals, L = grid_values(f, tag in DROP_LAST)
        body.append(f"{model} & {training} & " + " & ".join(three(v) for v in vals) + "\\\\")
        recs.append({"model": model, "training": training, "source": src, "n_layers": L,
                     **{c: three(v) for c, v in zip(COLS, vals)}})

    if missing:
        print("MISSING:\n  " + "\n  ".join(missing), file=sys.stderr)

    head = ("\\begin{tabular}{ll" + "r" * len(COLS) + "}\n\\toprule\n"
            "Model & Training & " + " & ".join(COLS) + "\\\\\n\\midrule")
    tex = head + "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
    Path(args.out_tex).write_text(tex)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)
    print(tex)
    print(f"wrote {args.out_tex} and {args.out_csv}"
          + (f"  [{len(missing)} rows missing]" if missing else ""))


if __name__ == "__main__":
    main()
