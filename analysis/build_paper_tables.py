#!/usr/bin/env python3
"""Table 1 (tab:main-results) and the random-seed table (tab:seeds) of the paper.

Trained rows: emotion-averaged rho_max from analysis/paper_stats/timelocked_maxr_over_layers.csv
(written by analysis/build_maxr_table.py from the 18-stimulus RSA grids in
analysis/rsa_grid/, rsa_grid_today/ and rsa_grid_facenet_original_vggface2_affectnet_16win/).

Random rows and seed ranges: analysis/seed_variability/results/no-ceg_summary.csv and
no-ceg_per_seed.csv (written by analysis/seed_variability/aggregate_seeds.py; 10 seeds per
family, 5 for Qwen3-VL, rho_max computed per seed before averaging).

Writes analysis/paper_stats/table_main_results.tex and table_seeds.tex.
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("FER_ROOT", "."))
MAXR = ROOT / "analysis/paper_stats/timelocked_maxr_over_layers.csv"
SEEDS = ROOT / "analysis/seed_variability/results"
OUT = ROOT / "analysis/paper_stats"

SCOPES = ["Ant-Left", "Ant-Right", "Post-Left", "Post-Right", "Temp-Left", "Temp-Right",
          "wholescalp", "ROI-avg"]
SEED_COLS = ["Ant-L", "Ant-R", "Post-L", "Post-R", "Temp-L", "Temp-R", "Whole", "ROI avg."]
# (family, label in the paper, (model, state) in timelocked_maxr_over_layers.csv)
ROWS = {
    "ResNet18": [("ImageNet", ("ResNet18", "ImageNet")), ("VGGFace2", ("ResNet18", "VGGFace2")),
                 ("AffectNet FT", ("ResNet18", "AffectNet FT"))],
    "ResNet50": [("ImageNet", ("ResNet50", "ImageNet")), ("VGGFace2", ("ResNet50", "VGGFace2")),
                 ("AffectNet FT", ("ResNet50", "AffectNet FT"))],
    "FaceNet": [("ImageNet", ("FaceNet", "ImageNet")),
                ("Complete VGGFace2", ("FaceNet", "VGGFace2")),
                ("AffectNet FT", ("FaceNet", "AffectNet FT"))],
    # ImageNet is the batch-norm VGG16 (vgg16bn_imagenet); the other two rows are the
    # non-BN VGG-Face model (vgg16_vggface) and its AffectNet fine-tune.
    "VGG16": [("ImageNet", ("VGG16-BN", "ImageNet")), ("VGGFace2", ("VGG16", "VGGFace")),
              ("AffectNet FT", ("VGG16", "AffectNet FT"))],
    "Qwen3-VL": [("Pretrained", ("Qwen3-VL", "pretrained"))],
    "M3DFEL": [("Kinetics-400", ("M3DFEL", "Kinetics")), ("DFEW FT", ("M3DFEL", "DFEW FT"))],
}
N_SEEDS = {"Qwen3-VL": 5}


def three(x):
    """.083 rather than 0.083, as in the paper."""
    return f"{float(x):.3f}".replace("0.", ".", 1)


def main():
    maxr = {}
    with MAXR.open() as fh:
        for r in csv.DictReader(fh):
            if r["emotion"] == "emo-avg":
                maxr[(r["model"], r["state"], r["roi"])] = float(r["max_r"])
    summary = {r["family"]: r for r in csv.DictReader((SEEDS / "no-ceg_summary.csv").open())}
    per_seed = defaultdict(list)
    for r in csv.DictReader((SEEDS / "no-ceg_per_seed.csv").open()):
        per_seed[r["family"]].append(float(r["ROI avg."]))

    lines, lowest = [], {}
    for fi, (family, rows) in enumerate(ROWS.items()):
        if fi:
            lines.append(r"\midrule")
        s = summary[family]
        cells = [f"{three(s[c + '_mean'])}$\\pm${three(s[c + '_sd'])}" for c in SEED_COLS]
        lines.append(f"{family} & Random ($n$={N_SEEDS.get(family, 10)}) & " + " & ".join(cells) + r" \\")
        trained = []
        for label, key in rows:
            vals = [maxr[key + (scope,)] for scope in SCOPES]
            trained.append(vals[-1])
            lines.append(f"{family} & {label} & " + " & ".join(three(v) for v in vals) + r" \\")
        lowest[family] = min(trained)
    (OUT / "table_main_results.tex").write_text("\n".join(lines) + "\n")

    seed_lines = []
    for family in ["ResNet18", "ResNet50", "FaceNet", "VGG16", "Qwen3-VL", "M3DFEL"]:
        s, v = summary[family], per_seed[family]
        seed_lines.append(f"{family} & {len(v)} & {three(s['ROI avg._mean'])} $\\pm$ "
                          f"{three(s['ROI avg._sd'])} & {three(min(v))}--{three(max(v))} & "
                          f"{three(lowest[family])} \\\\")
    (OUT / "table_seeds.tex").write_text("\n".join(seed_lines) + "\n")
    print("\n".join(lines + [""] + seed_lines))
    print(f"[ok] -> {OUT / 'table_main_results.tex'}, {OUT / 'table_seeds.tex'}")


if __name__ == "__main__":
    main()
