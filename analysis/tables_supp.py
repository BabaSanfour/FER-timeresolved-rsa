#!/usr/bin/env python3
import argparse
import collections
import csv
from pathlib import Path

import numpy as np

import rsa_core
import variants

STATS = rsa_core.STATS_DIRS["54"]
CLUSTERS = rsa_core.ROOT / "analysis"
SCOPES = ["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "CentroParietal-L", "CentroParietal-M",
          "CentroParietal-R", "OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R", "wholescalp"]
ORDER = ["VGG16", "ResNet18", "ResNet50", "FaceNet", "M3DFEL", "Qwen3-VL"]
LABEL = {"AffectNet FT": "AffectNet", "DFEW FT": "DFEW", "Pretrained": "pretrained"}
RDM_OF = {tag: (family, variant) for family, rows in variants.MAIN.items() for variant, tags in rows for tag in tags}


def point(v):
    """.25 or $-$.25."""
    s = f"{abs(v):.2f}".replace("0.", ".", 1)
    return s if v >= 0 else "$-$" + s


def variant_rows(name):
    rows = list(csv.DictReader((STATS / f"contrasts{name}_nc.csv").open()))
    mean = {(r["contrast"], r["scope"]): r for r in rows if r["hypothesis"] == "variant"}
    per_rdm = collections.defaultdict(list)
    for r in rows:
        if r["hypothesis"] == "rdm" and r["contrast"] in RDM_OF:
            per_rdm[RDM_OF[r["contrast"]], r["scope"]].append(float(r["observed"]))
    out = []
    for family in ORDER:
        for i, (variant, _) in enumerate(variants.MAIN[family]):
            cells = []
            for scope in SCOPES:
                r = mean[f"{family} {variant}", scope]
                p = float(r["p"] or 1) if scope == "wholescalp" else float(r["p_maxT"] or 1)
                runs = per_rdm[(family, variant), scope]
                sd = r"\,$\pm$\," + f"{np.std(runs, ddof=1):.2f}".lstrip("0") if len(runs) > 1 else ""
                cells.append(point(float(r["observed"])) + sd + (r"$^*$" if p < .05 else ""))
            n = len(per_rdm[(family, variant), "wholescalp"])
            label = LABEL.get(variant, variant) + (f" ({n})" if n > 1 else "")
            out.append(f"{family if i == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    return out[:-1]


def cluster_rows():
    def load(folder):
        with (CLUSTERS / folder / "model_survivors.csv").open() as fh:
            return {(r["family"], r["variant"]): r for r in csv.DictReader(fh)}
    raw, joint, v18 = (load(f) for f in ("cluster_permutation_54", "cluster_permutation_54_partial_joint",
                                         "cluster_permutation_18"))
    p = lambda r: f"{float(r['best_study_p']):.3f}".replace("0.", ".", 1)
    out = []
    for family in ORDER:
        for i, (variant, _) in enumerate(variants.MAIN[family]):
            k = (family, variant)
            out.append(f"{family if i == 0 else ''} & {LABEL.get(variant, variant)} & {raw[k]['n_study_survivors']} & "
                       f"{p(raw[k])} & {joint[k]['n_study_survivors']} & {p(joint[k])} & {v18[k]['n_study_survivors']} \\\\")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="folder for the .tex rows (the paper's tables/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("", "_joint"):
        (args.out / f"table_variants{name}_nc.tex").write_text("\n".join(variant_rows(name)) + "\n")
    (args.out / "table_clusters.tex").write_text("\n".join(cluster_rows()) + "\n")
    print(f"[ok] -> {args.out}/table_variants{{,_joint}}_nc.tex, table_clusters.tex")


if __name__ == "__main__":
    main()
