#!/usr/bin/env python3
"""Planned tests of the paper's comparisons between training regimes (--stimuli 18 | 54).

Four primary hypotheses, one test each, Holm across the four:
  H1 pretraining raises alignment over random initialisation: generic-pretrained
     minus random, pooled over the 6 families
  H2 face pretraining adds to generic pretraining: face minus ImageNet, pooled over
     the 4 image families
  H3 emotion fine-tuning adds nothing consistent: fine-tuned minus its starting
     model, pooled over the 5 fine-tuned models (95% CI; equivalence test with
     --sesoi)
  H4 video input does not beat image models: the two pretrained video models
     (M3DFEL Kinetics-400, Qwen3-VL) minus the 8 pretrained image variants
     (ImageNet and face, each image family), pooled over the 16 pairs
Secondary: each family's (or video model's) contrast alone, with p and 95% CI,
uncorrected.

The difference of a contrast A - B is computed on pairs of single RDMs, so both
sides of every selection carry the same noise: training run r of A with run r of
B (a fine-tune starts from pretraining run r), a released model with each of the
ten RDMs of the other variant, or two released models. In every ROI x emotion
cell (6 ROIs x fear, happiness, neutral) of a pair, layer and window are selected
on the participant mean of (A + B) / 2 when A and B share a layer axis (same
family), an unbiased selection for the difference; across families each model is
read at its own peak. Each participant's difference is averaged over the 18 cells,
then over the pairs, then (pooled tests) over the contrasts. Tests: two-sided
sign-flip (20,000 flips) on the participants' differences; 95% CI by bootstrap
over participants (10,000 resamples).

Output: analysis/paper_stats[_54]/contrasts.csv
"""
import argparse
import csv

import numpy as np

import rsa_core
import variants

N_FLIP = 20000
N_BOOT = 10000
SEED = 42
EMOTION_CELLS = (1, 2, 3)       # fear, happy, neutral on the grid's emotion axis (all, fear, happy, neutral)
GENERIC = {"ResNet18": "ImageNet", "ResNet50": "ImageNet", "FaceNet": "ImageNet", "VGG16": "ImageNet",
           "M3DFEL": "Kinetics-400", "Qwen3-VL": "Pretrained"}
FACE = {"ResNet18": "VGGFace2", "ResNet50": "VGGFace2", "FaceNet": "VGGFace2", "VGG16": "VGG-Face"}
FINE_TUNED = {"ResNet18": ("AffectNet FT", "VGGFace2"), "ResNet50": ("AffectNet FT", "VGGFace2"),
              "FaceNet": ("AffectNet FT", "VGGFace2"), "VGG16": ("AffectNet FT", "VGG-Face"),
              "M3DFEL": ("DFEW FT", "Kinetics-400")}
VIDEO = [("M3DFEL", "Kinetics-400"), ("Qwen3-VL", "Pretrained")]
IMAGE_PRETRAINED = [(f, v) for f in FACE for v in (GENERIC[f], FACE[f])]


def hypotheses():
    """{hypothesis: [(label, A, B)]}: the contrasts each primary test pools."""
    return {
        "H1 pretrained vs random": [(f, (f, v), (f, "Random")) for f, v in GENERIC.items()],
        "H2 face vs ImageNet": [(f, (f, v), (f, "ImageNet")) for f, v in FACE.items()],
        "H3 fine-tuned vs start": [(f, (f, ft), (f, start)) for f, (ft, start) in FINE_TUNED.items()],
        "H4 video vs image": [(f"{' '.join(v)} - {' '.join(i)}", v, i) for v in VIDEO for i in IMAGE_PRETRAINED],
    }


def cell_values(grid, select):
    """grid (S, O, L, E, W); select (O, L, E, W) participant mean that picks the cells
    -> (S,) each participant's value at the picked cells, mean over the 18 ROI x emotion cells."""
    vals = []
    for oi in range(rsa_core.N_ROI):
        for ei in EMOTION_CELLS:
            block = select[oi, :, ei, :]
            li, wi = np.unravel_index(np.nanargmax(block), block.shape)
            vals.append(grid[:, oi, li, ei, wi])
    return np.mean(vals, axis=0)


def sign_flip_p(d, rng, shift=0.0, side="two"):
    """Sign-flip p for the mean of d - shift: two-sided, or one-sided 'greater' / 'less'."""
    x = d - shift
    flips = (rng.choice(np.array([-1., 1.]), size=(N_FLIP, len(x))) * x).mean(1)
    obs = x.mean()
    hits = {"two": np.abs(flips) >= abs(obs), "greater": flips >= obs, "less": flips <= obs}[side]
    return float((1 + np.sum(hits)) / (N_FLIP + 1))


def bootstrap_ci(d, rng):
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(run, 1.)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--sesoi", type=float, default=None,
                    help="smallest effect of interest (rho): adds an equivalence test (TOST) for H3")
    args = ap.parse_args()
    s = args.stimuli
    tags = {(f, v): t for g, f, v, t in variants.variants(s) if g == "main"}
    cache = {}

    def grid(tag):
        if tag not in cache:
            cache[tag] = np.load(rsa_core.grid_path(s, tag))["rsa"].astype(np.float64)
        return cache[tag]

    def difference(a, b):
        """(S,) participant differences A - B, averaged over the equal-noise pairs."""
        ta, tb = tags[a], tags[b]
        pairs = list(zip(ta, tb)) if len(ta) == len(tb) else [(x, y) for x in ta for y in tb]
        diffs = []
        for x, y in pairs:
            ga, gb = grid(x), grid(y)
            if a[0] == b[0]:                       # same layers: cells chosen on the pair mean
                sel = np.nanmean((ga + gb) / 2, 0)
                diffs.append(cell_values(ga, sel) - cell_values(gb, sel))
            else:                                  # different layers: each at its own peak
                diffs.append(cell_values(ga, np.nanmean(ga, 0)) - cell_values(gb, np.nanmean(gb, 0)))
        return np.mean(diffs, axis=0), len(pairs)

    rng = np.random.default_rng(SEED)

    def row(level, hyp, label, a, b, d, n_pairs):
        lo, hi = bootstrap_ci(d, rng)
        return {"level": level, "hypothesis": hyp, "contrast": label, "a": a, "b": b, "n_pairs": n_pairs,
                "n": len(d), "mean_difference": round(float(d.mean()), 5), "ci95_low": round(lo, 5),
                "ci95_high": round(hi, 5), "n_positive": int((d > 0).sum()), "p": sign_flip_p(d, rng)}

    primary, secondary = [], []
    for hyp, items in hypotheses().items():
        ds = []
        for label, a, b in items:
            d, n_pairs = difference(a, b)
            ds.append(d)
            if not hyp.startswith("H4"):
                secondary.append(row("secondary", hyp, label, " ".join(a), " ".join(b), d, n_pairs))
        if hyp.startswith("H4"):                   # secondary: each video model against the 8 image variants
            for v in VIDEO:
                dv = np.mean([d for (label, a, b), d in zip(items, ds) if a == v], axis=0)
                secondary.append(row("secondary", hyp, " ".join(v), " ".join(v), "8 pretrained image variants",
                                     dv, len(IMAGE_PRETRAINED)))
        r = row("primary", hyp, f"pooled over {len(items)}", "", "", np.mean(ds, axis=0), len(items))
        if hyp.startswith("H3") and args.sesoi is not None:
            d = np.mean(ds, axis=0)
            r["p_equivalence"] = max(sign_flip_p(d, rng, -args.sesoi, "greater"),
                                     sign_flip_p(d, rng, args.sesoi, "less"))
            r["sesoi"] = args.sesoi
        primary.append(r)
    for r, ph in zip(primary, holm([r["p"] for r in primary])):
        r["p_holm"] = float(ph)
    rows = primary + secondary
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = rsa_core.ROOT / ("analysis/paper_stats" if s == "18" else "analysis/paper_stats_54") / "contrasts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        extra = f"  Holm={r['p_holm']:.4f}" if "p_holm" in r else ""
        extra += f"  p_equiv={r['p_equivalence']:.4f}" if "p_equivalence" in r else ""
        print(f"{r['level']:9s} {r['hypothesis']:24s} {r['contrast']:44s} {r['mean_difference']:+.4f} "
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] (+{r['n_positive']}/{r['n']}) p={r['p']:.4f}{extra}")
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
