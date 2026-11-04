#!/usr/bin/env python3
"""Planned tests of the paper's comparisons between training regimes (--stimuli 18 | 54).

Four primary hypotheses, one test each, Holm across the four:
  H1 pretraining raises alignment over random initialisation: generic-pretrained
     minus random, pooled over the 6 families
  H2 face pretraining adds to generic pretraining: face minus ImageNet, pooled over
     the 4 image families
  H3 emotion fine-tuning adds nothing consistent: fine-tuned minus its starting
     model, pooled over the 5 fine-tuned models
  H4 video input does not beat image models: the two pretrained video models
     (M3DFEL Kinetics-400, Qwen3-VL) minus the 8 pretrained image variants
     (ImageNet and face, each image family), pooled over the 16 pairs
Secondary: each family's (or video model's) contrast alone, uncorrected.

The difference of a contrast A - B is computed on pairs of single RDMs, so both
sides of every selection carry the same noise: training run r of A with run r of
B (a fine-tune starts from pretraining run r), a released model with each of the
ten RDMs of the other variant, or two released models. In every ROI x emotion
cell (6 ROIs x fear, happiness, neutral) of a pair, layer and window are selected
on the participant mean of (A + B) / 2 when A and B share a layer axis (same
family); across families each model is read at its own peak. Each participant's
difference is averaged over the 18 cells, then over the pairs, then (pooled tests)
over the contrasts.

Selecting the best of many layer x window cells favours the model whose cells
differ more from one another, so the difference is not centred on zero when
neither model relates to the EEG. Each contrast is therefore tested against its
own null: the stimulus labels are permuted within each emotion (1,000 times, the
same permutations for every RDM) and the whole contrast, selection included, is
recomputed on the participant-mean grids (linear in the participants' EEG).
Reported: the observed difference, the null mean, the corrected difference
(observed - null mean) with a 95% bootstrap CI over participants shifted by the
null mean, and the two-sided permutation p around the null mean.

Output: analysis/paper_stats[_54]/contrasts.csv
"""
import argparse
import csv

import numpy as np

import rsa_core
import variants
from rho_max_null import EMOTIONS, pair_permutation
from rsa_core import rank_normalise, upper

N_PERM = 1000
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


def pairs_of(ta, tb):
    return list(zip(ta, tb)) if len(ta) == len(tb) else [(x, y) for x in ta for y in tb]


def cell_values(grid, select):
    """grid (..., O, L, E, W) with E = the 3 emotions; select (O, L, E, W) participant mean that
    picks the cells -> (...) value at the picked cells, mean over the 18 ROI x emotion cells."""
    vals = []
    for oi in range(rsa_core.N_ROI):
        for ei in range(select.shape[2]):
            block = select[oi, :, ei, :]
            li, wi = np.unravel_index(np.nanargmax(block), block.shape)
            vals.append(grid[..., oi, li, ei, wi])
    return np.mean(vals, axis=0)


def pair_difference(ga, gb, same_family):
    """Grids (..., 6, L, 3, W) of A and B -> (...) A - B at the selected cells."""
    if same_family:
        sel = np.nanmean((ga + gb) / 2, 0) if ga.ndim == 5 else (ga + gb) / 2
        return cell_values(ga, sel) - cell_values(gb, sel)
    sa = np.nanmean(ga, 0) if ga.ndim == 5 else ga
    sb = np.nanmean(gb, 0) if gb.ndim == 5 else gb
    return cell_values(ga, sa) - cell_values(gb, sb)


def contrast(grid, tags, a, b):
    """grid(tag) -> participant (S,) or participant-mean () difference A - B, mean over pairs."""
    return np.mean([pair_difference(grid(x), grid(y), a[0] == b[0]) for x, y in pairs_of(tags[a], tags[b])], axis=0)


def null_contrasts(s, tags, hyp, W):
    """{(hypothesis, label) or hypothesis: (N_PERM,)} contrasts under within-emotion label permutation."""
    eeg, _, _ = rsa_core.load_eeg(s)
    rng = np.random.default_rng(SEED)
    eeg_mean, perms = {}, {}
    for e in EMOTIONS:
        idx = rsa_core.EMOTIONS[s][e]
        eeg_mean[e] = rank_normalise(upper(eeg[..., idx, :][..., idx]))[0].mean(0)[:rsa_core.N_ROI]   # (6, W, P)
        perms[e] = [pair_permutation(rng.permutation(len(idx))) for _ in range(N_PERM)]
    del eeg
    needed = sorted({t for items in hyp.values() for _, a, b in items for t in tags[a] + tags[b]})
    ranked = {}
    for tag in needed:
        model = rsa_core.load_model_rdm(s, tag)[:, :W]
        ranked[tag] = [rank_normalise(upper(model[..., rsa_core.EMOTIONS[s][e], :][..., rsa_core.EMOTIONS[s][e]]))
                       for e in EMOTIONS]
    out = {key: np.zeros(N_PERM) for h, items in hyp.items() for key in [h] + [(h, lab) for lab, _, _ in items]}
    for k in range(N_PERM):
        grids = {}
        for tag in needed:
            g = []
            for ei, e in enumerate(EMOTIONS):
                mz, flat = ranked[tag][ei]
                m = np.einsum("owp,lwp->olw", eeg_mean[e], mz[..., perms[e][k]], optimize=True)
                m[:, flat] = np.nan
                g.append(m)
            grids[tag] = np.stack(g, 2)                                   # (6, L, 3, W)
        for h, items in hyp.items():
            vals = [contrast(grids.get, tags, a, b) for _, a, b in items]
            for (lab, _, _), v in zip(items, vals):
                out[h, lab][k] = v
            out[h][k] = np.mean(vals)
        if (k + 1) % 100 == 0:
            print(f"  [null] {k + 1}/{N_PERM} permutations", flush=True)
    return out


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
    args = ap.parse_args()
    s = args.stimuli
    tags = {(f, v): t for g, f, v, t in variants.variants(s) if g == "main"}
    cache = {}

    def grid(tag):
        if tag not in cache:
            cache[tag] = np.load(rsa_core.grid_path(s, tag))["rsa"].astype(np.float64)[:, :, :, EMOTION_CELLS, :]
        return cache[tag]

    hyp = hypotheses()
    W = grid(tags["ResNet18", "Random"][0]).shape[-1]
    null = null_contrasts(s, tags, hyp, W)
    rng = np.random.default_rng(SEED)

    def row(level, h, label, a, b, d, n_pairs, nd):
        centre = float(nd.mean())
        lo, hi = bootstrap_ci(d, rng)
        obs = float(d.mean())
        return {"level": level, "hypothesis": h, "contrast": label, "a": a, "b": b, "n_pairs": n_pairs,
                "n": len(d), "mean_difference": round(obs, 5), "null_mean": round(centre, 5),
                "corrected_difference": round(obs - centre, 5), "ci95_low": round(lo - centre, 5),
                "ci95_high": round(hi - centre, 5), "n_above_null": int((d > centre).sum()),
                "p": float((1 + np.sum(np.abs(nd - centre) >= abs(obs - centre))) / (N_PERM + 1))}

    primary, secondary = [], []
    for h, items in hyp.items():
        ds = []
        for label, a, b in items:
            d = contrast(grid, tags, a, b)
            ds.append(d)
            if not h.startswith("H4"):
                secondary.append(row("secondary", h, label, " ".join(a), " ".join(b), d,
                                     len(pairs_of(tags[a], tags[b])), null[h, label]))
        if h.startswith("H4"):                   # secondary: each video model against the 8 image variants
            for v in VIDEO:
                keep = [i for i, (_, a, _) in enumerate(items) if a == v]
                secondary.append(row("secondary", h, " ".join(v), " ".join(v), "8 pretrained image variants",
                                     np.mean([ds[i] for i in keep], axis=0), len(IMAGE_PRETRAINED),
                                     np.mean([null[h, items[i][0]] for i in keep], axis=0)))
        primary.append(row("primary", h, f"pooled over {len(items)}", "", "", np.mean(ds, axis=0), len(items), null[h]))
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
        print(f"{r['level']:9s} {r['hypothesis']:24s} {r['contrast']:44s} observed {r['mean_difference']:+.4f} "
              f"null {r['null_mean']:+.4f} corrected {r['corrected_difference']:+.4f} "
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] p={r['p']:.4f}{extra}")
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
