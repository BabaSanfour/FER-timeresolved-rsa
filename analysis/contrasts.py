#!/usr/bin/env python3
"""Planned tests of the comparisons between training regimes, in every scalp ROI and on the
whole scalp (--stimuli 18 | 54; --partial <control set> for the partial-RSA grids).

Four hypotheses, each pooled over the families it concerns:
  H1 pretraining raises alignment over random initialisation: generic-pretrained
     minus random (6 families)
  H2 face pretraining against ImageNet pretraining (4 image families)
  H3 emotion fine-tuning against its starting model (5 fine-tuned models)
  H4 the two pretrained video models (M3DFEL Kinetics-400, Qwen3-VL) against the 8
     pretrained image variants (ImageNet and face, each image family), 16 pairs
Each family's contrast (each video model's for H4) is reported too, and each variant on its
own (hypothesis "variant": its rho_max at its own peak, mean over its RDMs, against the same
null), which puts every variant on the scale of the comparisons.

The difference A - B is computed on pairs of single RDMs, so both sides of every
selection carry the same noise: run r of A with run r of B (a fine-tune starts from
pretraining run r), a released model with each of the ten RDMs of the other variant, or
two released models. In each scope x emotion cell (fear, happiness, neutral) of a pair,
layer and window are selected on the participant mean of (A + B) / 2 when A and B share
a layer axis (same family); across families each model is read at its own peak. Each
participant's difference is averaged over the three emotions, then over the pairs, then
(pooled tests) over the contrasts. The time course of H1 uses the same procedure with
the layer selected in each window.

With --normalize, every cell (scope, emotion, window) of every grid, observed and permuted, is
divided by the upper noise ceiling of the EEG in that cell before the selection (the EEG after the
controls with --partial), so values are fractions of the ceiling (Abdelhedi et al., 2026).

Selecting the best of many cells favours the model whose cells differ more from one
another, so a difference is not centred on zero when neither model relates to the EEG.
Each difference is therefore tested against its own null: the stimulus labels are
permuted within each emotion (1,000 times, the same permutations for every RDM) and the
whole computation, selection included, is repeated on the participant-mean grids
(linear in the participants' EEG). With --partial, the permuted model vectors are
residualised on the control set of each window, as in partial_rsa.py.

The scopes are the 11 ROIs, the whole scalp and the four regions (the mean over each region's
ROIs, frontal, central, centro-parietal, occipito-temporal, maxT across the four).
Reported per hypothesis, contrast and scope: the observed difference, the null mean, the
corrected difference (observed - null mean) with a 95% bootstrap CI over participants
shifted by the null mean, the two-sided permutation p, and p corrected across the 11 ROIs
by the maximum statistic over ROIs in the same permutations (Westfall-Young maxT). The
whole-scalp test of each pooled hypothesis is Holm-corrected across the four hypotheses.
The H1 time course is corrected across the 11 x 13 ROI-window cells (whole scalp: across
its 13 windows).

--sizes: the FaceNet size curve instead: each training-set size (and each step-matched
subset) against the random FaceNet and against the full pool, same procedure and output
columns (the full-pool rows repeat H1 FaceNet).

Output: analysis/paper_stats[_54]/contrasts[_<set>].csv, contrasts_time[_<set>].csv and
contrasts_participants[_<set>].csv (each participant's pooled difference per scope, minus the null mean)
(--sizes: contrasts_sizes[_<set>].csv)

--design ckpt | ids: the FaceNet learning designs (variants.CHECKPOINTS, IDENTITY_COUNTS), each
model against the random FaceNet and the full pool -> contrasts_ckpt[...].csv, contrasts_ids[...].csv
"""
import argparse
import csv

import numpy as np

import rsa_core
import variants
from partial_rsa import control_sets, nuisance_basis, ranked_centered, residual_normalise
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
H1 = "H1 pretrained vs random"


def hypotheses():
    """{hypothesis: [(label, A, B)]}: the contrasts each pooled test averages."""
    return {
        H1: [(f, (f, v), (f, "Random")) for f, v in GENERIC.items()],
        "H2 face vs ImageNet": [(f, (f, v), (f, "ImageNet")) for f, v in FACE.items()],
        "H3 fine-tuned vs start": [(f, (f, ft), (f, start)) for f, (ft, start) in FINE_TUNED.items()],
        "H4 video vs image": [(f"{' '.join(v)} - {' '.join(i)}", v, i) for v in VIDEO for i in IMAGE_PRETRAINED],
    }


def size_hypotheses():
    """{hypothesis: [(label, A, B)]} of the size curve, one contrast each."""
    out = {}
    for group in ("size", "step"):
        for size in variants.SIZES if group == "size" else variants.SIZES[:-1]:
            a = ("FaceNet", f"{group} {size}")
            out[f"{group} {size} vs random"] = [("FaceNet", a, ("FaceNet", "Random"))]
            out[f"{group} {size} vs full pool"] = [("FaceNet", a, ("FaceNet", "VGGFace2"))]
    return out


def design_hypotheses(group):
    """{hypothesis: [(label, A, B)]} of a FaceNet learning design (checkpoints along training,
    group ckpt; identity counts, group ids): each model against the random FaceNet and against
    the full pool."""
    levels = variants.CHECKPOINT_STEPS if group == "ckpt" else variants.IDENTITIES
    out = {}
    for level in levels:
        a = ("FaceNet", f"{group} {level}")
        out[f"{group} {level} vs random"] = [("FaceNet", a, ("FaceNet", "Random"))]
        out[f"{group} {level} vs full pool"] = [("FaceNet", a, ("FaceNet", "VGGFace2"))]
    return out


def pairs_of(ta, tb):
    return list(zip(ta, tb)) if len(ta) == len(tb) else [(x, y) for x in ta for y in tb]


def cell_values(grid, select, by_window=False):
    """grid (..., O, L, 3, W), select (O, L, 3, W) -> the grid at the selected cells, mean over
    the 3 emotions: (..., O), or (..., O, W) with the layer selected in each window."""
    O, L, E, W = select.shape
    filled = np.where(np.isnan(select), -np.inf, select)
    if by_window:
        li = filled.argmax(1)                                               # (O, E, W)
        o, e, w = np.meshgrid(np.arange(O), np.arange(E), np.arange(W), indexing="ij")
        return grid[..., o, li, e, w].mean(-2)
    flat = filled.transpose(0, 2, 1, 3).reshape(O, E, L * W).argmax(-1)    # (O, E)
    li, wi = np.unravel_index(flat, (L, W))
    o, e = np.meshgrid(np.arange(O), np.arange(E), indexing="ij")
    return grid[..., o, li, e, wi].mean(-1)


def pair_difference(ga, gb, same_family, by_window=False):
    """Grids (..., O, L, 3, W) of A and B (participants first, or participant means) -> A - B."""
    mean = (lambda g: np.nanmean(g, 0)) if ga.ndim == 5 else (lambda g: g)
    if same_family:
        sel = mean((ga + gb) / 2)
        return cell_values(ga, sel, by_window) - cell_values(gb, sel, by_window)
    return cell_values(ga, mean(ga), by_window) - cell_values(gb, mean(gb), by_window)


def contrast(grid, tags, a, b, by_window=False):
    """grid(tag) -> the difference A - B, mean over pairs."""
    return np.mean([pair_difference(grid(x), grid(y), a[0] == b[0], by_window)
                    for x, y in pairs_of(tags[a], tags[b])], axis=0)


def own_peak(g):
    """A grid (participants first, or the participant mean) read at its own participant-mean peak."""
    return cell_values(g, np.nanmean(g, 0) if g.ndim == 5 else g)


def all_contrasts(grid, tags, hyp):
    """-> {key: value}: every contrast and pooled test per scope, each variant at its own peak
    (keys ('variant', 'family variant')), and the H1 time course (keys ('time', label) and ('time', H1))."""
    out = {}
    for key in sorted({x for items in hyp.values() for _, a, b in items for x in (a, b)}):
        out["variant", " ".join(key)] = np.mean([own_peak(grid(t)) for t in tags[key]], axis=0)
    for h, items in hyp.items():
        vals = [contrast(grid, tags, a, b) for _, a, b in items]
        out[h] = np.mean(vals, axis=0)
        if h.startswith("H4"):
            for v in VIDEO:
                out[h, " ".join(v)] = np.mean([x for x, (_, a, _) in zip(vals, items) if a == v], axis=0)
        else:
            for (label, _, _), x in zip(items, vals):
                out[h, label] = x
    if H1 in hyp:
        time = [contrast(grid, tags, a, b, by_window=True) for _, a, b in hyp[H1]]
        out["time", H1] = np.mean(time, axis=0)
        for (label, _, _), x in zip(hyp[H1], time):
            out["time", label] = x
    return out


def upper_ceiling(z):
    """Participants' unit-norm (ranked, centred, residualised) EEG vectors (S, ..., P) -> the upper
    noise ceiling (...): each participant's correlation with the mean of all, averaged (Nili et al., 2014)."""
    total = np.nansum(z, 0)
    unit = total / np.clip(np.linalg.norm(total, axis=-1, keepdims=True), 1e-12, None)
    return np.nanmean(np.sum(z * unit[None], -1), 0)


def permutation_nulls(s, tags, hyp, W, partial, normalize=False):
    """{key: (N_PERM, O[, W])} of all_contrasts under within-emotion label permutation, and the
    upper noise ceilings (O, 3, W) of the EEG (after the controls with --partial). With normalize,
    every cell is divided by its ceiling before the selection, as for the observed grids."""
    eeg, _, _ = rsa_core.load_eeg(s)
    O = eeg.shape[1]
    rng = np.random.default_rng(SEED)
    members = control_sets(s)[partial] if partial else []
    controls = {c: rsa_core.load_model_rdm(s, f"control_{c}")[0, :W] for c in members}
    eeg_mean, perms, bases, ceiling = {}, {}, {}, {}
    for e in EMOTIONS:
        idx = rsa_core.EMOTIONS[s][e]
        if partial:
            bases[e] = [nuisance_basis([upper(controls[c][k][idx][:, idx]) for c in members]) for k in range(W)]
            ev = ranked_centered(upper(eeg[..., idx, :][..., idx]))[:, :, :W]            # (S, O, W, P)
            z = np.stack([residual_normalise(ev[:, :, k], bases[e][k])[0] for k in range(W)], 2)
        else:
            z = rank_normalise(upper(eeg[..., idx, :][..., idx]))[0][:, :, :W]            # (S, O, W, P)
        eeg_mean[e] = z.mean(0)
        ceiling[e] = upper_ceiling(z)                                                    # (O, W)
        perms[e] = [pair_permutation(rng.permutation(len(idx))) for _ in range(N_PERM)]
    del eeg
    needed = sorted({t for items in hyp.values() for _, a, b in items for t in tags[a] + tags[b]})
    model = {}
    for tag in needed:
        m = rsa_core.load_model_rdm(s, tag)[:, :W]
        model[tag] = [(ranked_centered if partial else lambda v: rank_normalise(v)[0])(
            upper(m[..., rsa_core.EMOTIONS[s][e], :][..., rsa_core.EMOTIONS[s][e]])) for e in EMOTIONS]
    null = None
    for k in range(N_PERM):
        grids = {}
        for tag in needed:
            g = []
            for ei, e in enumerate(EMOTIONS):
                mv = model[tag][ei][..., perms[e][k]]                                  # (L, W, P)
                if partial:
                    x = np.empty((O, mv.shape[0], W))
                    for w in range(W):
                        mz, bad = residual_normalise(mv[:, w], bases[e][w])
                        x[:, :, w] = eeg_mean[e][:, w] @ mz.T
                        x[:, bad, w] = np.nan
                else:
                    flat = np.linalg.norm(mv, axis=-1) < 1e-12
                    x = np.einsum("owp,lwp->olw", eeg_mean[e], mv, optimize=True)
                    x[:, flat] = np.nan
                g.append(x / ceiling[e][:, None, :] if normalize else x)
            grids[tag] = np.stack(g, 2)                                                # (O, L, 3, W)
        vals = all_contrasts(grids.get, tags, hyp)
        if null is None:
            null = {key: np.zeros((N_PERM,) + v.shape) for key, v in vals.items()}
        for key, v in vals.items():
            null[key][k] = v
        if (k + 1) % 100 == 0:
            print(f"  [null] {k + 1}/{N_PERM} permutations", flush=True)
    return null, np.stack([ceiling[e] for e in EMOTIONS], 1)                              # (O, 3, W)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(run, 1.)
    return out


def tests(obs, null, rng):
    """obs (S, *shape), null (N, *shape) -> per-cell dict of arrays; maxT over the ROI cells
    (the first N_ROI scopes), separately over the whole-scalp cells, and over the region means."""
    centre, spread = null.mean(0), null.std(0)
    z = (obs.mean(0) - centre) / spread
    zn = np.abs((null - centre) / spread)
    p = (1 + (zn >= np.abs(z)).sum(0)) / (N_PERM + 1)
    p_max = np.empty_like(p)
    for part in (slice(0, rsa_core.N_ROI), slice(rsa_core.N_ROI, rsa_core.N_ROI + 1), slice(rsa_core.N_ROI + 1, None)):
        m = zn[:, part].reshape(N_PERM, -1).max(1)
        p_max[part] = (1 + (m[:, None] >= np.abs(z[part]).ravel()[None]).sum(0)).reshape(z[part].shape) / (N_PERM + 1)
    boot = obs[rng.integers(0, len(obs), size=(N_BOOT, len(obs)))].mean(1)
    return {"observed": obs.mean(0), "null_mean": centre, "corrected": obs.mean(0) - centre,
            "low": np.percentile(boot, 2.5, axis=0) - centre, "high": np.percentile(boot, 97.5, axis=0) - centre,
            "n_above": (obs > centre).sum(0), "p": p, "p_maxT": p_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--partial", default=None, help="a control set of partial_rsa.py (e.g. joint)")
    ap.add_argument("--sizes", action="store_true", help="the FaceNet size curve instead of H1-H4")
    ap.add_argument("--design", choices=["ckpt", "ids"], default=None,
                    help="a FaceNet learning design instead: checkpoints along training or identity counts")
    ap.add_argument("--n-perm", type=int, default=N_PERM, help="permutations (default 1,000; fewer for a quick check)")
    ap.add_argument("--normalize", action="store_true",
                    help="divide every cell by its upper noise ceiling before the selection (outputs *_nc)")
    args = ap.parse_args()
    globals()["N_PERM"] = args.n_perm
    s = args.stimuli
    tags = {(f, v if g == "main" else f"{g} {v}"): t for g, f, v, t in variants.variants(s) if g in ("main", "size", "step", "ckpt", "ids")}
    cache, scale = {}, {}

    def grid(tag):
        if tag not in cache:
            g = np.load(rsa_core.grid_path(s, tag, args.partial))["rsa"].astype(np.float64)[:, :, :, EMOTION_CELLS, :]
            cache[tag] = g / scale["ceiling"][None, :, None, :, :g.shape[-1]] if "ceiling" in scale else g
        return cache[tag]

    hyp = design_hypotheses(args.design) if args.design else size_hypotheses() if args.sizes else hypotheses()
    W = grid(tags["ResNet18", "Random"][0]).shape[-1]
    scopes = rsa_core.scopes()
    bounds = rsa_core.window_bounds()[:W]
    null, ceiling = permutation_nulls(s, tags, hyp, W, args.partial, args.normalize)
    if args.normalize:
        scale["ceiling"] = ceiling; cache.clear()
    obs = all_contrasts(grid, tags, hyp)
    # region means (the mean over a region's ROIs), tested against the same permutations
    regions = list(dict.fromkeys(sc.split("-")[0] for sc in scopes[:rsa_core.N_ROI]))
    members = [[i for i, sc in enumerate(scopes[:rsa_core.N_ROI]) if sc.startswith(rg + "-")] for rg in regions]
    with_regions = lambda a: np.concatenate([a] + [a[:, m].mean(1, keepdims=True) for m in members], axis=1)
    obs = {k: with_regions(v) for k, v in obs.items()}
    null = {k: with_regions(v) for k, v in null.items()}
    scopes = list(scopes) + regions
    rng = np.random.default_rng(SEED)
    n_pairs = {h: len(items) for h, items in hyp.items()}
    rows, time_rows, participant_rows = [], [], []
    for key, value in obs.items():
        r = tests(value, null[key], rng)
        if key[0] == "time":
            if key[1] == H1:
                centre = null[key].mean(0)
                for oi, sc in enumerate(scopes):
                    for wi in range(W):
                        for pi, v in enumerate(value[:, oi, wi] - centre[oi, wi]):
                            participant_rows.append({"hypothesis": "time " + H1, "scope": sc, "window": wi,
                                                     "participant": pi, "difference": round(float(v), 5)})
            for oi, sc in enumerate(scopes):
                for wi in range(W):
                    time_rows.append({"contrast": "pooled" if key[1] == H1 else key[1], "scope": sc,
                                      "window_start_ms": bounds[wi, 0], "window_end_ms": bounds[wi, 1],
                                      **{k: round(float(v[oi, wi]), 5) if k not in ("p", "p_maxT", "n_above") else
                                         (int(v[oi, wi]) if k == "n_above" else float(v[oi, wi])) for k, v in r.items()}})
            continue
        h, label = (key, "pooled") if isinstance(key, str) else key
        if label == "pooled" or h == "variant":
            centre = null[key].mean(0)
            for oi, sc in enumerate(scopes):
                for pi, v in enumerate(value[:, oi] - centre[oi]):
                    participant_rows.append({"hypothesis": h if h != "variant" else f"variant {label}", "scope": sc,
                                             "participant": pi, "difference": round(float(v), 5)})
        for oi, sc in enumerate(scopes):
            rows.append({"hypothesis": h, "contrast": label, "n_pairs": n_pairs[h] if label == "pooled" else "",
                         "scope": sc, "n": len(value),
                         **{k: round(float(v[oi]), 5) if k not in ("p", "p_maxT", "n_above") else
                            (int(v[oi]) if k == "n_above" else float(v[oi])) for k, v in r.items()}})
    whole = [r for r in rows if r["contrast"] == "pooled" and r["scope"] == "wholescalp"]
    for r, ph in zip(whole, holm([r["p"] for r in whole])):     # sizes: Holm across all size contrasts
        r["p_holm"] = float(ph)
    suffix = (f"_{args.design}" if args.design else "_sizes" if args.sizes else "") + \
        (f"_{args.partial}" if args.partial else "") + ("_nc" if args.normalize else "")
    out = rsa_core.ROOT / ("analysis/paper_stats" if s == "18" else "analysis/paper_stats_54")
    out.mkdir(parents=True, exist_ok=True)
    for name, table in ((f"contrasts{suffix}.csv", rows), (f"contrasts_time{suffix}.csv", time_rows),
                        (f"contrasts_participants{suffix}.csv", participant_rows)):
        if not table:
            continue
        fields = list(dict.fromkeys(k for r in table for k in r))
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(table)
    for r in rows:
        if r["contrast"] == "pooled" or r["scope"] == "wholescalp":
            extra = f"  Holm={r['p_holm']:.4f}" if "p_holm" in r else ""
            print(f"{r['hypothesis']:24s} {r['contrast']:22s} {r['scope']:19s} corrected {r['corrected']:+.4f} "
                  f"[{r['low']:+.4f}, {r['high']:+.4f}] p={r['p']:.4f} maxT={r['p_maxT']:.4f}{extra}")
    print(f"[ok] -> {out / ('contrasts' + suffix + '.csv')}, contrasts_time{suffix}.csv")


if __name__ == "__main__":
    main()
