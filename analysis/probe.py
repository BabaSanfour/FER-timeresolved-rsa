#!/usr/bin/env python3
"""Linear probe of the emotion label on each model's final-layer features (--stimuli 18 | 54).

Features: features/set<stimuli>/<name>/<name>.features.npy (n_stim, 16, d), written by
the extractors with --features: the input to the output layer for the CNNs, the
last backbone layer for M3DFEL, the last decoder block for Qwen3-VL. For each RDM,
a logistic regression (L2, C = 1, features standardised on the training set) is
trained on the frames of all videos except one actor's and predicts the held-out
actor's videos by averaging its class probabilities over their 16 frames (leave
one actor out, 6 folds). Accuracy is over all videos; a variant's value is the
mean +/- SD over its RDMs. Its p comes from 200 permutations of the emotion
labels within each actor (the same for every RDM of a variant), which keep every
training fold balanced, so the null is centred on chance (1/3); a permutation over
all videos would unbalance the folds and push the null below chance. The FaceNet
size-curve and step-matched models are probed too (accuracy only, no null).

Does task performance follow EEG alignment? Alignment is each RDM's rho_max (ROI and
emotion average, maxr_table.py), or rho_max above its chance level (rho_max_null.py)
when variants of different architectures are compared. Each test is a linear mixed
model on standardised variables, alignment ~ accuracy with a random intercept per group
(statsmodels MixedLM, Wald p for the slope), with a permutation test of the Spearman
correlation as a check (accuracy permuted across units, or within groups after
centring both variables on the group mean):
  size curve   the 70 FaceNets trained on 10k to 500k images and on the full pool (one
               architecture, graded training), accuracy = emotion probe or face-identity
               validation accuracy (identity_accuracy.py); groups = training run (the
               seven sizes of a run share its seed); also within size (groups = size)
  runs         the ten seeds or training runs of each variant (same architecture and
               recipe); groups = variant
  variants     across the 21 variants (and the 15 trained ones), on rho_max above chance:
               descriptive, since the variants differ in architecture and training at once

Outputs: analysis/paper_stats[_54]/probe.csv (per variant), probe_per_rdm.csv,
probe_vs_alignment.csv
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rsa_core
import variants

N_NULL = 200
N_PERM_RELATION = 10000
SEED = 42


def accuracy(features, emotion, actor):
    """features (n, T, d) -> leave-one-actor-out accuracy over the n videos."""
    n, T, d = features.shape
    correct = np.zeros(n, bool)
    for a in np.unique(actor):
        train, test = actor != a, actor == a
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(features[train].reshape(-1, d), np.repeat(emotion[train], T))
        proba = clf.predict_proba(features[test].reshape(-1, d)).reshape(test.sum(), T, -1).mean(1)
        correct[test] = clf.classes_[proba.argmax(1)] == emotion[test]
    return float(correct.mean())


def rdm_accuracies(job):
    """(features path, emotion, actor, label permutations) -> (observed, null accuracies)."""
    path, emotion, actor, perms = job
    f = np.load(path).astype(np.float64)
    if f.shape[0] != len(emotion):
        raise SystemExit(f"ABORT {path}: {f.shape[0]} videos, expected {len(emotion)}")
    return accuracy(f, emotion, actor), np.array([accuracy(f, emotion[p], actor) for p in perms])


def within_actor_permutations(actor, rng, n):
    perms = []
    for _ in range(n):
        p = np.arange(len(actor))
        for a in np.unique(actor):
            i = np.flatnonzero(actor == a)
            p[i] = rng.permutation(i)
        perms.append(p)
    return perms


def mixed(acc, align, groups):
    """Standardised slope of alignment ~ accuracy with a random intercept per group, and its Wald p."""
    import pandas as pd
    import statsmodels.formula.api as smf
    z = lambda v: (np.asarray(v, float) - np.mean(v)) / np.std(v)
    d = pd.DataFrame({"y": z(align), "x": z(acc), "g": np.asarray(groups)})
    fit = smf.mixedlm("y ~ x", d, groups=d["g"]).fit(reml=True)
    return float(fit.params["x"]), float(fit.pvalues["x"])


def relation(acc, align, groups=None, rng=None):
    """Spearman r between accuracy and alignment, with a permutation p; with groups, both are
    centred within groups and accuracy is permuted within groups."""
    acc, align = np.asarray(acc, float), np.asarray(align, float)
    if groups is not None:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            i = groups == g
            acc[i] -= acc[i].mean(); align[i] -= align[i].mean()
    r = spearmanr(acc, align)[0]
    null = np.empty(N_PERM_RELATION)
    for k in range(N_PERM_RELATION):
        a = acc.copy()
        if groups is None:
            a = rng.permutation(a)
        else:
            for g in np.unique(groups):
                i = np.flatnonzero(groups == g); a[i] = a[rng.permutation(i)]
        null[k] = spearmanr(a, align)[0]
    return float(r), float((1 + np.sum(np.abs(null) >= abs(r))) / (N_PERM_RELATION + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    args = ap.parse_args()
    s = args.stimuli
    names = rsa_core.MODEL_STIMULI[s]                       # all videos: the probe uses no EEG
    emotion = np.array([nm.split("_")[0] for nm in names])
    actor = np.array([nm.rsplit("_Actor", 1)[1] for nm in names])
    folder = rsa_core.ROOT / f"features/set{s}"
    out = rsa_core.ROOT / ("analysis/paper_stats" if s == "18" else "analysis/paper_stats_54")
    out.mkdir(parents=True, exist_ok=True)

    perms = within_actor_permutations(actor, np.random.default_rng(SEED), N_NULL)
    main_rows = [(f, v, t) for g, f, v, t in variants.variants(s) if g == "main"]
    size_rows = [(g, f, v, t) for g, f, v, t in variants.variants(s) if g in ("size", "step")
                 and all((folder / tag / f"{tag}.features.npy").exists() for tag in t)]
    jobs = [(folder / tag / f"{tag}.features.npy", emotion, actor, perms) for *_, tags in main_rows for tag in tags]
    jobs += [(folder / tag / f"{tag}.features.npy", emotion, actor, []) for *_, tags in size_rows for tag in tags]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = iter(list(pool.map(rdm_accuracies, jobs)))
    rows, per_rdm = [], []
    for family, variant, tags in main_rows:
        res = [next(results) for _ in tags]
        a = np.array([r[0] for r in res]); null = np.mean([r[1] for r in res], axis=0)
        for tag, r in zip(tags, res):
            per_rdm.append({"group": "main", "family": family, "variant": variant, "rdm": tag, "accuracy": round(r[0], 4)})
        rows.append({"family": family, "variant": variant, "n": len(tags), "accuracy": round(float(a.mean()), 4),
                     "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else "",
                     "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                     "null_mean": round(float(null.mean()), 4), "null_95": round(float(np.percentile(null, 95)), 4),
                     "p": float((1 + np.sum(null >= a.mean())) / (N_NULL + 1))})
        print(f"  {family:9s} {variant:13s} n={len(tags):2d} accuracy {a.mean():.3f} "
              f"(null {null.mean():.3f}, 95% {np.percentile(null, 95):.3f}, p {rows[-1]['p']:.3f})", flush=True)

    for group, family, variant, tags in size_rows:
        for tag in tags:
            per_rdm.append({"group": group, "family": family, "variant": variant, "rdm": tag,
                            "accuracy": round(next(results)[0], 4)})

    rng = np.random.default_rng(SEED)
    relations = []

    def add(test, units, accuracy_kind, alignment_kind, n, r, p, groups="", beta=None, p_mixed=None):
        relations.append({"test": test, "units": units, "accuracy": accuracy_kind, "alignment": alignment_kind,
                          "n": n, "groups": groups, "beta_mixed": "" if beta is None else round(beta, 3),
                          "p_mixed": "" if p_mixed is None else round(p_mixed, 4),
                          "spearman_r": round(r, 3), "p_permutation": round(p, 4)})
        m = "" if beta is None else f"mixed beta = {beta:+.3f}, p = {p_mixed:.4f}; "
        print(f"[{test}] {units}, {accuracy_kind} accuracy vs {alignment_kind} (n = {n}{', ' + groups if groups else ''}): "
              f"{m}Spearman r = {r:+.3f}, p = {p:.4f}", flush=True)

    with (out / "rho_max_per_rdm.csv").open() as fh:
        align = {r["rdm"]: float(r["max_r"]) for r in csv.DictReader(fh)
                 if r["roi"] == rsa_core.ROI_AVG and r["emotion"] == rsa_core.EMO_AVG and r["max_r"]}
    probe_acc = {r["rdm"]: r["accuracy"] for r in per_rdm}
    ident_path = rsa_core.ROOT / "analysis/identity_accuracy.csv"
    ident = {r["rdm"]: float(r["accuracy"]) for r in csv.DictReader(ident_path.open())} if ident_path.exists() else {}

    # size curve: the FaceNets of the size curve and the full pool (one architecture)
    curve = [(size, t) for g, f, size, tags in size_rows if g == "size" for t in tags]
    curve += [("full", t) for f, v, tags in main_rows if (f, v) == ("FaceNet", "VGGFace2") for t in tags]
    for kind, source in (("emotion probe", probe_acc), ("face identity", ident)):
        units = [(size, t) for size, t in curve if t in source and t in align]
        if len(units) < 10:
            continue
        a = [source[t] for _, t in units]; al = [align[t] for _, t in units]
        run = [t.rsplit("_r", 1)[1] for _, t in units]; size = [g for g, _ in units]
        add("size curve", "FaceNet size-curve models", kind, "rho_max", len(units), *relation(a, al, rng=rng),
            groups=f"{len(set(run))} runs", **dict(zip(("beta", "p_mixed"), mixed(a, al, run))))
        add("size curve, within size", "FaceNet size-curve models", kind, "rho_max", len(units),
            *relation(a, al, size, rng), groups=f"{len(set(size))} sizes", **dict(zip(("beta", "p_mixed"), mixed(a, al, size))))

    # runs: the ten seeds or training runs of each variant
    units = [(f"{f} {v}", t) for f, v, tags in main_rows if len(tags) == 10 for t in tags]
    a = [probe_acc[t] for _, t in units]; al = [align[t] for _, t in units]; g = [v for v, _ in units]
    add("runs, within variant", "seeds and training runs", "emotion probe", "rho_max", len(units),
        *relation(a, al, g, rng), groups=f"{len(set(g))} variants", **dict(zip(("beta", "p_mixed"), mixed(a, al, g))))

    # variants: descriptive, on rho_max above chance
    with (out / "rho_max_null.csv").open() as fh:
        excess = {(r["family"], r["variant"]): float(r["excess"]) for r in csv.DictReader(fh) if r["group"] == "main"}
    trained = [r for r in rows if r["variant"] != "Random"]
    for label, sel in (("all variants", rows), ("trained variants", trained)):
        r_s, p = spearmanr([r["accuracy"] for r in sel], [excess[r["family"], r["variant"]] for r in sel])
        add("variants (descriptive)", label, "emotion probe", "rho_max above chance", len(sel), float(r_s), float(p))
    relation_rows = relations
    for name, table in (("probe.csv", rows), ("probe_per_rdm.csv", per_rdm), ("probe_vs_alignment.csv", relation_rows)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(f"[ok] -> {out}/probe.csv, probe_per_rdm.csv, probe_vs_alignment.csv")


if __name__ == "__main__":
    main()
