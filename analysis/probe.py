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
all videos would unbalance the folds and push the null below chance. Across the
variants (all 21, and the 15 trained ones), the Spearman correlation between probe
accuracy and rho_max above its own chance level (excess over the permutation null,
ROI and emotion average; rho_max_null.py, run first) tests whether task performance
tracks EEG alignment. Raw rho_max is not used: its chance level grows with the number
of layers, which differs across families.

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
    jobs = [(folder / tag / f"{tag}.features.npy", emotion, actor, perms) for *_, tags in main_rows for tag in tags]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = iter(list(pool.map(rdm_accuracies, jobs)))
    rows, per_rdm = [], []
    for family, variant, tags in main_rows:
        res = [next(results) for _ in tags]
        a = np.array([r[0] for r in res]); null = np.mean([r[1] for r in res], axis=0)
        for tag, r in zip(tags, res):
            per_rdm.append({"family": family, "variant": variant, "rdm": tag, "accuracy": round(r[0], 4)})
        rows.append({"family": family, "variant": variant, "n": len(tags), "accuracy": round(float(a.mean()), 4),
                     "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else "",
                     "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                     "null_mean": round(float(null.mean()), 4), "null_95": round(float(np.percentile(null, 95)), 4),
                     "p": float((1 + np.sum(null >= a.mean())) / (N_NULL + 1))})
        print(f"  {family:9s} {variant:13s} n={len(tags):2d} accuracy {a.mean():.3f} "
              f"(null {null.mean():.3f}, 95% {np.percentile(null, 95):.3f}, p {rows[-1]['p']:.3f})", flush=True)

    with (out / "rho_max_null.csv").open() as fh:
        rho = {(r["family"], r["variant"]): float(r["excess"]) for r in csv.DictReader(fh) if r["group"] == "main"}
    acc = [r["accuracy"] for r in rows]
    al = [rho[r["family"], r["variant"]] for r in rows]
    trained = [i for i, r in enumerate(rows) if r["variant"] != "Random"]
    relation = []
    for label, idx in (("all variants", range(len(rows))), ("trained variants", trained)):
        r_s, p = spearmanr([acc[i] for i in idx], [al[i] for i in idx])
        relation.append({"variants": label, "n": len(idx), "spearman_r": round(float(r_s), 3), "p": round(float(p), 4)})
        print(f"[accuracy vs rho_max above chance] {label}: Spearman r = {r_s:.3f} (n = {len(idx)}, p = {p:.4f})")
    for name, table in (("probe.csv", rows), ("probe_per_rdm.csv", per_rdm), ("probe_vs_alignment.csv", relation)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(f"[ok] -> {out}/probe.csv, probe_per_rdm.csv, probe_vs_alignment.csv")


if __name__ == "__main__":
    main()
