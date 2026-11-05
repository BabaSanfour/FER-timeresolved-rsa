import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fer_rsa.analysis import rsa_core
from fer_rsa.analysis import variants

N_NULL = 200
SEED = 42


def accuracy(features, emotion, actor):
    """(n, T, d) features -> leave-one-actor-out accuracy over the n videos."""
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
    """n permutations of the videos within each actor (balanced folds, null at chance)."""
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
    names = rsa_core.MODEL_STIMULI[s]
    emotion = np.array([nm.split("_")[0] for nm in names])
    actor = np.array([nm.rsplit("_Actor", 1)[1] for nm in names])
    folder = rsa_core.ROOT / f"features/set{s}"
    out = rsa_core.STATS_DIRS[s]
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
            per_rdm.append({"group": "main", "family": family, "variant": variant, "rdm": tag, "accuracy": round(r[0], 4)})
        rows.append({"family": family, "variant": variant, "n": len(tags), "accuracy": round(float(a.mean()), 4),
                     "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else "",
                     "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                     "null_mean": round(float(null.mean()), 4), "null_95": round(float(np.percentile(null, 95)), 4),
                     "p": float((1 + np.sum(null >= a.mean())) / (N_NULL + 1))})
        print(f"  {family:9s} {variant:13s} n={len(tags):2d} accuracy {a.mean():.3f} "
              f"(null {null.mean():.3f}, 95% {np.percentile(null, 95):.3f}, p {rows[-1]['p']:.3f})", flush=True)

    for name, table in (("probe.csv", rows), ("probe_per_rdm.csv", per_rdm)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(f"[ok] -> {out}/probe.csv, probe_per_rdm.csv")


if __name__ == "__main__":
    main()
