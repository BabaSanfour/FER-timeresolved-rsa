#!/usr/bin/env python3
"""Grid-wide cluster permutation tests over layer × time RSA grids (--study 18 | 54).

One test per variant (analysis/variants.py, main group) on its participant-level
grid averaged over its ten seeds or training runs (rsa_core.variant_grid; a
released model has one RDM), for each of the 7 scopes x 4 conditions (28 grids).
Within a grid: one-sided sign-flip test of the cluster mass (sum of t over
4-neighbour clusters of cells with t above the one-sided .05 threshold), FWER
from the maximum cluster mass. Across grids: Holm over the 28 grids of a model
(p_holm_within_model) and over all grids of all variants (p_holm_study).

Every participant enters every grid (the EEG has no missing values, rsa_core.py).
A cell is NaN only where a model RDM is constant (no correlation), and then for
every participant; such cells are left out of the clustering.
"""
import csv
import json
import argparse

import numpy as np
from scipy.ndimage import label
from scipy.stats import t as tdist

import rsa_core
import variants

ROOT = rsa_core.ROOT
# 20,000 sign flips: the smallest p is 1/20,001, so a grid can survive Holm across
# all grids of all variants (21 x 28 = 588: 588 / 20,001 = .029 < .05)
N_PERM = 20000
SEED = 42
ALPHA = .05
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def t_values(x, signed_sum=None):
    n = x.shape[0]
    ss = np.sum(x * x, axis=0)
    total = np.sum(x, axis=0) if signed_sum is None else signed_sum
    mean = total / n
    var = np.maximum((ss - n * mean * mean) / (n - 1), 0)
    return np.divide(mean, np.sqrt(var / n), out=np.zeros_like(mean),
                     where=var > 0)


def clusters(tmap, threshold, valid=None):
    hit = tmap > threshold
    if valid is not None:
        hit &= valid
    labels, count = label(hit, structure=STRUCTURE)
    result = []
    for cluster_id in range(1, count + 1):
        mask = labels == cluster_id
        result.append((mask, float(tmap[mask].sum())))
    return result


def max_cluster_null(x, signs, threshold, valid=None):
    n, layers, windows = x.shape
    flat = x.reshape(n, -1)
    ss = np.sum(flat * flat, axis=0)
    null = np.zeros(len(signs), dtype=np.float32)
    for start in range(0, len(signs), 100):
        totals = signs[start:start + 100] @ flat
        means = totals / n
        var = np.maximum((ss[None, :] - n * means * means) / (n - 1), 0)
        tmaps = np.divide(means, np.sqrt(var / n),
                          out=np.zeros_like(means), where=var > 0)
        for j, tflat in enumerate(tmaps):
            found = clusters(tflat.reshape(layers, windows), threshold, valid)
            null[start + j] = max((mass for _, mass in found), default=0)
    return null


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(running, 1.)
    return adjusted


def signs_for(n, cache={}):
    """Sign matrix for a realised n. Seeded per n, so a grid's test is
    reproducible and a complete grid gives exactly the result it gave when the
    matrix was built once for the full sample."""
    if n not in cache:
        rng = np.random.default_rng(SEED)
        cache[n] = rng.choice(np.array([-1., 1.], np.float32), size=(N_PERM, n))
    return cache[n]


def threshold_for(n, cache={}):
    if n not in cache:
        cache[n] = float(tdist.ppf(1 - ALPHA, n - 1))
    return cache[n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["18", "54"], required=True)
    parser.add_argument("--partial", default=None,
                        help="test the partial-RSA grids of this control set (partial_rsa.py)")
    args = parser.parse_args()
    global OUT
    OUT = ROOT / (f"analysis/cluster_permutation_{args.study}"
                  + (f"_partial_{args.partial}" if args.partial else ""))
    tested = [(fam, var, tags) for g, fam, var, tags in variants.variants(args.study) if g == "main"]
    missing = [t for *_, tags in tested for t in tags
               if not rsa_core.grid_path(args.study, t, args.partial).exists()]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} grids missing, e.g. {missing[:3]}")
    OUT.mkdir(parents=True, exist_ok=True)
    grid_rows, cluster_rows = [], []

    for pi, (family, variant, tags) in enumerate(tested, 1):
        tag = rsa_core.slug(family, variant)
        rsa, z = rsa_core.variant_grid(args.study, tags, args.partial)
        scopes = [str(x) for x in z["scopes"]]
        emotions = [str(x) for x in z["emotions"]]
        n = rsa.shape[0]
        model_rows = []
        masks = np.zeros(rsa.shape[1:], dtype=np.uint8)  # scope, layer, emotion, window
        for si, scope in enumerate(scopes):
            for ei, emotion in enumerate(emotions):
                raw = rsa[:, si, :, ei, :]
                valid = np.isfinite(raw).all(axis=0)
                if not np.array_equal(valid, np.isfinite(raw).any(axis=0)):
                    raise SystemExit(f"ABORT {tag} {scope} {emotion}: NaN for some participants only")
                base = {"model": tag, "family": family, "variant": variant, "n_rdms": len(tags),
                        "scope": scope, "emotion": emotion, "n": n}
                # invalid cells are excluded from clustering; zero-filling touches no valid cell
                x = np.where(valid, raw, 0.).astype(np.float32)
                threshold = threshold_for(n)
                signs = signs_for(n)
                observed_t = np.where(valid, t_values(x), 0.)
                observed_clusters = clusters(observed_t, threshold, valid)
                null = max_cluster_null(x, signs, threshold, valid)
                cluster_ps = []
                for ci, (mask, mass) in enumerate(observed_clusters, 1):
                    p = float((1 + np.sum(null >= mass)) / (N_PERM + 1))
                    cluster_ps.append(p)
                    if p < ALPHA:
                        masks[si, :, ei, :][mask] = 1
                    where = np.argwhere(mask)
                    cluster_rows.append({
                        **base,
                        "cluster": ci, "mass": mass, "size": int(mask.sum()),
                        "layer_min": int(where[:, 0].min()),
                        "layer_max": int(where[:, 0].max()),
                        "window_min": int(where[:, 1].min()),
                        "window_max": int(where[:, 1].max()),
                        "peak_t": float(observed_t[mask].max()),
                        "mean_rho": float(np.mean(raw[:, mask])),
                        "p_grid_fwer": p,
                    })
                model_rows.append({
                    **base,
                    "n_layers": x.shape[1], "n_windows": x.shape[2],
                    "t_threshold": threshold, "n_permutations": N_PERM,
                    "n_valid_cells": int(valid.sum()),
                    "n_cells": int(valid.size),
                    "n_clusters": len(observed_clusters),
                    "max_t": float(observed_t[valid].max()) if valid.any() else float("nan"),
                    "min_cluster_p": min(cluster_ps, default=1.0),
                    "survives_grid_fwer": int(any(p < ALPHA for p in cluster_ps)),
                })
        adjusted = holm([row["min_cluster_p"] for row in model_rows])
        for row, adj in zip(model_rows, adjusted):
            row["p_holm_within_model"] = float(adj)
            row["survives_holm_within_model"] = int(adj < ALPHA)
        grid_rows.extend(model_rows)
        np.savez_compressed(
            OUT / f"{tag}_cluster_masks.npz", significant=masks,
            scopes=np.asarray(scopes), emotions=np.asarray(emotions),
            n=n, threshold=threshold_for(n),
            n_permutations=N_PERM,
        )
        print(f"[{pi}/{len(tested)}] {tag} ({len(tags)} RDMs): "
              f"{sum(r['survives_grid_fwer'] for r in model_rows)} grid-FWER, "
              f"{sum(r['survives_holm_within_model'] for r in model_rows)} Holm within model", flush=True)

    # study-wide: Holm over every grid of every variant
    for row, adj in zip(grid_rows, holm([row["min_cluster_p"] for row in grid_rows])):
        row["p_holm_study"] = float(adj)
        row["survives_holm_study"] = int(adj < ALPHA)

    with (OUT / "grid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
        writer.writeheader(); writer.writerows(grid_rows)
    with (OUT / "clusters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cluster_rows[0].keys())
        writer.writeheader(); writer.writerows(cluster_rows)

    survivors = []
    for tag in sorted(set(row["model"] for row in grid_rows)):
        rows = [row for row in grid_rows if row["model"] == tag]
        survivors.append({
            "model": tag, "family": rows[0]["family"], "variant": rows[0]["variant"],
            "n_rdms": rows[0]["n_rdms"], "n": rows[0]["n"],
            "n_grid_fwer_survivors": sum(r["survives_grid_fwer"] for r in rows),
            "n_holm_within_model_survivors": sum(r["survives_holm_within_model"] for r in rows),
            "n_holm_study_survivors": sum(r["survives_holm_study"] for r in rows),
            "best_cluster_p": min(r["min_cluster_p"] for r in rows),
            "best_holm_within_model_p": min(r["p_holm_within_model"] for r in rows),
            "best_holm_study_p": min(r["p_holm_study"] for r in rows),
        })
    with (OUT / "model_survivors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=survivors[0].keys())
        writer.writeheader(); writer.writerows(survivors)
    (OUT / "method.json").write_text(json.dumps({
        "test": "one-sample one-sided sign-flip",
        "grid": "full model-layer x EEG-window",
        "adjacency": "4-neighbour",
        "cluster_statistic": "sum of t",
        "cluster_forming_alpha": ALPHA,
        "within_grid_correction": "maximum cluster mass FWER",
        "n_permutations": N_PERM, "seed": SEED,
        "tested_grid": "per variant, the mean of its RDMs' participant-level grids (10 seeds or runs)",
        "across_grid_correction": {"within_model": "Holm across the 7 scopes x 4 conditions of a model",
                                   "study": f"Holm across all {len(grid_rows)} grids of all {len(tested)} variants"},
        "missing_data": "none: every participant in every grid; cells of a constant model RDM are excluded",
    }, indent=2))
    print(f"[ok] -> {OUT}")


if __name__ == "__main__":
    main()
