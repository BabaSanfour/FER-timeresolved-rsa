import csv
import json
import argparse

import numpy as np
from scipy.ndimage import label
from scipy.stats import t as tdist

from fer_rsa.analysis import rsa_core
from fer_rsa.analysis import variants

N_PERM = 20000
SEED = 42
ALPHA = .05
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def t_map(total, ss, n):
    """One-sample t from the sum and sum of squares over n (0 where the variance is 0)."""
    mean = total / n
    var = np.maximum((ss - n * mean * mean) / (n - 1), 0)
    return np.divide(mean, np.sqrt(var / n), out=np.zeros_like(mean), where=var > 0)


def clusters(tmap, threshold, valid):
    """-> [(mask, mass)] of the 4-neighbour clusters of valid cells above threshold."""
    hit = (tmap > threshold) & valid
    labels, count = label(hit, structure=STRUCTURE)
    result = []
    for cluster_id in range(1, count + 1):
        mask = labels == cluster_id
        result.append((mask, float(tmap[mask].sum())))
    return result


def max_cluster_null(x, signs, threshold, valid):
    """(n, L, W) grid -> largest cluster mass under each sign flip."""
    n, layers, windows = x.shape
    flat = x.reshape(n, -1)
    ss = np.sum(flat * flat, axis=0)
    null = np.zeros(len(signs), dtype=np.float32)
    for start in range(0, len(signs), 100):
        tmaps = t_map(signs[start:start + 100] @ flat, ss[None, :], n)
        for j, tflat in enumerate(tmaps):
            found = clusters(tflat.reshape(layers, windows), threshold, valid)
            null[start + j] = max((mass for _, mass in found), default=0)
    return null


def signs_for(n, cache={}):
    """(N_PERM, n) sign flips, the same for every grid."""
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
                        help="a control set of partial_rsa.py (e.g. joint)")
    args = parser.parse_args()
    out = rsa_core.ROOT / (f"analysis/cluster_permutation_{args.study}"
                           + (f"_partial_{args.partial}" if args.partial else ""))
    tested = [(fam, var, tags) for g, fam, var, tags in variants.variants(args.study) if g == "main"]
    missing = [t for *_, tags in tested for t in tags
               if not rsa_core.grid_path(args.study, t, args.partial).exists()]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} grids missing, e.g. {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    grid_rows, cluster_rows = [], []
    grid_nulls = []                      # (mean, sd, null) of each grid's maximum cluster mass

    for pi, (family, variant, tags) in enumerate(tested, 1):
        tag = rsa_core.slug(family, variant)
        rsa, z = rsa_core.variant_grid(args.study, tags, args.partial)
        scopes = [str(x) for x in z["scopes"]]
        emotions = [str(x) for x in z["emotions"]]
        n = rsa.shape[0]
        model_rows = []
        for si, scope in enumerate(scopes):
            for ei, emotion in enumerate(emotions):
                raw = rsa[:, si, :, ei, :]
                valid = np.isfinite(raw).all(axis=0)
                if not np.array_equal(valid, np.isfinite(raw).any(axis=0)):
                    raise SystemExit(f"ABORT {tag} {scope} {emotion}: NaN for some participants only")
                base = {"model": tag, "family": family, "variant": variant, "n_rdms": len(tags),
                        "scope": scope, "emotion": emotion, "n": n}
                x = np.where(valid, raw, 0.).astype(np.float32)
                threshold = threshold_for(n)
                signs = signs_for(n)
                observed_t = np.where(valid, t_map(x.sum(0), (x * x).sum(0), n), 0.)
                observed_clusters = clusters(observed_t, threshold, valid)
                null = max_cluster_null(x, signs, threshold, valid)
                grid_nulls.append((float(null.mean()), float(null.std()) or 1.0, null))
                cluster_ps = []
                for ci, (mask, mass) in enumerate(observed_clusters, 1):
                    p = float((1 + np.sum(null >= mass)) / (N_PERM + 1))
                    cluster_ps.append(p)
                    where = np.argwhere(mask)
                    cluster_rows.append({
                        **base, "grid": len(grid_nulls) - 1,
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
        grid_rows.extend(model_rows)
        print(f"[{pi}/{len(tested)}] {tag} ({len(tags)} RDMs): "
              f"{sum(r['survives_grid_fwer'] for r in model_rows)} grid-FWER", flush=True)

    # study-wide: per sign flip, the maximum over grids of the standardised maximum cluster mass
    study_max = np.max([(null - mu) / sd for mu, sd, null in grid_nulls], axis=0)
    for row in cluster_rows:
        mu, sd, _ = grid_nulls[row["grid"]]
        row["p_study"] = float((1 + np.sum(study_max >= (row["mass"] - mu) / sd)) / (N_PERM + 1))
    for gi, row in enumerate(grid_rows):
        row["p_study"] = min((c["p_study"] for c in cluster_rows if c["grid"] == gi), default=1.0)
        row["survives_study"] = int(row["p_study"] < ALPHA)
    with (out / "grid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
        writer.writeheader(); writer.writerows(grid_rows)
    with (out / "clusters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cluster_rows[0].keys())
        writer.writeheader(); writer.writerows(cluster_rows)

    survivors = []
    for tag in sorted(set(row["model"] for row in grid_rows)):
        rows = [row for row in grid_rows if row["model"] == tag]
        survivors.append({
            "model": tag, "family": rows[0]["family"], "variant": rows[0]["variant"],
            "n_rdms": rows[0]["n_rdms"], "n": rows[0]["n"],
            "n_grid_fwer_survivors": sum(r["survives_grid_fwer"] for r in rows),
            "n_study_survivors": sum(r["survives_study"] for r in rows),
            "best_cluster_p": min(r["min_cluster_p"] for r in rows),
            "best_study_p": min(r["p_study"] for r in rows),
        })
    with (out / "model_survivors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=survivors[0].keys())
        writer.writeheader(); writer.writerows(survivors)
    (out / "method.json").write_text(json.dumps({
        "test": "one-sample one-sided sign-flip",
        "grid": "full model-layer x EEG-window",
        "adjacency": "4-neighbour",
        "cluster_statistic": "sum of t",
        "cluster_forming_alpha": ALPHA,
        "within_grid_correction": "maximum cluster mass FWER",
        "n_permutations": N_PERM, "seed": SEED,
        "tested_grid": "per variant, the mean of its RDMs' participant-level grids (10 seeds or runs)",
        "across_grid_correction": f"maximum statistic across all {len(grid_rows)} grids of all {len(tested)} variants: "
                                  "each grid's maximum cluster mass standardised by its null mean and SD",
        "missing_data": "none: every participant in every grid; cells of a constant model RDM are excluded",
    }, indent=2))
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
