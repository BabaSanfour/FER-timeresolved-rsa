#!/usr/bin/env python3
"""rho_max against its own chance level, for every variant and control (--stimuli 18 | 54).

A model searches more cells (layers x windows) than a control (windows only), so
their rho_max values have different chance levels and cannot be compared directly.
For each RDM, the stimulus labels are permuted within each emotion (fear, happy,
neutral; the same permutations for every RDM) and the Table 1 statistic is
recomputed: the maximum over layer x window of the participant-mean Spearman in
each ROI x emotion cell, averaged over the 6 ROIs x 3 emotions. A variant's value
is the mean over its RDMs, per permutation. Reported: the observed value, the null
mean and 95th percentile, the permutation p, and the excess over chance
(observed - null mean), which puts models and controls on the same footing.

The participant-mean grid is linear in the participants' rank-normalised EEG
vectors, so the EEG is averaged once and a permutation only reorders the model's
ranked pairs.

Output: analysis/paper_stats[_54]/rho_max_null.csv
"""
import argparse
import csv

import numpy as np

import rsa_core
import variants
from rsa_core import upper, rank_normalise

N_PERM = 1000
SEED = 42
EMOTIONS = ("fear", "happy", "neutral")


def pair_permutation(perm):
    """Stimulus permutation -> index into the upper-triangle pairs of the permuted RDM."""
    n = len(perm)
    iu = np.triu_indices(n, 1)
    pos = np.full((n, n), -1)
    pos[iu] = np.arange(len(iu[0]))
    pos = np.maximum(pos, pos.T)
    return pos[perm[iu[0]], perm[iu[1]]]


def roi_cell_maxima(eeg_mean, model_z, model_flat, pairs=None):
    """eeg_mean (O, W, P), model_z (L, W, P) -> (6,) max over layer x window per ROI."""
    mz = model_z if pairs is None else model_z[..., pairs]
    g = np.einsum("owp,lwp->olw", eeg_mean[:rsa_core.N_ROI], mz, optimize=True)
    g[:, model_flat] = np.nan
    return np.nanmax(g.reshape(g.shape[0], -1), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    args = ap.parse_args()
    s = args.stimuli
    eeg, _, _ = rsa_core.load_eeg(s)
    W = eeg.shape[2]
    rng = np.random.default_rng(SEED)
    perms, eeg_mean = {}, {}
    for e in EMOTIONS:
        idx = rsa_core.EMOTIONS[s][e]
        ez, e_flat = rank_normalise(upper(eeg[..., idx, :][..., idx]))      # (S, O, W, P)
        if e_flat.any():
            raise SystemExit("ABORT: constant EEG RDM")
        eeg_mean[e] = ez.mean(0)
        perms[e] = [pair_permutation(rng.permutation(len(idx))) for _ in range(N_PERM)]
    rows = []
    for group, family, variant, tags in variants.variants(s):
        if group in ("size", "step"):
            continue
        obs, null = [], []
        for tag in tags:
            model = rsa_core.load_model_rdm(s, tag)[:, :W]
            o, n = 0.0, np.zeros(N_PERM)
            for e in EMOTIONS:
                idx = rsa_core.EMOTIONS[s][e]
                mz, m_flat = rank_normalise(upper(model[..., idx, :][..., idx]))   # (L, W, P)
                if m_flat.all():                                   # constant (identity within an emotion)
                    o, n = np.nan, np.full(N_PERM, np.nan)
                    break
                o += roi_cell_maxima(eeg_mean[e], mz, m_flat).mean()
                n += [roi_cell_maxima(eeg_mean[e], mz, m_flat, pp).mean() for pp in perms[e]]
            obs.append(o / len(EMOTIONS)); null.append(n / len(EMOTIONS))
        o, n = float(np.mean(obs)), np.mean(null, axis=0)
        defined = np.isfinite(o)
        rows.append({"group": group, "family": family, "variant": variant, "n": len(tags),
                     "rho_max": round(o, 5) if defined else "",
                     "null_mean": round(float(n.mean()), 5) if defined else "",
                     "null_95": round(float(np.percentile(n, 95)), 5) if defined else "",
                     "excess": round(o - float(n.mean()), 5) if defined else "",
                     "p": float((1 + np.sum(n >= o)) / (N_PERM + 1)) if defined else ""})
        r = rows[-1]
        print(f"  {group:7s} {family:9s} {variant:13s} rho_max {r['rho_max']}  null {r['null_mean']} "
              f"(95% {r['null_95']})  excess {r['excess']}  p {r['p']}", flush=True)
    out = rsa_core.ROOT / ("analysis/paper_stats" if s == "18" else "analysis/paper_stats_54") / "rho_max_null.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
