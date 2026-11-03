#!/usr/bin/env python3
"""Time-locked partial Spearman RSA (--stimuli 18 | 54).

For each participant, scalp scope, model layer, condition and window k, the
model and EEG vectors (upper triangles) are rank-transformed, residualised
against the ranked nuisance vectors of frame k, and correlated. Controls
(analysis/control_rdms.py, read like models): identity, occlusion (54 stimuli),
pixel, optical flow, motion energy; the pixel control differs per frame, the
others are the same at every frame. Control sets: each control alone, and all
of them jointly. A control that is constant in a condition (identity within one
emotion of the 18 stimuli) is left out of that condition's regression.

Outputs: one grid per RDM and control set, in the format of rsa_grid.py, so
maxr_table.py --partial <set> summarises them and cluster_permutation.py
--partial <set> tests them:
  analysis/partial_rsa[_54]/<set>/<model>_rsa_grid.npz   rsa (S, 7, L, 4, W), ..., controls
"""
import argparse

import numpy as np
from scipy.stats import rankdata

import rsa_core
import variants
from rsa_core import upper


def ranked_centered(x):
    ranks = rankdata(x, axis=-1)
    return ranks - ranks.mean(axis=-1, keepdims=True)


def nuisance_basis(control_vectors):
    """Orthonormal basis of the nonconstant, nonredundant ranked controls."""
    columns = [ranked_centered(v[None])[0] for v in control_vectors]
    columns = [c for c in columns if np.linalg.norm(c) > 1e-12]
    if not columns:
        return np.empty((len(control_vectors[0]), 0))
    design = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    return u[:, :int(np.sum(singular > tolerance))]


def residual_normalise(values, basis):
    """Ranked, centred vectors -> residuals on the basis, unit norm (and where they vanish)."""
    if basis.shape[1]:
        values = values - (values @ basis) @ basis.T
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, np.clip(norm, 1e-12, None)), norm[..., 0] < 1e-12


def control_sets(stimuli):
    names = [name for _, name in variants.controls(stimuli)]
    return {**{name: [name] for name in names}, "joint": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="RDM names (default: every RDM of the main variants, analysis/variants.py)")
    args = ap.parse_args()
    s = args.stimuli
    eeg, subjects, scopes = rsa_core.load_eeg(s)
    S, O, W = eeg.shape[:3]
    emotions = rsa_core.EMOTIONS[s]
    sets = control_sets(s)
    controls = {name: rsa_core.load_model_rdm(s, f"control_{name}")[0, :W] for name in sets["joint"]}  # (W, n, n)
    for name in sets:
        (rsa_core.PARTIAL_DIRS[s] / name).mkdir(parents=True, exist_ok=True)
    tags = args.models or [t for g, *_, ts in variants.variants(s) if g == "main" for t in ts]
    print(f"[eeg] {S} subjects x {O} scopes x {W} windows; {len(tags)} RDMs; control sets {list(sets)}")
    # the same for every model: ranked EEG vectors and nuisance bases, per emotion and window
    eeg_ranked = [ranked_centered(upper(eeg[..., idx, :][..., idx])) for idx in emotions.values()]
    bases = [{name: [nuisance_basis([upper(controls[c][k][idx][:, idx]) for c in members])
                     for k in range(W)] for name, members in sets.items()}
             for idx in emotions.values()]

    for tag in tags:
        model = rsa_core.load_model_rdm(s, tag)[:, :W]                       # (L, W, n, n)
        L = model.shape[0]
        grids = {name: np.full((S, O, L, len(emotions), W), np.nan, np.float32) for name in sets}
        for ei, idx in enumerate(emotions.values()):
            ev = eeg_ranked[ei]                                               # (S, O, W, P)
            mv = ranked_centered(upper(model[..., idx, :][..., idx]))        # (L, W, P)
            for name in sets:
                for k in range(W):
                    basis = bases[ei][name][k]
                    ez, e_invalid = residual_normalise(ev[:, :, k], basis)    # (S, O, P)
                    mz, m_invalid = residual_normalise(mv[:, k], basis)       # (L, P)
                    r = np.einsum("sop,lp->sol", ez, mz, optimize=True)
                    r[np.broadcast_to(e_invalid[:, :, None], r.shape)] = np.nan
                    r[np.broadcast_to(m_invalid[None, None], r.shape)] = np.nan
                    grids[name][:, :, :, ei, k] = r
        for name, grid in grids.items():
            np.savez_compressed(
                rsa_core.grid_path(s, tag, name), rsa=grid, subjects=np.array(subjects),
                scopes=np.array(scopes), emotions=np.array(list(emotions)), n_layers=L,
                window_bounds=rsa_core.window_bounds(s)[:W], controls=np.array(sets[name]))
        print(f"  {tag:28s} L={L:3d}", flush=True)


if __name__ == "__main__":
    main()
