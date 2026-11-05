#!/usr/bin/env python3
import argparse

import numpy as np

import rsa_core
import variants
from rsa_core import ranked_centered, unit, upper


def nuisance_basis(control_vectors):
    """Orthonormal basis of the nonconstant, nonredundant ranked controls."""
    columns = [ranked_centered(v) for v in control_vectors]
    columns = [c for c in columns if np.linalg.norm(c) > 1e-12]
    if not columns:
        return np.empty((len(control_vectors[0]), 0))
    design = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    return u[:, :int(np.sum(singular > tolerance))]


def residual_normalise(values, basis):
    """Ranked, centred vectors -> unit-norm residuals on the basis (and where they vanish)."""
    if basis.shape[1]:
        values = values - (values @ basis) @ basis.T
    return unit(values)


def window_bases(controls, members, idx, W):
    """Nuisance basis of each window k over the stimuli idx."""
    return [nuisance_basis([upper(controls[c][k][idx][:, idx]) for c in members]) for k in range(W)]


def control_sets(stimuli):
    """Each control alone, and all jointly."""
    names = [name for _, name in variants.controls(stimuli)]
    return {**{name: [name] for name in names}, "joint": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="RDM names (default: every RDM of the main variants)")
    args = ap.parse_args()
    s = args.stimuli
    eeg, subjects, scopes = rsa_core.load_eeg(s)
    S, O, W = eeg.shape[:3]
    emotions = rsa_core.EMOTIONS[s]
    sets = control_sets(s)
    controls = {name: rsa_core.load_model_rdm(s, f"control_{name}")[0, :W] for name in sets["joint"]}  # (W, n, n)
    for name in sets:
        (rsa_core.PARTIAL_DIRS[s] / name).mkdir(parents=True, exist_ok=True)
    tags = args.models or [t for t in variants.all_rdms(s) if not t.startswith("control_")]
    print(f"[eeg] {S} subjects x {O} scopes x {W} windows; {len(tags)} RDMs; control sets {list(sets)}")
    # shared by every model, per emotion (and window)
    eeg_ranked = [ranked_centered(upper(eeg[..., idx, :][..., idx])) for idx in emotions.values()]
    bases = [{name: window_bases(controls, members, idx, W) for name, members in sets.items()}
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
                window_bounds=rsa_core.window_bounds()[:W], controls=np.array(sets[name]))
        print(f"  {tag:28s} L={L:3d}", flush=True)


if __name__ == "__main__":
    main()
