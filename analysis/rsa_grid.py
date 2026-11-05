#!/usr/bin/env python3
"""Full time-locked RSA grid: subject x scope x layer x emotion x window.

Per cell: Spearman correlation between the upper triangles of
    EEG RDM  [subject s, scope o, window k]
    model RDM[layer l, frame k]
time-locked (EEG window k is always paired with model frame k). Spearman is a
dot product of rank-normalised vectors, so the whole grid is one einsum.
Every cell is kept, so nothing is collapsed before the summaries.

Scopes  : the 11 ROIs + "wholescalp" (eeg_rdms/meta.json)               -> 12
Emotions: all, fear, happy, neutral; the model RDM is sliced with the same
          indices as the EEG (rsa_core.EMOTIONS)                          -> 4
--stimuli 18 or 54: the same 25 participants (rsa_core.py).

Models: every RDM of analysis/variants.py (or --models), from rdms/set<stimuli>/<name>/;
each RDM's stimulus order is checked against the EEG's.

Output: <out-dir>/<model>_rsa_grid.npz
  rsa (S, 12, L, 4, W) float32, subjects, scopes, emotions, n_layers, window_bounds
"""
import argparse
from pathlib import Path

import numpy as np

import rsa_core
import variants
from rsa_core import upper, rank_normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="RDM names (default: every RDM of analysis/variants.py)")
    ap.add_argument("--rdm-dir", default=None, help="default: $FER_ROOT/rdms/set<stimuli>")
    ap.add_argument("--out-dir", default=None, help="default: rsa_core.GRID_DIRS")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else rsa_core.GRID_DIRS[args.stimuli]
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg, subjects, scopes = rsa_core.load_eeg(args.stimuli)
    S, O, W = eeg.shape[:3]
    window_bounds = rsa_core.window_bounds(args.stimuli)[:W]
    emotions = rsa_core.EMOTIONS[args.stimuli]
    print(f"[eeg] {args.stimuli} stimuli: {S} subjects x {O} scopes x {W} windows")
    # the EEG side is the same for every model: rank it once per emotion
    eeg_ranked = [rank_normalise(upper(eeg[..., idx, :][..., idx]))     # (S, O, W, P)
                  for idx in emotions.values()]

    for tag in args.models or variants.all_rdms(args.stimuli):
        model = rsa_core.load_model_rdm(args.stimuli, tag, args.rdm_dir)   # (L, frames, n, n)
        if model.shape[1] < W or model.shape[-1] != eeg.shape[-1]:
            raise SystemExit(f"ABORT {tag}: model {model.shape}, EEG {eeg.shape}")
        model = model[:, :W]
        L = model.shape[0]
        grid = np.full((S, O, L, len(emotions), W), np.nan, dtype=np.float32)
        for ei, idx in enumerate(emotions.values()):
            ez, e_flat = eeg_ranked[ei]
            mz, m_flat = rank_normalise(upper(model[..., idx, :][..., idx]))   # (L, W, P)
            r = np.einsum("sowp,lwp->solw", ez, mz, optimize=True)
            # Constant RDMs have no correlation (the dot product would be 0).
            r[np.broadcast_to(m_flat[None, None], r.shape)] = np.nan
            r[np.broadcast_to(e_flat[:, :, None], r.shape)] = np.nan
            grid[:, :, :, ei] = r.astype(np.float32)
        np.savez_compressed(
            out_dir / f"{tag}_rsa_grid.npz", rsa=grid, subjects=np.array(subjects),
            scopes=np.array(scopes), emotions=np.array(list(emotions)),
            n_layers=L, window_bounds=window_bounds)
        best = np.nanmax(np.nanmean(grid[:, :, :, 0], axis=0))
        print(f"  {tag:28s} L={L:3d}  best mean RSA (all) = {best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
