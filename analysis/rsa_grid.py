#!/usr/bin/env python3
"""Full time-locked RSA grid: subject x scope x layer x emotion x window.

Per cell: Spearman correlation between the upper triangles of
    EEG RDM  [subject s, scope o, window k]
    model RDM[layer l, frame k]
time-locked (EEG window k is always paired with model frame k). Spearman is a
dot product of rank-normalised vectors, so the whole grid is one einsum.
Every cell is kept, so nothing is collapsed before the summaries.

Scopes  : the 6 ROIs (roi_order.npy) + "wholescalp"                      -> 7
Emotions: all, fear, happy, neutral; the model RDM is sliced with the same
          indices as the EEG (rsa_core.EMOTIONS)                          -> 4
--stimuli 18: 26 participants; 54: 25 participants, and participants with a
missing video are analysed on the stimulus pairs they have (rsa_core.py).

Output: <out-dir>/<model>_rsa_grid.npz
  rsa (S, 7, L, 4, W) float32, subjects, scopes, emotions, n_layers, window_bounds
"""
import argparse
from pathlib import Path

import numpy as np

import rsa_core
from rsa_core import upper, rank_normalise

DEFAULTS = {"18": ("analysis/model_rdms_canonical", "analysis/rsa_grid"),
            "54": ("analysis/model_rdms_54", "analysis/rsa_grid_54")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="RDM basenames in --model-dir (default: every *_weights.npy)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else rsa_core.ROOT / DEFAULTS[args.stimuli][0]
    out_dir = Path(args.out_dir) if args.out_dir else rsa_core.ROOT / DEFAULTS[args.stimuli][1]
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg, subjects, scopes = rsa_core.load_eeg(args.stimuli)
    S, O, W = eeg.shape[:3]
    window_bounds = rsa_core.window_bounds(args.stimuli)[:W]
    emotions = rsa_core.EMOTIONS[args.stimuli]
    print(f"[eeg] {args.stimuli} stimuli: {S} subjects x {O} scopes x {W} windows")

    tags = args.models or sorted(p.name[:-4] for p in model_dir.glob("*_weights.npy"))
    for tag in tags:
        model = np.load(model_dir / f"{tag}.npy")               # (L, frames, n, n)
        if model.shape[1] < W or model.shape[-1] != eeg.shape[-1]:
            raise SystemExit(f"ABORT {tag}: model {model.shape}, EEG {eeg.shape}")
        model = model[:, :W]
        L = model.shape[0]
        grid = np.full((S, O, L, len(emotions), W), np.nan, dtype=np.float32)
        for ei, idx in enumerate(emotions.values()):
            ev = upper(eeg[..., idx, :][..., idx])                # (S, O, W, P)
            mv = upper(model[..., idx, :][..., idx])              # (L, W, P)
            for sidx, keep in rsa_core.pair_groups(ev):
                ez, e_flat = rank_normalise(ev[sidx][..., keep])
                mz, m_flat = rank_normalise(mv[..., keep])
                r = np.einsum("sowp,lwp->solw", ez, mz, optimize=True)
                # Constant RDMs have no correlation (the dot product would be 0).
                r[np.broadcast_to(m_flat[None, None], r.shape)] = np.nan
                r[np.broadcast_to(e_flat[:, :, None], r.shape)] = np.nan
                grid[sidx, :, :, ei] = r.astype(np.float32)
        np.savez_compressed(
            out_dir / f"{tag}_rsa_grid.npz", rsa=grid, subjects=np.array(subjects),
            scopes=np.array(scopes), emotions=np.array(list(emotions)),
            n_layers=L, window_bounds=window_bounds)
        best = np.nanmax(np.nanmean(grid[:, :, :, 0], axis=0))
        print(f"  {tag:46s} L={L:3d}  best mean RSA (all) = {best:+.4f}")


if __name__ == "__main__":
    main()
