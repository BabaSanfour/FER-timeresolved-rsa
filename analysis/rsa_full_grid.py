#!/usr/bin/env python3
"""Full time-locked RSA grid: subject x scope x layer x emotion x window.

Every cell is kept, so nothing is collapsed before the summaries.

Estimator, per cell: Spearman correlation between the upper triangles of
    EEG RDM  [subject s, scope o, window k]
    model RDM[layer l, frame k]
i.e. TIME-LOCKED - EEG window k is always paired with model frame k, never
with any other frame. Spearman is computed as a dot product of rank-normalised
vectors, which is exact (ties get average ranks) and lets the whole grid be one
einsum.

Scopes  : the 6 ROIs (from roi_order.npy) + "wholescalp"      -> 7
Emotions: fear (0-5), happy (6-11), neutral (12-17) -> 6x6 RDMs (15 pairs)
          all (0-17)                                -> 18x18 RDM (153 pairs)

Outputs (analysis/rsa_grid/):
  <model>_rsa_grid.npz   rsa (S, 7, L, 4, 16) float32 + subjects/scopes/layers/
                         emotions/window_bounds coordinates
  <model>_rsa.csv        one row per subject x scope x layer x emotion, with the
                         time-locked score = mean over the 16 windows
  rsa_summary.csv        subject-mean +/- SEM for every model x scope x layer x emotion
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import rsa_core

ROOT = rsa_core.ROOT
MDIR = ROOT / "analysis/model_rdms_canonical"
OUT_DIR = ROOT / "analysis/rsa_grid"

EMOTIONS = rsa_core.EMOTIONS["18"]


def upper(r):
    """(..., k, k) -> (..., k*(k-1)/2) upper triangle."""
    iu = np.triu_indices(r.shape[-1], 1)
    return r[..., iu[0], iu[1]]


def rank_normalise(v):
    """Rank -> mean-centre -> unit norm, so a dot product IS Spearman's rho."""
    r = np.apply_along_axis(rankdata, -1, v)
    rc = r - r.mean(-1, keepdims=True)
    n = np.linalg.norm(rc, axis=-1, keepdims=True)
    return np.divide(rc, np.clip(n, 1e-12, None)), (n[..., 0] < 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="RDM basenames under model_rdms_canonical/ (default: all *_weights)")
    ap.add_argument("--model-dir", default=None,
                    help="directory containing model RDM .npy files (default: model_rdms_canonical)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-windows", type=int, default=None,
                    help="use only the first N matched EEG windows/model frames "
                         "(for legacy 15-frame RDMs; default: all EEG windows)")
    args = ap.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else MDIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg, subjects, scopes = rsa_core.load_eeg("18")
    if args.n_windows is not None:
        if args.n_windows <= 0 or args.n_windows > eeg.shape[2]:
            raise SystemExit(f"ABORT: --n-windows must be in [1,{eeg.shape[2]}]")
        eeg = eeg[:, :, :args.n_windows]
    S, O, W = eeg.shape[0], eeg.shape[1], eeg.shape[2]
    print(f"[eeg] {S} subjects x {O} scopes x {W} windows: {scopes}")

    window_bounds = rsa_core.window_bounds("18")[:W]

    tags = args.models or sorted(p.name[:-4] for p in model_dir.glob("*_weights.npy")
                                 if not p.name.endswith(("_fear.npy", "_happy.npy",
                                                         "_neutral.npy")))
    print(f"[models] {len(tags)}")

    emo_names = list(EMOTIONS)
    summary = []

    for tag in tags:
        model = np.load(model_dir / f"{tag}.npy")               # (L, 16, 18, 18)
        if model.shape[1] < W:
            print(f"  SKIP {tag}: {model.shape[1]} frames < {W} EEG windows")
            continue
        model = model[:, :W]
        L = model.shape[0]
        grid = np.full((S, O, L, len(emo_names), W), np.nan, dtype=np.float32)

        for ei, emo in enumerate(emo_names):
            idx = EMOTIONS[emo]
            ev = upper(eeg[:, :, :, idx][:, :, :, :, idx])      # (S, O, W, P)
            mv = upper(model[:, :, idx][:, :, :, idx])          # (L, W, P)
            ez, e_flat = rank_normalise(ev)
            mz, m_flat = rank_normalise(mv)
            # Spearman for every (subject, scope, layer, window) at once.
            r = np.einsum("sowp,lwp->solw", ez, mz, optimize=True)
            # Constant RDMs have undefined correlation; the dot product would
            # silently return 0. Mark them NaN instead.
            r[np.broadcast_to(m_flat[None, None, :, :], r.shape)] = np.nan
            r[np.broadcast_to(e_flat[:, :, None, :], r.shape)] = np.nan
            grid[:, :, :, ei, :] = r.astype(np.float32)

        np.savez_compressed(
            out_dir / f"{tag}_rsa_grid.npz", rsa=grid, subjects=np.array(subjects),
            scopes=np.array(scopes), emotions=np.array(emo_names),
            n_layers=L, window_bounds=window_bounds)

        # per-subject time-locked score = mean over the 16 window<->frame pairs
        tl = np.nanmean(grid, axis=-1)                          # (S, O, L, E)
        with (out_dir / f"{tag}_rsa.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "subject", "scope", "layer", "emotion", "rsa_timelocked"])
            for si, sub in enumerate(subjects):
                for oi, sc in enumerate(scopes):
                    for li in range(L):
                        for ei, emo in enumerate(emo_names):
                            w.writerow([tag, sub, sc, li, emo, f"{tl[si, oi, li, ei]:.6f}"])

        m = np.nanmean(tl, axis=0)                              # (O, L, E)
        sd = np.nanstd(tl, axis=0, ddof=1)
        sem = sd / np.sqrt(np.sum(~np.isnan(tl), axis=0))
        for oi, sc in enumerate(scopes):
            for li in range(L):
                for ei, emo in enumerate(emo_names):
                    summary.append({"model": tag, "scope": sc, "layer": li, "emotion": emo,
                                    "n_subj": int(np.sum(~np.isnan(tl[:, oi, li, ei]))),
                                    "rsa_mean": float(m[oi, li, ei]),
                                    "rsa_sem": float(sem[oi, li, ei])})
        best = np.nanmax(m[:, :, emo_names.index("all")])
        print(f"  {tag:38s} L={L:3d} grid={grid.shape}  best mean RSA (all) = {best:+.4f}")

    # A targeted --models run must not erase summaries for previously computed models.
    summary_path = out_dir / "rsa_summary.csv"
    if args.models and summary_path.exists():
        with summary_path.open(newline="") as f:
            prior = list(csv.DictReader(f))
        selected = set(args.models)
        summary = [r for r in prior if r.get("model") not in selected] + summary
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scope", "layer", "emotion",
                                          "n_subj", "rsa_mean", "rsa_sem"])
        w.writeheader()
        for r in summary:
            def formatted(value):
                try:
                    return round(float(value), 6)
                except (TypeError, ValueError):
                    return value
            w.writerow({**r, "rsa_mean": formatted(r["rsa_mean"]),
                        "rsa_sem": formatted(r["rsa_sem"])})
    print(f"\n[ok] {len(summary)} summary rows -> {summary_path}")


if __name__ == "__main__":
    main()
