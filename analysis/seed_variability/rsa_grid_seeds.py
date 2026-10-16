#!/usr/bin/env python3
"""RSA grids for the seed-variability campaign, with an explicit cohort switch.

analysis/rsa_full_grid.py currently cannot run: its ROI loader was edited to
drop CEG ("the preregistered/common 25-participant cohort") but the wholescalp
loader was not, so the subject-set check aborts every time. The published
18-stimulus grids -- and therefore Table 1 -- contain 26 subjects INCLUDING
CEG, while the manuscript states 25 participants. Rather than silently pick
one, this driver takes --cohort and reuses rsa_full_grid's own computation, so
both readings can be produced and compared.

  --cohort with-ceg   26 subjects; reproduces the published grids bit-for-bit
  --cohort no-ceg     25 subjects; matches the manuscript text
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FER_ROOT", "."))
sys.path.insert(0, str(ROOT / "analysis"))
import rsa_full_grid as R  # noqa: E402


def load_eeg(cohort):
    roi_files = sorted(p for p in R.EEG_ROI_DIR.glob("*_windowed_rdms.npy")
                       if not p.name.startswith("group"))
    ws_files = sorted(R.EEG_WS_DIR.glob("*_wholescalp_rdms.npy"))
    if cohort == "no-ceg":
        roi_files = [p for p in roi_files if not p.name.startswith("CEG_")]
        ws_files = [p for p in ws_files if not p.name.startswith("CEG_")]
    roi_subs = [p.name.replace("_windowed_rdms.npy", "") for p in roi_files]
    ws_subs = [p.name.replace("_wholescalp_rdms.npy", "") for p in ws_files]
    if roi_subs != ws_subs:
        raise SystemExit(f"ABORT: subject mismatch: "
                         f"roi-only={sorted(set(roi_subs)-set(ws_subs))} "
                         f"ws-only={sorted(set(ws_subs)-set(roi_subs))}")
    roi = np.stack([np.load(p) for p in roi_files], 0)
    ws = np.stack([np.load(p) for p in ws_files], 0)
    rois = [str(x) for x in np.load(R.EEG_ROI_DIR / "roi_order.npy", allow_pickle=True)]
    return (np.concatenate([roi, ws[:, None]], axis=1), roi_subs,
            rois + ["wholescalp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["with-ceg", "no-ceg"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    eeg, subjects, scopes = load_eeg(args.cohort)
    S, O, W = eeg.shape[0], eeg.shape[1], eeg.shape[2]
    print(f"[eeg] cohort={args.cohort}  {S} subjects x {O} scopes x {W} windows")

    model_dir, out_dir = Path(args.model_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = np.load(R.EEG_ROI_DIR / "window_bounds.npy")[:W]
    tags = args.models or sorted(p.name[:-4] for p in model_dir.glob("*_weights.npy")
                                 if not p.name.endswith(("_fear.npy", "_happy.npy",
                                                         "_neutral.npy")))
    emo_names = list(R.EMOTIONS)
    for tag in tags:
        f = model_dir / f"{tag}.npy"
        if not f.exists():
            print(f"  MISSING {tag}"); continue
        model = np.load(f)[:, :W]
        L = model.shape[0]
        grid = np.full((S, O, L, len(emo_names), W), np.nan, np.float32)
        for ei, emo in enumerate(emo_names):
            idx = R.EMOTIONS[emo]
            ev = R.upper(eeg[:, :, :, idx][:, :, :, :, idx])
            mv = R.upper(model[:, :, idx][:, :, :, idx])
            ez, e_flat = R.rank_normalise(ev)
            mz, m_flat = R.rank_normalise(mv)
            r = np.einsum("sowp,lwp->solw", ez, mz, optimize=True)
            r[np.broadcast_to(m_flat[None, None, :, :], r.shape)] = np.nan
            r[np.broadcast_to(e_flat[:, :, None, :], r.shape)] = np.nan
            grid[:, :, :, ei, :] = r.astype(np.float32)
        np.savez_compressed(out_dir / f"{tag}_rsa_grid.npz", rsa=grid,
                            subjects=np.array(subjects), scopes=np.array(scopes),
                            emotions=np.array(emo_names), n_layers=L,
                            window_bounds=wb)
        print(f"  {tag:38s} L={L:3d}")


if __name__ == "__main__":
    main()
