#!/usr/bin/env python3
"""Time-locked RSA of the control RDMs with the EEG, and the noise-ceiling table.

Controls (identity, graded identity, pixel difference, motion energy, optical
flow) are static n x n RDMs without a frame dimension, so the same control RDM
is correlated with each EEG window; max_r is the maximum over windows of the
participant-mean Spearman (the one-layer case of the model rho_max), reported
with the noise ceilings at that window.

Within one emotion of the 18-stimulus set the six stimuli are six different
actors, so the identity RDM is constant there and its Spearman undefined: those
cells are written as "undefined" rather than 0.

Outputs (--stimuli 18 | 54):
  analysis/motion_controls[_54]/results/timelocked_control_rsa[_54].csv
  analysis/paper_stats[_54]/noise_ceiling_16win.csv  (ceilings averaged over windows)
"""
import argparse
import csv

import numpy as np

import rsa_core
from rsa_core import upper, spearman

ROOT = rsa_core.ROOT
CONTROLS = ["identity", "identity_graded", "pixel_diff", "motion_energy", "optical_flow"]
PATHS = {
    "18": ("analysis/motion_controls/rdms", "analysis/motion_controls/results/timelocked_control_rsa.csv",
           "analysis/paper_stats/noise_ceiling_16win.csv"),
    "54": ("analysis/motion_controls_54/rdms",
           "analysis/motion_controls_54/results/timelocked_control_rsa_54.csv",
           "analysis/paper_stats_54/noise_ceiling_16win.csv"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    args = ap.parse_args()
    rdm_dir, out_path, nc_path = (ROOT / p for p in PATHS[args.stimuli])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    eeg, subjects, scopes = rsa_core.load_eeg(args.stimuli)
    S, W = eeg.shape[0], eeg.shape[2]
    wb = rsa_core.window_bounds(args.stimuli)
    controls = {}
    for c in CONTROLS:
        rdm = np.load(rdm_dir / f"{c}_rdm.npy")
        if rdm.shape != (len(rsa_core.MODEL_STIMULI[args.stimuli]),) * 2:
            raise SystemExit(f"ABORT {c}: shape {rdm.shape}, expected one row per stimulus")
        controls[c] = rsa_core.select_stimuli(rdm, args.stimuli)
    print(f"[eeg] {eeg.shape}; controls {list(controls)}")

    nc_lower, nc_upper = rsa_core.noise_ceilings(eeg, args.stimuli)   # (7, 4, W)
    nc_rows, rows = [], []
    for oi, sc in enumerate(scopes):
        for ei, (emo, idx) in enumerate(rsa_core.EMOTIONS[args.stimuli].items()):
            ev = upper(eeg[:, oi][..., idx, :][..., idx])               # (S, W, P)
            lo, up = nc_lower[oi, ei], nc_upper[oi, ei]
            nc_rows.append({"scope": sc, "roi": sc, "emotion": emo,
                            "lower_mean": round(float(np.nanmean(lo)), 5),
                            "upper_mean": round(float(np.nanmean(up)), 5)})
            for c, rdm in controls.items():
                cv = upper(rdm[np.ix_(idx, idx)])
                per_win = np.array([np.nanmean([spearman(cv, ev[s, k]) for s in range(S)])
                                    for k in range(W)])
                if np.all(np.isnan(per_win)):
                    rows.append({"control": c, "emotion": emo, "roi": sc, "scope": sc,
                                 "max_r": "", "max_window": "", "win_start_ms": "",
                                 "win_end_ms": "", "nc_lower": "", "nc_upper": "",
                                 "within_nc": "undefined"})
                    continue
                ki = int(np.nanargmax(per_win)); v = float(per_win[ki])
                rows.append({
                    "control": c, "emotion": emo, "roi": sc, "scope": sc,
                    "max_r": round(v, 5), "max_window": ki,
                    "win_start_ms": int(wb[ki][0]), "win_end_ms": int(wb[ki][1]),
                    "nc_lower": round(float(lo[ki]), 5), "nc_upper": round(float(up[ki]), 5),
                    "within_nc": "ABOVE" if v > up[ki] else ("below" if v < lo[ki] else "within"),
                })
        print(f"  {sc} done", flush=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with nc_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(nc_rows[0]))
        w.writeheader(); w.writerows(nc_rows)
    n_undef = sum(r["within_nc"] == "undefined" for r in rows)
    print(f"[ok] {len(rows)} control rows ({n_undef} undefined) -> {out_path}")
    print(f"[ok] {len(nc_rows)} noise-ceiling rows -> {nc_path}")


if __name__ == "__main__":
    main()
