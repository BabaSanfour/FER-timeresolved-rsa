#!/usr/bin/env python3
"""Time-locked RSA for the low-level control RDMs, + the noise-ceiling table.

Controls (identity / pixel_diff / motion_energy / optical_flow) are STATIC 18x18
RDMs - they have no frame dimension - so the same control RDM is correlated with
each of the 16 EEG windows, and max_r is the max over those 16 windows of the
subject-mean Spearman. That is the L=1 case of the model convention in
build_maxr_table.py, so the two are directly comparable.

Note: within a single emotion the 6 stimuli are 6 DIFFERENT actors, so the
identity RDM's 6x6 block is constant and its Spearman is undefined. Those cells
are emitted as NaN rather than 0.

Outputs:
  analysis/motion_controls/results/timelocked_control_rsa.csv
  analysis/paper_stats/noise_ceiling_16win.csv
"""
import csv
import os
from pathlib import Path

import numpy as np

import rsa_core
from rsa_core import upper, spearman

ROOT = Path(os.environ.get("FER_ROOT", "."))
RDM_DIR = ROOT / "analysis/motion_controls/rdms"
RES_DIR = ROOT / "analysis/motion_controls/results"
NC_OUT = ROOT / "analysis/paper_stats/noise_ceiling_16win.csv"

EMOTIONS = rsa_core.EMOTIONS["18"]
CONTROLS = ["identity", "identity_graded", "pixel_diff", "motion_energy", "optical_flow"]


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    NC_OUT.parent.mkdir(parents=True, exist_ok=True)

    eeg, _, scopes = rsa_core.load_eeg("18")
    S, W = eeg.shape[0], eeg.shape[2]
    wb = rsa_core.window_bounds("18")
    print(f"[eeg] {eeg.shape} scopes={scopes}")

    ctrl_rdms = {}
    for c in CONTROLS:
        p = RDM_DIR / f"{c}_rdm.npy"
        if not p.exists():
            print(f"  [skip] {c}: {p} missing"); continue
        r = np.load(p)
        if r.shape != (18, 18):
            print(f"  [skip] {c}: shape {r.shape} != (18,18)"); continue
        ctrl_rdms[c] = r
    print(f"[controls] {list(ctrl_rdms)}")

    nc_lower, nc_upper = rsa_core.noise_ceilings(eeg, "18")     # (7, 4, W)
    nc_rows, rows = [], []
    for oi, sc in enumerate(scopes):
        for ei, (emo, idx) in enumerate(EMOTIONS.items()):
            ev = upper(eeg[:, oi][:, :, idx][:, :, :, idx])         # (S, W, P)
            lo, up = nc_lower[oi, ei], nc_upper[oi, ei]
            nc_rows.append({"scope": sc, "roi": sc, "emotion": emo,
                            "lower_mean": round(float(np.nanmean(lo)), 5),
                            "upper_mean": round(float(np.nanmean(up)), 5)})

            for c, rdm in ctrl_rdms.items():
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
        print(f"  {sc} done")

    with (RES_DIR / "timelocked_control_rsa.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["control", "emotion", "roi", "scope", "max_r",
                                          "max_window", "win_start_ms", "win_end_ms",
                                          "nc_lower", "nc_upper", "within_nc"])
        w.writeheader(); w.writerows(rows)
    with NC_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "roi", "emotion", "lower_mean", "upper_mean"])
        w.writeheader(); w.writerows(nc_rows)

    n_undef = sum(r["within_nc"] == "undefined" for r in rows)
    print(f"\n[ok] {len(rows)} control rows -> {RES_DIR / 'timelocked_control_rsa.csv'}"
          f"  ({n_undef} undefined)")
    print(f"[ok] {len(nc_rows)} nc rows -> {NC_OUT}")


if __name__ == "__main__":
    main()
