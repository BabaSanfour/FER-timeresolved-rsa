#!/usr/bin/env python3
"""Time-locked RSA between five 54-stimulus control RDMs and complete EEG RDMs.

EEG in the model stimulus order; participants with a missing video use the pairs
they have, in the control RSA and in the noise ceilings (see rsa_core.py).
"""
import csv

import numpy as np

import rsa_core
from rsa_core import upper, spearman

ROOT = rsa_core.ROOT
CTRL = ROOT / "analysis/motion_controls_54/rdms"
OUT = ROOT / "analysis/motion_controls_54/results/timelocked_control_rsa_54.csv"
EMOTIONS = ["all", "fear", "happy", "neutral"]
CONTROLS = ["identity", "identity_graded", "pixel_diff", "motion_energy", "optical_flow"]
SLICES = rsa_core.EMOTIONS["54"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    eeg_by_emotion = {emotion: rsa_core.load_eeg("54", emotion)[0] for emotion in EMOTIONS}
    subs, scopes = rsa_core.subjects("54"), rsa_core.scopes("54")
    controls = {c: np.load(CTRL / f"{c}_rdm.npy") for c in CONTROLS}
    rows = []
    for oi, scope in enumerate(scopes):
        for emotion in EMOTIONS:
            eeg = eeg_by_emotion[emotion][:, oi]  # subject, window, n, n
            ev = upper(eeg)
            group = np.nanmean(ev, 0)            # per pair, over the participants who have it
            total = np.nansum(ev, 0)
            count = np.isfinite(ev).sum(0).astype(ev.dtype)
            lower, upper_nc = [], []
            for w in range(ev.shape[1]):
                upper_nc.append(np.nanmean([spearman(ev[s, w], group[w]) for s in range(len(subs))]))
                lower.append(np.nanmean([
                    spearman(ev[s, w], (total[w] - ev[s, w]) / (count[w] - 1))
                    for s in range(len(subs))
                ]))
            idx = SLICES[emotion]
            for control, full_rdm in controls.items():
                cv = upper(full_rdm[np.ix_(idx, idx)])
                per_window = np.asarray([
                    np.nanmean([spearman(cv, ev[s, w]) for s in range(len(subs))])
                    for w in range(ev.shape[1])
                ])
                if np.all(np.isnan(per_window)):
                    best = win = np.nan
                else:
                    win = int(np.nanargmax(per_window))
                    best = float(per_window[win])
                rows.append({
                    "control": control, "emotion": emotion, "roi": scope,
                    "max_r": best, "max_window": win,
                    "nc_lower": np.nan if np.isnan(win) else lower[win],
                    "nc_upper": np.nan if np.isnan(win) else upper_nc[win],
                })
        print(f"[ok] {scope}", flush=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
