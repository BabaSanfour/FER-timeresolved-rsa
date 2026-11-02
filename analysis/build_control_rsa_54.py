#!/usr/bin/env python3
"""Time-locked RSA between five 54-stimulus control RDMs and complete EEG RDMs.

EEG in the model stimulus order; participants with a missing video use the pairs
they have, in the control RSA and in the noise ceilings (see eeg54.py).
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import eeg54

ROOT = Path(".")
EEG = ROOT / "eeg_rdms_complete"
CTRL = ROOT / "analysis/motion_controls_54/rdms"
OUT = ROOT / "analysis/motion_controls_54/results/timelocked_control_rsa_54.csv"
ROIS = [str(x) for x in np.load(EEG / "roi_order.npy", allow_pickle=True)]
EMOTIONS = ["all", "fear", "happy", "neutral"]
CONTROLS = ["identity", "identity_graded", "pixel_diff", "motion_energy", "optical_flow"]
SLICES = {"all": np.arange(54), "fear": np.arange(18),
          "happy": np.arange(18, 36), "neutral": np.arange(36, 54)}


def upper(x):
    ij = np.triu_indices(x.shape[-1], 1)
    return x[..., ij[0], ij[1]]


def spear(x, y):
    """Spearman on the pairs finite in both (a participant's available stimuli)."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = rankdata(x[keep]), rankdata(y[keep])
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subs = eeg54.subjects(EEG)
    eeg_by_emotion = {emotion: eeg54.load(subs, emotion, EEG) for emotion in EMOTIONS}
    controls = {c: np.load(CTRL / f"{c}_rdm.npy") for c in CONTROLS}
    rows = []
    for oi, scope in enumerate(ROIS + ["wholescalp"]):
        for emotion in EMOTIONS:
            eeg = eeg_by_emotion[emotion][:, oi]  # subject, window, n, n
            ev = upper(eeg)
            group = np.nanmean(ev, 0)            # per pair, over the participants who have it
            total = np.nansum(ev, 0)
            count = np.isfinite(ev).sum(0).astype(ev.dtype)
            lower, upper_nc = [], []
            for w in range(ev.shape[1]):
                upper_nc.append(np.nanmean([spear(ev[s, w], group[w]) for s in range(len(subs))]))
                lower.append(np.nanmean([
                    spear(ev[s, w], (total[w] - ev[s, w]) / (count[w] - 1))
                    for s in range(len(subs))
                ]))
            idx = SLICES[emotion]
            for control, full_rdm in controls.items():
                cv = upper(full_rdm[np.ix_(idx, idx)])
                per_window = np.asarray([
                    np.nanmean([spear(cv, ev[s, w]) for s in range(len(subs))])
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
