#!/usr/bin/env python3
"""rho_max of every model variant, with noise ceilings (--stimuli 18 | 54).

rho_max = maximum over (layer x window) of the participant-mean time-locked
Spearman, taken separately in each scope x emotion cell (rsa_core.cell_maxima).
Scopes: 6 ROIs, whole scalp, ROI-avg (mean of the six ROI values); emotions:
all, fear, happy, neutral, emo-avg (mean of the three emotion values). The noise
ceilings are read at the window of each maximum and averaged like the values.

A variant with ten RDMs (random seeds or training runs, analysis/variants.py)
gets rho_max for each RDM and then the mean, SD and range over the ten; the
selection is never made after averaging grids, so a random seed, a training run
and a released model go through the same maximum.

Outputs (analysis/paper_stats for 18 stimuli, analysis/paper_stats_54 for 54):
  rho_max.csv          one row per variant x scope x emotion (group: main, size, step)
  rho_max_per_rdm.csv  one row per RDM x scope x emotion
"""
import argparse
import csv

import numpy as np

import rsa_core
import variants

OUT = {"18": rsa_core.ROOT / "analysis/paper_stats", "54": rsa_core.ROOT / "analysis/paper_stats_54"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    args = ap.parse_args()
    out = OUT[args.stimuli]
    out.mkdir(parents=True, exist_ok=True)
    missing = [t for t in variants.all_rdms(args.stimuli)
               if not rsa_core.grid_path(args.stimuli, t).exists()]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} grids missing, e.g. {missing[:3]}")
    eeg, _, scopes = rsa_core.load_eeg(args.stimuli)
    wb = rsa_core.window_bounds(args.stimuli)
    print(f"[eeg] {eeg.shape}; computing noise ceilings ...")
    lo, up = rsa_core.noise_ceilings(eeg, args.stimuli)              # (7, 4, W)
    scopes_x = scopes + [rsa_core.ROI_AVG]
    emos_x = list(rsa_core.EMOTIONS[args.stimuli]) + [rsa_core.EMO_AVG]

    cache, rows, per_rdm = {}, [], []
    for group, family, variant, tags in variants.variants(args.stimuli):
        cells = []
        for tag in tags:
            if tag not in cache:                                       # full-pool runs appear twice
                rsa = np.load(rsa_core.grid_path(args.stimuli, tag))["rsa"]
                cache[tag] = rsa_core.averaged_cells(rsa_core.cell_maxima(rsa, lo, up))
            cells.append(cache[tag])
        for oi, sc in enumerate(scopes_x):
            for ei, emo in enumerate(emos_x):
                v = np.array([c[oi, ei][0] for c in cells])
                ncl = float(np.mean([c[oi, ei][1] for c in cells]))
                ncu = float(np.mean([c[oi, ei][2] for c in cells]))
                single = cells[0][oi, ei] if len(cells) == 1 else None
                ki = single[4] if single else -1
                m = float(v.mean())
                rows.append({
                    "group": group, "family": family, "variant": variant, "n": len(tags),
                    "emotion": emo, "roi": sc, "max_r": round(m, 5),
                    "sd": round(float(v.std(ddof=1)), 5) if len(v) > 1 else "",
                    "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                    "max_layer": single[3] if single else -1, "max_window": ki,
                    "win_start_ms": int(wb[ki][0]) if ki >= 0 else -1,
                    "win_end_ms": int(wb[ki][1]) if ki >= 0 else -1,
                    "nc_lower": round(ncl, 5), "nc_upper": round(ncu, 5),
                    "within_nc": "ABOVE" if m > ncu else ("below" if m < ncl else "within"),
                })
                for tag, c in zip(tags, cells):
                    per_rdm.append({"group": group, "family": family, "variant": variant,
                                    "rdm": tag, "emotion": emo, "roi": sc,
                                    "max_r": round(c[oi, ei][0], 5), "max_layer": c[oi, ei][3],
                                    "max_window": c[oi, ei][4], "nc_lower": round(c[oi, ei][1], 5),
                                    "nc_upper": round(c[oi, ei][2], 5)})
        roi = [r for r in rows if r["variant"] == variant and r["family"] == family
               and r["group"] == group and r["roi"] == rsa_core.ROI_AVG and r["emotion"] == rsa_core.EMO_AVG][0]
        print(f"  {group:4s} {family:9s} {variant:13s} n={len(tags):2d}  ROI-avg rho_max "
              f"{roi['max_r']:.4f}" + (f" +/- {roi['sd']:.4f}" if roi["sd"] != "" else ""))

    for name, table in (("rho_max.csv", rows), ("rho_max_per_rdm.csv", per_rdm)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    n_below = sum(r["within_nc"] == "below" for r in rows)
    print(f"\n[ok] {len(rows)} rows -> {out / 'rho_max.csv'}; {len(per_rdm)} -> rho_max_per_rdm.csv")
    print(f"     {n_below} below the lower ceiling, {sum(r['within_nc'] == 'ABOVE' for r in rows)} above the upper")


if __name__ == "__main__":
    main()
