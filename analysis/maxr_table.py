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

The nuisance controls are one-layer RDMs (group "control"), so their rho_max is
the maximum over windows; a cell where a control is constant (identity within
one emotion of the 18 stimuli) is "undefined". --partial SET summarises the
partial-RSA grids of that control set (partial_rsa.py) for the main variants,
without ceilings (they bound the raw RSA, not partial RSA).

Outputs (analysis/paper_stats for 18 stimuli, analysis/paper_stats_54 for 54):
  rho_max.csv          one row per variant x scope x emotion (groups: main, size, step, control)
  rho_max_per_rdm.csv  one row per RDM x scope x emotion
  noise_ceilings.csv   lower and upper ceiling per scope x emotion, mean over windows
  rho_max_partial_<SET>.csv, rho_max_partial_<SET>_per_rdm.csv   with --partial
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
    ap.add_argument("--partial", default=None, help="a control set of partial_rsa.py (e.g. joint)")
    args = ap.parse_args()
    out = OUT[args.stimuli]
    out.mkdir(parents=True, exist_ok=True)
    rows_in = [v for v in variants.variants(args.stimuli) if args.partial is None or v[0] == "main"]
    missing = [t for *_, tags in rows_in for t in tags
               if not rsa_core.grid_path(args.stimuli, t, args.partial).exists()]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} grids missing, e.g. {missing[:3]}")
    eeg, _, scopes = rsa_core.load_eeg(args.stimuli)
    wb = rsa_core.window_bounds(args.stimuli)
    print(f"[eeg] {eeg.shape}; computing noise ceilings ...")
    lo, up = rsa_core.noise_ceilings(eeg, args.stimuli)              # (7, 4, W)
    if args.partial:          # the ceilings bound the raw RSA, not partial RSA: left empty
        lo, up = np.full_like(lo, np.nan), np.full_like(up, np.nan)
    scopes_x = scopes + [rsa_core.ROI_AVG]
    emos_x = list(rsa_core.EMOTIONS[args.stimuli]) + [rsa_core.EMO_AVG]

    def r5(x):
        return "" if np.isnan(x) else round(float(x), 5)

    cache, rows, per_rdm = {}, [], []
    for group, family, variant, tags in rows_in:
        cells = []
        for tag in tags:
            if tag not in cache:                                       # full-pool runs appear twice
                rsa = np.load(rsa_core.grid_path(args.stimuli, tag, args.partial))["rsa"]
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
                    "emotion": emo, "roi": sc, "max_r": r5(m),
                    "sd": r5(v.std(ddof=1)) if len(v) > 1 else "",
                    "min": r5(v.min()), "max": r5(v.max()),
                    "max_layer": single[3] if single else -1, "max_window": ki,
                    "win_start_ms": int(wb[ki][0]) if ki >= 0 else -1,
                    "win_end_ms": int(wb[ki][1]) if ki >= 0 else -1,
                    "nc_lower": r5(ncl), "nc_upper": r5(ncu),
                    "within_nc": "undefined" if np.isnan(m) else "" if np.isnan(ncl) else
                                 "ABOVE" if m > ncu else ("below" if m < ncl else "within"),
                })
                for tag, c in zip(tags, cells):
                    per_rdm.append({"group": group, "family": family, "variant": variant,
                                    "rdm": tag, "emotion": emo, "roi": sc,
                                    "max_r": r5(c[oi, ei][0]), "max_layer": c[oi, ei][3],
                                    "max_window": c[oi, ei][4], "nc_lower": r5(c[oi, ei][1]),
                                    "nc_upper": r5(c[oi, ei][2])})
        roi = [r for r in rows if r["variant"] == variant and r["family"] == family
               and r["group"] == group and r["roi"] == rsa_core.ROI_AVG and r["emotion"] == rsa_core.EMO_AVG][0]
        print(f"  {group:7s} {family:9s} {variant:13s} n={len(tags):2d}  ROI-avg rho_max "
              f"{roi['max_r']}" + (f" +/- {roi['sd']}" if roi["sd"] != "" else ""))

    stem = "rho_max" if args.partial is None else f"rho_max_partial_{args.partial}"
    tables = [(f"{stem}.csv", rows), (f"{stem}_per_rdm.csv", per_rdm)]
    if args.partial is None:
        tables.append(("noise_ceilings.csv", [
            {"scope": sc, "emotion": emo, "lower_mean": r5(np.nanmean(lo[oi, ei])),
             "upper_mean": r5(np.nanmean(up[oi, ei]))}
            for oi, sc in enumerate(scopes) for ei, emo in enumerate(rsa_core.EMOTIONS[args.stimuli])]))
    for name, table in tables:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    n_below = sum(r["within_nc"] == "below" for r in rows)
    print(f"\n[ok] {len(rows)} rows -> {out / tables[0][0]}; {len(per_rdm)} -> {tables[1][0]}")
    print(f"     {n_below} below the lower ceiling, {sum(r['within_nc'] == 'ABOVE' for r in rows)} above the upper")


if __name__ == "__main__":
    main()
