#!/usr/bin/env python3
"""Supplementary figure: the 18 unoccluded videos.

The five planned comparisons, pooled over families, at their peak over layers and windows, before and after
removing all nuisance RDMs, over the scalp (fraction of the noise ceiling above chance; dots, p < .05 after
correction across the 11 ROIs).

Reads analysis/paper_stats/contrasts{,_joint}_nc.csv (contrasts.py --stimuli 18 --normalize [--partial joint]);
writes <out>.pdf and <out>.png.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

import figure_style as st
import rsa_core
from fig2_training import COMPARISONS, DIVERGING, SHORT_NAMES, topo_scalp

STATS18 = rsa_core.ROOT / "analysis/paper_stats"
VMAX = 0.2


def pooled(name):
    with (STATS18 / f"contrasts{name}_nc.csv").open() as fh:
        return {(r["hypothesis"], r["scope"]): r for r in csv.DictReader(fh) if r["contrast"] == "pooled"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS18 / "figS_18"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    info, roi_of = topo_scalp()
    H = 1.9
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    x0 = 0.16; step = (0.9 - x0) / len(COMPARISONS); hw_in = 0.6; hw = hw_in / 5.5; hh = hw_in / H
    for i, (name, label) in enumerate((("", "raw"), ("_joint", "controls\nremoved"))):
        t = pooled(name)
        y = 0.42 + i * 0.72
        for k, (hyp, _, _) in enumerate(COMPARISONS):
            ax = fig.add_axes([x0 + k * step + (step - hw) / 2, f(y + hw_in), hw, hh])
            vals = np.array([float(t[hyp, r]["corrected"]) for r in roi_of])
            sig = np.array([float(t[hyp, r]["p_maxT"]) < .05 for r in roi_of])
            im = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                      sensors=False, extrapolate="head", mask=sig,
                                      mask_params=dict(marker="o", markersize=1.3, markerfacecolor=st.INK,
                                                       markeredgecolor="none"))[0]
            if i == 0:
                ax.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0)
        fig.text(x0 - 0.02, f(y + hw_in / 2), label, ha="right", va="center", fontsize=6.6, linespacing=1.0)
    cax = fig.add_axes([0.89, f(1.72), 0.012, 1.3 / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.set_label("difference above chance\n(fraction of noise ceiling)", fontsize=5.4, labelpad=2, linespacing=1.0)
    st.black_text(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
