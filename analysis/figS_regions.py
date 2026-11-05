#!/usr/bin/env python3
"""Supplementary figures completing Figs. 2a and 3a for the scalp regions not shown there (54 videos,
noise-ceiling scale, as in the main figures).

  S-regimes   Fig. 2a for the frontal, central and centro-parietal ROIs: rho_max by training regime, raw (filled)
              and after all controls (open), family markers, chance dashes
  S-time      Fig. 3a-b for the frontal and centro-parietal ROIs, occipito-temporal midline and whole scalp: the
              five comparisons after all controls, pooled and best model, bars over windows with p < .05

Reads analysis/paper_stats_54/contrasts{,_joint}_nc.csv and contrasts_time_joint_nc.csv; writes
analysis/paper_stats_54/figS_regimes.{pdf,png} and figS_time.{pdf,png}.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figure_style as st
from fig2_training import (BEST, COMPARISONS, STATS, full_label, legend_block, regime_panel, rows, sensor_inset,
                           sensor_layout)
from fig3_time_course import COLORS, LINE_STYLE, N_SHOWN, load, series, time_panel

REGIME_SCOPES = ["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "CentroParietal-L",
                 "CentroParietal-M", "CentroParietal-R"]
TIME_SCOPES = ["Frontal-L", "Frontal-M", "Frontal-R", "OccipitoTemporal-M", "CentroParietal-L", "CentroParietal-M",
               "CentroParietal-R", "wholescalp"]


def regimes(out):
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    raw, joint = variant(rows("")), variant(rows("joint"))
    H = 4.4
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(2, 4, left=0.1, right=0.99, top=0.92, bottom=0.3, wspace=0.1, hspace=0.75)
    layout = sensor_layout()
    for k, sc in enumerate(REGIME_SCOPES):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        regime_panel(ax, sc, raw, joint)
        sensor_inset(ax, sc, layout)
        if k % 4:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\rho_{\max}$ (fraction of noise ceiling)", fontsize=6.4)
    legend_block(fig, H)
    st.black_text(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")


def times(out):
    t = load("joint")
    layout = sensor_layout()
    groups = [("all models (pooled)", ["pooled"] * len(COMPARISONS)), ("best model", [BEST[h] for h, _, _ in COMPARISONS])]
    H = 7.2
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(4, 4, left=0.13, right=0.99, top=0.93, bottom=0.12, wspace=0.1, hspace=0.45)
    for g, (gname, contrasts) in enumerate(groups):
        means = [v for sc in TIME_SCOPES for (h, _, _), c in zip(COMPARISONS, contrasts)
                 for v in series(t[h, c, sc][:N_SHOWN], "corrected")]
        span = max(means) - min(means)
        top, bottom = max(means) + 0.4 * span, min(means) - 0.15 * span
        for k, sc in enumerate(TIME_SCOPES):
            r = 2 * g + k // 4
            ax = fig.add_subplot(gs[r, k % 4])
            time_panel(ax, t, sc, contrasts, top, bottom)
            sensor_inset(ax, sc, layout)
            ax.set_title(full_label(sc), fontsize=6.4, pad=2, linespacing=1.0)
            if k % 4:
                ax.tick_params(labelleft=False)
            if r < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time from onset (ms)", fontsize=6.4, labelpad=1)
        y1 = gs[2 * g, 0].get_position(fig).y1; y0 = gs[2 * g + 1, 0].get_position(fig).y0
        fig.text(0.01, y1 + 0.03, "ab"[g], fontsize=9, fontweight="bold", va="bottom")
        fig.text(0.015, (y0 + y1) / 2, gname.replace(" (", "\n("), rotation=90, ha="center", va="center", fontsize=6.8,
                 linespacing=1.0)
    fig.text(0.055, 0.53, "difference above chance (fraction of noise ceiling)", rotation=90, ha="center", va="center",
             fontsize=6.4, color=st.INK2)
    handles = [Line2D([], [], color=COLORS[h], lw=1.3, ls=LINE_STYLE[h], label=n) for h, n, _ in COMPARISONS]
    handles.append(Line2D([], [], color=st.INK2, lw=1.6, label="p < .05"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.55, 0.0))
    st.black_text(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS), help="output folder")
    args = ap.parse_args()
    st.setup()
    regimes(f"{args.out}/figS_regimes")
    times(f"{args.out}/figS_time")
    print(f"[ok] -> {args.out}/figS_regimes, figS_time")


if __name__ == "__main__":
    main()
