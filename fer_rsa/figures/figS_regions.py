import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fer_rsa.figures import figure_style as st
from fer_rsa.figures.fig2_training import (GROUPS, STATS, by_variant, full_label, legend_block, regime_panel, rows, sensor_inset,
                           sensor_layout)
from fer_rsa.figures.fig3_time_course import legend_handles, limits, load, time_panel

REGIME_SCOPES = ["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "CentroParietal-L",
                 "CentroParietal-M", "CentroParietal-R"]
TIME_SCOPES = ["Frontal-L", "Frontal-M", "Frontal-R", "OccipitoTemporal-M", "CentroParietal-L", "CentroParietal-M",
               "CentroParietal-R", "wholescalp"]


def regimes(out):
    raw, joint = by_variant(rows("")), by_variant(rows("joint"))
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
    st.save(fig, out)


def times(out):
    t = load()
    layout = sensor_layout()
    H = 7.2
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(4, 4, left=0.13, right=0.99, top=0.93, bottom=0.12, wspace=0.1, hspace=0.45)
    for g, (gname, contrasts) in enumerate(GROUPS):
        top, bottom = limits(t, TIME_SCOPES, contrasts)
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
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.55, 0.0))
    st.save(fig, out)


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
