import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fer_rsa.figures import figure_style as st
from fer_rsa.figures.fig2_training import (COMPARISONS, GROUPS, SHORT_NAMES, STATS, VMAX, full_label, scalp_map, sensor_inset,
                           sensor_layout, topo_scalp)

SCOPES = ["OccipitoTemporal-L", "OccipitoTemporal-R", "Central-L", "Central-R"]
# one colour per comparison (validated as a set)
COLORS = {"H1 pretrained vs random": "#e87ba4", "H5 face vs random": "#2a78d6", "H2 face vs ImageNet": "#d3891a",
          "H4 video vs image": "#5a3d9a", "H3 fine-tuned vs start": "#008300"}
WINDOW = 3                                             # 187.5-250 ms
N_SHOWN = 8                                            # windows up to 500 ms


def load():
    """{(hypothesis, contrast, scope): rows sorted by window}."""
    out = {}
    with (STATS / "contrasts_time_joint_nc.csv").open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["hypothesis"], r["contrast"], r["scope"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: float(r["window_start_ms"]))
    return out


def series(rs, key):
    return np.array([float(r[key]) for r in rs])


def smooth(x, y, n=200):
    """PCHIP interpolation through the window values (no overshoot); display only."""
    from scipy.interpolate import PchipInterpolator
    xx = np.linspace(x[0], x[-1], n)
    return xx, PchipInterpolator(x, y)(xx)


def limits(t, scopes, contrasts):
    """y limits shared by the panels, with room for the significance bars."""
    means = [v for sc in scopes for (h, _), c in zip(COMPARISONS, contrasts)
             for v in series(t[h, c, sc][:N_SHOWN], "corrected")]
    span = max(means) - min(means)
    return max(means) + 0.4 * span, min(means) - 0.15 * span


def time_panel(ax, t, scope, contrasts, top, bottom):
    """The comparisons in one ROI; bars over windows with p < .05, one row per comparison."""
    for k, ((hyp, _), contrast) in enumerate(zip(COMPARISONS, contrasts)):
        color = COLORS[hyp]
        rs = t[hyp, contrast, scope][:N_SHOWN]
        t0, t1 = series(rs, "window_start_ms"), series(rs, "window_end_ms")
        cx = (t0 + t1) / 2
        sx, lo = smooth(cx, series(rs, "low")); _, hi = smooth(cx, series(rs, "high"))
        ax.fill_between(sx, lo, hi, color=color, alpha=0.08, lw=0)
        ax.plot(*smooth(cx, series(rs, "corrected")), color=color, lw=1.2)
        sig = series(rs, "p_maxT") < .05
        yb = top - k * (top - bottom) * 0.045
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
            ax.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=1.6, solid_capstyle="butt")
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 500); ax.set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
    ax.set_ylim(bottom, top + (top - bottom) * 0.04)
    ax.tick_params(labelsize=6, length=2, pad=1)
    st.point_ticks(ax)


WRAPPED = {"H1 pretrained vs random": "generic pretraining\nvs random", "H5 face vs random": "face recognition\nvs random",
           "H2 face vs ImageNet": "face vs object\nrecognition", "H4 video vs image": "video vs\nimage tasks",
           "H3 fine-tuned vs start": "emotion recognition\nvs base task"}


def legend_handles(wrap=False):
    handles = [Line2D([], [], color=COLORS[h], lw=1.3, label=WRAPPED[h] if wrap else n) for h, n in COMPARISONS]
    return handles + [Line2D([], [], color=st.INK2, lw=1.6, label="p < .05")]


def window_map(ax, t, hyp, contrast, scalp):
    info, roi_of = scalp
    return scalp_map(ax, [float(t[hyp, contrast, r][WINDOW]["corrected"]) for r in roi_of],
                     [float(t[hyp, contrast, r][WINDOW]["p_maxT"]) < .05 for r in roi_of], info, dot=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    t = load()
    layout, scalp = sensor_layout(), topo_scalp()
    H = 4.95
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    tops = (0.5, 1.72)
    for i, (rname, contrasts) in enumerate(GROUPS):
        top, bottom = limits(t, SCOPES, contrasts)
        gs = fig.add_gridspec(1, 4, left=0.11, right=0.99, top=f(tops[i]), bottom=f(tops[i] + 1.05), wspace=0.1)
        for j, sc in enumerate(SCOPES):
            ax = fig.add_subplot(gs[0, j])
            time_panel(ax, t, sc, contrasts, top, bottom)
            if i == 0:
                sensor_inset(ax, sc, layout)
                ax.set_title(full_label(sc), fontsize=6.6, pad=3, linespacing=1.0)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time from onset (ms)", fontsize=6.4, labelpad=1)
            if j:
                ax.tick_params(labelleft=False)
        fig.text(0.035, f(tops[i] + 0.525), rname.replace(" (", "\n("), rotation=90, ha="center", va="center",
                 fontsize=6.8, linespacing=1.0)
    fig.text(0.075, f(1.62), "difference above chance (fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.2, color=st.INK2)
    # c: maps on the left, the legend of a-b on the right
    n = len(COMPARISONS)
    hw_in, step_in, x0_in = 0.52, 0.58, 0.62
    hw = hw_in / 5.5; hh = hw_in / H; step = step_in / 5.5; x0 = x0_in / 5.5
    for i, (rname, contrasts) in enumerate(GROUPS):
        y = f(3.98 + i * 0.62)
        for k, ((hyp, _), contrast) in enumerate(zip(COMPARISONS, contrasts)):
            axm = fig.add_axes([x0 + k * step, y, hw, hh])
            im = window_map(axm, t, hyp, contrast, scalp)
            if i == 0:
                axm.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0)
            else:                                                   # model name under its map
                axm.text(0.5, -0.04, contrast.split()[0], transform=axm.transAxes, ha="center", va="top",
                         fontsize=5.8)
        fig.text(0.035, y + hh / 2, rname.replace(" (", "\n("), rotation=90, ha="center", va="center", fontsize=6.8,
                 linespacing=1.0)
    xc_end = x0 + (n - 1) * step + hw
    cax = fig.add_axes([xc_end + 0.02, f(4.6), 0.012, 1.14 / H])        # right of the maps
    cb = fig.colorbar(im, cax=cax)
    ticks = [-VMAX, 0, VMAX]
    cb.set_ticks(ticks, labels=[st.point_format(v) for v in ticks])
    cb.ax.tick_params(labelsize=5.6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference above chance\n(fraction of noise ceiling)", fontsize=5.8, labelpad=2, linespacing=1.0)
    fig.legend(handles=legend_handles(wrap=True), loc="center left", bbox_to_anchor=(xc_end + 0.13, f(4.03)), ncol=1,
               fontsize=6.2, frameon=False, handlelength=1.8, handletextpad=0.5, labelspacing=0.5)
    for letter, y in (("a", 0.05), ("b", 1.6), ("c", 3.06)):
        fig.text(0.01, f(y), letter, fontsize=9, fontweight="bold", va="top")
    fig.text(x0 + (xc_end - x0) / 2, f(3.08), "187–250 ms", fontsize=7, ha="center", va="top")
    st.save(fig, args.out)
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
