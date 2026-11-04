#!/usr/bin/env python3
"""Fig. 3: when the training effects appear (54 videos, noise-ceiling scale, all controls removed).

Every value is the difference above chance as a fraction of the upper EEG noise ceiling (contrasts.py
--normalize), window by window with the layer selected in each window, after removing identity, occlusion,
pixel, optical-flow and motion-energy structure jointly. Statistics are over participants (n = 25):
permutation null with the same selection, 95% bootstrap CI, p corrected across the 11 x 13 ROI-window cells
(maxT). Families are combined within each participant's value; the family markers in b are descriptive.

  a     the five comparisons of Fig. 2, pooled over families, in the two occipito-temporal and the two central
        ROIs, 0-500 ms (curves interpolated between window centres); bars over windows with p < .05
  b     the same comparisons in the 187-250 ms window: pooled estimate with 95% CI (bar), each family
        (marker), * p < .05
  c     the pooled comparisons over the scalp in the 187-250 ms window; dots on ROIs with p < .05

Reads analysis/paper_stats_54/contrasts_time_joint_nc.csv; writes analysis/paper_stats_54/fig3_time_course.{pdf,png}.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figure_style as st
from fig2_training import (BEST, COMPARISONS, DIVERGING, REGION_COLORS, SHORT_NAMES, STATS, full_label,
                           region_of, sensor_inset, sensor_layout, topo_scalp)

SCOPES = ["OccipitoTemporal-L", "OccipitoTemporal-R", "Central-L", "Central-R"]
# one colour per comparison (validated as a set, all pairs, light surface); line style as a second cue
COLORS = {"H1 pretrained vs random": "#e87ba4", "H5 face vs random": "#2a78d6", "H2 face vs ImageNet": "#d3891a",
          "H4 video vs image": "#5a3d9a", "H3 fine-tuned vs start": "#008300"}
LINE_STYLE = {"H1 pretrained vs random": "-", "H5 face vs random": "-", "H2 face vs ImageNet": (0, (3, 1.5)),
              "H4 video vs image": (0, (3, 1.5)), "H3 fine-tuned vs start": "-"}
WINDOW = 3                                             # 187.5-250 ms
N_SHOWN = 8                                            # windows up to 500 ms
VMAX = 0.2


def load(name="joint"):
    """{(hypothesis, contrast, scope): rows sorted by window}."""
    out = {}
    with (STATS / f"contrasts_time{'_' + name if name else ''}_nc.csv").open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["hypothesis"], r["contrast"], r["scope"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: float(r["window_start_ms"]))
    return out


def series(rs, key):
    return np.array([float(r[key]) for r in rs])


def smooth(x, y, n=200):
    """Shape-preserving cubic interpolation through the window values (display only; PCHIP does not overshoot)."""
    from scipy.interpolate import PchipInterpolator
    xx = np.linspace(x[0], x[-1], n)
    return xx, PchipInterpolator(x, y)(xx)


def time_panel(ax, t, scope, contrasts, top, bottom):
    """The five comparisons (contrast per comparison: pooled or the best model) in one ROI; bars above the traces
    over windows with p < .05, one row per comparison."""
    for k, ((hyp, _, _), contrast) in enumerate(zip(COMPARISONS, contrasts)):
        color, ls = COLORS[hyp], LINE_STYLE[hyp]
        rs = t[hyp, contrast, scope][:N_SHOWN]
        t0, t1 = series(rs, "window_start_ms"), series(rs, "window_end_ms")
        cx = (t0 + t1) / 2
        sx, lo = smooth(cx, series(rs, "low")); _, hi = smooth(cx, series(rs, "high"))
        ax.fill_between(sx, lo, hi, color=color, alpha=0.08, lw=0)
        ax.plot(*smooth(cx, series(rs, "corrected")), color=color, lw=1.2, ls=ls)
        sig = series(rs, "p_maxT") < .05
        yb = top - k * (top - bottom) * 0.045
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
            ax.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=1.6, solid_capstyle="butt")
    ax.axvspan(187.5, 250, color=st.RULE, alpha=0.4, lw=0, zorder=0)
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 500); ax.set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
    ax.set_ylim(bottom, top + (top - bottom) * 0.04)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def window_map(ax, t, hyp, contrast, scalp):
    import mne
    info, roi_of = scalp
    vals = np.array([float(t[hyp, contrast, r][WINDOW]["corrected"]) for r in roi_of])
    sig = np.array([float(t[hyp, contrast, r][WINDOW]["p_maxT"]) < .05 for r in roi_of])
    im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                 sensors=False, extrapolate="head", mask=sig,
                                 mask_params=dict(marker="o", markersize=1.0, markerfacecolor=st.INK,
                                                  markeredgecolor="none"))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    t = load("joint")
    layout, scalp = sensor_layout(), topo_scalp()
    rows = [("all models (pooled)", ["pooled"] * len(COMPARISONS)),
            ("best model", [BEST[h] for h, _, _ in COMPARISONS])]
    H = 5.85
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    tops = (0.5, 2.15)
    for i, (rname, contrasts) in enumerate(rows):
        vals = [v for sc in SCOPES for (h, _, _), c in zip(COMPARISONS, contrasts) for k in ("low", "high")
                for v in series(t[h, c, sc][:N_SHOWN], k)]
        top, bottom = max(vals) + 0.22, min(vals) - 0.03
        gs = fig.add_gridspec(1, 4, left=0.11, right=0.99, top=f(tops[i]), bottom=f(tops[i] + 1.2), wspace=0.1)
        for j, sc in enumerate(SCOPES):
            ax = fig.add_subplot(gs[0, j])
            time_panel(ax, t, sc, contrasts, top, bottom)
            sensor_inset(ax, sc, layout) if i == 0 else None
            if i == 0:
                ax.set_title(full_label(sc), fontsize=6.6, pad=3, linespacing=1.0, color=REGION_COLORS[region_of(sc)])
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time from onset (ms)", fontsize=6.4, labelpad=1)
            if j:
                ax.tick_params(labelleft=False)
        fig.text(0.035, f(tops[i] + 0.6), rname.replace(" (", "\n("), rotation=90, ha="center", va="center",
                 fontsize=6.8, linespacing=1.0)
    fig.text(0.075, f(1.75), "difference above chance (fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.2, color=st.INK2)
    handles = [Line2D([], [], color=COLORS[h], lw=1.3, ls=LINE_STYLE[h], label=n) for h, n, _ in COMPARISONS]
    handles.append(Line2D([], [], color=st.INK2, lw=1.6, label="p < .05"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.55, f(3.82)), ncol=3, fontsize=6.2, frameon=False,
               handlelength=2.0, columnspacing=1.2)
    # c: maps in the window, pooled and best model
    n = len(COMPARISONS)
    hw_in = 0.6; hw = hw_in / 5.5; hh = hw_in / H; x0 = 0.11; step = (0.92 - x0) / n
    for i, (rname, contrasts) in enumerate(rows):
        y = f(5.02 + i * 0.7)
        for k, ((hyp, name, _), contrast) in enumerate(zip(COMPARISONS, contrasts)):
            axm = fig.add_axes([x0 + k * step + (step - hw) / 2, y, hw, hh])
            im = window_map(axm, t, hyp, contrast, scalp)
            if i == 0:
                axm.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0, color=COLORS[hyp])
            else:
                axm.text(1.02, 0.5, contrast.split()[0], transform=axm.transAxes, rotation=90, ha="left",
                         va="center", fontsize=5.8, color=st.INK2)
        fig.text(0.035, y + hh / 2, rname.replace(" (", "\n("), rotation=90, ha="center", va="center", fontsize=6.8,
                 linespacing=1.0)
    cax = fig.add_axes([0.94, f(5.7), 0.012, 1.25 / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5.6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    for letter, y in (("a", 0.05), ("b", 1.95), ("c", 4.08)):
        fig.text(0.01, f(y), letter, fontsize=9, fontweight="bold", va="top")
    fig.text(0.04, f(4.1), "187–250 ms", fontsize=6.8, va="top", color=st.INK2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
