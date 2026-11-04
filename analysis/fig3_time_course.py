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
from matplotlib.patches import Patch

import figure_style as st
from fig2_training import (COMPARISONS, DIVERGING, REGION_COLORS, SHORT_NAMES, STATS, full_label, region_of,
                           sensor_inset, sensor_layout, topo_scalp)

SCOPES = ["OccipitoTemporal-L", "OccipitoTemporal-R", "Central-L", "Central-R"]
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


def time_panel(ax, t, scope, top, bottom):
    for k, (hyp, _, regime) in enumerate(COMPARISONS):
        color, ls = st.REGIME_COLORS[regime], LINE_STYLE[hyp]
        rs = t[hyp, "pooled", scope][:N_SHOWN]
        t0, t1 = series(rs, "window_start_ms"), series(rs, "window_end_ms")
        cx = (t0 + t1) / 2
        sx, lo = smooth(cx, series(rs, "low")); _, hi = smooth(cx, series(rs, "high"))
        ax.fill_between(sx, lo, hi, color=color, alpha=0.08, lw=0)
        ax.plot(*smooth(cx, series(rs, "corrected")), color=color, lw=1.2, ls=ls)
        p, m = series(rs, "p_maxT"), series(rs, "corrected")
        for sign, yb in ((1, top - k * 0.032), (-1, bottom + k * 0.032)):
            sig = (p < .05) & (np.sign(m) == sign)
            for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
                ax.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=1.6, ls=ls, solid_capstyle="butt")
    ax.axvspan(187.5, 250, color=st.RULE, alpha=0.4, lw=0, zorder=0)
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 500); ax.set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
    ax.set_ylim(bottom - 0.03, top + 0.03)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def window_panel(ax, t, scope, lim):
    """The window: pooled estimate (bar, 95% CI) and each family (marker) per comparison."""
    families = {}
    for (hyp, contrast, sc), rs in t.items():
        if sc == scope and contrast != "pooled":
            families.setdefault(hyp, []).append((contrast, float(rs[WINDOW]["corrected"])))
    for x, (hyp, _, regime) in enumerate(COMPARISONS):
        color = st.REGIME_COLORS[regime]
        r = t[hyp, "pooled", scope][WINDOW]
        m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
        ax.bar(x, m, width=0.62, color=color, alpha=0.3, lw=0, zorder=1)
        ax.plot([x, x], [lo, hi], color=color, lw=1.0, zorder=2)
        fams = sorted(families.get(hyp, []))
        for k, (contrast, v) in enumerate(fams):
            mk = st.MARKERS[contrast.split()[0]]
            ax.plot(x + (k - (len(fams) - 1) / 2) * 0.1, v, mk, ms=st.size(mk) - 1.8, mfc="white", mec=color,
                    mew=0.6, zorder=3)
        if float(r["p_maxT"]) < .05:
            ax.text(x, lim * 0.93, "*", ha="center", va="center", fontsize=8, color=st.INK)
    ax.axhline(0, color=st.INK2, lw=0.6)
    ax.set_xticks([])                                    # comparisons in the order and colours of the legend
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1)
    ax.set_xlim(-0.6, len(COMPARISONS) - 0.4); ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def window_map(ax, t, hyp, scalp):
    import mne
    info, roi_of = scalp
    vals = np.array([float(t[hyp, "pooled", r][WINDOW]["corrected"]) for r in roi_of])
    sig = np.array([float(t[hyp, "pooled", r][WINDOW]["p_maxT"]) < .05 for r in roi_of])
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
    H = 5.45
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))

    # a: time courses
    vals = [v for sc in SCOPES for h, _, _ in COMPARISONS for k in ("low", "high")
            for v in series(t[h, "pooled", sc][:N_SHOWN], k)]
    top, bottom = max(vals) + 0.12, min(vals) - 0.12
    gs = fig.add_gridspec(1, 4, left=0.11, right=0.99, top=f(0.55), bottom=f(1.85), wspace=0.1)
    for j, sc in enumerate(SCOPES):
        ax = fig.add_subplot(gs[0, j])
        time_panel(ax, t, sc, top, bottom)
        sensor_inset(ax, sc, layout)
        ax.set_title(full_label(sc), fontsize=6.6, pad=3, linespacing=1.0, color=REGION_COLORS[region_of(sc)])
        ax.set_xlabel("time from onset (ms)", fontsize=6.4, labelpad=1)
        if j:
            ax.tick_params(labelleft=False)
    fig.text(0.025, f(1.2), "difference above chance\n(fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.4, color=st.INK2, linespacing=1.0)
    handles = [Line2D([], [], color=st.REGIME_COLORS[r], lw=1.3, ls=LINE_STYLE[h], label=n) for h, n, r in COMPARISONS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.55, f(2.3)), ncol=3, fontsize=6.2, frameon=False,
               handlelength=2.0, columnspacing=1.2)

    # b: the 187-250 ms window
    lim = max(abs(float(r[WINDOW][k])) for (h, c, sc), r in t.items() if sc in SCOPES
              for k in ("low", "high", "corrected")) * 1.15
    gs = fig.add_gridspec(1, 4, left=0.11, right=0.99, top=f(2.85), bottom=f(3.95), wspace=0.1)
    for j, sc in enumerate(SCOPES):
        ax = fig.add_subplot(gs[0, j])
        window_panel(ax, t, sc, lim)
        ax.set_title(full_label(sc), fontsize=6.6, pad=3, linespacing=1.0, color=REGION_COLORS[region_of(sc)])
        if j:
            ax.tick_params(labelleft=False)
    fig.text(0.025, f(3.42), "difference above chance\n(fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.4, color=st.INK2, linespacing=1.0)
    fams = [Line2D([], [], marker=m, ls="none", ms=st.size(m) - 1.5, mfc="white", mec=st.INK2, mew=0.6, label=fam)
            for fam, m in st.MARKERS.items()]
    fig.legend(handles=[Patch(fc=st.INK2, alpha=0.3, ec="none", label="pooled (95% CI)")] + fams,
               loc="center", bbox_to_anchor=(0.55, f(4.22)), ncol=7, fontsize=6.0, frameon=False,
               handletextpad=0.3, columnspacing=0.9)

    # c: maps in the window
    n = len(COMPARISONS)
    hw_in = 0.68; hw = hw_in / 5.5; hh = hw_in / H; x0 = 0.11; step = (0.92 - x0) / n
    for k, (hyp, name, regime) in enumerate(COMPARISONS):
        axm = fig.add_axes([x0 + k * step + (step - hw) / 2, f(5.35), hw, hh])
        im = window_map(axm, t, hyp, scalp)
        axm.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0, color=st.REGIME_COLORS[regime])
    cax = fig.add_axes([0.94, f(5.3), 0.012, hh * 0.9])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5.6, width=0.4, length=2); cb.outline.set_linewidth(0.4)

    for letter, y in (("a", 0.05), ("b", 2.5), ("c", 4.38)):
        fig.text(0.01, f(y), letter, fontsize=9, fontweight="bold", va="top")
    fig.text(0.04, f(2.52), "187–250 ms", fontsize=6.8, va="top", color=st.INK2)
    fig.text(0.04, f(4.4), "187–250 ms, pooled", fontsize=6.8, va="top", color=st.INK2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
