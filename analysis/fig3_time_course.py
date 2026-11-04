#!/usr/bin/env python3
"""Fig. 3: at 187-250 ms, left occipito-temporal sites carry the effects of training that survive the
controls (54 videos, noise-ceiling scale).

Every value is the difference above chance as a fraction of the upper EEG noise ceiling (contrasts.py
--normalize), window by window with the layer selected in each window, pooled over families.

  a     each comparison (rows) in the 11 ROIs and on the whole scalp (WS) x 13 windows, before and after
        removing identity, occlusion, pixel, optical-flow and motion-energy structure jointly; dots: p < .05
        (maxT across the 11 x 13 ROI-window cells; whole scalp across its 13 windows); the 187-250 ms
        window is outlined
  b     the same comparisons over the scalp in the 187-250 ms window
  c     the three comparisons at occipito-temporal left after the controls (95% CI), bars over the windows
        with p < .05 (above the traces: above chance; below: below chance); the ROI is chosen from a

Reads analysis/paper_stats_54/contrasts_time{,_joint}_nc.csv; writes analysis/paper_stats_54/fig3_time_course.{pdf,png}.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figure_style as st
from fig2_training import DIVERGING, REGION_COLORS, STATS, full_label, region_of, topo_scalp

ROWS = ["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "CentroParietal-L", "CentroParietal-M",
        "CentroParietal-R", "OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R", "wholescalp"]
SHORT = {"Frontal": "F", "Central": "C", "CentroParietal": "CP", "OccipitoTemporal": "OT"}
COMPARISONS = [("H1 pretrained vs random", "pretrained\nvs random", "Generic"),
               ("H2 face vs ImageNet", "face identity\nvs ImageNet", "Face"),
               ("H3 fine-tuned vs start", "emotion-tuned\nvs base model", "Emotion")]
WINDOW = 3                                               # 187.5-250 ms
FOCUS = "OccipitoTemporal-L"
VMAX = 0.2


def load(name):
    out = {}
    with (STATS / f"contrasts_time{'_' + name if name else ''}_nc.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["contrast"] == "pooled":
                out.setdefault((r["hypothesis"], r["scope"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: float(r["window_start_ms"]))
    return out


def series(rs, key):
    return np.array([float(r[key]) for r in rs])


def label(scope):
    return "WS" if scope == "wholescalp" else f"{SHORT[region_of(scope)]}-{scope.split('-')[1]}"


def grid(ax, table, hyp, labels):
    vals = np.array([series(table[hyp, sc], "corrected") for sc in ROWS])
    sig = np.array([series(table[hyp, sc], "p_maxT") < .05 for sc in ROWS])
    n = len(ROWS)
    y = np.arange(n, dtype=float); y[-1] += 0.5                      # whole scalp set apart
    for i in range(n):
        ax.imshow(vals[i:i + 1], cmap=DIVERGING, vmin=-VMAX, vmax=VMAX, aspect="auto", interpolation="nearest",
                  extent=(-0.5, vals.shape[1] - 0.5, y[i] + 0.5, y[i] - 0.5))
        xs = np.flatnonzero(sig[i])
        ax.plot(xs, [y[i]] * len(xs), "o", ms=1.3, color=st.INK, mec="none")
    ax.add_patch(Rectangle((WINDOW - 0.5, -0.5), 1, y[-1] + 1, fill=False, ec=st.INK, lw=0.7))
    ax.set_xlim(-0.5, vals.shape[1] - 0.5); ax.set_ylim(y[-1] + 0.5, -0.5)
    ax.set_xticks([0, 4, 8, 12], ["0", "250", "500", "750"], fontsize=5.6)
    ax.set_yticks(y, [label(s) for s in ROWS] if labels else [""] * n, fontsize=5.2)
    if labels:
        for tick, sc in zip(ax.get_yticklabels(), ROWS):
            tick.set_color(REGION_COLORS["wholescalp" if sc == "wholescalp" else region_of(sc)])
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)


def head(ax, table, hyp, scalp):
    import mne
    info, roi_of = scalp
    vals = np.array([float(table[hyp, r][WINDOW]["corrected"]) for r in roi_of])
    sig = np.array([float(table[hyp, r][WINDOW]["p_maxT"]) < .05 for r in roi_of])
    im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                 sensors=False, extrapolate="head", mask=sig,
                                 mask_params=dict(marker="o", markersize=1.0, markerfacecolor=st.INK,
                                                  markeredgecolor="none"))
    return im


CURVES = ["OccipitoTemporal-L", "OccipitoTemporal-R", "Central-R"]
BIN = (2, 4)                                             # windows 125-250 ms


def smooth(x, y, n=200):
    """Shape-preserving cubic interpolation through the window values (display only; PCHIP does not overshoot)."""
    from scipy.interpolate import PchipInterpolator
    xx = np.linspace(x[0], x[-1], n)
    return xx, PchipInterpolator(x, y)(xx)


def curve(ax, raw, joint, hyp, regime, scope, best, top, bottom, best_joint=None):
    from fig2_training import BEST
    color = st.REGIME_COLORS[regime]
    r, j = raw[hyp, scope], joint[hyp, scope]
    t0, t1 = series(r, "window_start_ms"), series(r, "window_end_ms")
    cx = (t0 + t1) / 2
    ax.axvspan(125, 250, color=st.RULE, alpha=0.45, lw=0, zorder=0)
    sx, lo = smooth(cx, series(r, "low")); _, hi = smooth(cx, series(r, "high"))
    ax.fill_between(sx, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(*smooth(cx, series(r, "corrected")), color=color, lw=1.0)
    ax.plot(*smooth(cx, series(j, "corrected")), color=color, lw=1.0, ls=(0, (3, 1.5)))
    if best is not None:
        ax.plot(*smooth(cx, series(best[hyp, scope], "corrected")), color=st.INK2, lw=0.6, alpha=0.8)
    if best_joint is not None:
        ax.plot(*smooth(cx, series(best_joint[hyp, scope], "corrected")), color=st.INK2, lw=0.6, alpha=0.8,
                ls=(0, (3, 1.5)))
    for rs, lw, k in ((r, 1.8, 0), (j, 0.9, 1)):                    # significant windows, raw and controls removed
        p, m = series(rs, "p_maxT"), series(rs, "corrected")
        for sign, yb in ((1, top - k * 0.035), (-1, bottom + k * 0.035)):
            sig = (p < .05) & (np.sign(m) == sign)
            for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
                ax.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=lw, solid_capstyle="butt")
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 500); ax.set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
    ax.set_ylim(bottom - 0.02, top + 0.02)
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def bin_head(ax, table, hyp, scalp):
    import mne
    info, roi_of = scalp
    a, b = BIN
    vals = np.array([np.mean(series(table[hyp, r][a:b], "corrected")) for r in roi_of])
    sig = np.array([bool(np.any(series(table[hyp, r][a:b], "p_maxT") < .05)) for r in roi_of])
    im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                 sensors=False, extrapolate="head", mask=sig,
                                 mask_params=dict(marker="o", markersize=1.0, markerfacecolor=st.INK,
                                                  markeredgecolor="none"))
    return im


def load_family(name=""):
    from fig2_training import BEST
    out = {}
    with (STATS / f"contrasts_time{'_' + name if name else ''}_nc.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["contrast"] == BEST.get(r["hypothesis"]):
                out.setdefault((r["hypothesis"], r["scope"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: float(r["window_start_ms"]))
    return out


ROI_LINES = {"OccipitoTemporal-L": "#eda100", "OccipitoTemporal-R": "#9a6400", "Central-R": "#e87ba4"}


def roi_panel(ax, raw, joint, hyp, contrast, top, bottom):
    """Three ROIs (colour) for one comparison: after controls (bold, 95% CI) and raw (faint); bars over
    significant windows (controls removed bold, raw faint), above or below the traces by sign."""
    for k, (scope, color) in enumerate(ROI_LINES.items()):
        r, j = raw[hyp, contrast, scope][:8], joint[hyp, contrast, scope][:8]
        t0, t1 = series(j, "window_start_ms"), series(j, "window_end_ms")
        cx = (t0 + t1) / 2
        sx, lo = smooth(cx, series(j, "low")); _, hi = smooth(cx, series(j, "high"))
        ax.fill_between(sx, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(*smooth(cx, series(r, "corrected")), color=color, lw=0.8, alpha=0.4)
        ax.plot(*smooth(cx, series(j, "corrected")), color=color, lw=1.3)
        p, m = series(j, "p_maxT"), series(j, "corrected")                 # significance after the controls
        for sign, yb in ((1, top - k * 0.03), (-1, bottom + k * 0.03)):
            sig = (p < .05) & (np.sign(m) == sign)
            for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
                ax.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=1.8, solid_capstyle="butt")
    ax.axvspan(125, 250, color=st.RULE, alpha=0.35, lw=0, zorder=0)
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 500); ax.set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
    ax.set_ylim(bottom - 0.03, top + 0.03)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def inset_map(ax, joint, hyp, contrast, scalp):
    import mne
    info, roi_of = scalp
    a, b = BIN
    ins = ax.inset_axes([0.7, 0.62, 0.3, 0.38])
    vals = np.array([np.mean(series(joint[hyp, contrast, r][a:b], "corrected")) for r in roi_of])
    sig = np.array([bool(np.any(series(joint[hyp, contrast, r][a:b], "p_maxT") < .05)) for r in roi_of])
    im, _ = mne.viz.plot_topomap(vals, info, axes=ins, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                 sensors=False, extrapolate="head", mask=sig,
                                 mask_params=dict(marker="o", markersize=0.7, markerfacecolor=st.INK,
                                                  markeredgecolor="none"))
    return im


def load_all(name):
    out = {}
    with (STATS / f"contrasts_time{'_' + name if name else ''}_nc.csv").open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["hypothesis"], r["contrast"], r["scope"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: float(r["window_start_ms"]))
    return out


def main():
    from fig2_training import BEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    raw, joint = load_all(""), load_all("joint")
    scalp = topo_scalp()
    H = 3.55
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(2, 3, left=0.12, right=0.99, top=0.9, bottom=0.27, wspace=0.14, hspace=0.25)
    for j, (hyp, name, regime) in enumerate(COMPARISONS):
        vals = [v for c in ("pooled", BEST[hyp]) for t in (raw, joint) for sc in ROI_LINES
                for k in (("corrected", "low", "high") if t is joint else ("corrected",))
                for v in series(t[hyp, c, sc][:8], k)]
        top, bottom = max(vals) + 0.06, min(vals) - 0.06
        for i, contrast in enumerate(("pooled", BEST[hyp])):
            ax = fig.add_subplot(gs[i, j])
            roi_panel(ax, raw, joint, hyp, contrast, top, bottom)
            im = inset_map(ax, joint, hyp, contrast, scalp)
            if i == 0:
                ax.set_title(name.replace("\n", " "), fontsize=7, color=st.REGIME_COLORS[regime], pad=3)
                ax.tick_params(labelbottom=False)
            if i == 1:
                ax.text(0.03, 0.04, BEST[hyp].split()[0], transform=ax.transAxes, fontsize=6.6, color=st.INK2,
                        va="bottom")
    fig.text(0.02, 0.76, "all models\n(pooled)", rotation=90, ha="center", va="center", fontsize=6.8, linespacing=1.0)
    fig.text(0.02, 0.45, "strongest\nfamily", rotation=90, ha="center", va="center", fontsize=6.8, linespacing=1.0)
    fig.text(0.555, 0.19, "time from onset (ms)", ha="center", fontsize=6.8)
    fig.text(0.06, 0.6, "difference above chance\n(fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.2, color=st.INK2, linespacing=1.0)
    handles = [Line2D([], [], color=c, lw=1.3, label=full_label(sc).replace("\n", " ")) for sc, c in ROI_LINES.items()]
    handles += [Line2D([], [], color=st.INK2, lw=1.3, label="controls removed (95% CI)"),
                Line2D([], [], color=st.INK2, lw=0.8, alpha=0.4, label="raw"),
                Line2D([], [], color=st.INK2, lw=1.8, label="p < .05, controls removed")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.42, 0.08), ncol=3, fontsize=6.2, frameon=False,
               handlelength=1.6, columnspacing=1.0)
    cax = fig.add_axes([0.8, 0.105, 0.17, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5.6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("maps: 125–250 ms,\ncontrols removed", fontsize=6, linespacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
