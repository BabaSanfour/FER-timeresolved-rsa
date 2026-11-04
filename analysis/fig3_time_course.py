#!/usr/bin/env python3
"""Fig. 3: when each training regime changes EEG alignment (54 videos, noise-ceiling scale).

Every value is a fraction of the upper EEG noise ceiling (contrasts.py --normalize), as in Fig. 2.

  a     the three time-resolved comparisons (rows: pretrained vs random, face identity vs ImageNet,
        emotion-tuned vs base model) in the three occipito-temporal ROIs, the two central ROIs and on the
        whole scalp (columns), window by window with the layer selected in each window: pooled over
        families before (solid, 95% CI) and after (dashed) removing all nuisance structure, and the
        strongest family of Fig. 2c (thin grey); bars above the traces mark windows with p < .05 (maxT
        across the 11 x 13 ROI-window cells; whole scalp across its 13 windows), raw (thick) and after
        the controls (thin)
  b     the pooled comparisons over the scalp in five time bins, before and after the controls: the
        difference above chance per ROI (mean over the bin's windows), dots on ROIs with a significant
        window in the bin

Reads analysis/paper_stats_54/contrasts_time{,_joint}_nc.csv; writes analysis/paper_stats_54/fig3_time_course.{pdf,png}.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figure_style as st
from fig2_training import BEST, DIVERGING, REGION_COLORS, STATS, full_label, region_of, topo_scalp

SCOPES = ["OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R", "Central-L", "Central-R", "wholescalp"]
COMPARISONS = [("H1 pretrained vs random", "pretrained\nvs random", "Generic"),
               ("H2 face vs ImageNet", "face identity\nvs ImageNet", "Face"),
               ("H3 fine-tuned vs start", "emotion-tuned\nvs base model", "Emotion")]
BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 13)]          # window indices: 0-125, 125-250, 250-375, 375-500, 500-800 ms
MAP_ROWS = [("H1 pretrained vs random", "", "pretrained vs random\nraw"),
            ("H1 pretrained vs random", "joint", "controls\nremoved"),
            ("H2 face vs ImageNet", "", "face identity\nvs ImageNet, raw"),
            ("H2 face vs ImageNet", "joint", "controls\nremoved")]
VMAX = 0.2


def load(name):
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


def sig_bars(ax, rs, y, color, lw):
    t0, t1 = series(rs, "window_start_ms"), series(rs, "window_end_ms")
    sig = series(rs, "p_maxT") < .05
    for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
        ax.plot([t0[a], t1[b - 1]], [y, y], color=color, lw=lw, solid_capstyle="butt")


def trace_panel(ax, hyp, regime, scope, raw, joint, top):
    color = st.REGIME_COLORS[regime]
    r = raw[hyp, "pooled", scope]
    c = (series(r, "window_start_ms") + series(r, "window_end_ms")) / 2
    ax.fill_between(c, series(r, "low"), series(r, "high"), color=color, alpha=0.2, lw=0)
    ax.plot(c, series(r, "corrected"), color=color, lw=1.0)
    ax.plot(c, series(joint[hyp, "pooled", scope], "corrected"), color=color, lw=0.9, ls=(0, (3, 1.5)))
    ax.plot(c, series(raw[hyp, BEST[hyp], scope], "corrected"), color=st.INK2, lw=0.6, alpha=0.8)
    sig_bars(ax, r, top * 0.93, color, 1.8)
    sig_bars(ax, joint[hyp, "pooled", scope], top * 0.84, color, 0.8)
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0, 800); ax.set_xticks([0, 400, 800], ["0", "400", "800"])
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def maps(fig, box, raw, joint):
    import mne
    info, roi_of = topo_scalp()
    x0, y0, w, h = box
    lab_w = 0.2
    cw = (w - lab_w) / len(BINS)
    rh = h / len(MAP_ROWS)
    ref = raw["H1 pretrained vs random", "pooled", "wholescalp"]
    starts, ends = series(ref, "window_start_ms"), series(ref, "window_end_ms")
    for i, (hyp, name, label) in enumerate(MAP_ROWS):
        t = raw if name == "" else joint
        yy = y0 + h - (i + 1) * rh
        fig.text(x0 + lab_w - 0.01, yy + rh / 2, label, ha="right", va="center", fontsize=6.4, linespacing=1.0,
                 color=st.INK if name == "" else st.INK2)
        for j, (a, b) in enumerate(BINS):
            ax = fig.add_axes([x0 + lab_w + j * cw, yy, cw * 0.95, rh * 0.95])
            vals = [np.mean([float(r["corrected"]) for r in t[hyp, "pooled", roi][a:b]]) for roi in roi_of]
            sig = [any(float(r["p_maxT"]) < .05 for r in t[hyp, "pooled", roi][a:b]) for roi in roi_of]
            im, _ = mne.viz.plot_topomap(np.array(vals), info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX),
                                         contours=0, sensors=False, extrapolate="head", mask=np.array(sig),
                                         mask_params=dict(marker="o", markersize=0.9, markerfacecolor=st.INK,
                                                          markeredgecolor="none"))
            if i == 0:
                ax.set_title(f"{starts[a]:.0f}–{ends[b - 1]:.0f} ms", fontsize=6.4, pad=2)
    cax = fig.add_axes([x0 + lab_w + 0.05, y0 - 0.025, len(BINS) * cw - 0.1, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference above chance (fraction of noise ceiling)", fontsize=6.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    raw, joint = load(""), load("joint")
    H = 6.95
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(3, 6, left=0.15, right=0.99, top=f(0.45), bottom=f(3.15), wspace=0.15, hspace=0.35)
    for i, (hyp, label, regime) in enumerate(COMPARISONS):
        vals = [v for sc in SCOPES for t in (raw, joint) for k in ("low", "high", "corrected")
                for v in series(t[hyp, "pooled", sc], k)] + \
               [v for sc in SCOPES for v in series(raw[hyp, BEST[hyp], sc], "corrected")]
        top = max(vals) * 1.25
        for j, sc in enumerate(SCOPES):
            ax = fig.add_subplot(gs[i, j])
            trace_panel(ax, hyp, regime, sc, raw, joint, top)
            ax.set_ylim(min(vals) * 1.1, top)
            if j:
                ax.tick_params(labelleft=False)
            if i < 2:
                ax.tick_params(labelbottom=False)
            if i == 0:
                ax.set_title(full_label(sc), fontsize=6.4, pad=2, linespacing=1.0,
                             color=REGION_COLORS["wholescalp" if sc == "wholescalp" else region_of(sc)])
            if j == 0:
                ax.set_ylabel(label, fontsize=6.6, color=st.REGIME_COLORS[regime], linespacing=1.0, labelpad=3)
    fig.text(0.565, f(3.38), "time from onset (ms)", ha="center", fontsize=7)
    fig.text(0.015, f(1.8), "difference above chance (fraction of noise ceiling)", rotation=90, ha="center",
             va="center", fontsize=6.6, color=st.INK2)
    handles = [Line2D([], [], color=st.INK2, lw=1.0, label="pooled, raw (95% CI)"),
               Line2D([], [], color=st.INK2, lw=0.9, ls=(0, (3, 1.5)), label="pooled, controls removed"),
               Line2D([], [], color=st.INK2, lw=0.6, alpha=0.8, label="strongest family (Qwen3-VL, ResNet50, VGG16)"),
               Line2D([], [], color=st.INK2, lw=1.8, label="p < .05, raw"),
               Line2D([], [], color=st.INK2, lw=0.8, label="p < .05, controls removed")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.55, f(3.68)), ncol=3, fontsize=6.4, frameon=False,
               handlelength=1.8, columnspacing=1.0)
    maps(fig, (0.02, f(6.3), 0.96, 2.2 / H), raw, joint)
    fig.text(0.01, f(0.05), "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.01, f(3.95), "b", fontsize=9, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
