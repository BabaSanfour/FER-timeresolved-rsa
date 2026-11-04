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
from fig2_training import DIVERGING, REGION_COLORS, STATS, region_of, topo_scalp

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    raw, joint = load(""), load("joint")
    H = 5.9
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    scalp = topo_scalp()
    row_h, gap = 1.12, 0.13
    xg = [0.17, 0.42]                                    # grid columns (raw, controls removed)
    gw = 0.23
    hw = 0.16                                            # head width (figure fraction)
    xh = [0.69, 0.84]
    for i, (hyp, name, regime) in enumerate(COMPARISONS):
        top = 0.55 + i * (row_h + gap)
        y0, h = f(top + row_h), row_h / H
        fig.text(0.055, y0 + h / 2, name, ha="center", va="center", fontsize=6.8, color=st.REGIME_COLORS[regime],
                 linespacing=1.0, rotation=90)
        for j, t in enumerate((raw, joint)):
            ax = fig.add_axes([xg[j], y0, gw, h])
            grid(ax, t, hyp, labels=j == 0)
            if i == len(COMPARISONS) - 1:
                ax.set_xlabel("time from onset (ms)", fontsize=6.4, labelpad=1)
            else:
                ax.tick_params(labelbottom=False)
            hh = hw * 5.5 / H
            axh = fig.add_axes([xh[j], y0 + (h - hh) / 2, hw, hh])
            im = head(axh, t, hyp, scalp)
    for j, lab in enumerate(("raw", "controls removed")):
        fig.text(xg[j] + gw / 2, f(0.5), lab, ha="center", va="bottom", fontsize=6.8, color=st.INK2)
        fig.text(xh[j] + hw / 2, f(0.5), lab, ha="center", va="bottom", fontsize=6.8, color=st.INK2)
    fig.text(xh[0] + hw, f(0.3), "187–250 ms", ha="center", va="bottom", fontsize=7.2)
    fig.text(xg[0] + gw + 0.01, f(0.3), "11 ROIs and whole scalp × 13 windows", ha="center", va="bottom", fontsize=7.2)
    cax = fig.add_axes([0.69, f(4.42), 0.3, 0.06 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5.8, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference above chance\n(fraction of noise ceiling)", fontsize=6.2, linespacing=1.0)
    # c: the focus ROI after the controls
    c = fig.add_axes([0.17, f(5.55), 0.48, 0.95 / H])
    for k, (hyp, name, regime) in enumerate(COMPARISONS):
        rs = joint[hyp, FOCUS]
        color = st.REGIME_COLORS[regime]
        t0, t1 = series(rs, "window_start_ms"), series(rs, "window_end_ms")
        cx = (t0 + t1) / 2
        c.fill_between(cx, series(rs, "low"), series(rs, "high"), color=color, alpha=0.15, lw=0)
        c.plot(cx, series(rs, "corrected"), color=color, lw=1.0)
        p, m = series(rs, "p_maxT"), series(rs, "corrected")
        for sign, yb in ((1, 0.34 - k * 0.035), (-1, -0.27 + k * 0.035)):   # above / below chance
            sig = (p < .05) & (np.sign(m) == sign)
            for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
                c.plot([t0[a], t1[b - 1]], [yb, yb], color=color, lw=1.8, solid_capstyle="butt")
    c.axvspan(187.5, 250, color=st.RULE, alpha=0.4, lw=0, zorder=0)
    c.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    c.set_xlim(0, 800); c.set_ylim(-0.3, 0.37)
    c.set_xlabel("time from onset (ms)", fontsize=6.6)
    c.set_ylabel("above chance\n(fraction of ceiling)", fontsize=6.4)
    c.set_title("occipito-temporal left, controls removed", fontsize=6.8, pad=2, color=REGION_COLORS["OccipitoTemporal"])
    c.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    fig.legend(handles=[Line2D([], [], color=st.REGIME_COLORS[r], lw=1.2, label=n.replace("\n", " "))
                        for _, n, r in COMPARISONS] +
               [Line2D([], [], color=st.INK2, lw=1.8, label="p < .05, corrected")],
               loc="center left", bbox_to_anchor=(0.69, f(5.05)), fontsize=6.4, frameon=False, handlelength=1.6)
    for letter, x, y in (("a", 0.01, 0.05), ("b", 0.66, 0.05), ("c", 0.01, 4.45)):
        fig.text(x, f(y), letter, fontsize=9, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
