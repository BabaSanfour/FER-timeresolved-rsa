#!/usr/bin/env python3
"""Fig. 2: what each training step contributes to EEG alignment, region by region (54 videos).

  a  the planned comparisons (contrasts.py) in each of the 11 ROIs and on the whole scalp:
     pooled test of each hypothesis and each family's contrast, corrected for layer-selection
     bias (observed minus null mean); * maxT-corrected p < .05 across ROIs (whole scalp: its
     own permutation p)
  b  the whole-scalp tests: each family's contrast with its 95% CI and the pooled test with
     its Holm-corrected p
  c  the pooled tests after removing identity, occlusion, pixel, optical-flow and
     motion-energy structure jointly (contrasts.py --partial joint)

Reads analysis/paper_stats_54/contrasts.csv and contrasts_joint.csv; writes
analysis/paper_stats_54/fig2_training.{pdf,png}.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import figure_style as st
import rsa_core

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
HYPOTHESES = [("H1 pretrained vs random", "Pretrained − random", "Generic"),
              ("H2 face vs ImageNet", "Face − ImageNet", "Face"),
              ("H3 fine-tuned vs start", "Fine-tuned − start", "Emotion"),
              ("H4 video vs image", "Video − image", "Generic")]
SCOPE_LABELS = {"Frontal": "F", "Central": "C", "CentroParietal": "CP", "OccipitoTemporal": "OT"}
VMAX = .06


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def p_text(p):
    return "p < .001" if p < .001 else "p = " + (f"{p:.3f}" if p < .01 else f"{p:.2f}").replace("0.", ".", 1)


def scope_order(scopes):
    """ROIs grouped by region (left, midline, right), then the whole scalp."""
    rois = [f"{r}-{s}" for r in SCOPE_LABELS for s in "LMR" if f"{r}-{s}" in scopes]
    return rois + ["wholescalp"]


def scope_label(scope):
    if scope == "wholescalp":
        return "whole\nscalp"
    region, side = scope.split("-")
    return f"{SCOPE_LABELS[region]}-{side}"


def heatmap(ax, table, pooled_only=False, show_cbar=False, fig=None):
    scopes = scope_order({r["scope"] for r in table})
    cell = {(r["hypothesis"], r["contrast"], r["scope"]): r for r in table}
    lines, labels, colors = [], [], []
    for hyp, title, regime in HYPOTHESES:
        contrasts = ["pooled"] + ([] if pooled_only else list(dict.fromkeys(
            r["contrast"] for r in table if r["hypothesis"] == hyp and r["contrast"] != "pooled")))
        for c in contrasts:
            lines.append((hyp, c))
            labels.append(title if c == "pooled" else "   " + c.replace(" Pretrained", "").replace(" Kinetics-400", ""))
            colors.append(st.REGIME_COLORS[regime])
    values = np.array([[float(cell[h, c, s]["corrected"]) for s in scopes] for h, c in lines])
    im = ax.imshow(values, cmap=DIVERGING, vmin=-VMAX, vmax=VMAX, aspect="auto", interpolation="nearest")
    for i, (h, c) in enumerate(lines):
        for j, s in enumerate(scopes):
            r = cell[h, c, s]
            p = float(r["p"]) if s == "wholescalp" else float(r["p_maxT"])
            if p < .05:
                ax.text(j, i + 0.12, "*", ha="center", va="center", fontsize=6.5, color=st.INK)
    ax.set_xticks(range(len(scopes)), [scope_label(s) for s in scopes], fontsize=5.4)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(lines)), labels, fontsize=5.6)
    for tick, (h, c) in zip(ax.get_yticklabels(), lines):
        tick.set_fontweight("bold" if c == "pooled" else "normal")
    for i, (h, c) in enumerate(lines):
        if i and c == "pooled":
            ax.axhline(i - 0.5, color="white", lw=1.6)
    ax.axvline(len(scopes) - 1.5, color="white", lw=1.6)
    for x in (2.5, 4.5, 7.5):
        ax.axvline(x, color="white", lw=0.8)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_cbar:
        cax = fig.add_axes([ax.get_position().x1 + 0.012, ax.get_position().y0 - 0.235, 0.01, 0.18])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
        cb.ax.tick_params(labelsize=5, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label(r"$\Delta\rho$ − null mean", fontsize=5.5)


def forest(ax, table):
    table = [r for r in table if r["scope"] == "wholescalp"]
    y, ticks, labels = 0, [], []
    for hyp, title, regime in HYPOTHESES:
        color = st.REGIME_COLORS[regime]
        pooled = next(r for r in table if r["contrast"] == "pooled" and r["hypothesis"] == hyp)
        ph = float(pooled["p_holm"])
        for r in (r for r in table if r["contrast"] != "pooled" and r["hypothesis"] == hyp):
            m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
            ax.plot([lo, hi], [y, y], color=color, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot(m, y, "o", ms=2.8, color=color, mec="white", mew=0.3, zorder=3)
            ticks.append(y); labels.append(r["contrast"].replace(" Pretrained", "").replace(" Kinetics-400", ""))
            y -= 1
        m, lo, hi = (float(pooled[k]) for k in ("corrected", "low", "high"))
        ax.fill([lo, m, hi, m], [y, y + 0.32, y, y - 0.32], color=st.INK if ph < .05 else st.MUTED, lw=0, zorder=3)
        ticks.append(y); labels.append(f"{title.split(' ')[0]}: Holm {p_text(ph)}")
        y -= 1.6
    ax.axvline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ticks, labels, fontsize=5.3)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 1.1, 0.6)
    ax.set_xlim(-0.05, 0.07)
    ax.set_xticks([-0.04, 0, 0.04])
    ax.xaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.set_xlabel(r"$\Delta\rho$ − null mean, whole scalp", fontsize=5.8)


def headline(fig, ax, letter, text, dx=0.0, dy=0.03):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.05 + dx, pos.y1 + dy, letter, fontsize=8, fontweight="bold", va="bottom")
    fig.text(pos.x0 - 0.025 + dx, pos.y1 + dy, text, fontsize=6.6, fontweight="bold", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    raw, joint = rows(STATS / "contrasts.csv"), rows(STATS / "contrasts_joint.csv")
    fig = plt.figure(figsize=(5.5, 4.6))
    a = fig.add_axes([0.2, 0.33, 0.42, 0.58])
    b = fig.add_axes([0.79, 0.33, 0.19, 0.58])
    c = fig.add_axes([0.2, 0.05, 0.42, 0.13])
    heatmap(a, raw, show_cbar=True, fig=fig)
    forest(b, raw)
    heatmap(c, joint, pooled_only=True)
    headline(fig, a, "a", "Each training step, region by region", dx=-0.13, dy=0.055)
    headline(fig, b, "b", "Whole scalp", dx=-0.07, dy=0.055)
    headline(fig, c, "c", "After removing identity, occlusion and low-level structure", dx=-0.13, dy=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
