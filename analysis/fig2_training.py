#!/usr/bin/env python3
"""Fig. 2: what each step of training contributes to EEG alignment, in every ROI.

Values are the observed rho_max (each variant read at its own peak, mean over its RDMs and the
three emotions) and observed differences; the grey dash marks the chance level, the mean of
the permutation null with the same selection (contrasts.py), which is high for models with
many layers.

  a     each variant in each of the 11 ROIs and on the whole scalp, by training regime (54
        videos): one marker per family, joined across a family's regimes; bars are the mean over
        families, the dark dash its chance level
  b-e   the planned comparisons in each ROI and on the whole scalp: each participant's
        difference (54 videos, grey), the mean and 95% bootstrap CI (54 videos; b also 18 videos,
        open), each with its chance level (dash); * maxT-corrected p < .05 across ROIs (whole scalp: Holm across H1-H4), in the
        order 54 | 18. c shows each video model against the image models (family markers).
ROI labels carry the region colours of Fig. 1.

Reads analysis/paper_stats[_54]/contrasts.csv and contrasts_participants.csv (54); writes
analysis/paper_stats_54/fig2_training.{pdf,png}.
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
import rsa_core
import variants

STATS = {"54": rsa_core.ROOT / "analysis/paper_stats_54", "18": rsa_core.ROOT / "analysis/paper_stats"}
REGIONS = [("Frontal", "frontal"), ("Central", "central"), ("CentroParietal", "centro-parietal"),
           ("OccipitoTemporal", "occipito-temporal")]
REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300",
                 "OccipitoTemporal": "#eda100", "wholescalp": st.INK}          # as in Fig. 1
SHORT = {"Frontal": "F", "Central": "C", "CentroParietal": "CP", "OccipitoTemporal": "OT"}
GRID = [["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "wholescalp"],
        ["CentroParietal-L", "CentroParietal-M", "CentroParietal-R",
         "OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R"]]
VIDEO = (("Qwen3-VL Pretrained", "*"), ("M3DFEL Kinetics-400", "p"))
PANELS = [("H1 pretrained vs random", "pretrained − random", "Generic", (("pooled", "o"),)),
          ("H4 video vs image", "video model − image models", "Generic", VIDEO),
          ("H2 face vs ImageNet", "face − ImageNet pretraining", "Face", (("pooled", "o"),)),
          ("H3 fine-tuned vs start", "emotion fine-tuned − its start", "Emotion", (("pooled", "o"),))]


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def region_of(scope):
    return scope.split("-")[0]


def scope_label(scope):
    return "whole scalp" if scope == "wholescalp" else f"{SHORT[region_of(scope)]}-{scope.split('-')[1]}"


def variant_panel(ax, scope, table):
    """table[s]: {(family variant, scope): row} of the variant rows."""
    xs = {regime: i for i, regime in enumerate(st.REGIMES)}
    by_regime = {regime: [] for regime in st.REGIMES}
    by_regime18 = {regime: [] for regime in st.REGIMES}
    chance = {regime: [] for regime in st.REGIMES}
    for fi, (family, vrows) in enumerate(variants.MAIN.items()):
        pts = []
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            x = xs[regime] - 0.12 + (fi - 2.5) * 0.07
            y54 = float(table["54"][f"{family} {variant}", scope]["observed"])
            y18 = float(table["18"][f"{family} {variant}", scope]["observed"])
            chance[regime].append(float(table["54"][f"{family} {variant}", scope]["null_mean"]))
            pts.append((x, y54, regime)); by_regime[regime].append(y54); by_regime18[regime].append(y18)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st.RULE, lw=0.5, zorder=2)
        m = st.MARKERS[family]
        for x, y54, regime in pts:
            ax.plot(x, y54, m, ms=st.size(m) - 1.2, mfc=st.REGIME_COLORS[regime], mec="white", mew=0.3, zorder=4)
    for regime in st.REGIMES:
        c = st.REGIME_COLORS[regime]
        ax.bar(xs[regime] - 0.12, np.mean(by_regime[regime]), width=0.46, color=c, alpha=0.25, lw=0, zorder=1)
        ax.plot([xs[regime] - 0.35, xs[regime] + 0.11], [np.mean(chance[regime])] * 2, color=st.INK2, lw=1.1,
                solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r][0] for r in st.REGIMES], fontsize=5.2)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.55, len(st.REGIMES) - 0.65)
    ax.set_title(scope_label(scope), fontsize=6, color=REGION_COLORS[region_of(scope)], pad=2,
                 fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def scope_positions(scopes):
    """x position of each ROI, grouped by region with a gap between regions, then the whole scalp."""
    pos, x = {}, 0.0
    for region, _ in REGIONS:
        for side in "LMR":
            if f"{region}-{side}" in scopes:
                pos[f"{region}-{side}"] = x; x += 1
        x += 0.6
    pos["wholescalp"] = x + 0.2
    return pos


def significant(r, contrast):
    if contrast == "pooled" and r["scope"] == "wholescalp":
        return float(r["p_holm"] or 1) < .05
    return float(r["p_maxT"]) < .05


def contrast_panel(ax, hyp, title, regime, comparisons, participants, series, rng, sets=("54",)):
    color = st.REGIME_COLORS[regime]
    pick = {s: {(r["contrast"], r["scope"]): r for r in comparisons[s] if r["hypothesis"] == hyp} for s in STATS}
    pos = scope_positions({sc for c, sc in pick["54"] if c == series[0][0]})
    n = len(sets) * len(series)
    step = 0.34 if n == 2 else 0.3
    top = 0.105
    for sc, x in pos.items():
        if series[0][0] == "pooled":
            d = np.array([float(r["difference"]) for r in participants if r["hypothesis"] == hyp and r["scope"] == sc])
            d = d + float(pick["54"]["pooled", sc]["null_mean"])
            ax.scatter(x - 0.36 + rng.uniform(-0.06, 0.06, len(d)), d, s=1.4, color=st.PARTICIPANT, lw=0, zorder=2)
        k = 0
        for contrast, marker in series:
            for s, filled in ((x, x == "54") for x in sets):
                r = pick[s][contrast, sc]
                dx = (k - (n - 1) / 2) * step + (0.08 if series[0][0] == "pooled" else 0); k += 1
                c0 = float(r["null_mean"])
                m, lo, hi = float(r["observed"]), float(r["low"]) + c0, float(r["high"]) + c0
                ax.plot([x + dx - 0.1, x + dx + 0.1], [c0, c0], color=st.INK2, lw=1.0, solid_capstyle="butt", zorder=5)
                ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], fmt=marker,
                            ms=st.size(marker) - (1.8 if marker == "*" else 1.2), mfc=color if filled else "white",
                            mec=color, mew=0.6, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
                if significant(r, contrast):
                    ax.text(x + dx, top, "*", ha="center", va="center", fontsize=6.5, color=st.INK)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(list(pos.values()), ["whole\nscalp" if s == "wholescalp" else s.split("-")[1] for s in pos],
                  fontsize=5.4)
    for tick, sc in zip(ax.get_xticklabels(), pos):
        tick.set_color(REGION_COLORS[region_of(sc)])
    for region, label in REGIONS:
        xs = [x for s, x in pos.items() if s.startswith(region + "-")]
        ax.text(np.mean(xs), -0.2, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.4,
                color=REGION_COLORS[region])
    ax.set_xlim(-0.7, pos["wholescalp"] + 0.6)
    ax.set_ylim(-0.1, top + 0.012)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def legend_block(fig):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=f"{st.REGIME_LABELS[r]} ({st.REGIME_LABELS[r][0]})")
               for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    sets = [Line2D([], [], marker="o", ls="none", ms=3, mfc=st.INK2, mec=st.INK2, label="54 videos"),
            Line2D([], [], marker="o", ls="none", ms=3, mfc="white", mec=st.INK2, label="18 videos (b)"),
            Line2D([], [], marker="o", ls="none", ms=1.8, mfc=st.PARTICIPANT, mec="none", label="participants (54)"),
            Line2D([], [], color=st.INK2, lw=1.1, label="chance")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=1.1, borderaxespad=0, fontsize=6, loc="lower left")
    for y, title, handles in ((0.072, "Training regime", regimes), (0.038, "Model family", families),
                              (0.004, "Stimulus set", sets)):
        fig.text(0.02, y + 0.008, title, fontsize=6, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS["54"] / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    comparisons = {s: rows(STATS[s] / "contrasts.csv") for s in STATS}
    participants = rows(STATS["54"] / "contrasts_participants.csv")
    table = {s: {(r["contrast"], r["scope"]): r for r in comparisons[s] if r["hypothesis"] == "variant"} for s in STATS}
    fig = plt.figure(figsize=(5.5, 6.6))
    gs_a = fig.add_gridspec(2, 6, left=0.08, right=0.99, top=0.955, bottom=0.66, wspace=0.12, hspace=0.42)
    axes_a = np.array([[fig.add_subplot(gs_a[i, j]) for j in range(6)] for i in range(2)])
    for i, row in enumerate(GRID):
        for j, scope in enumerate(row):
            variant_panel(axes_a[i, j], scope, table)
    lo = min(ax.get_ylim()[0] for ax in axes_a.flat); hi = max(ax.get_ylim()[1] for ax in axes_a.flat)
    for i, j in np.ndindex(axes_a.shape):
        axes_a[i, j].set_ylim(lo, hi)
        if j:
            axes_a[i, j].tick_params(labelleft=False)
    fig.text(0.015, 0.81, r"$\rho_{\max}$", rotation=90, va="center", fontsize=6.5)
    gs_b = fig.add_gridspec(2, 2, left=0.08, right=0.99, top=0.58, bottom=0.165, hspace=0.72, wspace=0.08)
    rng = np.random.default_rng(0)
    axes_b = []
    for k, (hyp, title, regime, series) in enumerate(PANELS):
        ax = fig.add_subplot(gs_b[k // 2, k % 2])
        contrast_panel(ax, hyp, title, regime, comparisons, participants, series, rng,
                       ("54", "18") if hyp.startswith("H1") else ("54",))
        if k % 2:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"difference in $\rho_{\max}$")
        axes_b.append(ax)
    st.panel_label(axes_a[0, 0], "a", x=-0.5)
    for ax, letter in zip(axes_b, "bcde"):
        st.panel_label(ax, letter, x=-0.16 if ax in axes_b[::2] else -0.05)
    legend_block(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
