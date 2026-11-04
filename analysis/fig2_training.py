#!/usr/bin/env python3
"""Fig. 2: what training contributes to EEG alignment.

  a, b  ROI- and emotion-averaged rho_max of every variant by training regime (18 and
        54 videos): one marker per family (mean over its seeds or runs), lines join a
        family's regimes, bars are the mean over families
  c     planned contrasts H1-H4 (contrasts.py): each participant's difference (grey),
        mean and 95% bootstrap CI, Holm-corrected p, for 18 and 54 videos
  d     rho_max minus the mean of its own permutation null (rho_max_null.py, 54 videos):
        every variant and nuisance control; filled when p < .05
  e     FaceNet face training by training-set size (18 videos): size curve and the
        step-matched check, mean +/- SD over ten repeats; random seeds as a band

Reads analysis/paper_stats[_54]/rho_max.csv, contrasts.csv, contrasts_participants.csv,
rho_max_null.csv; writes analysis/paper_stats/fig2_training.{pdf,png}.
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

STATS = {"18": rsa_core.ROOT / "analysis/paper_stats", "54": rsa_core.ROOT / "analysis/paper_stats_54"}
HYPOTHESES = [("H1 pretrained vs random", "H1\npretrained\n− random"), ("H2 face vs ImageNet", "H2\nface\n− ImageNet"),
              ("H3 fine-tuned vs start", "H3\nemotion\n− start"), ("H4 video vs image", "H4\nvideo\n− image")]
CONTROL_NUDGE = {"Pixel": 0.0012, "Identity": -0.0012}       # label offsets where two controls nearly coincide
SIZE_IMAGES = {"10k": 1e4, "50k": 5e4, "100k": 1e5, "200k": 2e5, "300k": 3e5, "500k": 5e5, "715k": 715810}


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def summary(s):
    """{(group, family, variant): row} at ROI average x emotion average."""
    return {(r["group"], r["family"], r["variant"]): r for r in rows(STATS[s] / "rho_max.csv")
            if r["roi"] == rsa_core.ROI_AVG and r["emotion"] == rsa_core.EMO_AVG}


def regime_panel(ax, s):
    table = summary(s)
    xs = {regime: i for i, regime in enumerate(st.REGIMES)}
    by_regime = {regime: [] for regime in st.REGIMES}
    for fi, (family, vrows) in enumerate(variants.MAIN.items()):
        pts = []
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            pts.append((xs[regime] + (fi - 2.5) * 0.07, float(table["main", family, variant]["max_r"]), regime))
            by_regime[regime].append(pts[-1][1])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st.RULE, lw=0.6, zorder=2)
        m = st.MARKERS[family]
        for x, y, regime in pts:
            ax.plot(x, y, m, ms=st.size(m), mfc=st.REGIME_COLORS[regime], mec="white", mew=0.4, zorder=4)
    for regime, vals in by_regime.items():
        ax.bar(xs[regime], np.mean(vals), width=0.62, color=st.REGIME_COLORS[regime], alpha=0.25, lw=0, zorder=1)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r] for r in st.REGIMES], rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(st.REGIMES) - 0.45)
    ax.set_ylabel(r"$\rho_{\max}$")
    ax.set_title(f"{s} videos")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def contrast_panel(ax):
    rng = np.random.default_rng(0)
    for k, (s, mfc) in enumerate((("18", st.INK), ("54", "white"))):
        result = {r["hypothesis"]: r for r in rows(STATS[s] / "contrasts.csv") if r["level"] == "primary"}
        part = rows(STATS[s] / "contrasts_participants.csv")
        for hi, (hyp, _) in enumerate(HYPOTHESES):
            x = hi + (k - 0.5) * 0.36
            d = np.array([float(r["difference"]) for r in part if r["hypothesis"] == hyp])
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(d)), d, s=3, color=st.PARTICIPANT, lw=0, zorder=2)
            r = result[hyp]
            mean, lo, hi_ = float(r["mean_difference"]), float(r["ci95_low"]), float(r["ci95_high"])
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi_ - mean]], fmt="o", ms=3.2, mfc=mfc, mec=st.INK,
                        mew=0.7, ecolor=st.INK, elinewidth=0.8, capsize=0, zorder=4)
            p = float(r["p_holm"])
            ax.text(x, 0.071, st.stars(p), ha="center", va="bottom", fontsize=6 if p < .05 else 5, color=st.INK)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(HYPOTHESES)), [label for _, label in HYPOTHESES], fontsize=5.6)
    ax.set_ylim(-0.06, 0.082)
    ax.set_ylabel(r"difference in $\rho$ (per participant)")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.2, mfc=st.INK, mec=st.INK, label="18 videos"),
                       Line2D([], [], marker="o", ls="none", ms=3.2, mfc="white", mec=st.INK, label="54 videos")],
              loc="lower left", handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=0.8)


def chance_panel(ax):
    null = rows(STATS["54"] / "rho_max_null.csv")
    xs = {regime: i for i, regime in enumerate(st.REGIMES + ["Control"])}
    counts = {}
    for r in null:
        if not r["excess"]:
            continue
        excess, p = float(r["excess"]), float(r["p"])
        if r["group"] == "control":
            x = xs["Control"]
            ax.plot(x, excess, "o", ms=3.6, mfc=st.MUTED if p < .05 else "white", mec=st.MUTED, mew=0.7, zorder=4)
            ax.text(x + 0.13, excess + CONTROL_NUDGE.get(r["variant"], 0), r["variant"].lower(), va="center",
                    fontsize=5.3, color=st.INK2)
            continue
        regime = st.REGIME_OF[r["variant"]]
        k = counts.setdefault(regime, 0); counts[regime] += 1
        m = st.MARKERS[r["family"]]
        c = st.REGIME_COLORS[regime]
        ax.plot(xs[regime] + (k - 2.5) * 0.08, excess, m, ms=st.size(m), mfc=c if p < .05 else "white",
                mec=c, mew=0.7, zorder=4)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(xs)), [st.REGIME_LABELS[r] for r in st.REGIMES] + ["controls"])
    ax.set_xlim(-0.55, len(xs) - 0.1)
    ax.set_ylabel(r"$\rho_{\max}$ − chance (54 videos)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.6, mfc=st.INK2, mec=st.INK2, label="p < .05"),
                       Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec=st.INK2, label="n.s.")],
              loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def size_panel(ax):
    table = summary("18")
    rnd = table["main", "FaceNet", "Random"]
    m, sd = float(rnd["max_r"]), float(rnd["sd"])
    ax.axhspan(m - sd, m + sd, color=st.REGIME_COLORS["Random"], alpha=0.25, lw=0)
    ax.text(7e5, m - sd + 0.0012, "random (10 seeds)", fontsize=5.3, color=st.INK2, va="bottom", ha="right")
    for group, mfc, label in (("size", st.REGIME_COLORS["Face"], "fixed epochs"),
                              ("step", "white", "matched steps")):
        pts = [(SIZE_IMAGES[v], float(r["max_r"]), float(r["sd"])) for (g, f, v), r in table.items() if g == group]
        if group == "step":                                  # the full pool is the end point of both
            pts.append(next((x, y, e) for x, y, e in
                            [(SIZE_IMAGES[v], float(r["max_r"]), float(r["sd"]))
                             for (g, f, v), r in table.items() if g == "size" and v == variants.FULL_POOL]))
        pts.sort()
        x, y, e = map(np.array, zip(*pts))
        ax.errorbar(x, y, yerr=e, fmt="^", ms=3.6, mfc=mfc, mec=st.REGIME_COLORS["Face"], mew=0.7,
                    color=st.REGIME_COLORS["Face"], lw=0.9, elinewidth=0.7, capsize=0,
                    ls="-" if group == "size" else (0, (3, 1.5)), label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([1e4, 1e5, 715810], ["10k", "100k", "716k"])
    ax.minorticks_off()
    ax.set_xlabel("face training images (FaceNet)")
    ax.set_ylabel(r"$\rho_{\max}$ (18 videos)")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.14), handlelength=2.2, borderaxespad=0.1)


def legend_block(fig):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=1.1, borderaxespad=0, fontsize=6, loc="lower left")
    for y, title, handles in ((0.045, "Training regime", regimes), (0.0, "Model family", families)):
        fig.text(0.02, y + 0.012, title, fontsize=6, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS["18"] / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    fig = plt.figure(figsize=(5.5, 4.1))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.99, top=0.95, bottom=0.18, hspace=0.8, wspace=0.38,
                          width_ratios=[1, 1, 1.45])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    d = fig.add_subplot(gs[1, :2]); e = fig.add_subplot(gs[1, 2])
    regime_panel(a, "18"); regime_panel(b, "54"); b.set_ylabel("")
    contrast_panel(c); chance_panel(d); size_panel(e)
    for ax, letter in zip((a, b, c, d, e), "abcde"):
        st.panel_label(ax, letter, x=-0.3 if ax in (a, b) else -0.2 if ax is not d else -0.1)
    legend_block(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
