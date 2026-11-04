#!/usr/bin/env python3
"""Fig. 2: pretraining, not emotion training, raises EEG alignment (54 videos).

  a  the planned contrasts (contrasts.py), corrected for layer-selection bias: each
     family's contrast with its 95% CI, and the pooled test (diamond) with its
     Holm-corrected p
  b  emotion decoding does not predict alignment: held-out linear-probe accuracy
     (probe.py) against rho_max above its own permutation null (rho_max_null.py),
     one point per variant

Reads analysis/paper_stats_54/contrasts.csv, probe.csv, rho_max_null.csv; writes
analysis/paper_stats_54/fig2_training.{pdf,png}.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import figure_style as st
import rsa_core

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
HYPOTHESES = [("H1 pretrained vs random", "Pretrained − random", "Generic"),
              ("H2 face vs ImageNet", "Face − ImageNet", "Face"),
              ("H3 fine-tuned vs start", "Emotion fine-tuned − its start", "Emotion"),
              ("H4 video vs image", "Video − image models", "Generic")]


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def p_text(p):
    return "p < .001" if p < .001 else ("p = " + (f"{p:.3f}" if p < .01 else f"{p:.2f}").replace("0.", ".", 1))


def forest(ax):
    table = rows(STATS / "contrasts.csv")
    y, ticks, labels = 0, [], []
    for hyp, title, regime in HYPOTHESES:
        color = st.REGIME_COLORS[regime]
        pooled = next(r for r in table if r["level"] == "primary" and r["hypothesis"] == hyp)
        ph = float(pooled["p_holm"])
        ax.text(-0.0255, y, title, fontsize=6.3, fontweight="bold", va="center", ha="left")
        ax.text(0.0365, y, f"Holm {p_text(ph)}", fontsize=5.8, va="center", ha="left",
                color=st.INK if ph < .05 else st.MUTED, fontweight="bold" if ph < .05 else "normal")
        y -= 1
        for r in (r for r in table if r["level"] == "secondary" and r["hypothesis"] == hyp):
            m, lo, hi = float(r["corrected_difference"]), float(r["ci95_low"]), float(r["ci95_high"])
            ax.plot([lo, hi], [y, y], color=color, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot(m, y, "o", ms=3.2, color=color, mec="white", mew=0.3, zorder=3)
            ticks.append(y); labels.append(r["contrast"].replace(" Pretrained", "").replace(" Kinetics-400", ""))
            y -= 1
        m, lo, hi = (float(pooled[k]) for k in ("corrected_difference", "ci95_low", "ci95_high"))
        ax.fill([lo, m, hi, m], [y, y + 0.32, y, y - 0.32], color=st.INK if ph < .05 else st.MUTED, lw=0, zorder=3)
        ticks.append(y); labels.append("pooled")
        y -= 1.6
    ax.axvline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ticks, labels, fontsize=5.6)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 1.1, 0.7)
    ax.set_xlim(-0.026, 0.05)
    ax.set_xticks([-0.02, 0, 0.02, 0.04])
    ax.xaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.set_xlabel(r"difference in $\rho$, corrected for selection bias")


def probe(ax):
    acc = {(r["family"], r["variant"]): float(r["accuracy"]) for r in rows(STATS / "probe.csv")}
    exc = {(r["family"], r["variant"]): float(r["excess"]) for r in rows(STATS / "rho_max_null.csv")
           if r["group"] == "main"}
    trained = [k for k in acc if k[1] != "Random"]
    for (family, variant), a in acc.items():
        regime = st.REGIME_OF[variant]
        m = st.MARKERS[family]
        ax.plot(a, exc[family, variant], m, ms=st.size(m) + 0.5, mfc=st.REGIME_COLORS[regime], mec="white",
                mew=0.4, zorder=3 if regime != "Random" else 2)
    r_t, p_t = spearmanr([acc[k] for k in trained], [exc[k] for k in trained])
    ax.text(0.97, 0.05, f"trained: r = {st.point_format(round(r_t, 2))}, {p_text(p_t)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=st.INK2)
    ax.axvline(1 / 3, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(1 / 3 + 0.005, ax.get_ylim()[1], "chance", fontsize=5.3, color=st.MUTED, va="top")
    ax.set_xlabel("held-out emotion decoding (accuracy)")
    ax.set_ylabel(r"$\rho_{\max}$ above chance")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    regimes = [Line2D([], [], marker="o", ls="none", ms=4, mfc=st.REGIME_COLORS[r], mec="white", label=st.REGIME_LABELS[r])
               for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", label=f)
                for f, m in st.MARKERS.items()]
    ax.legend(handles=regimes + families, ncol=3, loc="upper left", bbox_to_anchor=(-0.2, -0.3),
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0, fontsize=5.6)


def headline(fig, ax, letter, text, dx=0.0):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.05 + dx, pos.y1 + 0.03, letter, fontsize=8, fontweight="bold", va="bottom")
    fig.text(pos.x0 - 0.025 + dx, pos.y1 + 0.03, text, fontsize=6.8, fontweight="bold", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    fig = plt.figure(figsize=(5.5, 3.3))
    a = fig.add_axes([0.1, 0.12, 0.38, 0.8])
    b = fig.add_axes([0.62, 0.46, 0.35, 0.46])
    forest(a); probe(b)
    headline(fig, a, "a", "Pretraining raises alignment; specialization does not", dx=-0.05)
    headline(fig, b, "b", "Emotion decoding does not predict alignment", dx=-0.02)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
