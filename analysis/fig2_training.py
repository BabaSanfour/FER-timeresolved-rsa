#!/usr/bin/env python3
"""Fig. 2: what each step of training contributes to EEG alignment (54 videos).

  a     rho_max of every variant by training regime in four scalp regions (frontal,
        central, centro-parietal, occipito-temporal; the mean of the region's ROIs,
        emotion average): one marker per family (mean over its seeds or runs), lines join
        a family's regimes, bars are the mean over families
  b-e   the planned comparisons (contrasts.py) in each ROI and on the whole scalp (c: each
        video model against the image models, by family marker, before the controls):
        each participant's difference minus the null mean (grey), the mean and 95%
        bootstrap CI (filled), and the same after removing identity, occlusion, pixel,
        optical-flow and motion-energy structure jointly (open); * maxT-corrected p < .05
        across ROIs (whole scalp: Holm across H1-H4), grey * after the controls

Reads analysis/paper_stats_54/rho_max.csv, contrasts.csv, contrasts_participants.csv and
contrasts_joint.csv; writes analysis/paper_stats_54/fig2_training.{pdf,png}.
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

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
REGIONS = [("Frontal", "frontal"), ("Central", "central"), ("CentroParietal", "centro-parietal"),
           ("OccipitoTemporal", "occipito-temporal")]
VIDEO = (("Qwen3-VL Pretrained", "*"), ("M3DFEL Kinetics-400", "p"))
PANELS = [("H1 pretrained vs random", "pretrained − random", "Generic"),
          ("H4 video vs image", "video model − image models (Qwen3-VL, M3DFEL)", "Generic"),
          ("H2 face vs ImageNet", "face − ImageNet pretraining", "Face"),
          ("H3 fine-tuned vs start", "emotion fine-tuned − its start", "Emotion")]


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


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


def regime_panel(ax, table, region, label):
    rois = {r["roi"] for r in table if r["roi"].startswith(region + "-")}
    value = {}
    for r in table:
        if r["group"] == "main" and r["emotion"] == rsa_core.EMO_AVG and r["roi"] in rois:
            value.setdefault((r["family"], r["variant"]), []).append(float(r["max_r"]))
    xs = {regime: i for i, regime in enumerate(st.REGIMES)}
    by_regime = {regime: [] for regime in st.REGIMES}
    for fi, (family, vrows) in enumerate(variants.MAIN.items()):
        pts = []
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            y = float(np.mean(value[family, variant]))
            pts.append((xs[regime] + (fi - 2.5) * 0.07, y, regime)); by_regime[regime].append(y)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st.RULE, lw=0.6, zorder=2)
        m = st.MARKERS[family]
        for x, y, regime in pts:
            ax.plot(x, y, m, ms=st.size(m) - 0.6, mfc=st.REGIME_COLORS[regime], mec="white", mew=0.4, zorder=4)
    for regime, vals in by_regime.items():
        ax.bar(xs[regime], np.mean(vals), width=0.62, color=st.REGIME_COLORS[regime], alpha=0.25, lw=0, zorder=1)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r] for r in st.REGIMES], rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(st.REGIMES) - 0.45)
    ax.set_title(label)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def contrast_panel(ax, hyp, title, regime, raw, joint, participants, rng, series=(("pooled", "o"),)):
    """series: (contrast, marker) pairs; the pooled test carries the participants, a family its marker."""
    color = st.REGIME_COLORS[regime]
    pick = lambda table, c: {r["scope"]: r for r in table if r["hypothesis"] == hyp and r["contrast"] == c}
    pos = scope_positions(set(pick(raw, series[0][0])))
    tables = ((raw, True), (joint, False)) if len(series) == 1 else ((raw, True),)
    step = 0.34
    top = 0.11
    for sc, x in pos.items():
        k = 0
        for contrast, marker in series:
            if contrast == "pooled":
                d = np.array([float(r["difference"]) for r in participants
                              if r["hypothesis"] == hyp and r["scope"] == sc])
                ax.scatter(x - 0.14 + rng.uniform(-0.08, 0.08, len(d)), d, s=1.6, color=st.PARTICIPANT, lw=0,
                           zorder=2)
            for table, filled in tables:
                r = pick(table, contrast)[sc]
                dx = (k - (len(series) * len(tables) - 1) / 2) * step + (0.03 if len(series) == 1 else 0); k += 1
                m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
                ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=st.size(marker) - (1.6 if marker == "*" else 1),
                            mfc=color if filled else "white", mec=color, mew=0.7, ecolor=color, elinewidth=0.8,
                            capsize=0, zorder=4)
                if contrast == "pooled" and sc == "wholescalp":
                    p = float(r["p_holm"] or 1)
                else:
                    p = float(r["p_maxT"])
                if p < .05:
                    ax.text(x + dx, top - (0 if filled else 0.012), "*", ha="center", va="center", fontsize=6.5,
                            color=st.INK if filled else st.MUTED)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(list(pos.values()), ["whole\nscalp" if s == "wholescalp" else s.split("-")[1] for s in pos],
                  fontsize=5.4)
    for region, label in REGIONS:
        xs = [x for s, x in pos.items() if s.startswith(region + "-")]
        ax.text(np.mean(xs), -0.22, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.4,
                color=st.INK2)
    ax.set_xlim(-0.6, pos["wholescalp"] + 0.6)
    ax.set_ylim(-0.1, top + 0.012)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def legend_block(fig):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    estimates = [Line2D([], [], marker="o", ls="none", ms=1.8, mfc=st.PARTICIPANT, mec="none", label="participants"),
                 Line2D([], [], marker="o", ls="none", ms=3, mfc=st.INK2, mec=st.INK2, label="mean, 95% CI"),
                 Line2D([], [], marker="o", ls="none", ms=3, mfc="white", mec=st.INK2, label="after controls")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=1.1, borderaxespad=0, fontsize=6, loc="lower left")
    for y, title, handles in ((0.085, "Training regime", regimes), (0.05, "Model family", families),
                              (0.015, "Comparisons", estimates)):
        fig.text(0.02, y + 0.008, title, fontsize=6, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    table = rows(STATS / "rho_max.csv")
    raw, joint = rows(STATS / "contrasts.csv"), rows(STATS / "contrasts_joint.csv")
    participants = rows(STATS / "contrasts_participants.csv")
    fig = plt.figure(figsize=(5.5, 5.8))
    gs_a = fig.add_gridspec(1, 4, left=0.09, right=0.99, top=0.95, bottom=0.75, wspace=0.12)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    for i, (ax, (region, label)) in enumerate(zip(axes_a, REGIONS)):
        regime_panel(ax, table, region, label)
        if i:
            ax.tick_params(labelleft=False)
    top = max(ax.get_ylim()[1] for ax in axes_a)
    for ax in axes_a:
        ax.set_ylim(0, top)
    axes_a[0].set_ylabel(r"$\rho_{\max}$")
    gs_b = fig.add_gridspec(2, 2, left=0.09, right=0.99, top=0.65, bottom=0.2, hspace=0.75, wspace=0.08)
    rng = np.random.default_rng(0)
    axes_b = []
    for k, (hyp, title, regime) in enumerate(PANELS):
        ax = fig.add_subplot(gs_b[k // 2, k % 2])
        contrast_panel(ax, hyp, title, regime, raw, joint, participants, rng,
                       VIDEO if hyp.startswith("H4") else (("pooled", "o"),))
        if k % 2:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\Delta\rho$ − null mean")
        axes_b.append(ax)
    st.panel_label(axes_a[0], "a", x=-0.36)
    for ax, letter in zip(axes_b, "bcde"):
        st.panel_label(ax, letter, x=-0.16 if ax in axes_b[::2] else -0.05)
    legend_block(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
