#!/usr/bin/env python3
"""Fig. 4: study-wide clusters, and face training by amount of data (54 videos).

  a     number of layer x time grids (of 12 scopes x 4 emotion conditions) with a cluster that
        survives the study-wide maximum statistic (cluster_permutation.py), by training regime:
        one marker per family, lines join a family's regimes, bars are the mean over families
  b     FaceNet by number of face training images, occipito-temporal rho_max (the mean of the
        three occipito-temporal ROIs, emotion average): fixed epochs and step-matched training,
        mean +/- SD over ten runs; the ten random seeds as a band

Reads analysis/cluster_permutation_54/grid_summary.csv and
analysis/paper_stats_54/rho_max_per_rdm.csv; writes analysis/paper_stats_54/fig4_clusters.{pdf,png}.
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
CLUSTERS = rsa_core.ROOT / "analysis/cluster_permutation_54/grid_summary.csv"
REGION = "OccipitoTemporal"
SIZE_IMAGES = {"10k": 1e4, "50k": 5e4, "100k": 1e5, "200k": 2e5, "300k": 3e5, "500k": 5e5, "715k": 715810}


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def cluster_panel(ax, path):
    counts, n_grids = {}, {}
    for r in rows(path):
        key = (r["family"], r["variant"])
        counts[key] = counts.get(key, 0) + int(r["survives_study"])
        n_grids[key] = n_grids.get(key, 0) + 1
    xs = {regime: i for i, regime in enumerate(st.REGIMES)}
    by_regime = {regime: [] for regime in st.REGIMES}
    for fi, (family, vrows) in enumerate(variants.MAIN.items()):
        pts = []
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            y = counts[family, variant]
            pts.append((xs[regime] + (fi - 2.5) * 0.07, y, regime)); by_regime[regime].append(y)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st.RULE, lw=0.6, zorder=2)
        m = st.MARKERS[family]
        for x, y, regime in pts:
            ax.plot(x, y, m, ms=st.size(m), mfc=st.REGIME_COLORS[regime], mec="white", mew=0.4, zorder=4)
    for regime, vals in by_regime.items():
        ax.bar(xs[regime], np.mean(vals), width=0.62, color=st.REGIME_COLORS[regime], alpha=0.25, lw=0, zorder=1)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r] for r in st.REGIMES], rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(st.REGIMES) - 0.45)
    ax.set_ylabel(f"grids with a study-wide\ncluster (of {max(n_grids.values())})")


def region_values(per_rdm, group, variant=None):
    """{variant: [occipito-temporal rho_max of each RDM]} for one group (FaceNet only)."""
    by_rdm = {}
    for r in per_rdm:
        if (r["group"] == group and r["family"] == "FaceNet" and r["emotion"] == rsa_core.EMO_AVG
                and r["roi"].startswith(REGION + "-") and (variant is None or r["variant"] == variant)):
            by_rdm.setdefault((r["variant"], r["rdm"]), []).append(float(r["max_r"]))
    out = {}
    for (v, _), vals in by_rdm.items():
        out.setdefault(v, []).append(np.mean(vals))
    return out


def size_panel(ax, per_rdm):
    rnd = region_values(per_rdm, "main", "Random")["Random"]
    m, sd = np.mean(rnd), np.std(rnd, ddof=1)
    ax.axhspan(m - sd, m + sd, color=st.REGIME_COLORS["Random"], alpha=0.25, lw=0)
    ax.text(7e5, m - sd - 0.002, "random (10 seeds)", fontsize=6.5, color=st.INK2, va="top", ha="right")
    size = region_values(per_rdm, "size")
    color = st.REGIME_COLORS["Face"]
    for group, mfc, label in (("size", color, "fixed epochs"), ("step", "white", "matched steps")):
        vals = region_values(per_rdm, group)
        if group == "step":                                  # the full pool is the end point of both
            vals[variants.FULL_POOL] = size[variants.FULL_POOL]
        pts = sorted((SIZE_IMAGES[v], np.mean(x), np.std(x, ddof=1)) for v, x in vals.items())
        x, y, e = map(np.array, zip(*pts))
        ax.errorbar(x, y, yerr=e, fmt="^", ms=3.6, mfc=mfc, mec=color, mew=0.7, color=color, lw=0.9,
                    elinewidth=0.7, capsize=0, ls="-" if group == "size" else (0, (3, 1.5)), label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([1e4, 1e5, 715810], ["10k", "100k", "716k"])
    ax.minorticks_off()
    ax.set_xlabel("face training images (FaceNet)")
    ax.set_ylabel(r"$\rho_{\max}$, occipito-temporal")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.legend(loc="lower right", handlelength=2.2, borderaxespad=0.1)


def legend_block(fig):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=1.1, borderaxespad=0, fontsize=7.2, loc="lower left")
    for y, title, handles in ((0.07, "Training regime", regimes), (0.0, "Model family", families)):
        fig.text(0.02, y + 0.022, title, fontsize=7.2, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig4_clusters"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    fig = plt.figure(figsize=(5.5, 2.2))
    gs = fig.add_gridspec(1, 3, left=0.1, right=0.99, top=0.88, bottom=0.36, wspace=0.55, width_ratios=[1, 1.2, 1.2])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    cluster_panel(a, CLUSTERS)
    a.set_ylim(bottom=0)
    size_panel(b, rows(STATS / "rho_max_per_rdm.csv"))
    c.text(0.5, 0.5, "accuracy x alignment\n(probe running)", ha="center", va="center", transform=c.transAxes,
           fontsize=7.2, color=st.MUTED)
    for ax, letter in ((a, "a"), (b, "b"), (c, "c")):
        st.panel_label(ax, letter, x=-0.36)
    legend_block(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
