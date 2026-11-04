#!/usr/bin/env python3
"""Fig. 3: when and where pretraining raises EEG alignment (54 videos).

  a-d  the pretraining gain (H1, pooled over the six families, minus the null mean) over time in
       each scalp region (mean of the region's ROIs): mean over participants with a 95% bootstrap
       interval (54 videos), the same for 18 videos (dashed), a bar over the windows in which one of the region's ROIs is significant (maxT
       across the 11 x 13 ROI-window cells) and a triangle at the peak
  e    the gain over the scalp in three time bins (each electrode takes its ROI's value; P9 and P10,
       in no ROI, are left out); electrodes of ROIs with a significant window are marked

Reads analysis/paper_stats[_54]/contrasts_participants.csv and contrasts_time.csv, and the ROIs of eeg_rdms/meta.json; writes
analysis/paper_stats_54/fig3_time_course.{pdf,png}.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import figure_style as st
import rsa_core

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
STATS18 = rsa_core.ROOT / "analysis/paper_stats"
ROIS = json.loads((rsa_core.ROOT / "eeg_rdms/meta.json").read_text())["roi_channels"]
REGIONS = [("OccipitoTemporal", "occipito-temporal"), ("CentroParietal", "centro-parietal"),
           ("Central", "central"), ("Frontal", "frontal")]
BINS = [(0, 2), (2, 4), (4, 8)]                       # window indices: 0-125, 125-250, 250-500 ms
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
VMAX = .05
N_BOOT = 10000
H1 = "H1 pretrained vs random"


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load(stats=STATS):
    cells = [r for r in rows(stats / "contrasts_time.csv") if r["contrast"] == "pooled"]
    joint = cells
    starts = sorted({float(r["window_start_ms"]) for r in cells})
    ends = sorted({float(r["window_end_ms"]) for r in cells})
    cell = {(r["scope"], starts.index(float(r["window_start_ms"]))): r for r in cells}
    jcell = {(r["scope"], starts.index(float(r["window_start_ms"]))): r for r in joint}
    part = {}
    for r in rows(stats / "contrasts_participants.csv"):
        if r["hypothesis"] == "time " + H1:
            part.setdefault((r["scope"], int(r["window"])), {})[int(r["participant"])] = float(r["difference"])
    return cell, jcell, part, np.array(starts), np.array(ends)


REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300",
                 "OccipitoTemporal": "#eda100"}                        # as in Fig. 1
BAR, PEAK = .085, .094                                # height of the significance bars and peak marks


def region_mean(part, rois, W, rng):
    n = len(part[rois[0], 0])
    x = np.array([[np.mean([part[s, w][p] for s in rois]) for w in range(W)] for p in range(n)])   # (S, W)
    boot = x[rng.integers(0, n, (N_BOOT, n))].mean(1)
    return x.mean(0), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


def region_panel(ax, region, label, cell, part, part18, starts, ends, rng):
    rois = [s for s in ROIS if s.startswith(region + "-")]
    W = len(starts)
    centres = (starts + ends) / 2
    color = st.REGIME_COLORS["Generic"]
    m18, lo18, hi18 = region_mean(part18, rois, W, rng)
    ax.plot(centres, m18, color=color, lw=0.7, ls=(0, (2, 1.2)), alpha=0.9, zorder=2)
    m, lo, hi = region_mean(part, rois, W, rng)
    ax.fill_between(centres, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(centres, m, color=color, lw=1.1, zorder=3)
    sig = np.array([any(float(cell[s, w]["p_maxT"]) < .05 for s in rois) for w in range(W)])
    for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, sig.astype(int), 0]) == v) for v in (1, -1)]):
        ax.plot([starts[a], ends[b - 1]], [BAR, BAR], color=st.INK, lw=1.6, solid_capstyle="butt")
    if sig.any():
        peak = int(np.argmax(np.where(sig, m, -np.inf)))
        ax.plot(centres[peak], PEAK, "v", ms=3.5, color=st.INK)
        ax.text(centres[peak] + 40, PEAK, f"{centres[peak]:.0f} ms", ha="left", va="center", fontsize=5.6)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0, ends[-1])
    ax.set_xticks([0, 200, 400, 600, 800], ["0", "", "400", "", "800"])
    ax.set_ylim(-0.03, 0.105)
    ax.set_title(label, color=REGION_COLORS[region])
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def scalp_info():
    """MNE info of the electrodes that belong to an ROI, with each one's ROI."""
    import mne
    member = {ch: roi for roi, chans in ROIS.items() for ch in chans}
    montage = mne.channels.make_standard_montage("biosemi64")
    names = [c for c in montage.ch_names if c in member]
    info = mne.create_info(names, 256.0, "eeg")
    info.set_montage(montage)
    return info, [member[c] for c in names]


def head(ax, values, significant, scalp):
    import mne
    info, roi_of = scalp
    data = np.array([values[r] for r in roi_of])
    mask = np.array([significant[r] for r in roi_of])
    im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False, cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0,
                                 sensors=False, mask=mask, extrapolate="head",
                                 mask_params=dict(marker="o", markersize=1.6, markerfacecolor=st.INK,
                                                  markeredgecolor="none"))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    cell, _, part, starts, ends = load()
    _, _, part18, _, _ = load(STATS18)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(5.5, 3.3))
    gs = fig.add_gridspec(1, 4, left=0.08, right=0.97, top=0.9, bottom=0.52, wspace=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (region, label) in zip(axes, REGIONS):
        region_panel(ax, region, label, cell, part, part18, starts, ends, rng)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\Delta\rho$ − chance")
    fig.text(0.535, 0.43, "time from onset (ms)", ha="center", fontsize=6.5)
    color = st.REGIME_COLORS["Generic"]
    handles = [Line2D([], [], color=color, lw=1.1, label="pretrained − random,\n54 videos, 95% CI"),
               Line2D([], [], color=color, lw=0.7, ls=(0, (2, 1.2)), label="18 videos"),
               Line2D([], [], color=st.INK, lw=1.6, label="significant (maxT)"),
               Line2D([], [], color=st.INK, marker="v", ls="none", ms=3.5, label="peak")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.05), frameon=False, fontsize=5.6,
               handlelength=1.8, labelspacing=0.5)
    scalp = scalp_info()
    for k, (a, b) in enumerate(BINS):
        ax = fig.add_axes([0.34 + k * 0.17, 0.02, 0.15, 0.3])
        values = {s: np.mean([float(cell[s, w]["corrected"]) for w in range(a, b)]) for s in ROIS}
        significant = {s: any(float(cell[s, w]["p_maxT"]) < .05 for w in range(a, b)) for s in ROIS}
        im = head(ax, values, significant, scalp)
        ax.set_title(f"{starts[a]:.0f}–{ends[b - 1]:.0f} ms", fontsize=6, pad=1)
    cax = fig.add_axes([0.86, 0.07, 0.01, 0.2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label(r"$\Delta\rho$ − chance", fontsize=5.5)
    st.panel_label(axes[0], "a", x=-0.32)
    fig.text(0.31, 0.33, "b", fontsize=8, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
