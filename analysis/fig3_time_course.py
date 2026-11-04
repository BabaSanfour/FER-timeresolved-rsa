#!/usr/bin/env python3
"""Fig. 3: where and when pretraining raises EEG alignment (54 videos).

  a  scalp maps of the pretraining gain (H1 pooled over the six families, corrected for
     layer-selection bias) in six time bins; each ROI is coloured by its mean gain over the
     bin's windows and outlined when one of them is significant (maxT over the 11 x 13
     ROI-window cells)
  b  the gain over time in each of the 11 ROIs, grouped by region, with its 95% bootstrap
     interval over participants; dots mark the significant windows

Reads analysis/paper_stats_54/contrasts_time.csv (contrasts.py) and the ROIs of eeg_rdms/meta.json;
writes analysis/paper_stats_54/fig3_time_course.{pdf,png}.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon

import figure_style as st
import rsa_core

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
ROIS = json.loads((rsa_core.ROOT / "eeg_rdms/meta.json").read_text())["roi_channels"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300", "OccipitoTemporal": "#eda100"}
REGION_NAMES = {"Frontal": "Frontal", "Central": "Central", "CentroParietal": "Centro-par.", "OccipitoTemporal": "Occ.-temp."}
SIDES = {"L": "left", "M": "midline", "R": "right"}
BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13)]          # window indices (62.5 ms each)
VMAX = .06


def load():
    rows = [r for r in csv.DictReader((STATS / "contrasts_time.csv").open()) if r["contrast"] == "pooled"]
    starts = sorted({float(r["window_start_ms"]) for r in rows})
    ends = sorted({float(r["window_end_ms"]) for r in rows})
    data = {}
    for r in rows:
        w = starts.index(float(r["window_start_ms"]))
        data[r["scope"], w] = r
    return data, np.array(starts), np.array(ends)


def head_layout():
    import mne
    from mne.channels.layout import _find_topomap_coords
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(montage.ch_names, 256.0, "eeg")
    info.set_montage(montage)
    xy = np.array(_find_topomap_coords(info, picks=list(range(64))))
    radius = np.linalg.norm(xy, axis=1).max() * 1.07
    member = {ch: roi for roi, chans in ROIS.items() for ch in chans}
    n = 220
    gx, gy = np.meshgrid(np.linspace(-radius, radius, n), np.linspace(-radius, radius, n))
    nearest = np.argmin((gx[..., None] - xy[:, 0]) ** 2 + (gy[..., None] - xy[:, 1]) ** 2, -1)
    rois = list(ROIS)
    label = np.array([rois.index(member[c]) if c in member else -1 for c in montage.ch_names])[nearest]
    label = np.where(gx ** 2 + gy ** 2 <= radius ** 2, label, -2)
    return gx, gy, label, radius, rois


def head(ax, values, significant, layout):
    gx, gy, label, radius, rois = layout
    img = np.full(label.shape, np.nan)
    for i, roi in enumerate(rois):
        img[label == i] = values[roi]
    ax.imshow(img, origin="lower", extent=(-radius, radius, -radius, radius), cmap=DIVERGING,
              vmin=-VMAX, vmax=VMAX, interpolation="nearest")
    ax.contour(gx, gy, label, levels=np.arange(-1.5, len(rois)), colors="white", linewidths=0.4)
    for i, roi in enumerate(rois):
        if significant[roi]:
            ax.contour(gx, gy, (label == i).astype(float), levels=[.5], colors=st.INK, linewidths=0.9)
    ax.add_patch(Circle((0, 0), radius, fill=False, lw=0.6, ec=st.INK2))
    ax.add_patch(Polygon([[-0.12 * radius, 0.99 * radius], [0, 1.13 * radius], [0.12 * radius, 0.99 * radius]],
                         closed=False, fill=False, lw=0.6, ec=st.INK2))
    ax.set_xlim(-1.15 * radius, 1.15 * radius)
    ax.set_ylim(-1.1 * radius, 1.2 * radius)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig3_time_course"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    data, starts, ends = load()
    centres = (starts + ends) / 2
    layout = head_layout()
    fig = plt.figure(figsize=(5.5, 4.4))

    for k, (a, b) in enumerate(BINS):
        ax = fig.add_axes([0.01 + k * 0.15, 0.72, 0.146, 0.2])
        values = {roi: np.mean([float(data[roi, w]["corrected"]) for w in range(a, b)]) for roi in ROIS}
        significant = {roi: any(float(data[roi, w]["p_maxT"]) < .05 for w in range(a, b)) for roi in ROIS}
        head(ax, values, significant, layout)
        ax.set_title(f"{starts[a]:.0f}–{ends[b - 1]:.0f} ms", fontsize=6, pad=1)
    cax = fig.add_axes([0.925, 0.75, 0.01, 0.14])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-VMAX, VMAX)), cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label(r"$\Delta\rho$ − null mean", fontsize=5.5)
    fig.text(0.02, 0.955, "a", fontsize=8, fontweight="bold")
    fig.text(0.05, 0.955, "Pretraining gain over the scalp (outlined: significant)", fontsize=6.8, fontweight="bold")

    ax = fig.add_axes([0.17, 0.08, 0.62, 0.55])
    order = [f"{r}-{s}" for r in REGION_COLORS for s in "LMR" if f"{r}-{s}" in ROIS]
    step = 0.075
    gaps = np.cumsum([0] + [0.5 * step if order[i].split("-")[0] != order[i - 1].split("-")[0] else 0
                            for i in range(1, len(order))])
    top = (len(order) - 1) * step + gaps[-1]
    color = st.REGIME_COLORS["Generic"]
    for k, roi in enumerate(order):
        base = top - k * step - gaps[k]
        region, side = roi.split("-")
        m = np.array([float(data[roi, w]["corrected"]) for w in range(len(starts))])
        lo = np.array([float(data[roi, w]["low"]) for w in range(len(starts))])
        hi = np.array([float(data[roi, w]["high"]) for w in range(len(starts))])
        sig = np.array([float(data[roi, w]["p_maxT"]) < .05 for w in range(len(starts))])
        ax.axhline(base, color=st.RULE, lw=0.5, zorder=1)
        ax.fill_between(centres, base + lo, base + hi, color=color, alpha=0.25, lw=0, zorder=2)
        ax.plot(centres, base + m, color="#c2457a", lw=0.9, zorder=3)
        ax.plot(centres[sig], base + m[sig], "o", ms=2.6, color=st.INK, zorder=4)
        ax.text(-15, base + 0.012, f"{REGION_NAMES[region]} {SIDES[side]}", ha="right", va="center", fontsize=5.6)
        ax.plot(-8, base + 0.012, "o", ms=3, color=REGION_COLORS[region], clip_on=False)
    for a, _ in BINS[1:]:
        ax.axvline(starts[a], color=st.RULE, lw=0.4, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(0, ends[-1])
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("time from video onset (ms)")
    ax.set_ylim(-0.03, top + 0.09)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.plot([ends[-1] + 15] * 2, [0, 0.05], color=st.INK, lw=0.8, clip_on=False)
    ax.text(ends[-1] + 25, 0.025, r"$\Delta\rho$ = .05", fontsize=5.5, va="center", ha="left")
    fig.text(0.02, 0.665, "b", fontsize=8, fontweight="bold")
    fig.text(0.05, 0.665, "Pretraining gain in each region (six families, 95% CI; dots: significant)",
             fontsize=6.8, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
