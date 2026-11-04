#!/usr/bin/env python3
"""Fig. 2: what each step of training contributes to EEG alignment (54 videos).

Every value is a fraction of the upper EEG noise ceiling: contrasts.py --normalize divides each
cell of each grid (scope, emotion, window) by its ceiling before the peak is selected, the
permuted grids alike, so the chance levels (dark dashes, the mean of the permutation null with
the same selection) and the p values are on the same scale. "After controls": identity,
occlusion, pixel, optical-flow and motion-energy structure removed jointly from the EEG and model
RDMs, with the ceiling of the EEG after the same removal.

  a     each variant at the three occipito-temporal ROIs and on the whole scalp, by training regime:
        raw (filled bar and markers) and after controls (open), one marker per family; bars are the
        mean over families, dashes their chance level; a sensor map in each panel
  b     the four comparisons in the 11 ROIs, raw and with all controls removed: the difference
        above chance per ROI, dots on ROIs with p < .05 (maxT across ROIs)
  c     what each control removes: the four comparisons at the occipito-temporal and central region
        means (mean over the region's ROIs), raw, with each control removed alone and with all
        removed; difference above chance with 95% CI, filled where p < .05 (maxT across the four regions)
ROI labels carry the region colours of Fig. 1.

Reads analysis/paper_stats_54/contrasts{,_identity,_occlusion,_pixel,_optical_flow,_motion_energy,_joint}_nc.csv and the ROIs of
eeg_rdms/meta.json; writes analysis/paper_stats_54/fig2_training.{pdf,png}.
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
from matplotlib.patches import Patch

import figure_style as st
import rsa_core
import variants

STATS = rsa_core.ROOT / "analysis/paper_stats_54"
ROI_CHANNELS = json.loads((rsa_core.ROOT / "eeg_rdms/meta.json").read_text())["roi_channels"]
REGIONS = [("Frontal", "frontal"), ("Central", "central"), ("CentroParietal", "centro-parietal"),
           ("OccipitoTemporal", "occipito-temporal")]
REGION_NAMES = dict(REGIONS)
REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300",
                 "OccipitoTemporal": "#eda100", "wholescalp": st.INK}          # as in Fig. 1
SIDES = {"L": "left", "M": "midline", "R": "right"}
PANEL_A = ["OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R", "wholescalp"]
COMPARISONS = [("H1 pretrained vs random", "generic pretraining\n− random", "Generic"),
               ("H4 video vs image", "video\n− image models", "Generic"),
               ("H2 face vs ImageNet", "face identity\n− ImageNet", "Face"),
               ("H3 fine-tuned vs start", "emotion fine-tuning\n− its start", "Emotion")]
CONDITIONS = [("", "raw"), ("joint", "all controls\nremoved")]                    # rows of b
CONTROL_SETS = [("", "raw"), ("identity", "identity"), ("occlusion", "occlusion"), ("pixel", "pixel"),
                ("optical_flow", "optical flow"), ("motion_energy", "motion energy"), ("joint", "all")]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
VMAX = 0.2


def rows(name=""):
    with (STATS / f"contrasts{'_' + name if name else ''}_nc.csv").open() as fh:
        return list(csv.DictReader(fh))


def region_of(scope):
    return scope.split("-")[0]


def full_label(scope):
    if scope == "wholescalp":
        return "whole scalp"
    region, side = scope.split("-")
    return f"{REGION_NAMES[region]}\n{SIDES[side]}"


def fmt(ax):
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


# ---------------------------------------------------------------- a: regimes, raw and after controls

def sensor_layout():
    """2-D positions of the 64 electrodes (MNE topomap projection) and the head radius."""
    import mne
    from mne.channels.layout import _find_topomap_coords
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(montage.ch_names, 256.0, "eeg")
    info.set_montage(montage)
    xy = np.array(_find_topomap_coords(info, picks=list(range(64))))
    return dict(zip(montage.ch_names, xy)), np.linalg.norm(xy, axis=1).max() * 1.08


def sensor_inset(ax, scope, layout):
    """A small head in the panel's upper left with the scope's electrodes in its region colour."""
    pos, radius = layout
    chans = set(pos) if scope == "wholescalp" else set(ROI_CHANNELS[scope])
    color = REGION_COLORS[region_of(scope)] if scope != "wholescalp" else st.INK2
    ins = ax.inset_axes([0.0, 0.74, 0.24, 0.26])
    t = np.linspace(0, 2 * np.pi, 100)
    ins.plot(radius * np.cos(t), radius * np.sin(t), color=st.INK2, lw=0.4)
    ins.plot([-0.12 * radius, 0, 0.12 * radius], [0.99 * radius, 1.15 * radius, 0.99 * radius], color=st.INK2, lw=0.4)
    for ch, (x, y) in pos.items():
        on = ch in chans
        ins.plot(x, y, "o", ms=1.3 if on else 0.7, mfc=color if on else st.RULE, mec="none")
    ins.set_xlim(-1.2 * radius, 1.2 * radius); ins.set_ylim(-1.15 * radius, 1.25 * radius)
    ins.set_aspect("equal"); ins.axis("off")


def regime_panel(ax, scope, raw, joint):
    """Each regime: the raw bar (filled) and the bar after controls (open), family markers on each,
    chance dashes."""
    for xi, regime in enumerate(st.REGIMES):
        c = st.REGIME_COLORS[regime]
        for table, dx, filled in ((raw, -0.2, True), (joint, 0.2, False)):
            vals, chance = [], []
            for fi, (family, vrows) in enumerate(variants.MAIN.items()):
                for variant, _ in vrows:
                    if st.REGIME_OF[variant] != regime:
                        continue
                    r = table[f"{family} {variant}", scope]
                    y = float(r["observed"]); vals.append(y); chance.append(float(r["null_mean"]))
                    m = st.MARKERS[family]
                    ax.plot(xi + dx + (fi - 2.5) * 0.04, y, m, ms=st.size(m) - 1.6, mfc=c if filled else "white",
                            mec="white" if filled else c, mew=0.3 if filled else 0.5, zorder=4)
            ax.bar(xi + dx, np.mean(vals), width=0.36, fc=c if filled else "white", alpha=0.3 if filled else 1,
                   ec="none" if filled else c, lw=0.6, zorder=1)
            ax.plot([xi + dx - 0.18, xi + dx + 0.18], [np.mean(chance)] * 2, color=st.INK2, lw=1.0,
                    solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r] for r in st.REGIMES], fontsize=6.4, rotation=35,
                  ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.5, len(st.REGIMES) - 0.5)
    ax.set_ylim(0, 0.76); ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_title(full_label(scope), fontsize=7.2, color=REGION_COLORS[region_of(scope)], pad=2, linespacing=1.0)
    fmt(ax)


# ---------------------------------------------------------------- b: topomap grid

def topo_scalp():
    """MNE info of the electrodes in an ROI, and the ROI of each (each electrode takes its ROI's value;
    P9 and P10, in no ROI, are left out)."""
    import mne
    member = {ch: roi for roi, chans in ROI_CHANNELS.items() for ch in chans}
    montage = mne.channels.make_standard_montage("biosemi64")
    names = [c for c in montage.ch_names if c in member]
    info = mne.create_info(names, 256.0, "eeg")
    info.set_montage(montage)
    return info, [member[c] for c in names]


def topo_grid(fig, box, scalp, tables):
    """Rows: raw and all controls removed; columns: the four comparisons."""
    import mne
    x0, y0, w, h = box
    info, roi_of = scalp
    cw, rh = w / (len(COMPARISONS) + 0.9), h / len(CONDITIONS)
    for i, (name, rlabel) in enumerate(CONDITIONS):
        fig.text(x0, y0 + h - (i + 0.5) * rh, rlabel, ha="left", va="center", fontsize=6.8, color=st.INK2)
        t = {(r["hypothesis"], r["scope"]): r for r in tables[name] if r["contrast"] == "pooled"}
        for j, (hyp, clabel, regime) in enumerate(COMPARISONS):
            ax = fig.add_axes([x0 + (j + 0.9) * cw, y0 + h - (i + 1) * rh, cw * 0.95, rh * 0.95])
            im, _ = mne.viz.plot_topomap(
                np.array([float(t[hyp, r]["corrected"]) for r in roi_of]), info, axes=ax, show=False,
                cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0, sensors=False, extrapolate="head",
                mask=np.array([float(t[hyp, r]["p_maxT"]) < .05 for r in roi_of]),
                mask_params=dict(marker="o", markersize=1.2, markerfacecolor=st.INK, markeredgecolor="none"))
            if i == 0:
                ax.set_title(clabel, fontsize=6.8, pad=1, color=st.REGIME_COLORS[regime], linespacing=0.95)
    cax = fig.add_axes([x0 + 0.9 * cw + 0.03, y0 - 0.035, len(COMPARISONS) * cw - 0.06, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=6.2, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference − chance (fraction of the noise ceiling)", fontsize=6.6)


# ---------------------------------------------------------------- c: what each control removes

def control_panel(ax, region, tables):
    """The four comparisons at a region mean under each control set (x): difference above chance,
    95% CI; filled where p < .05 (maxT across the four regions)."""
    for ci, (hyp, label, regime) in enumerate(COMPARISONS):
        color = st.REGIME_COLORS[regime]
        xs = np.arange(len(CONTROL_SETS)) + (ci - 1.5) * 0.09
        ys = []
        for k, (name, _) in enumerate(CONTROL_SETS):
            r = next(r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == "pooled"
                     and r["scope"] == region)
            m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
            ys.append(m)
            sig = float(r["p_maxT"]) < .05                    # filled: p < .05 (maxT); open: n.s.
            ax.errorbar(xs[k], m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.2, mfc=color if sig else "white",
                        mec=color, mew=0.7, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
        ax.plot(xs, ys, color=color, lw=0.9, ls=(0, (3, 1.5)) if ci == 1 else "-", zorder=3, label=label)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(CONTROL_SETS)), [lab for _, lab in CONTROL_SETS], fontsize=6.6, rotation=35,
                  ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(CONTROL_SETS) - 0.5)
    ax.set_title(REGION_NAMES[region], color=REGION_COLORS[region])
    fmt(ax)


def legend_block(fig, height):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    marks = [Patch(fc=st.INK2, alpha=0.3, ec="none", label="raw"),
             Patch(fc="white", ec=st.INK2, lw=0.6, label="all controls removed"),
             Line2D([], [], color=st.INK2, lw=1.1, label="chance"),
             Line2D([], [], marker="o", ls="none", ms=3.2, mfc=st.INK2, mec=st.INK2, label="p < .05, corrected"),
             Line2D([], [], marker="o", ls="none", ms=3.2, mfc="white", mec=st.INK2, label="n.s.")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=0.9, borderaxespad=0, fontsize=7.2, loc="lower left")
    dy = 0.16 / height
    for y, title, handles in ((2 * dy, "Training regime", regimes), (dy, "Model family", families),
                              (0.0, "Estimates", marks)):
        fig.text(0.02, y + 0.04 / height, title, fontsize=7.2, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    tables = {name: rows(name) for name, _ in CONTROL_SETS}
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    fig = plt.figure(figsize=(5.5, 6.9))
    gs = fig.add_gridspec(1, 4, left=0.1, right=0.99, top=0.95, bottom=0.79, wspace=0.1)
    a = [fig.add_subplot(gs[0, j]) for j in range(4)]
    layout = sensor_layout()
    for j, (ax, sc) in enumerate(zip(a, PANEL_A)):
        regime_panel(ax, sc, variant(tables[""]), variant(tables["joint"]))
        sensor_inset(ax, sc, layout)
        if j:
            ax.tick_params(labelleft=False)
    a[0].set_ylabel(r"$\rho_{\max}$ / noise ceiling")
    topo_grid(fig, (0.03, 0.44, 0.94, 0.2), topo_scalp(), tables)
    gs = fig.add_gridspec(1, 2, left=0.1, right=0.74, top=0.31, bottom=0.17, wspace=0.1)
    c1, c2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    control_panel(c1, "OccipitoTemporal", tables); control_panel(c2, "Central", tables)
    lo = min(c1.get_ylim()[0], c2.get_ylim()[0]); hi = max(c1.get_ylim()[1], c2.get_ylim()[1])
    for ax in (c1, c2):
        ax.set_ylim(lo, hi)
    c2.tick_params(labelleft=False)
    c1.set_ylabel("difference − chance")
    fig.legend(*c1.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0.76, 0.245), frameon=False,
               fontsize=6.8, handlelength=2.0, labelspacing=0.9, title="comparisons", title_fontsize=7)
    st.panel_label(a[0], "a", x=-0.42)
    fig.text(0.02, 0.665, "b", fontsize=9, fontweight="bold")
    st.panel_label(c1, "c", x=-0.32)
    legend_block(fig, fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
