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
  b     the four comparisons in the 11 ROIs, raw, with occlusion removed and with all controls
        removed: the difference above chance per ROI, dots on ROIs with p < .05 (maxT across ROIs)
  c, d  face-identity - ImageNet training and emotion fine-tuning - its start per ROI and on the
        whole scalp, raw (filled) and after controls (open), 95% CI, chance dash; * p < .05 (maxT
        across ROIs; whole scalp: Holm across the four comparisons), grey after controls
ROI labels carry the region colours of Fig. 1.

Reads analysis/paper_stats_54/contrasts{,_occlusion,_joint}_nc.csv and the ROIs of
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
CONDITIONS = [("", "raw"), ("occlusion", "occlusion\nremoved"), ("joint", "all controls\nremoved")]
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
    """Rows: raw, occlusion removed, all controls removed; columns: the four comparisons."""
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


# ---------------------------------------------------------------- c, d: one comparison per ROI

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


def roi_panel(ax, hyp, title, regime, tables):
    """Raw (filled) and after controls (open) per ROI: observed difference, 95% CI, chance dash."""
    color = st.REGIME_COLORS[regime]
    pick = {name: {r["scope"]: r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == "pooled"}
            for name in ("", "joint")}
    pos = scope_positions(set(pick[""]))
    top = 0.42
    for sc, x in pos.items():
        for k, (name, filled) in enumerate((("", True), ("joint", False))):
            r = pick[name][sc]
            dx = (k - 0.5) * 0.34
            c0 = float(r["null_mean"])
            m, lo, hi = float(r["observed"]), float(r["low"]) + c0, float(r["high"]) + c0
            ax.plot([x + dx - 0.1, x + dx + 0.1], [c0, c0], color=st.INK2, lw=1.0, solid_capstyle="butt", zorder=5)
            ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.3, mfc=color if filled else "white",
                        mec=color, mew=0.6, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
            p = float(r["p_holm"] or 1) if sc == "wholescalp" else float(r["p_maxT"])
            if p < .05:
                ax.text(x + dx, top, "*", ha="center", va="center", fontsize=7.7, color=st.INK if filled else st.MUTED)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(list(pos.values()), ["whole\nscalp" if s == "wholescalp" else s.split("-")[1] for s in pos],
                  fontsize=6.6)
    for tick, sc in zip(ax.get_xticklabels(), pos):
        tick.set_color(REGION_COLORS[region_of(sc)])
    for region, label in REGIONS:
        xs = [x for s, x in pos.items() if s.startswith(region + "-")]
        ax.text(np.mean(xs), -0.2, label.replace("-", "-\n") if len(label) > 8 else label,
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.6,
                color=REGION_COLORS[region], linespacing=0.95)
    ax.set_xlim(-0.7, pos["wholescalp"] + 0.6)
    ax.set_ylim(-0.24, top + 0.05)
    ax.set_title(title)
    fmt(ax)


def legend_block(fig, height):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    marks = [Line2D([], [], marker="o", ls="none", ms=3, mfc=st.INK2, mec=st.INK2, label="raw"),
             Line2D([], [], marker="o", ls="none", ms=3, mfc="white", mec=st.INK2, label="after controls"),
             Line2D([], [], color=st.INK2, lw=1.1, label="chance"),
             Line2D([], [], marker="$*$", ls="none", ms=4, color=st.INK, label="p < .05, corrected")]
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
    tables = {name: rows(name) for name, _ in CONDITIONS}
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    fig = plt.figure(figsize=(5.5, 7.5))
    gs = fig.add_gridspec(1, 4, left=0.1, right=0.99, top=0.95, bottom=0.79, wspace=0.1)
    a = [fig.add_subplot(gs[0, j]) for j in range(4)]
    layout = sensor_layout()
    for j, (ax, sc) in enumerate(zip(a, PANEL_A)):
        regime_panel(ax, sc, variant(tables[""]), variant(tables["joint"]))
        sensor_inset(ax, sc, layout)
        if j:
            ax.tick_params(labelleft=False)
    a[0].set_ylabel(r"$\rho_{\max}$ / noise ceiling")
    topo_grid(fig, (0.03, 0.385, 0.94, 0.28), topo_scalp(), tables)
    gs = fig.add_gridspec(1, 2, left=0.1, right=0.99, top=0.265, bottom=0.13, wspace=0.08)
    c, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    roi_panel(c, "H2 face vs ImageNet", "face-identity − ImageNet training", "Face", tables)
    roi_panel(d, "H3 fine-tuned vs start", "emotion fine-tuning − its start", "Emotion", tables)
    c.set_ylabel("difference − chance"); d.tick_params(labelleft=False)
    st.panel_label(a[0], "a", x=-0.42)
    fig.text(0.02, 0.69, "b", fontsize=9, fontweight="bold")
    st.panel_label(c, "c", x=-0.16); st.panel_label(d, "d", x=-0.05)
    legend_block(fig, fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
