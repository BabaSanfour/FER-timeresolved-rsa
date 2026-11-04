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
        above chance per ROI, dots on ROIs with p < .05 (maxT across ROIs). Emotion-tuned vs base
        model: each fine-tuned network against the network it was fine-tuned from
  c     what each control removes, one panel per comparison: raw, each control removed alone and all
        removed (shaded), at the occipito-temporal and central region means (mean over the region's
        ROIs); difference above chance with 95% CI, filled where p < .05 (maxT across the four regions)
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
COMPARISONS = [("H1 pretrained vs random", "pretrained vs random", "Generic"),
               ("H4 video vs image", "video vs image models", "Generic"),
               ("H2 face vs ImageNet", "face identity vs ImageNet", "Face"),
               ("H3 fine-tuned vs start", "emotion-tuned vs base model", "Emotion")]
CONDITIONS = [("", "raw"), ("joint", "controls\nremoved")]                          # rows of b
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
    cw, rh = w / (len(COMPARISONS) + 0.15), h / len(CONDITIONS)
    for i, (name, rlabel) in enumerate(CONDITIONS):
        fig.text(x0 + 0.1 * cw, y0 + h - (i + 0.5) * rh, rlabel, ha="center", va="center", fontsize=7,
                 color=st.INK2, rotation=90, linespacing=1.0)
        t = {(r["hypothesis"], r["scope"]): r for r in tables[name] if r["contrast"] == "pooled"}
        for j, (hyp, clabel, regime) in enumerate(COMPARISONS):
            ax = fig.add_axes([x0 + (j + 0.15) * cw, y0 + h - (i + 1) * rh, cw * 0.95, rh * 0.95])
            im, _ = mne.viz.plot_topomap(
                np.array([float(t[hyp, r]["corrected"]) for r in roi_of]), info, axes=ax, show=False,
                cmap=DIVERGING, vlim=(-VMAX, VMAX), contours=0, sensors=False, extrapolate="head",
                mask=np.array([float(t[hyp, r]["p_maxT"]) < .05 for r in roi_of]),
                mask_params=dict(marker="o", markersize=1.2, markerfacecolor=st.INK, markeredgecolor="none"))
            if i == 0:
                ax.set_title(clabel.replace(" vs ", "\nvs "), fontsize=7, pad=2, linespacing=1.0)
    cax = fig.add_axes([x0 + 0.15 * cw + 0.12, y0 - 0.03, len(COMPARISONS) * cw - 0.24, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=6.2, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference above chance (fraction of noise ceiling)", fontsize=6.6)


FAMILY_ROWS = ["ResNet18", "ResNet50", "FaceNet", "VGG16", "M3DFEL", "Qwen3-VL"]
VIDEO_CONTRAST = {"M3DFEL": "M3DFEL Kinetics-400", "Qwen3-VL": "Qwen3-VL Pretrained"}


def family_contrast(hyp, family, table):
    """The family's contrast name for a comparison, or None if the family has none."""
    name = VIDEO_CONTRAST.get(family) if hyp.startswith("H4") else family
    return name if name and any(r["hypothesis"] == hyp and r["contrast"] == name for r in table) else None


def head_rows(fig, box, scalp, spec, col_titles, row_label_size=6.6):
    """Generic grid of heads: spec = [(row label, [(table, hyp, contrast) or None per column])]."""
    import mne
    x0, y0, w, h = box
    info, roi_of = scalp
    ncol = len(col_titles)
    cw, rh = w / (ncol + 0.35), h / len(spec)
    for i, (rlabel, cells) in enumerate(spec):
        fig.text(x0 + 0.3 * cw, y0 + h - (i + 0.5) * rh, rlabel, ha="right", va="center", fontsize=row_label_size,
                 color=st.INK2)
        for j, cell in enumerate(cells):
            if cell is None:
                continue
            table, hyp, contrast = cell
            t = {r["scope"]: r for r in table if r["hypothesis"] == hyp and r["contrast"] == contrast}
            ax = fig.add_axes([x0 + (j + 0.35) * cw, y0 + h - (i + 1) * rh, cw * 0.95, rh * 0.95])
            mne.viz.plot_topomap(
                np.array([float(t[r]["corrected"]) for r in roi_of]), info, axes=ax, show=False, cmap=DIVERGING,
                vlim=(-VMAX, VMAX), contours=0, sensors=False, extrapolate="head",
                mask=np.array([float(t[r]["p_maxT"]) < .05 for r in roi_of]),
                mask_params=dict(marker="o", markersize=1.0, markerfacecolor=st.INK, markeredgecolor="none"))
    for j, title in enumerate(col_titles):
        fig.text(x0 + (j + 0.35 + 0.475) * cw, y0 + h + 0.004, title, ha="center", va="bottom", fontsize=6.8,
                 linespacing=1.0)


BEST = {"H1 pretrained vs random": "Qwen3-VL", "H4 video vs image": "Qwen3-VL Pretrained",
        "H2 face vs ImageNet": "ResNet50", "H3 fine-tuned vs start": "VGG16"}


def pooled_and_best(fig, box, scalp, tables):
    """b and c side by side, each 2 rows (raw, controls removed) x 4 columns (the comparisons); b = all
    models pooled, c = the strongest family of each comparison (named in the column title)."""
    import mne
    x0, y0, w, h = box
    info, roi_of = scalp
    lab_w, gap = 0.055, 0.03                                      # row-label column and gap between b and c
    hw = (w - lab_w - gap) / 8                                     # head width (figure fraction)
    rh = hw * fig.get_figwidth() / fig.get_figheight()             # square heads
    xb = [x0 + lab_w + j * hw for j in range(4)]
    xc = [x0 + lab_w + 4 * hw + gap + j * hw for j in range(4)]
    for i, (name, rlabel) in enumerate(CONDITIONS):
        yy = y0 + h - (i + 1) * rh
        fig.text(x0 + lab_w * 0.6, yy + rh / 2, rlabel, ha="center", va="center", fontsize=7, color=st.INK2,
                 rotation=90, linespacing=1.0)
        for xs, pick in ((xb, lambda hyp: "pooled"), (xc, lambda hyp: BEST[hyp])):
            for xx, (hyp, _, _) in zip(xs, COMPARISONS):
                t = {r["scope"]: r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == pick(hyp)}
                ax = fig.add_axes([xx + 0.03 * hw, yy, hw * 0.94, rh * 0.94])
                im, _ = mne.viz.plot_topomap(
                    np.array([float(t[r]["corrected"]) for r in roi_of]), info, axes=ax, show=False, cmap=DIVERGING,
                    vlim=(-VMAX, VMAX), contours=0, sensors=False, extrapolate="head",
                    mask=np.array([float(t[r]["p_maxT"]) < .05 for r in roi_of]),
                    mask_params=dict(marker="o", markersize=0.9, markerfacecolor=st.INK, markeredgecolor="none"))
    H = fig.get_figheight()
    top = y0 + h - 0.12 * rh                                        # just above the heads
    bottom = y0 + h - 2 * rh
    for xs, group, fam in ((xb, "all models (pooled)", False), (xc, "strongest family", True)):
        for xx, (hyp, label, _) in zip(xs, COMPARISONS):
            fig.text(xx + hw / 2, top, label.replace(" vs ", "\nvs "), ha="center", va="bottom", fontsize=6.2,
                     linespacing=1.0)
            if fam:                                                 # the family shown, under its heads
                fig.text(xx + hw / 2, bottom + 0.01 / H, BEST[hyp].split()[0], ha="center", va="top", fontsize=6.6,
                         color=st.INK2)
        fig.text((xs[0] + xs[-1] + hw) / 2, top + 0.24 / H, group, ha="center", va="bottom", fontsize=7.2)
    cax = fig.add_axes([xb[0] + hw, bottom - 0.25 / H, xc[-1] - xb[0] - hw, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(-VMAX), "0", st.point_format(VMAX)])
    cb.ax.tick_params(labelsize=6.2, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("difference above chance (fraction of noise ceiling)", fontsize=6.6)
    return xc[0], 2 * rh


# ---------------------------------------------------------------- c: what each control removes

C_REGIONS = [("OccipitoTemporal", "o"), ("Central", "s")]


def control_panel(ax, hyp, title, tables, labels=True, families=False):
    """One comparison: rows = control sets (raw, each control alone, all), the occipito-temporal and
    central region means (difference above chance, 95% CI), filled where p < .05 (maxT across the
    four regions)."""
    n = len(CONTROL_SETS)
    for k, (name, _) in enumerate(CONTROL_SETS):
        for j, (region, marker) in enumerate(C_REGIONS):
            r = next(r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == "pooled"
                     and r["scope"] == region)
            m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
            y = (n - 1 - k) + (0.5 - j) * 0.34
            color = REGION_COLORS[region]
            sig = float(r["p_maxT"]) < .05
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=marker, ms=3.2, mfc=color if sig else "white", mec=color,
                        mew=0.7, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
            if families and name in ("", "joint"):                 # each family at raw and all removed
                for fam in FAMILY_ROWS:
                    c = family_contrast(hyp, fam, tables[name])
                    if c is None:
                        continue
                    f = next(r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == c
                             and r["scope"] == region)
                    fm = st.MARKERS[fam]
                    ax.plot(float(f["corrected"]), y + (0.13 if j == 0 else -0.13), fm, ms=st.size(fm) - 2.2,
                            mfc=color if float(f["p_maxT"]) < .05 else "white", mec=color, mew=0.4, alpha=0.9,
                            zorder=3)
    ax.axvline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhspan(-0.5, 0.5, color=st.RULE, alpha=0.35, lw=0, zorder=0)          # all removed
    ax.set_yticks(range(n), [lab for _, lab in CONTROL_SETS][::-1] if labels else [""] * n, fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.2, 0.45 if families else 0.25); ax.set_xticks([-0.2, 0, 0.2, 0.4] if families else [-0.2, -0.1, 0, 0.1, 0.2])
    ax.xaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.set_title(title, fontsize=7, pad=3)
    ax.spines["left"].set_visible(False)


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


def layout_best(args, tables, variant):
    """a; b (pooled) and c (strongest family) side by side; d what each control removes."""
    H = 7.1
    f = lambda inch: 1 - inch / H                                  # inches from the top -> figure fraction
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(1, 4, left=0.1, right=0.99, top=f(0.3), bottom=f(1.55), wspace=0.1)
    a = [fig.add_subplot(gs[0, j]) for j in range(4)]
    layout = sensor_layout()
    for j, (ax, sc) in enumerate(zip(a, PANEL_A)):
        regime_panel(ax, sc, variant(tables[""]), variant(tables["joint"]))
        sensor_inset(ax, sc, layout)
        if j:
            ax.tick_params(labelleft=False)
    a[0].set_ylabel(r"$\rho_{\max}$ (fraction of noise ceiling)")
    xc, _ = pooled_and_best(fig, (0.02, f(3.97), 0.97, 1.4 / H), topo_scalp(), tables)
    top_d = f(4.88)
    gs = fig.add_gridspec(1, 4, left=0.16, right=0.99, top=top_d, bottom=f(5.88), wspace=0.12)
    ds = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, (ax, (hyp, title, _)) in enumerate(zip(ds, COMPARISONS)):
        control_panel(ax, hyp, title.replace(" vs ", "\nvs "), tables, labels=j == 0)
    ds[0].text(-0.62, 0.5, "removed", rotation=90, transform=ds[0].transAxes, ha="center", va="center", fontsize=7,
               color=st.INK2)
    fig.text(0.16, f(6.21), "difference above chance (fraction of noise ceiling)", ha="left", va="center", fontsize=7)
    fig.legend(handles=[Line2D([], [], marker=m, ls="none", ms=3.2, mfc=REGION_COLORS[r], mec=REGION_COLORS[r],
                               label=f"{REGION_NAMES[r]} (region mean)") for r, m in C_REGIONS],
               loc="center right", bbox_to_anchor=(0.99, f(6.38)), ncol=2, fontsize=6.8, handletextpad=0.2,
               columnspacing=1.0, frameon=False)
    for letter, x, y in (("a", 0.01, 0.05), ("b", 0.01, 2.12), ("c", xc - 0.02, 2.12), ("d", 0.01, 4.42)):
        fig.text(x, f(y), letter, fontsize=9, fontweight="bold", va="top")   # each panel's top-left corner
    legend_block(fig, H)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    ap.add_argument("--families", choices=["none", "grid", "best", "forest"], default="best",
                    help="per-family results: a head grid after controls, the best family per comparison, "
                         "or family markers in the control panel")
    args = ap.parse_args()
    st.setup()
    tables = {name: rows(name) for name, _ in CONTROL_SETS}
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    if args.families == "best":
        return layout_best(args, tables, variant)
    H = {"none": 7.0, "forest": 7.0, "grid": 9.6}[args.families]
    k = 7.0 / H                                     # vertical positions below are for a 7-inch figure
    Y = lambda y: 1 - (1 - y) * k
    fig = plt.figure(figsize=(5.5, H))
    gs = fig.add_gridspec(1, 4, left=0.1, right=0.99, top=Y(0.95), bottom=Y(0.79), wspace=0.1)
    a = [fig.add_subplot(gs[0, j]) for j in range(4)]
    layout = sensor_layout()
    for j, (ax, sc) in enumerate(zip(a, PANEL_A)):
        regime_panel(ax, sc, variant(tables[""]), variant(tables["joint"]))
        sensor_inset(ax, sc, layout)
        if j:
            ax.tick_params(labelleft=False)
    a[0].set_ylabel(r"$\rho_{\max}$ (fraction of noise ceiling)")
    scalp = topo_scalp()
    if args.families == "best":
        bc_h = 3.3 / H                                             # four rows of heads
        xc = pooled_and_best(fig, (0.03, Y(0.70) - bc_h, 0.96, bc_h), scalp, tables)
        extra = (3.3 - 0.185 * 7.0 - 0.1) / H + 0.02
    else:
        topo_grid(fig, (0.06, Y(0.47), 0.93, 0.185 * k), scalp, tables)
    extra = extra if args.families == "best" else {"none": 0, "forest": 0, "grid": 2.85}[args.families] / H
    if args.families == "grid":
        spec = [(fam, [(tables["joint"], hyp, c) if (c := family_contrast(hyp, fam, tables["joint"])) else None
                       for hyp, _, _ in COMPARISONS]) for fam in FAMILY_ROWS]
        head_rows(fig, (0.06, Y(0.42) - extra, 0.93, extra - 0.075), scalp, spec,
                  [t.replace(" vs ", "\nvs ") for _, t, _ in COMPARISONS])
        fig.text(0.02, Y(0.42) - 0.045, "c", fontsize=9, fontweight="bold")
        fig.text(0.06, Y(0.42) - 0.045, "each family, after all controls", fontsize=7, color=st.INK2)
    elif args.families == "best":
        fig.text(0.02, Y(0.685), "b", fontsize=9, fontweight="bold")
        fig.text(xc - 0.03, Y(0.685), "c", fontsize=9, fontweight="bold")
    top_c = Y(0.36) - extra
    gs = fig.add_gridspec(1, 4, left=0.16, right=0.99, top=top_c, bottom=top_c - 0.19 * k, wspace=0.12)
    cs = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, (ax, (hyp, title, _)) in enumerate(zip(cs, COMPARISONS)):
        control_panel(ax, hyp, title.replace(" vs ", "\nvs "), tables, labels=j == 0,
                      families=args.families == "forest")
    fig.text(0.16, top_c - 0.24 * k, "difference above chance (fraction of noise ceiling)", ha="left", fontsize=7)
    cs[0].text(-0.62, 0.5, "removed", rotation=90, transform=cs[0].transAxes, ha="center", va="center", fontsize=7,
               color=st.INK2)
    fig.legend(handles=[Line2D([], [], marker=m, ls="none", ms=3.2, mfc=REGION_COLORS[r], mec=REGION_COLORS[r],
                               label=f"{REGION_NAMES[r]} (region mean)") for r, m in C_REGIONS],
               loc="lower right", bbox_to_anchor=(0.99, top_c - 0.283 * k), ncol=2, fontsize=6.8, handletextpad=0.2,
               columnspacing=1.0, frameon=False)
    st.panel_label(a[0], "a", x=-0.42)
    if args.families != "best":
        fig.text(0.02, Y(0.685), "b", fontsize=9, fontweight="bold")
    fig.text(0.02, top_c + 0.025 * k, "c" if args.families in ("none", "forest") else "d", fontsize=9,
             fontweight="bold")
    legend_block(fig, fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
