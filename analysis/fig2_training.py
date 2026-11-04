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
  b     the four comparisons (columns) in the 11 ROIs, all models pooled, raw and with all controls
        removed (rows): the difference above chance per ROI, dots on ROIs with p < .05 (maxT across ROIs)
  c     the same for the strongest family of each comparison (named under its maps; chosen after seeing
        the results). Emotion-tuned vs base model: each fine-tuned network against the network it was
        fine-tuned from
  d     what each control removes: for each comparison, the occipito-temporal (solid) and central
        (hatched) region means (mean over the region's ROIs), raw, with each control removed alone and
        with all removed; difference above chance with 95% CI, dot: p < .05 (maxT across the four regions);
        markers: the strongest family of c (family shape, filled where p < .05)
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

D_REGIONS = [("OccipitoTemporal", None), ("Central", "//////")]       # solid, hatched


def bar_panel(ax, hyp, title, regime, tables):
    """One comparison: x = control sets (raw, each alone, all); bars = the occipito-temporal (solid) and
    central (hatched) region means, difference above chance with 95% CI; * p < .05 (maxT across the four
    regions)."""
    color = st.REGIME_COLORS[regime]
    for k, (name, _) in enumerate(CONTROL_SETS):
        for j, (region, hatch) in enumerate(D_REGIONS):
            r = next(r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == "pooled"
                     and r["scope"] == region)
            m, lo, hi = float(r["corrected"]), float(r["low"]), float(r["high"])
            x = k + (j - 0.5) * 0.38
            ax.bar(x, m, width=0.36, fc=color if hatch is None else "white", ec=color, lw=0.6, hatch=hatch,
                   alpha=0.85 if hatch is None else 1, zorder=2)
            ax.plot([x, x], [lo, hi], color=st.INK2, lw=0.6, zorder=3)
            if float(r["p_maxT"]) < .05:                       # a dot, as on the maps
                ax.plot(x, (hi + 0.02) if m >= 0 else (lo - 0.02), "o", ms=1.8, color=st.INK, zorder=4)
            f = next(r for r in tables[name] if r["hypothesis"] == hyp and r["contrast"] == BEST[hyp]
                     and r["scope"] == region)                     # the strongest family (as in c)
            fm = st.MARKERS[BEST[hyp].split()[0]]
            ax.plot(x, float(f["corrected"]), fm, ms=min(st.size(fm) - 1.4, 4.2), mec=st.INK, mew=0.5,
                    mfc=st.INK if float(f["p_maxT"]) < .05 else "white", zorder=5)
    ax.axhline(0, color=st.INK2, lw=0.6, zorder=1)
    ax.set_xticks(range(len(CONTROL_SETS)), [lab for _, lab in CONTROL_SETS], fontsize=6.4, rotation=45,
                  ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.6, len(CONTROL_SETS) - 0.4)
    ax.set_title(title, fontsize=7, pad=3, linespacing=1.0)
    fmt(ax)


def legend_block(fig, height):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=st.REGIME_LABELS[r]) for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    marks = [Patch(fc=st.INK2, alpha=0.3, ec="none", label="raw"),
             Patch(fc="white", ec=st.INK2, lw=0.6, label="all controls removed"),
             Line2D([], [], color=st.INK2, lw=1.1, label="chance"),
             Line2D([], [], marker="o", ls="none", ms=2.2, color=st.INK, label="p < .05, corrected")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=0.9, borderaxespad=0, fontsize=7.2, loc="lower left")
    dy = 0.16 / height
    for y, title, handles in ((2 * dy, "Training regime", regimes), (dy, "Model family", families),
                              (0.0, "Estimates", marks)):
        fig.text(0.02, y + 0.04 / height, title, fontsize=7.2, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def layout_best(args, tables, variant):
    """a; b (pooled) and c (strongest family) side by side; d what each control removes."""
    H = 7.35
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
    gs = fig.add_gridspec(1, 4, left=0.1, right=0.99, top=top_d, bottom=f(5.88), wspace=0.1)
    ds = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, (ax, (hyp, title, regime)) in enumerate(zip(ds, COMPARISONS)):
        bar_panel(ax, hyp, title.replace(" vs ", "\nvs "), regime, tables)
        if j:
            ax.tick_params(labelleft=False)
    lo = min(ax.get_ylim()[0] for ax in ds); hi = max(ax.get_ylim()[1] for ax in ds)
    for ax in ds:
        ax.set_ylim(lo - 0.03, hi + 0.04)
    ds[-1].legend(handles=[Line2D([], [], marker=st.MARKERS[BEST[h].split()[0]], ls="none", ms=3.5, mfc=st.INK,
                                  mec=st.INK, label=BEST[h].split()[0]) for h in ("H1 pretrained vs random",
                                                                                 "H2 face vs ImageNet",
                                                                                 "H3 fine-tuned vs start")],
                  title="strongest family\n(filled: p < .05)", title_fontsize=6.2, fontsize=6.2, loc="upper right",
                  handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
    ds[0].set_ylabel("difference above chance\n(fraction of noise ceiling)", fontsize=7)
    fig.text(0.545, f(6.42), "removed from the EEG and model RDMs", ha="center", va="center", fontsize=7,
             color=st.INK2)
    fig.legend(handles=[Patch(fc=st.INK2, alpha=0.85, ec=st.INK2, label="occipito-temporal (region mean)"),
                        Patch(fc="white", ec=st.INK2, hatch="//////", label="central (region mean)")],
               loc="center right", bbox_to_anchor=(0.99, f(6.62)), ncol=2, fontsize=6.8, handletextpad=0.4,
               columnspacing=1.2, frameon=False)
    for letter, x, y in (("a", 0.01, 0.05), ("b", 0.01, 2.12), ("c", xc - 0.02, 2.12), ("d", 0.01, 4.42)):
        fig.text(x, f(y), letter, fontsize=9, fontweight="bold", va="top")   # each panel's top-left corner
    legend_block(fig, H)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig2_training"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    tables = {name: rows(name) for name, _ in CONTROL_SETS}
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    layout_best(args, tables, variant)


if __name__ == "__main__":
    main()
