#!/usr/bin/env python3
"""Fig. 2: what each step of training contributes to EEG alignment, in every ROI.

Values are the observed rho_max (each variant read at its own peak, mean over its RDMs and the
three emotions) and observed differences; the grey dash marks the chance level, the mean of
the permutation null with the same selection (contrasts.py), which is high for models with
many layers.

  a     each variant in each of the 11 ROIs and on the whole scalp, by training regime (54
        videos): one marker per family, joined across a family's regimes; bars are the mean over
        families, the dark dash its chance level
  b-e   the planned comparisons in each ROI and on the whole scalp: each participant's
        difference (54 videos, grey), the mean and 95% bootstrap CI (54 videos; b also 18 videos,
        open), each with its chance level (dash); * maxT-corrected p < .05 across ROIs (whole scalp: Holm across H1-H4), in the
        order 54 | 18. c shows each video model against the image models (family markers).
ROI labels carry the region colours of Fig. 1.

Reads analysis/paper_stats[_54]/contrasts.csv and contrasts_participants.csv (54); writes
analysis/paper_stats_54/fig2_training.{pdf,png}.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figure_style as st
import rsa_core
import variants

STATS = {"54": rsa_core.ROOT / "analysis/paper_stats_54", "18": rsa_core.ROOT / "analysis/paper_stats"}
REGIONS = [("Frontal", "frontal"), ("Central", "central"), ("CentroParietal", "centro-parietal"),
           ("OccipitoTemporal", "occipito-temporal")]
REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300",
                 "OccipitoTemporal": "#eda100", "wholescalp": st.INK}          # as in Fig. 1
SHORT = {"Frontal": "F", "Central": "C", "CentroParietal": "CP", "OccipitoTemporal": "OT"}
REGION_NAMES = {"Frontal": "frontal", "Central": "central", "CentroParietal": "centro-parietal",
                "OccipitoTemporal": "occipito-temporal"}
GRID = [["Frontal-L", "Frontal-M", "Frontal-R", "Central-L", "Central-R", "wholescalp"],
        ["CentroParietal-L", "CentroParietal-M", "CentroParietal-R",
         "OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R"]]
VIDEO = (("Qwen3-VL Pretrained", "*"), ("M3DFEL Kinetics-400", "p"))
PANELS = [("H1 pretrained vs random", "pretrained − random", "Generic", (("pooled", "o"),)),
          ("H4 video vs image", "video model − image models", "Generic", VIDEO),
          ("H2 face vs ImageNet", "face − ImageNet pretraining", "Face", (("pooled", "o"),)),
          ("H3 fine-tuned vs start", "emotion fine-tuned − its start", "Emotion", (("pooled", "o"),))]


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


ROI_CHANNELS = json.loads((rsa_core.ROOT / "eeg_rdms/meta.json").read_text())["roi_channels"]


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
    ins = ax.inset_axes([0.0, 0.7, 0.3, 0.3])
    t = np.linspace(0, 2 * np.pi, 100)
    ins.plot(radius * np.cos(t), radius * np.sin(t), color=st.INK2, lw=0.4)
    ins.plot([-0.12 * radius, 0, 0.12 * radius], [0.99 * radius, 1.15 * radius, 0.99 * radius], color=st.INK2, lw=0.4)
    for ch, (x, y) in pos.items():
        on = ch in chans
        ins.plot(x, y, "o", ms=1.3 if on else 0.7, mfc=color if on else st.RULE, mec="none")
    ins.set_xlim(-1.2 * radius, 1.2 * radius); ins.set_ylim(-1.15 * radius, 1.25 * radius)
    ins.set_aspect("equal"); ins.axis("off")


def region_of(scope):
    return scope.split("-")[0]


def scope_label(scope):
    return "whole scalp" if scope == "wholescalp" else f"{SHORT[region_of(scope)]}-{scope.split('-')[1]}"


def variant_panel(ax, scope, table):
    """table: {(family variant, scope): row} of the variant rows of one stimulus set / control set."""
    xs = {regime: i for i, regime in enumerate(st.REGIMES)}
    by_regime = {regime: [] for regime in st.REGIMES}
    chance = {regime: [] for regime in st.REGIMES}
    for fi, (family, vrows) in enumerate(variants.MAIN.items()):
        pts = []
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            x = xs[regime] - 0.12 + (fi - 2.5) * 0.07
            y54 = float(table[f"{family} {variant}", scope]["observed"])
            chance[regime].append(float(table[f"{family} {variant}", scope]["null_mean"]))
            pts.append((x, y54, regime)); by_regime[regime].append(y54)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st.RULE, lw=0.5, zorder=2)
        m = st.MARKERS[family]
        for x, y54, regime in pts:
            ax.plot(x, y54, m, ms=st.size(m) - 1.2, mfc=st.REGIME_COLORS[regime], mec="white", mew=0.3, zorder=4)
    for regime in st.REGIMES:
        c = st.REGIME_COLORS[regime]
        ax.bar(xs[regime] - 0.12, np.mean(by_regime[regime]), width=0.46, color=c, alpha=0.25, lw=0, zorder=1)
        ax.plot([xs[regime] - 0.35, xs[regime] + 0.11], [np.mean(chance[regime])] * 2, color=st.INK2, lw=1.1,
                solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r][0] for r in st.REGIMES], fontsize=5.2)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.55, len(st.REGIMES) - 0.65)
    ax.set_title(scope_label(scope), fontsize=6, color=REGION_COLORS[region_of(scope)], pad=2,
                 fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def ceilings(scope, partial_rows, per_rdm_raw, per_rdm_joint):
    """Lower and upper noise ceiling at the peak windows of the main variants (mean over RDMs and the
    three emotions), raw and after the joint controls (partial_rsa.py --ceilings)."""
    emotions = ("fear", "happy", "neutral")
    raw = [(float(r["nc_lower"]), float(r["nc_upper"])) for r in per_rdm_raw
           if r["group"] == "main" and r["roi"] == scope and r["emotion"] in emotions and r["nc_lower"]]
    part = {(r["emotion"], int(r["window"])): (float(r["lower"]), float(r["upper"])) for r in partial_rows
            if r["set"] == "joint" and r["scope"] == scope}
    joint = [part[r["emotion"], int(r["max_window"])] for r in per_rdm_joint
             if r["group"] == "main" and r["roi"] == scope and r["emotion"] in emotions and r["max_window"]]
    return np.mean(raw, 0), np.mean(joint, 0)


def paired_panel(ax, scope, raw, joint, ceiling):
    """Each regime: the raw bar (filled) and the bar after joint controls (open), family markers on
    each, chance dashes, and the noise ceilings (grey band raw, dashed after controls)."""
    (rl, ru), (jl, ju) = ceiling
    ax.axhspan(rl, ru, color=st.RULE, alpha=0.45, lw=0, zorder=0)
    for y in (jl, ju):
        ax.axhline(y, color=st.MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=0)
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
    ax.set_xticks(range(len(st.REGIMES)), [st.REGIME_LABELS[r] for r in st.REGIMES], fontsize=5.2, rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(-0.5, len(st.REGIMES) - 0.5)
    ax.set_title(scope_label(scope), fontsize=6, color=REGION_COLORS[region_of(scope)], pad=2, fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


COMPARISONS = [("H1 pretrained vs random", "pretrained\n− random", "Generic"),
               ("H4 video vs image", "video\n− image", "Generic"),
               ("H2 face vs ImageNet", "face\n− ImageNet", "Face"),
               ("H3 fine-tuned vs start", "emotion FT\n− start", "Emotion")]
CONDITIONS = [("", "raw"), ("occlusion", "occlusion removed"), ("joint", "all controls removed")]


def region_panel(ax, regions):
    """The four comparisons at the region means, raw, with occlusion removed and with all controls
    removed; maxT across the four regions."""
    tables = {name: {(r["hypothesis"], r["scope"]): r
                     for r in rows(STATS["54"] / f"contrasts{'_' + name if name else ''}.csv") if r["contrast"] == "pooled"}
              for name, _ in CONDITIONS}
    x0, ticks, labels = 0.0, [], []
    for region in regions:
        start = x0
        for hyp, label, regime in COMPARISONS:
            color = st.REGIME_COLORS[regime]
            for k, (name, _) in enumerate(CONDITIONS):
                r = tables[name][hyp, region]
                x = x0 + (k - 1) * 0.24
                c0 = float(r["null_mean"])
                m, lo, hi = float(r["observed"]), float(r["low"]) + c0, float(r["high"]) + c0
                ax.plot([x - 0.08, x + 0.08], [c0, c0], color=st.INK2, lw=0.9, solid_capstyle="butt", zorder=5)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3,
                            mfc=[color, st.blend(color, 0.45), "white"][k], mec=color, mew=0.6, ecolor=color,
                            elinewidth=0.7, capsize=0, zorder=4)
                if float(r["p_maxT"]) < .05:
                    ax.text(x, 0.075, "*", ha="center", va="center", fontsize=6.5, color=st.INK)
            ticks.append(x0); labels.append(label); x0 += 1
        ax.text((start + x0 - 1) / 2, 0.092, REGION_NAMES[region], ha="center", fontsize=6,
                color=REGION_COLORS[region])
        x0 += 0.5
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(ticks, labels, fontsize=5.2)
    ax.set_xlim(-0.5, x0 - 0.9)
    ax.set_ylim(-0.045, 0.1)
    ax.set_ylabel(r"difference in $\rho_{\max}$")
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3, mfc=f, mec=st.INK2, label=lab)
                       for f, (_, lab) in zip([st.INK2, st.blend(st.INK2, 0.45), "white"], CONDITIONS)],
              loc="lower left", ncol=3, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2, fontsize=5.4)


def scope_positions(scopes):
    """x position of each ROI, grouped by region with a gap between regions, then the whole scalp."""
    pos, x = {}, 0.0
    for region, _ in REGIONS:
        for side in "LMR":
            if f"{region}-{side}" in scopes:      # region-mean scopes (no side) are not drawn here
                pos[f"{region}-{side}"] = x; x += 1
        x += 0.6
    pos["wholescalp"] = x + 0.2
    return pos


def significant(r, contrast):
    if contrast == "pooled" and r["scope"] == "wholescalp":
        return float(r["p_holm"] or 1) < .05
    return float(r["p_maxT"]) < .05


def contrast_panel(ax, hyp, title, regime, comparisons, participants, series, rng, sets=("54",)):
    color = st.REGIME_COLORS[regime]
    pick = {s: {(r["contrast"], r["scope"]): r for r in comparisons[s] if r["hypothesis"] == hyp} for s in sets}
    pos = scope_positions({sc for c, sc in pick[sets[0]] if c == series[0][0]})
    n = len(sets) * len(series)
    step = 0.34 if n == 2 else 0.3
    top = 0.105
    for sc, x in pos.items():
        if series[0][0] == "pooled" and participants is not None:
            d = np.array([float(r["difference"]) for r in participants if r["hypothesis"] == hyp and r["scope"] == sc])
            d = d + float(pick[sets[0]]["pooled", sc]["null_mean"])
            ax.scatter(x - 0.36 + rng.uniform(-0.06, 0.06, len(d)), d, s=1.4, color=st.PARTICIPANT, lw=0, zorder=2)
        k = 0
        for contrast, marker in series:
            for s, filled in ((x, x == sets[0]) for x in sets):
                r = pick[s][contrast, sc]
                dx = (k - (n - 1) / 2) * step + (0.08 if series[0][0] == "pooled" else 0); k += 1
                c0 = float(r["null_mean"])
                m, lo, hi = float(r["observed"]), float(r["low"]) + c0, float(r["high"]) + c0
                ax.plot([x + dx - 0.1, x + dx + 0.1], [c0, c0], color=st.INK2, lw=1.0, solid_capstyle="butt", zorder=5)
                ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], fmt=marker,
                            ms=st.size(marker) - (1.8 if marker == "*" else 1.2), mfc=color if filled else "white",
                            mec=color, mew=0.6, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
                if significant(r, contrast):
                    ax.text(x + dx, top, "*", ha="center", va="center", fontsize=6.5,
                            color=st.INK if filled else st.MUTED)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(list(pos.values()), ["whole\nscalp" if s == "wholescalp" else s.split("-")[1] for s in pos],
                  fontsize=5.4)
    for tick, sc in zip(ax.get_xticklabels(), pos):
        tick.set_color(REGION_COLORS[region_of(sc)])
    for region, label in REGIONS:
        xs = [x for s, x in pos.items() if s.startswith(region + "-")]
        ax.text(np.mean(xs), -0.2, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.4,
                color=REGION_COLORS[region])
    ax.set_xlim(-0.7, pos["wholescalp"] + 0.6)
    ax.set_ylim(-0.1, top + 0.012)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


CONTROL_SETS = [("", "raw"), ("identity", "identity"), ("occlusion", "occlusion"), ("pixel", "pixel"),
                ("optical_flow", "optical flow"), ("motion_energy", "motion energy"), ("joint", "all jointly")]
OT = ["OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R"]


def control_panel(ax, hyp, scopes, regime):
    """One comparison at a few ROIs under each control set: observed difference, 95% CI, chance dash."""
    color = st.REGIME_COLORS[regime]
    markers = {"OccipitoTemporal-L": "<", "OccipitoTemporal-M": "o", "OccipitoTemporal-R": ">"}
    for k, (name, label) in enumerate(CONTROL_SETS):
        table = rows(STATS["54"] / f"contrasts{'_' + name if name else ''}.csv")
        for j, sc in enumerate(scopes):
            r = next(r for r in table if r["hypothesis"] == hyp and r["contrast"] == "pooled" and r["scope"] == sc)
            x = k + (j - 1) * 0.24
            c0 = float(r["null_mean"])
            m, lo, hi = float(r["observed"]), float(r["low"]) + c0, float(r["high"]) + c0
            ax.plot([x - 0.08, x + 0.08], [c0, c0], color=st.INK2, lw=1.0, solid_capstyle="butt", zorder=5)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=markers[sc], ms=3, mfc=color, mec=color,
                        ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
            if float(r["p_maxT"]) < .05:
                ax.text(x, 0.105, "*", ha="center", va="center", fontsize=6.5, color=st.INK)
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(CONTROL_SETS)), [label for _, label in CONTROL_SETS], fontsize=5.4, rotation=30,
                  ha="right", rotation_mode="anchor")
    ax.set_xlabel("removed from the EEG and model RDMs", fontsize=5.6)
    ax.set_xlim(-0.6, len(CONTROL_SETS) - 0.4)
    ax.set_ylim(-0.03, 0.117)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    ax.legend(handles=[Line2D([], [], marker=markers[sc], ls="none", ms=3, mfc=color, mec=color,
                              label=scope_label(sc)) for sc in scopes],
              loc="lower left", ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.2, fontsize=5.4)


def regime_row(fig, gs_row, scopes, table, layout, title=None):
    axes = [fig.add_subplot(gs_row[0, j]) for j in range(len(scopes))]
    for ax, sc in zip(axes, scopes):
        variant_panel(ax, sc, table)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for j, ax in enumerate(axes):
        ax.set_ylim(0, hi * 1.2)
        sensor_inset(ax, scopes[j], layout)
        if j:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\rho_{\max}$")
    if title:
        fig.text(0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1), axes[0].get_position().y1 + 0.045,
                 title, ha="center", fontsize=6.5)
    return axes


def legend_block(fig, height, ceiling=False):
    regimes = [Patch(fc=st.REGIME_COLORS[r], ec="none", label=f"{st.REGIME_LABELS[r]} ({st.REGIME_LABELS[r][0]})")
               for r in st.REGIMES]
    families = [Line2D([], [], marker=m, ls="none", ms=st.size(m), mfc=st.INK2, mec="white", mew=0.4, label=f)
                for f, m in st.MARKERS.items()]
    marks = [Line2D([], [], marker="o", ls="none", ms=3, mfc=st.INK2, mec=st.INK2, label="raw"),
             Line2D([], [], marker="o", ls="none", ms=3, mfc="white", mec=st.INK2, label="after joint controls"),
             Line2D([], [], color=st.INK2, lw=1.1, label="chance"),
             Line2D([], [], marker="$*$", ls="none", ms=4, color=st.INK, label="p < .05, corrected")]
    if ceiling:
        marks += [Patch(fc=st.RULE, alpha=0.45, ec="none", label="noise ceiling"),
                  Line2D([], [], color=st.MUTED, lw=0.6, ls=(0, (2, 1.5)), label="ceiling after controls")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=0.9, borderaxespad=0, fontsize=6, loc="lower left")
    dy = 0.16 / height
    for y, title, handles in ((2 * dy, "Training regime", regimes), (dy, "Model family", families),
                              (0.0, "Estimates", marks)):
        fig.text(0.02, y + 0.04 / height, title, fontsize=6, fontweight="bold", va="center")
        fig.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.16, y), handlelength=1.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["occlusion", "occlusion_v1", "face"], default="occlusion",
                    help="occlusion: raw and controlled OT panels, control breakdown, face per ROI; "
                         "face: raw OT panels, face and Qwen3-VL per ROI after controls")
    ap.add_argument("--out", default=None, help="output path without extension")
    args = ap.parse_args()
    out = args.out or str(STATS["54"] / f"fig2_{args.layout}")
    st.setup()
    raw, joint = rows(STATS["54"] / "contrasts.csv"), rows(STATS["54"] / "contrasts_joint.csv")
    variant = lambda t: {(r["contrast"], r["scope"]): r for r in t if r["hypothesis"] == "variant"}
    scopes = OT + ["wholescalp"]
    layout = sensor_layout()
    comparisons = {"raw": raw, "joint": joint}
    if args.layout == "occlusion":
        partial = rows(STATS["54"] / "noise_ceilings_partial.csv")
        per_raw = rows(STATS["54"] / "rho_max_per_rdm.csv")
        per_joint = rows(STATS["54"] / "rho_max_partial_joint_per_rdm.csv")
        fig = plt.figure(figsize=(5.5, 6.6))
        gs_a = fig.add_gridspec(1, 4, left=0.09, right=0.99, top=0.95, bottom=0.74, wspace=0.1)
        a = [fig.add_subplot(gs_a[0, j]) for j in range(4)]
        for ax, sc in zip(a, scopes):
            paired_panel(ax, sc, variant(raw), variant(joint), ceilings(sc, partial, per_raw, per_joint))
        for j, ax in enumerate(a):
            ax.set_ylim(0, 0.3)
            sensor_inset(ax, scopes[j], layout)
            if j:
                ax.tick_params(labelleft=False)
        a[0].set_ylabel(r"$\rho_{\max}$")
        b = fig.add_subplot(fig.add_gridspec(1, 1, left=0.09, right=0.99, top=0.62, bottom=0.46)[0, 0])
        region_panel(b, ["OccipitoTemporal", "Central"])
        gs = fig.add_gridspec(1, 2, left=0.09, right=0.99, top=0.36, bottom=0.17, wspace=0.08)
        c, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        contrast_panel(c, "H2 face vs ImageNet", "face − ImageNet pretraining", "Face", comparisons, None,
                       (("pooled", "o"),), None, ("raw", "joint"))
        contrast_panel(d, "H3 fine-tuned vs start", "emotion fine-tuned − its start", "Emotion", comparisons, None,
                       (("pooled", "o"),), None, ("raw", "joint"))
        c.set_ylabel(r"difference in $\rho_{\max}$"); d.tick_params(labelleft=False)
        for ax in (c, d):
            ax.set_ylim(-0.06, 0.117)
        for ax, letter, x in ((a[0], "a", -0.42), (b, "b", -0.09), (c, "c", -0.16), (d, "d", -0.05)):
            st.panel_label(ax, letter, x=x)
    elif args.layout == "occlusion_v1":
        fig = plt.figure(figsize=(5.5, 6.4))
        a = regime_row(fig, fig.add_gridspec(1, 4, left=0.09, right=0.99, top=0.93, bottom=0.75, wspace=0.12),
                       scopes, variant(raw), layout, "raw")
        b = regime_row(fig, fig.add_gridspec(1, 4, left=0.09, right=0.99, top=0.66, bottom=0.48, wspace=0.12),
                       scopes, variant(joint), layout, "after removing identity, occlusion, pixel, optical flow and motion energy")
        gs = fig.add_gridspec(1, 2, left=0.09, right=0.99, top=0.39, bottom=0.17, wspace=0.18, width_ratios=[1, 1.3])
        c, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        control_panel(c, "H1 pretrained vs random", OT, "Generic")
        c.set_title("pretrained − random, occipito-temporal")
        c.set_ylabel(r"difference in $\rho_{\max}$")
        contrast_panel(d, "H2 face vs ImageNet", "face − ImageNet pretraining", "Face", comparisons, None,
                       (("pooled", "o"),), None, ("raw", "joint"))
        d.set_ylim(-0.06, 0.117)
        for ax, letter, x in ((a[0], "a", -0.38), (b[0], "b", -0.38), (c, "c", -0.2), (d, "d", -0.1)):
            st.panel_label(ax, letter, x=x)
    else:
        fig = plt.figure(figsize=(5.5, 4.6))
        a = regime_row(fig, fig.add_gridspec(1, 4, left=0.09, right=0.99, top=0.92, bottom=0.66, wspace=0.12),
                       scopes, variant(raw), layout)
        gs = fig.add_gridspec(1, 2, left=0.09, right=0.99, top=0.54, bottom=0.24, wspace=0.08)
        b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        contrast_panel(b, "H2 face vs ImageNet", "face − ImageNet pretraining", "Face", comparisons, None,
                       (("pooled", "o"),), None, ("raw", "joint"))
        contrast_panel(c, "H4 video vs image", "Qwen3-VL − image models", "Generic", comparisons, None,
                       (("Qwen3-VL Pretrained", "*"),), None, ("raw", "joint"))
        b.set_ylabel(r"difference in $\rho_{\max}$"); c.tick_params(labelleft=False)
        for ax in (b, c):
            ax.set_ylim(-0.06, 0.117)
        for ax, letter, x in ((a[0], "a", -0.38), (b, "b", -0.16), (c, "c", -0.05)):
            st.panel_label(ax, letter, x=x)
    legend_block(fig, fig.get_figheight(), ceiling=args.layout == "occlusion")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    print(f"[ok] -> {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
