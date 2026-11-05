#!/usr/bin/env python3
"""Figure 1 (study design), drawn from code.

Panels
  a  stimuli: real frames of one actor (fear / happiness / neutral) across the 1-s clip,
     plus the sunglasses and mask versions used in the 54-video set
  b  EEG: Biosemi-64 layout with the 11 scalp ROIs (eeg/roi_definitions.json) and the
     13 analysis windows of 62.5 ms from onset (the last ends at 800 ms)
  c  model families x training regimes actually analysed, cells naming the training data
  d  EEG RDMs, e  model RDMs, f  layer x time RSA map with its maximum (rho_max)
     -- d, e, f are schematic: synthetic matrices generated here, not data

Usage
  python analysis/figures/fig1_study_design.py --stimuli <dir with set18/ and set54/> \
      --out analysis/figures/out/fig1_study_design
Writes <out>.pdf and <out>.png.
"""
import argparse
import json
from pathlib import Path

import av
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

# Validated categorical slots (dataviz reference palette, light mode, --pairs all).
EMOTION_COLORS = {"fear": "#eb6834", "happy": "#1baf7a", "neutral": "#2a78d6"}
# ROI regions (dataviz slots violet, magenta, green, yellow; validated --pairs all, labelled)
ROI_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300", "OccipitoTemporal": "#eda100"}
ROI_LABELS = {"Frontal": "Frontal", "Central": "Central", "CentroParietal": "Centro-parietal",
              "OccipitoTemporal": "Occipito-temporal"}
WINDOW_MS, N_WINDOWS, END_MS = 62.5, 13, 800
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

ACTOR = "11"      # all nine actor-11 clips show the same person (several other actors' files are mislabelled)
FRAMES_SHOWN = (0, 5, 10, 15)            # of the 16 sampled frames -> 0, 333, 667, 1000 ms
CROP = (640, 60, 1320, 967)              # portrait box (3:4) around the face in the 1920x1080 frames
ASPECT = (CROP[3] - CROP[1]) / (CROP[2] - CROP[0])


def sixteen_frames(path):
    """The 16 frames the pipeline samples (prepare_stimuli_54.sample_16 rule)."""
    with av.open(str(path)) as c:
        frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
    idx = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int)
    return [frames[i] for i in idx]


def thumb(img, width=240):
    return np.asarray(img.crop(CROP).resize((width, round(width * ASPECT)), Image.Resampling.LANCZOS))


def panel_label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", color=INK, va="bottom")
    fig.text(x + 0.016, y, title, fontsize=7, fontweight="bold", color=INK, va="bottom")


def rule(fig, x0, x1, y):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=RULE, lw=0.6))


def synthetic_rdm(rng, emotion_strength, n=18):
    """Block-structured dissimilarities: three emotion blocks of six, plus noise."""
    emo = np.repeat(np.arange(3), n // 3)
    actor = np.tile(np.arange(n // 3), 3)
    base = 1.0 - emotion_strength * (emo[:, None] == emo[None, :]) - 0.12 * (actor[:, None] == actor[None, :])
    noise = rng.normal(0, 0.09, (n, n))
    rdm = base + (noise + noise.T) / 2
    np.fill_diagonal(rdm, 0)
    return rdm


def draw_stack(ax, rdms, label_blocks):
    """Offset stack of RDM images, front one with emotion block labels."""
    k = len(rdms)
    off = 0.09
    for i, r in enumerate(rdms[::-1]):
        j = k - 1 - i
        x0, y0 = j * off, (k - 1 - j) * off
        ax.imshow(r, cmap=SEQ, vmin=0.0, vmax=1.6, extent=(x0, x0 + 1, y0, y0 + 1),
                  interpolation="nearest", zorder=10 + i)
        ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False, lw=0.5, ec="white", zorder=10 + i))
    if label_blocks:
        x0, y0 = 0, (k - 1) * off
        for b, emo in enumerate(EMOTION_COLORS):
            ax.add_patch(Rectangle((x0 + b / 3, y0 + 1.01), 1 / 3 - 0.01, 0.035,
                                   color=EMOTION_COLORS[emo], lw=0, clip_on=False, zorder=40))
            ax.add_patch(Rectangle((x0 - 0.045, y0 + 1 - (b + 1) / 3 + 0.005), 0.035, 1 / 3 - 0.01,
                                   color=EMOTION_COLORS[emo], lw=0, clip_on=False, zorder=40))
    ax.set_xlim(-0.06, 1 + (k - 1) * off + 0.02)
    ax.set_ylim(-0.02, 1 + (k - 1) * off + 0.06)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_head(ax):
    """Biosemi ActiveTwo 64-channel layout (10-10 labels), standard azimuthal projection (MNE).

    The 11 ROIs (eeg/roi_definitions.json) are coloured by region; left and right ROIs are filled
    dots, midline ROIs open dots; channels outside the ROIs (P9, P10: whole scalp only) are small
    grey dots.
    """
    from mne.channels.layout import _find_topomap_coords
    roi = json.loads((REPO / "eeg/roi_definitions.json").read_text())
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(montage.ch_names, 256.0, "eeg")
    info.set_montage(montage)
    xy = dict(zip(montage.ch_names, _find_topomap_coords(info, picks=list(range(len(montage.ch_names))))))
    centre = np.zeros(2)
    radius = max(np.linalg.norm(p) for p in xy.values()) * 1.07     # just outside P9 / P10 / Iz
    ax.add_patch(Circle(centre, radius, fill=False, lw=0.7, ec=INK2))
    nose = np.array([[-0.12 * radius, radius * 0.99], [0, radius * 1.13], [0.12 * radius, radius * 0.99]])
    ax.add_patch(Polygon(nose, closed=False, fill=False, lw=0.7, ec=INK2))
    member = {ch: name for name, chans in roi.items() if not name.startswith("_") for ch in chans}
    for ch, p in xy.items():
        name = member.get(ch)
        if name is None:
            ax.plot(*p, "o", ms=1.5, mfc="#b9b8b2", mec="none")
            continue
        region, side = name.split("-")
        color = ROI_COLORS[region]
        if side == "M":
            ax.plot(*p, "o", ms=2.4, mfc="white", mec=color, mew=0.8)
        else:
            ax.plot(*p, "o", ms=2.6, mfc=color, mec="white", mew=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", ms=2.6, mfc=ROI_COLORS[r], mec="white", mew=0.3, label=lab)
               for r, lab in ROI_LABELS.items()]
    handles.append(Line2D([], [], marker="o", ls="none", ms=2.4, mfc="white", mec=INK2, mew=0.8, label="midline"))
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.1), frameon=False,
              fontsize=5, handletextpad=0.1, columnspacing=0.6, borderaxespad=0, labelspacing=0.3)
    ax.set_xlim(centre[0] - radius * 1.6, centre[0] + radius * 1.6)
    ax.set_ylim(centre[1] - radius * 1.45, centre[1] + radius * 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_windows(ax):
    for k in range(N_WINDOWS):
        start, stop = k * WINDOW_MS, min((k + 1) * WINDOW_MS, END_MS)
        ax.add_patch(Rectangle((start + 3, 0), stop - start - 6, 1, color=BLUE_RAMP[1] if k % 2 else BLUE_RAMP[2], lw=0))
    ax.set_xlim(0, END_MS)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 400, 800])
    ax.set_xticklabels(["0", "400", "800 ms"])
    ax.tick_params(axis="x", labelsize=5.5, length=2, width=0.5, colors=INK, pad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color(INK2)
    ax.set_xlabel("13 windows × 62.5 ms from onset", fontsize=5.5, color=INK, labelpad=1)


def draw_models(ax):
    """Families x training regimes; cells name the training data (random = dot, none = small grey dot)."""
    families = ["ResNet18", "ResNet50", "FaceNet", "VGG16", "M3DFEL", "Qwen3-VL"]
    regimes = ["Random", "Generic", "Face", "Emotion"]
    img = ["*", "ImageNet", "VGGFace2", "AffectNet"]
    cells = {"ResNet18": img, "ResNet50": img, "FaceNet": img, "VGG16": img[:2] + ["VGG-Face"] + img[3:],
             "M3DFEL": ["*", "Kinetics", "", "DFEW"], "Qwen3-VL": ["*", "Released", "", ""]}
    n_r = len(families)
    xs = [0.25, 0.415, 0.635, 0.855]         # column centres, axes fraction
    top = n_r - 0.4
    for x, r in zip(xs, regimes):
        ax.text(x - 0.03, top, r, fontsize=5.5, color=INK, ha="left", va="bottom", rotation=35,
                rotation_mode="anchor")
    for i, fam in enumerate(families):
        y = n_r - 1 - i
        ax.text(0.205, y, fam, fontsize=5.5, color=INK, ha="right", va="center")
        for x, c in zip(xs, cells[fam]):
            if c == "*":
                ax.plot(x, y, "o", ms=4, color=INK, mew=0)
            elif c:
                ax.text(x, y, c, fontsize=4.8, color=INK, ha="center", va="center")
            else:
                ax.plot(x, y, "o", ms=1.3, color=RULE, mew=0)
    ax.plot([0.02, 0.99], [1.5, 1.5], color=RULE, lw=0.6)   # image | video models
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n_r + 2.2)
    ax.axis("off")


def draw_rsa_map(ax, rng):
    """Tall layer x time map: layers on y, EEG window / model frame on x."""
    L, W = 40, N_WINDOWS
    li, wi = np.meshgrid(np.linspace(0, 1, L), np.arange(W), indexing="ij")
    peak = 0.16 * np.exp(-((li - 0.72) ** 2) / 0.05 - ((wi - 3) ** 2) / 6)
    early = 0.05 * np.exp(-((li - 0.25) ** 2) / 0.04 - ((wi - 1.5) ** 2) / 3)
    grid = peak + early + rng.normal(0, 0.012, (L, W))
    ax.imshow(grid, cmap=DIV, vmin=-0.2, vmax=0.2, origin="lower", aspect="auto",
              extent=(0, W * WINDOW_MS, 0, 1), interpolation="nearest")
    l, w = np.unravel_index(np.argmax(grid), grid.shape)
    xy = ((w + 0.5) * WINDOW_MS, (l + 0.5) / L)
    ax.plot(*xy, marker="o", ms=5, mfc="none", mec=INK, mew=0.8)
    ax.annotate(r"$\rho_{\max}$", xy=xy, xytext=(W * WINDOW_MS / 2, 1.07), textcoords="data", fontsize=6.5, color=INK,
                ha="center", va="bottom", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_xticks([0, END_MS])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["first", "last"])
    ax.tick_params(labelsize=5.5, length=2, width=0.5, colors=INK, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(INK2)
    ax.set_xlabel("Time (ms)", fontsize=5.5, color=INK, labelpad=1)
    ax.set_ylabel("Layer", fontsize=5.5, color=INK, labelpad=-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True, type=Path)
    ap.add_argument("--out", default=str(REPO / "analysis/figures/out/fig1_study_design"))
    args = ap.parse_args()

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5})
    rng = np.random.default_rng(7)
    W_IN, H_IN = 5.5, 3.2
    fig = plt.figure(figsize=(W_IN, H_IN))
    sq = W_IN / H_IN                         # height (fig fraction) of a square of width w is w * sq

    # Columns: a | b over c | d over e | f (narrow, both rows)
    A0, A1 = 0.010, 0.262
    B0, B1 = 0.292, 0.572
    C0, C1 = 0.600, 0.790
    D0, D1 = 0.835, 0.990
    TOP, MID = 0.945, 0.455                   # panel-title baselines of the two rows

    # ---- a: stimuli (both rows) ----------------------------------------------------
    panel_label(fig, A0, TOP, "a", "Stimuli")
    rule(fig, A0, A1, TOP - 0.007)
    names = {"fear": "Fear", "happy": "Hapiness", "neutral": "Neutral"}      # stimuli/set18 file names
    gap, lab_w = 0.003, 0.028                    # emotion colour bar + rotated label strip
    size = 0.86 * (A1 - A0 - lab_w - 3 * gap) / 4
    block = lab_w + 4 * size + 3 * gap
    xa = A0 + (A1 - A0 - block) / 2              # whole stimulus block centred in column 1
    x_img0 = xa + lab_w
    h = size * sq * ASPECT
    row_gap, axis_h = 0.010, 0.075                # space under the rows for the time arrow + labels
    note_top = 0.030 + 0.85 * size * sq * ASPECT  # top of the occlusion note (see below)
    block_h = 3 * h + 2 * row_gap + axis_h
    y_block_top = (TOP - 0.015 + note_top) / 2 + block_h / 2   # centred between title rule and note
    for r, emo in enumerate(EMOTION_COLORS):
        frames = sixteen_frames(args.stimuli / "set18" / f"{names[emo]}_Actor{ACTOR}.mp4")
        y = y_block_top - (r + 1) * h - r * row_gap
        fig.add_artist(Rectangle((xa + 0.002, y), 0.005, h, transform=fig.transFigure,
                                 color=EMOTION_COLORS[emo], lw=0))
        fig.text(xa + 0.013, y + h / 2, {"fear": "Fear", "happy": "Happy", "neutral": "Neutral"}[emo],
                 fontsize=6, color=INK, va="center", rotation=90)
        for c, k in enumerate(FRAMES_SHOWN):
            ax = fig.add_axes([x_img0 + c * (size + gap), y, size, h])
            ax.imshow(thumb(frames[k], width=300), interpolation="none")
            ax.axis("off")
    y_arrow = y - 0.035
    ax_t = fig.add_axes([x_img0, y_arrow, 4 * size + 3 * gap, 0.02])
    ax_t.annotate("", xy=(1, 0.5), xytext=(0, 0.5), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))
    ax_t.axis("off")
    for c, ms in enumerate((0, 333, 667, 1000)):
        fig.text(x_img0 + c * (size + gap) + size / 2, y_arrow - 0.025, f"{ms} ms" if c == 3 else f"{ms}",
                 fontsize=5.5, color=INK, ha="center")
    # 54-video set: a small note at the bottom of the column (two tiny occluded examples + one line)
    size_o, gap_o = 0.85 * size, 0.004
    h_o = size_o * sq * ASPECT
    y_o = 0.030
    x_o = xa
    for c, fname in enumerate([f"Fear_Glasses_Actor{ACTOR}.mp4", f"Fear_Masked_Actor{ACTOR}.mp4"]):
        frames = sixteen_frames(args.stimuli / "set54" / fname)
        ax = fig.add_axes([x_o + c * (size_o + gap_o), y_o, size_o, h_o])
        ax.imshow(thumb(frames[15], width=300), interpolation="none")
        ax.axis("off")
    fig.text(x_o + 2 * (size_o + gap_o) + 0.006, y_o + h_o / 2,
             "54-video set: each clip\nalso with sunglasses\nor a mask", fontsize=5.5, color=INK,
             va="center", linespacing=1.15)

    # ---- b: EEG (top) / c: models (bottom) ------------------------------------------
    panel_label(fig, B0, TOP, "b", "Electroencephalography")
    rule(fig, B0, B1, TOP - 0.007)
    draw_head(fig.add_axes([B0 + 0.03, 0.640, B1 - B0 - 0.06, 0.285]))
    draw_windows(fig.add_axes([B0 + 0.012, 0.590, B1 - B0 - 0.024, 0.026]))

    panel_label(fig, B0, MID, "c", "Models")
    rule(fig, B0, B1, MID - 0.007)
    draw_models(fig.add_axes([B0, 0.015, B1 - B0, MID - 0.03]))

    # ---- d: EEG RDMs (top) / e: model RDMs (bottom) ----------------------------------
    panel_label(fig, C0, TOP, "d", "EEG RDMs")
    rule(fig, C0, C1, TOP - 0.007)
    draw_stack(fig.add_axes([C0 + 0.012, 0.555, C1 - C0 - 0.02, 0.35]),
               [synthetic_rdm(rng, s) for s in (0.40, 0.50, 0.55, 0.45)], label_blocks=True)

    panel_label(fig, C0, MID, "e", "Model RDMs")
    rule(fig, C0, C1, MID - 0.007)
    draw_stack(fig.add_axes([C0 + 0.012, 0.065, C1 - C0 - 0.02, 0.35]),
               [synthetic_rdm(rng, s) for s in (0.2, 0.35, 0.5, 0.65)], label_blocks=True)

    # ---- f: RSA map (both rows) ------------------------------------------------------
    panel_label(fig, D0, TOP, "f", "RSA")
    rule(fig, D0, D1, TOP - 0.007)
    ax_f = fig.add_axes([D0 + 0.050, 0.30, D1 - D0 - 0.058, 0.40])
    draw_rsa_map(ax_f, rng)
    for y_from in (0.74, 0.25):               # from the d and e stacks into the map
        fig.add_artist(FancyArrowPatch((C1 + 0.004, y_from), (D0 + 0.044, 0.48 + (0.05 if y_from > 0.5 else -0.05)),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                                       lw=0.7, color="#1c5cab", connectionstyle="arc3,rad=0"))
    fig.text(C1 + 0.018, 0.49, "Spearman ρ", fontsize=5.5, color="#1c5cab",
             rotation=90, ha="center", va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.text import Text
    for t in fig.findobj(Text):                                  # every text in black ink
        t.set_color(INK)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"[ok] {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
