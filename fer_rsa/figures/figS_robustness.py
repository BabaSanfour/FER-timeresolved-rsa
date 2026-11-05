import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from fer_rsa.figures import figure_style as st
from fer_rsa.analysis import rsa_core
from fer_rsa.figures.fig2_training import COMPARISONS, SHORT_NAMES, STATS, VMAX, scalp_map, topo_scalp
from fer_rsa.figures.fig3_time_course import COLORS

WIN = "187.5"                                           # 187.5-250 ms
COL = {h[:2]: c for h, c in COLORS.items()}             # the comparison colours of Fig. 3
EMK = {"fear": "^", "happy": "o", "neutral": "s"}; ENAME = {"fear": "fear", "happy": "happiness", "neutral": "neutral"}
ECOL = {"neutral": "#7a7975", "happy": "#1baf7a", "fear": "#eb6834"}
KEYS = [("", "H1", "OccipitoTemporal-R", None, "generic vs random, OT-R"),
        ("", "H2", "OccipitoTemporal-R", None, "face vs object, OT-R"),
        ("", "H4", "OccipitoTemporal-M", None, "video vs image, OT-M"),
        ("_joint", "H1", "OccipitoTemporal-L", WIN, "generic vs random, OT-L"),
        ("_joint", "H5", "OccipitoTemporal-L", WIN, "face vs random, OT-L"),
        ("_joint", "H2", "OccipitoTemporal-L", WIN, "face vs object, OT-L"),
        ("_joint", "H2", "Central-R", WIN, "face vs object, C-R"),
        ("_joint", "H4", "OccipitoTemporal-L", WIN, "video vs image, OT-L"),
        ("_joint", "H3", "OccipitoTemporal-L", WIN, "emotion vs base, OT-L")]


def get(tag, hyp, sc, win, suffix=""):
    f = STATS / f"contrasts{'_time' if win else ''}{tag}_nc{suffix}.csv"
    for r in csv.DictReader(f.open()):
        if r["hypothesis"].startswith(hyp) and r["contrast"] == "pooled" and r["scope"] == sc and (win is None or r["window_start_ms"] == win):
            return float(r["corrected"]), float(r["low"]), float(r["high"]), float(r["p_maxT"])


def table(name):
    f = STATS / f"contrasts_time_joint_nc{name}.csv"
    return {(r["hypothesis"], r["scope"]): r for r in csv.DictReader(f.open()) if r["contrast"] == "pooled" and r["window_start_ms"] == WIN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "figS_robustness"), help="output path without extension")
    ap.add_argument("--recognition", default=str(rsa_core.ROOT / "behavior/recognition_frames.csv"),
                    help="recognition frames per participant and video")
    args = ap.parse_args()
    info, roi_of = topo_scalp()
    st.setup()
    H = 4.75; f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))

    # a: key results within each emotion
    n = len(KEYS); ys = np.arange(n)[::-1]
    ax = fig.add_axes([0.34, f(2.55), 0.3, 2.2 / H])
    for y, (tag, h, sc, win, _) in zip(ys, KEYS):
        m_, lo, hi, _ = get(tag, h, sc, win)
        ax.plot([lo, hi], [y, y], color=COL[h], lw=1.4, alpha=0.45, solid_capstyle="butt")
        ax.plot(m_, y, "D", ms=3.8, color=COL[h], mec="white", mew=0.3, zorder=4)
        for k, e in enumerate(EMK):
            r = get(tag, h, sc, win, f"_{e}")
            ax.plot(r[0], y + (1 - k) * 0.24, EMK[e], ms=2.8, mfc=st.INK if r[3] < .05 else "white", mec=st.INK, mew=0.5, zorder=3)
    ax.axvline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2))); ax.axhline(n - 3.5, color=st.RULE, lw=0.6)
    ax.set_ylim(-0.7, n - 0.3); ax.set_xlim(-0.42, 0.42); ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    st.point_ticks(ax, "x")
    ax.set_yticks(ys, [k[4] for k in KEYS], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference above chance\n(fraction of noise ceiling)", fontsize=6.2, linespacing=1.0)
    for yc, lab in ((n - 2, "peak, raw"), (2.5, "187–250 ms,\ncontrols removed")):
        ax.text(-0.97, yc, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=6.2, linespacing=1.0)
    ax.legend(handles=[Line2D([], [], marker="D", ls="none", ms=3.6, color=st.INK2, mec="white", label="all")] +
              [Line2D([], [], marker=EMK[e], ls="none", ms=3, mfc=st.INK, mec=st.INK, label=ENAME[e]) for e in EMK],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, fontsize=5.6, frameon=False, handletextpad=0.1, columnspacing=0.6)
    ax.text(1.0, 0.01, "filled: p < .05", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=st.INK2)
    ax.set_title("each emotion", fontsize=7, pad=12)

    # b: recognition time (emotion hue, occlusion lightness)
    rec = pd.read_csv(args.recognition); rec["ms"] = (rec["Frame value"] - 1) * 1000 / 30
    ax = fig.add_axes([0.8, f(2.55), 0.18, 2.2 / H])
    yy = 0; ticks, labs = [], []
    for ek, e in (("Peur", "fear"), ("Joie", "happy"), ("Neutre", "neutral")):
        for ok, on, a in (("Aucune", "plain", 1.0), ("Lunettes", "glasses", 0.65), ("Masque", "mask", 0.35)):
            v = rec[(rec.Emotion == ek) & (rec.Occlusion == ok)]["ms"].dropna(); q = np.percentile(v, [25, 50, 75])
            ax.plot([q[0], q[2]], [yy, yy], color=ECOL[e], lw=2.4, alpha=a, solid_capstyle="butt")
            ax.plot(q[1], yy, "|", ms=5, color=st.INK, mew=1, zorder=3)
            ticks.append(yy); labs.append(on); yy -= 1
        ax.text(-0.55, yy + 2, ENAME[e], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
        yy -= 0.6
    ax.set_yticks(ticks, labs, fontsize=5.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 400); ax.set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
    ax.set_ylim(yy + 0.3, ticks[0] + 1.4)
    ax.plot([187.5, 250], [ticks[0] + 1.0] * 2, color=st.INK, lw=1.6, solid_capstyle="butt")
    ax.text(218.75, ticks[0] + 1.15, "EEG effects", ha="center", va="bottom", fontsize=5.6)
    ax.set_xlabel("expression recognised\n(ms into video)", fontsize=6.2, linespacing=1.0)
    ax.set_title("recognition time", fontsize=7, pad=12)

    # c: actor folds and correct trials
    x0 = 0.16; step = (0.9 - x0) / len(COMPARISONS); hw_in = 0.6; hw = hw_in / 5.5; hh = hw_in / H
    def row(y_in, label, vals_of, sig_of, vmax=VMAX, titles=False):
        for k, (hyp, _) in enumerate(COMPARISONS):
            axm = fig.add_axes([x0 + k * step + (step - hw) / 2, f(y_in + hw_in), hw, hh])
            im = scalp_map(axm, vals_of(hyp), sig_of(hyp), info, vmax)
            if titles: axm.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0)
        fig.text(x0 - 0.02, f(y_in + hw_in / 2), label, ha="right", va="center", fontsize=6.6, linespacing=1.0)
        return im
    full = table(""); folds = [table(f"_dropactor{k}") for k in range(6)]; ct = table("_eeg_rdms_correct")
    def stab(h):
        out = []
        for r in roi_of:
            s_ = np.sign(float(full[h, r]["corrected"]))
            out.append(s_ * sum(1 for t in folds if float(t[h, r]["p_maxT"]) < .05 and np.sign(float(t[h, r]["corrected"])) == s_) / 6)
        return out
    yc = 3.38
    im = row(yc, "actor folds\nreproducing", stab, lambda h: [False] * len(roi_of), vmax=1, titles=True)
    cax = fig.add_axes([0.89, f(yc + 0.6), 0.012, 0.6 / H]); cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-1, 0, 1], labels=["6/6 −", "0", "6/6 +"]); cb.ax.tick_params(labelsize=5.2, length=2)
    cb.set_label("folds with\nthe same effect", fontsize=5.4, labelpad=2, linespacing=1.0)
    im = row(yc + 0.72, "correct\ntrials only", lambda h: [float(ct[h, r]["corrected"]) for r in roi_of],
             lambda h: [float(ct[h, r]["p_maxT"]) < .05 for r in roi_of])
    cax = fig.add_axes([0.89, f(yc + 1.32), 0.012, 0.6 / H]); cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-VMAX, 0, VMAX], labels=[st.point_format(v) for v in (-VMAX, 0, VMAX)]); cb.ax.tick_params(labelsize=5.2, length=2)
    cb.set_label("above chance\n(fraction of ceiling)", fontsize=5.4, labelpad=2, linespacing=1.0)
    fig.text(0.53, f(3.02), "187–250 ms, controls removed", ha="center", va="top", fontsize=7)

    for letter, x, y_ in (("a", 0.01, 0.05), ("b", 0.64, 0.05), ("c", 0.01, 2.95)):
        fig.text(x, f(y_), letter, fontsize=9, fontweight="bold", va="top")
    st.save(fig, args.out)
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
