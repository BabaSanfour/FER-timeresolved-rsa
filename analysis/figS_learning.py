#!/usr/bin/env python3
"""Supplementary figure: task accuracy and the amount of face-recognition training do not predict alignment
(54 videos, left occipito-temporal ROI, fractions of the noise ceiling).

  a  every network with seeds or runs: alignment above chance against the leave-one-actor-out accuracy of a linear
     emotion probe on its final-layer features, with the within-variant slope of a linear mixed model
     (statsmodels MixedLM, both standardized, random intercept per variant)
  b  FaceNet rho_max and its chance level against held-out identity accuracy on the full pool's 5,346 identities,
     along training (nine checkpoints) and over training-set sizes (40 epochs, or the full pool's 55,920 steps);
     raw (top) and after all nuisance RDMs (bottom)
  c  FaceNet trained on 50 to 3,500 identities (100 images each) and on the full pool, against the random FaceNet
  d  the within-variant slope of a in every ROI, raw and after the controls

Reads analysis/paper_stats_54/probe_per_rdm.csv (probe.py), contrasts{,_joint}_nc.csv (hypothesis "rdm"),
contrasts_{ckpt,sizes,ids}{,_joint}_nc.csv (contrasts.py --normalize --design ckpt|sizes|ids), and
analysis/{checkpoint_accuracy,identity_accuracy,identity_count_accuracy}.csv (identity_accuracy.py, train.py);
writes <out>.pdf and <out>.png.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import figure_style as st
import rsa_core
from fig2_training import STATS, scalp_map, topo_scalp

SC = "OccipitoTemporal-L"
FACE = st.REGIME_COLORS["Face"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "figS_learning"), help="output path without extension")
    args = ap.parse_args()
    L = str(STATS); A = str(rsa_core.ROOT / "analysis")
    info, roi_of = topo_scalp()
    st.setup()
    H = 5.15; f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    fmt = st.point_ticks
    top, bot = 0.45, 2.3

    # a: emotion probe accuracy vs alignment (every network), within-variant slope; maps of the slope per ROI
    pr = {r["rdm"]: (r["family"], r["variant"], 100 * float(r["accuracy"])) for r in csv.DictReader(open(f"{L}/probe_per_rdm.csv")) if r["group"] == "main"}
    def frame(name):
        rows = [dict(sc=r["scope"], fam=pr[r["contrast"]][0], vn=pr[r["contrast"]][1], acc=pr[r["contrast"]][2], al=float(r["corrected"]))
                for r in csv.DictReader(open(f"{L}/contrasts{name}_nc.csv")) if r["hypothesis"] == "rdm" and r["contrast"] in pr]
        d = pd.DataFrame(rows); d["v"] = d.fam + " " + d.vn; return d
    def slope(g):
        g = g.groupby("v").filter(lambda x: len(x) > 1).copy()
        for k in ("acc", "al"): g["z" + k] = (g[k] - g[k].mean()) / g[k].std()
        m = smf.mixedlm("zal ~ zacc", g, groups=g["v"]).fit(); return m.params["zacc"], m.pvalues["zacc"]
    raw = frame(""); d = raw[raw.sc == SC]
    ax = fig.add_axes([0.085, f(bot), 0.32, (bot - top) / H])
    for (fam, vn), g in d.groupby(["fam", "vn"]):
        c = st.REGIME_COLORS[st.REGIME_OF[vn]]; mk = st.MARKERS[fam]
        ax.plot(g.acc, g.al, mk, ms=1.8, mfc=c, mec="none", alpha=0.35)
        ax.plot(g.acc.mean(), g.al.mean(), mk, ms=st.size(mk) - 1, mfc=c, mec="white", mew=0.4)
    b, p = slope(d)
    ax.axvline(100 / 3, color=st.MUTED, lw=0.5, ls=(0, (3, 2))); ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("emotion probe accuracy (%)", fontsize=6.4)
    ax.set_ylabel("OT-L, difference above chance\n(fraction of noise ceiling)", fontsize=6, linespacing=1.0)
    ax.text(0.03, 0.97, f"within variants\nβ = {b:+.2f}, p = {p:.2f}", transform=ax.transAxes, va="top", fontsize=5.8, linespacing=1.0)
    ax.text(100 / 3, ax.get_ylim()[0], " chance", fontsize=5.4, color=st.INK2, va="bottom"); fmt(ax)
    ax.set_title("emotion probe, every network", fontsize=6.6, pad=3)
    hw = 0.85 / 5.5; hh = 0.85 / H                                  # d: the slope of a in every ROI
    for i, (name, lab) in enumerate((("", "raw"), ("_joint", "controls removed"))):
        bt = {sc: slope(g) for sc, g in frame(name).groupby("sc")}
        axm = fig.add_axes([0.07 + i * 0.19, f(bot + 2.45 - 0.55), hw, hh])
        imb = scalp_map(axm, [bt[r][0] for r in roi_of], [bt[r][1] * len(roi_of) < .05 for r in roi_of], info, 0.3)
        axm.set_title(lab, fontsize=6.2, pad=2)
    cax = fig.add_axes([0.105, f(bot + 2.45 - 0.25), 0.28, 0.06 / H]); cb = fig.colorbar(imb, cax=cax, orientation="horizontal")
    cb.set_ticks([-0.3, 0, 0.3], labels=["-.3", "0", ".3"]); cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
    cb.set_label("within-variant slope of alignment on probe accuracy", fontsize=5.8, labelpad=2)
    fig.text(0.245, f(bot + 0.5), "slope in every ROI", ha="center", va="top", fontsize=6.6)

    JOINT_DIR = L
    for ri, (SUF, LL) in enumerate((("_nc", L), ("_joint_nc", JOINT_DIR))):
        # b: similarity against held-out identity accuracy (same 5,346-way test): along training and over training-set size
        acc_ck = {}
        for line in open(f"{A}/checkpoint_accuracy.csv"):
            if line.strip(): run, step, t1, _ = line.strip().split(","); acc_ck.setdefault(int(step), []).append(float(t1))
        acc_rdm = {}
        for r in csv.DictReader(open(f"{A}/identity_accuracy.csv")):
            acc_rdm.setdefault(r["rdm"].rsplit("_", 1)[0], []).append(100 * float(r["accuracy"]))
        Rc = {(r["contrast"], r["scope"]): r for r in csv.DictReader(open(f"{LL}/contrasts_ckpt{SUF}.csv")) if r["hypothesis"] == "variant"}
        Rs = {(r["contrast"], r["scope"]): r for r in csv.DictReader(open(f"{LL}/contrasts_sizes{SUF}.csv")) if r["hypothesis"] == "variant"}
        SZ = ["10k", "50k", "100k", "200k", "300k", "500k"]
        designs = [("along training", "o", FACE, [(np.mean(acc_ck[s_]), Rc[f"FaceNet ckpt {s_}", SC]) for s_ in sorted(acc_ck)]),
                   ("images, 40 epochs", "^", st.INK2, [(np.mean(acc_rdm[f"facenet_size{k}"]), Rs[f"FaceNet size {k}", SC]) for k in SZ]),
                   ("images, matched steps", "s", st.MUTED, [(np.mean(acc_rdm[f"facenet_step{k}"]), Rs[f"FaceNet step {k}", SC]) for k in SZ[:-1]])]
        ax = fig.add_axes([0.52, f(bot + ri * 2.45), 0.19, (bot - top) / H])
        for lab, mk, c, pts in designs:
            pts.sort(key=lambda t: t[0]); x_ = [a for a, _ in pts]
            ax.plot(x_, [float(r["observed"]) for _, r in pts], "-" + mk, color=c, ms=2.6, lw=0.9, label=lab)
            ax.plot(x_, [float(r["null_mean"]) for _, r in pts], "--" + mk, color=c, ms=2.4, mfc="white", lw=0.7)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"]); ax.set_xlim(0.01, 150)
        ax.axvline(100 / 5346, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
        ax.set_ylim(0, 0.75); ax.set_xlabel("" if ri == 0 else "held-out identity accuracy (%)", fontsize=6.4)
        ax.set_ylabel(r"OT-L, $\rho_{\max}$ (fraction of noise ceiling)", fontsize=6)
        ax.legend(fontsize=5.4, loc="lower right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3); fmt(ax)
        ax.text(0.1, 0.97, "open: chance", transform=ax.transAxes, va="top", fontsize=5.4, color=st.INK2)
        ax.set_title("FaceNet, identity accuracy" if ri == 0 else "controls removed", fontsize=6.6, pad=3)

        # c: FaceNet trained on more identities (100 images each), and the full pool
        R = {(r["contrast"], r["scope"]): r for r in csv.DictReader(open(f"{LL}/contrasts_ids{SUF}.csv")) if r["hypothesis"] == "variant"}
        ids_acc = [float(l.split(",")[1]) for l in open(f"{A}/identity_count_accuracy.csv") if l.strip()]
        IDS = [("ids 50ids", 50), ("ids 300ids", 300), ("ids 1500ids", 1500), ("ids 3500ids", 3500)]
        ax = fig.add_axes([0.8, f(bot + ri * 2.45), 0.18, (bot - top) / H])
        xs = [v for _, v in IDS]
        ax.plot(xs, [float(R["FaceNet " + k, SC]["observed"]) for k, _ in IDS], "-o", color=FACE, ms=2.4, lw=1.0, label=r"$\rho_{\max}$")
        ax.plot(xs, [float(R["FaceNet " + k, SC]["null_mean"]) for k, _ in IDS], "--o", color=FACE, ms=2.2, mfc="white", lw=0.8, label="chance")
        full = R["FaceNet VGGFace2", SC]
        ax.plot(5346, float(full["observed"]), "D", ms=3, color=FACE); ax.plot(5346, float(full["null_mean"]), "D", ms=2.8, mfc="white", mec=FACE)
        rr = R["FaceNet Random", SC]
        ax.plot(10, float(rr["observed"]), "o", ms=2.6, color=st.MUTED); ax.plot(10, float(rr["null_mean"]), "o", ms=2.4, mfc="white", mec=st.MUTED)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([10, 50, 300, 1500, 5346], ["random", "50", "300", "1.5k", "5.3k\n(full)"], fontsize=5.6); ax.set_xlim(7, 9000)
        ax.set_ylim(0, 0.75); ax.set_xlabel("" if ri == 0 else "identities trained on", fontsize=6.4)
        ax.set_ylabel(r"OT-L, $\rho_{\max}$ (fraction of noise ceiling)", fontsize=6)
        ax.text(0.03, 0.97, f"every network learns its task\n(held-out accuracy {min(ids_acc):.0f}–{max(ids_acc):.0f}%)", transform=ax.transAxes, va="top", fontsize=5.4, color=st.INK2, linespacing=1.0)
        ax.legend(fontsize=5.6, loc="lower right", handlelength=1.6, borderaxespad=0.2); fmt(ax)
        ax.set_title("FaceNet, identities" if ri == 0 else "controls removed", fontsize=6.6, pad=3)


        #__END__
    for letter, x_, y_ in (("a", 0.01, 0.05), ("b", 0.455, 0.05), ("c", 0.73, 0.05), ("d", 0.01, bot + 0.4)):
        fig.text(x_, f(y_), letter, fontsize=9, fontweight="bold", va="top")
    st.black_text(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
