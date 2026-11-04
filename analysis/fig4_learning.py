#!/usr/bin/env python3
"""Fig. 4: learning the face or emotion task does not raise EEG alignment (54 videos).

Values are rho_max above its chance level (the permutation-null mean, contrasts.py), as a fraction of the
upper noise ceiling (--normalize tables; --raw for the rho scale), in the three occipito-temporal ROIs, the
two central ROIs and on the whole scalp (Fig. 1 region colours; left solid, midline dotted, right dashed).

  a     FaceNet along full-pool training (three runs, weights kept at nine steps): held-out identity
        accuracy (top) and alignment (bottom) against the training step
  b     FaceNet trained on 50 to 3,500 identities (100 images each, the full pool's step count, three runs),
        and on the full pool (5,346 identities)
  c     every FaceNet of a, b and the size curve: alignment on the whole scalp against held-out identity
        accuracy
  d     every variant: alignment on the whole scalp against the accuracy of a linear emotion probe on its
        final-layer features (colour = training regime, marker = family)

Reads analysis/paper_stats_54/contrasts{_ckpt,_ids,_sizes,}[_nc].csv, probe.csv,
analysis/checkpoint_accuracy.csv, identity_count_accuracy.csv, identity_accuracy.csv.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figure_style as st
import rsa_core
import variants
from fig2_training import REGION_COLORS, STATS, full_label, region_of

SCOPES = ["OccipitoTemporal-L", "OccipitoTemporal-M", "OccipitoTemporal-R", "Central-L", "Central-R", "wholescalp"]
SIDE_LS = {"L": "-", "M": (0, (1, 1.2)), "R": (0, (3, 1.5))}
STEPS = [0, 30, 100, 300, 1000, 3000, 10000, 30000, 55920]
IDS = [50, 300, 1500, 3500]
SIZES = {"10k": 1e4, "50k": 5e4, "100k": 1e5, "200k": 2e5, "300k": 3e5, "500k": 5e5}
SUFFIX = {"s": "_nc"}


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def variant_rows(name):
    path = STATS / f"contrasts{'_' + name if name else ''}{SUFFIX['s']}.csv"
    return {(r["contrast"], r["scope"]): r for r in rows(path) if r["hypothesis"] == "variant"}


def style(scope):
    if scope == "wholescalp":
        return st.MUTED, "-"
    return REGION_COLORS[region_of(scope)], SIDE_LS[scope.split("-")[1]]


def scope_lines(ax, xs, keys, table):
    for scope in SCOPES:
        color, ls = style(scope)
        y = [float(table[k, scope]["corrected"]) for k in keys]
        ax.plot(xs, y, color=color, ls=ls, lw=1.0, marker="o", ms=2.2, mfc=color, mec="none", zorder=3)
    ax.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))


def mean_by(path_rows, key, value):
    out = {}
    for r in path_rows:
        out.setdefault(key(r), []).append(value(r))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS / "fig4_learning"), help="output path without extension")
    ap.add_argument("--raw", action="store_true", help="rho scale instead of fractions of the noise ceiling")
    args = ap.parse_args()
    if args.raw:
        SUFFIX["s"] = ""
    st.setup()
    A = rsa_core.ROOT / "analysis"
    ckpt, ids, sizes, main_t = variant_rows("ckpt"), variant_rows("ids"), variant_rows("sizes"), variant_rows("")
    ckpt_acc = mean_by([dict(zip(("run", "step", "top1", "top5"), line.strip().split(",")))
                        for line in (A / "checkpoint_accuracy.csv").read_text().splitlines() if line.strip()],
                       lambda r: int(r["step"]), lambda r: float(r["top1"]))
    ids_runs = [float(line.split(",")[1]) for line in (A / "identity_count_accuracy.csv").read_text().splitlines()
                if line.strip()]
    ids_acc = mean_by([line.split(",") for line in (A / "identity_count_accuracy.csv").read_text().splitlines()
                       if line.strip()], lambda r: int(r[0].split("/")[1].replace("ids", "")), lambda r: float(r[1]))
    rdm_acc = {r["rdm"]: 100 * float(r["accuracy"]) for r in rows(A / "identity_accuracy.csv")}
    size_acc = {g: mean_by([(n, a) for n, a in rdm_acc.items() if n.startswith(f"facenet_{g}")],
                           lambda r: r[0].split("_")[1][len(g):], lambda r: r[1]) for g in ("size", "step")}
    full_acc = float(np.mean([a for n, a in rdm_acc.items() if n.startswith("facenet_vggface2")]))
    unit = "fraction of noise ceiling" if not args.raw else r"$\rho$"

    H = 4.85
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    # a: along training
    xs = [10] + STEPS[1:]                                    # step 0 drawn at 10 on the log axis
    a_top = fig.add_axes([0.1, f(0.95), 0.36, 0.55 / H])
    a_top.plot(xs, [ckpt_acc[s] for s in STEPS], color=st.REGIME_COLORS["Face"], marker="o", ms=2.5, lw=1.0)
    a_top.set_xscale("log"); a_top.set_xticks([]); a_top.minorticks_off()
    a_top.set_ylabel("identity\naccuracy (%)", fontsize=6.6)
    a_top.set_ylim(0, 100); a_top.set_yticks([0, 50, 100])
    a_top.spines["bottom"].set_visible(False)
    a = fig.add_axes([0.1, f(2.15), 0.36, 1.12 / H])
    scope_lines(a, xs, [f"FaceNet ckpt {s}" for s in STEPS], ckpt)
    a.set_xscale("log"); a.minorticks_off()
    a.set_xticks([10, 100, 1000, 10000, 55920], ["0", "100", "1k", "10k", "56k"])
    a.set_xlabel("training step (FaceNet, full pool)")
    a.set_ylabel(f"alignment above chance\n({unit})", fontsize=6.6)
    # b: identity count
    b = fig.add_axes([0.6, f(2.15), 0.36, 1.67 / H])
    xs_b = IDS + [5346]
    keys = [f"FaceNet ids {n}ids" for n in IDS]
    for scope in SCOPES:
        color, ls = style(scope)
        y = [float(ids[k, scope]["corrected"]) for k in keys]
        b.plot(IDS, y, color=color, ls=ls, lw=1.0, marker="o", ms=2.2, mfc=color, mec="none", zorder=3)
        b.plot(5346, float(ids["FaceNet VGGFace2", scope]["corrected"]), "o", ms=3, mfc="white", mec=color, zorder=4)
    b.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    b.set_xscale("log"); b.minorticks_off()
    b.set_xticks(xs_b, ["50", "300", "1.5k", "3.5k", "full"])
    b.set_xlabel("identities trained on (FaceNet)")
    b.text(0.02, 0.97, f"held-out identity accuracy {min(ids_runs):.0f}–{max(ids_runs):.0f}% in every run",
           transform=b.transAxes, fontsize=6.2, color=st.INK2, va="top")
    b.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    hi = max(a.get_ylim()[1], b.get_ylim()[1])
    a.set_ylim(min(0, a.get_ylim()[0]), hi); b.set_ylim(min(0, b.get_ylim()[0]), hi)
    # c: every FaceNet, whole scalp, against identity accuracy
    c = fig.add_axes([0.1, f(4.05), 0.36, 1.35 / H])
    face = st.REGIME_COLORS["Face"]
    pts = [("o", [(ckpt_acc[s], float(ckpt[f"FaceNet ckpt {s}", "wholescalp"]["corrected"])) for s in STEPS],
            "along training"),
           ("s", [(ids_acc[n], float(ids[f"FaceNet ids {n}ids", "wholescalp"]["corrected"])) for n in IDS],
            "identity count"),
           ("^", [(size_acc["size"][k], float(sizes[f"FaceNet size {k}", "wholescalp"]["corrected"])) for k in SIZES],
            "training-set size"),
           ("v", [(size_acc["step"][k], float(sizes[f"FaceNet step {k}", "wholescalp"]["corrected"]))
                  for k in list(SIZES)[:-1]], "size, matched steps")]
    for marker, xy, label in pts:
        x, y = zip(*xy)
        c.plot(x, y, marker, ms=3.2, mfc="white", mec=face, mew=0.7, label=label)
    c.plot(full_acc, float(main_t["FaceNet VGGFace2", "wholescalp"]["corrected"]), "D", ms=3.2, color=face,
           label="full pool")
    c.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    c.set_xlabel("held-out identity accuracy (%)")
    c.set_ylabel(f"whole scalp\n({unit})", fontsize=6.6)
    c.legend(fontsize=5.8, loc="upper left", handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    c.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    # d: every variant, emotion probe accuracy
    d = fig.add_axes([0.6, f(4.05), 0.36, 1.35 / H])
    probe = {(r["family"], r["variant"]): 100 * float(r["accuracy"]) for r in rows(STATS / "probe.csv")}
    for family, vrows in variants.MAIN.items():
        for variant, _ in vrows:
            regime = st.REGIME_OF[variant]
            m = st.MARKERS[family]
            d.plot(probe[family, variant], float(main_t[f"{family} {variant}", "wholescalp"]["corrected"]), m,
                   ms=st.size(m) - 1, mfc=st.REGIME_COLORS[regime], mec="white", mew=0.3)
    d.axvline(100 / 3, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    d.text(100 / 3, d.get_ylim()[1], " chance", fontsize=5.8, color=st.INK2, va="top")
    d.axhline(0, color=st.MUTED, lw=0.5, ls=(0, (3, 2)))
    d.set_xlabel("emotion probe accuracy (%)")
    d.yaxis.set_major_formatter(lambda v, _: st.point_format(round(v, 3)))
    # legends and letters
    handles = [Line2D([], [], color=style(s)[0], ls=style(s)[1], lw=1.0, label=full_label(s).replace("\n", " "))
               for s in SCOPES]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, f(4.66)), ncol=3, fontsize=6.2, frameon=False,
               handlelength=2.2, columnspacing=1.0)
    for letter, x, y in (("a", 0.01, 0.05), ("b", 0.52, 0.05), ("c", 0.01, 2.55), ("d", 0.52, 2.55)):
        fig.text(x, f(y), letter, fontsize=9, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
