import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fer_rsa.figures import figure_style as st
from fer_rsa.analysis import rsa_core
from fer_rsa.figures.fig2_training import COMPARISONS, CONDITIONS, SHORT_NAMES, VMAX, scalp_map, topo_scalp

STATS18 = rsa_core.STATS_DIRS["18"]


def pooled(name):
    with (STATS18 / f"contrasts{'_' + name if name else ''}_nc.csv").open() as fh:
        return {(r["hypothesis"], r["scope"]): r for r in csv.DictReader(fh) if r["contrast"] == "pooled"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS18 / "figS_18"), help="output path without extension")
    args = ap.parse_args()
    st.setup()
    info, roi_of = topo_scalp()
    H = 1.9
    f = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(5.5, H))
    x0 = 0.16; step = (0.9 - x0) / len(COMPARISONS); hw_in = 0.6; hw = hw_in / 5.5; hh = hw_in / H
    for i, (name, label) in enumerate(CONDITIONS):
        t = pooled(name)
        y = 0.42 + i * 0.72
        for k, (hyp, _) in enumerate(COMPARISONS):
            ax = fig.add_axes([x0 + k * step + (step - hw) / 2, f(y + hw_in), hw, hh])
            im = scalp_map(ax, [float(t[hyp, r]["corrected"]) for r in roi_of],
                           [float(t[hyp, r]["p_maxT"]) < .05 for r in roi_of], info)
            if i == 0:
                ax.set_title(SHORT_NAMES[hyp], fontsize=6, pad=2, linespacing=1.0)
        fig.text(x0 - 0.02, f(y + hw_in / 2), label, ha="right", va="center", fontsize=6.6, linespacing=1.0)
    cax = fig.add_axes([0.89, f(1.72), 0.012, 1.3 / H])
    cb = fig.colorbar(im, cax=cax)
    ticks = [-VMAX, 0, VMAX]
    cb.set_ticks(ticks, labels=[st.point_format(v) for v in ticks])
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.set_label("difference above chance\n(fraction of noise ceiling)", fontsize=5.4, labelpad=2, linespacing=1.0)
    st.save(fig, args.out)
    print(f"[ok] -> {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
