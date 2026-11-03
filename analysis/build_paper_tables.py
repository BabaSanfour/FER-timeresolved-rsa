#!/usr/bin/env python3
"""Table 1 (tab:main-results) and the random-seed table (tab:seeds) of the paper.

Both read analysis/paper_stats/rho_max.csv (maxr_table.py --stimuli 18): the
emotion-averaged rho_max of every variant, mean +/- SD over its ten random seeds
or training runs (a released model has one value).

Writes analysis/paper_stats/table_main_results.tex and table_seeds.tex.
"""
import csv

import rsa_core
import variants

STATS = rsa_core.ROOT / "analysis/paper_stats"
SCOPES = ["Ant-Left", "Ant-Right", "Post-Left", "Post-Right", "Temp-Left", "Temp-Right",
          "wholescalp", rsa_core.ROI_AVG]


def three(x):
    """.083 rather than 0.083, as in the paper."""
    return f"{float(x):.3f}".replace("0.", ".", 1)


def main():
    with (STATS / "rho_max.csv").open() as fh:
        table = {(r["family"], r["variant"], r["roi"]): r for r in csv.DictReader(fh)
                 if r["group"] == "main" and r["emotion"] == rsa_core.EMO_AVG}

    lines = []
    for fi, (family, rows) in enumerate(variants.MAIN.items()):
        if fi:
            lines.append(r"\midrule")
        for variant, tags in rows:
            cells = [table[family, variant, scope] for scope in SCOPES]
            values = [three(c["max_r"]) if len(tags) == 1 else
                      f"{three(c['max_r'])}$\\pm${three(c['sd'])}" for c in cells]
            label = f"{variant} ($n$={len(tags)})" if len(tags) > 1 else variant
            lines.append(f"{family} & {label} & " + " & ".join(values) + r" \\")
    (STATS / "table_main_results.tex").write_text("\n".join(lines) + "\n")

    seed_lines = []
    for family, rows in variants.MAIN.items():
        (_, seeds), *trained = rows
        r = table[family, "Random", rsa_core.ROI_AVG]
        lowest = min(float(table[family, v, rsa_core.ROI_AVG]["max_r"]) for v, _ in trained)
        seed_lines.append(f"{family} & {len(seeds)} & {three(r['max_r'])} $\\pm$ {three(r['sd'])} & "
                          f"{three(r['min'])}--{three(r['max'])} & {three(lowest)} \\\\")
    (STATS / "table_seeds.tex").write_text("\n".join(seed_lines) + "\n")
    print("\n".join(lines + [""] + seed_lines))
    print(f"[ok] -> {STATS / 'table_main_results.tex'}, {STATS / 'table_seeds.tex'}")


if __name__ == "__main__":
    main()
