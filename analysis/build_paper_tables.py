#!/usr/bin/env python3
"""The paper's rho_max tables, from maxr_table.py's outputs (both stimulus sets).

  table_main_results[_54].tex  Table 1 (18 videos) and its 54-video counterpart: the
                               emotion-averaged rho_max of every variant in each of the 11
                               ROIs and on the whole scalp (mean over its ten random seeds
                               or training runs; the SDs are in the seed table)
  table_controls[_54].tex      the same columns for the nuisance RDMs
  table_partial[_54].tex       ROI- and emotion-averaged rho_max of every variant, raw
                               and after partialling out each control and all jointly
  table_seeds.tex              18 videos, ROI- and emotion-averaged rho_max of each
                               regime (random, generic, face, emotion): mean +/- SD
                               [range] over its seeds or training runs

Reads analysis/paper_stats[_54]/rho_max.csv and rho_max_partial_<set>.csv; writes the
.tex rows next to them.
"""
import csv

import rsa_core
import variants

REGIMES = [{"Random"}, {"ImageNet", "Kinetics-400", "Pretrained"}, {"VGGFace2", "VGG-Face"},
           {"AffectNet FT", "DFEW FT"}]


def three(x):
    """.083 rather than 0.083, as in the paper."""
    return f"{float(x):.3f}".replace("0.", ".", 1)


def value(row, n, sd=True):
    if not row["max_r"]:
        return "--"                                   # undefined: a constant control
    return three(row["max_r"]) if n == 1 or not sd else f"{three(row['max_r'])}$\\pm${three(row['sd'])}"


def read(path):
    with path.open() as fh:
        return {(r["group"], r["family"], r["variant"], r["roi"]): r for r in csv.DictReader(fh)
                if r["emotion"] == rsa_core.EMO_AVG}


def roi_rows(table, group, rows_by_family, scopes, sd=True):
    lines = []
    for fi, (family, rows) in enumerate(rows_by_family.items()):
        if fi:
            lines.append(r"\midrule")
        for variant, tags in rows:
            values = [value(table[group, family, variant, scope], len(tags), sd) for scope in scopes]
            label = f"{variant} ($n$={len(tags)})" if len(tags) > 1 else variant
            lines.append(f"{family} & {label} & " + " & ".join(values) + r" \\")
    return lines


def main():
    written = []
    for s in ("18", "54"):
        stats = rsa_core.ROOT / ("analysis/paper_stats" if s == "18" else "analysis/paper_stats_54")
        suffix = "" if s == "18" else "_54"
        table = read(stats / "rho_max.csv")
        scopes = rsa_core.scopes()
        outputs = {f"table_main_results{suffix}.tex": ["% columns: " + ", ".join(scopes)] +
                   roi_rows(table, "main", variants.MAIN, scopes, sd=False)}
        controls = variants.controls(s)
        outputs[f"table_controls{suffix}.tex"] = [
            f"{label} & " + " & ".join(value(table["control", "Control", label, sc], 1) for sc in scopes) + r" \\"
            for label, _ in controls]

        sets = [name for _, name in controls] + ["joint"]
        partial = {name: read(stats / f"rho_max_partial_{name}.csv") for name in sets}
        lines = []
        for fi, (family, rows) in enumerate(variants.MAIN.items()):
            if fi:
                lines.append(r"\midrule")
            for variant, tags in rows:
                key = ("main", family, variant, rsa_core.ROI_AVG)
                cells = [value(table[key], len(tags))] + [value(partial[name][key], len(tags)) for name in sets]
                label = f"{variant} ($n$={len(tags)})" if len(tags) > 1 else variant
                lines.append(f"{family} & {label} & " + " & ".join(cells) + r" \\")
        outputs[f"table_partial{suffix}.tex"] = [
            "% columns: raw, " + ", ".join(label for label, _ in controls) + ", joint"] + lines

        if s == "18":
            seed_lines = []
            for family, rows in variants.MAIN.items():
                cells = []
                for regime in REGIMES:
                    match = [(v, t) for v, t in rows if v in regime]
                    if not match:
                        cells.append("--")
                        continue
                    (variant, tags), = match
                    r = table["main", family, variant, rsa_core.ROI_AVG]
                    cells.append(three(r["max_r"]) + " (1 model)" if len(tags) == 1 else
                                 f"{three(r['max_r'])}$\\pm${three(r['sd'])} [{three(r['min'])}, {three(r['max'])}]")
                seed_lines.append(f"{family} & " + " & ".join(cells) + r" \\")
            outputs["table_seeds.tex"] = seed_lines
        for name, lines in outputs.items():
            (stats / name).write_text("\n".join(lines) + "\n")
            written.append(stats / name)
    print("[ok] ->", ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
