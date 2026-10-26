#!/usr/bin/env python3
"""Subset image lists for the FaceNet training-set-size analysis.

Repeat r shuffles the full training list with seed 1000 + r and takes its first
10k, 50k, 100k, 200k, 300k and 500k images (so smaller subsets are nested in
larger ones); each subset keeps the validation images of the identities it covers.
"""
import argparse
import random
from pathlib import Path

SIZES = {"10k": 10_000, "50k": 50_000, "100k": 100_000,
         "200k": 200_000, "300k": 300_000, "500k": 500_000}


def read(path):
    return [(p, int(l)) for p, l in (line.split("\t") for line in path.read_text().splitlines() if line)]


def write(path, rows):
    path.write_text("".join(f"{p}\t{l}\n" for p, l in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("vggface_train/index"))
    ap.add_argument("--out", type=Path, default=Path("vggface_train/indexes_repeated"))
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()

    train, val = read(a.source / "train.txt"), read(a.source / "val.txt")
    classes = (a.source / "classes.txt").read_text()
    for r in range(1, a.repeats + 1):
        shuffled = train.copy()
        random.Random(1000 + r).shuffle(shuffled)
        for name, size in SIZES.items():
            selected = shuffled[:size]
            labels = {l for _, l in selected}
            dest = a.out / f"repeat_{r:02d}" / name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "classes.txt").write_text(classes)
            write(dest / "train.txt", selected)
            write(dest / "val.txt", [row for row in val if row[1] in labels])
            print(f"{dest}: {len(labels)} identities")


if __name__ == "__main__":
    main()
