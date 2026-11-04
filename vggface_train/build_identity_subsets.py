#!/usr/bin/env python3
"""Identity subsets for the FaceNet identity-count analysis.

Repeat r draws identities with at least 100 training images in the order of a shuffle
with seed 2000 + r and keeps the first 50, 300, 1,500 and 3,500 (nested), each with 100
training images per identity (drawn with the same seed) and all of its validation
images. Labels are renumbered 0..N-1, so each model has only its own identities as
output classes and learns its task (train.py --steps at the full pool's step count).
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

COUNTS = (50, 300, 1500, 3500)
PER_IDENTITY = 100


def read(path):
    return [(p, int(l)) for p, l in (line.split("\t") for line in path.read_text().splitlines() if line)]


def write(path, rows):
    path.write_text("".join(f"{p}\t{l}\n" for p, l in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("vggface_train/index"))
    ap.add_argument("--out", type=Path, default=Path("vggface_train/indexes_identities"))
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    train, val = read(a.source / "train.txt"), read(a.source / "val.txt")
    names = [line for line in (a.source / "classes.txt").read_text().splitlines() if line.strip()]
    images, val_of = defaultdict(list), defaultdict(list)
    for p, l in train:
        images[l].append(p)
    for p, l in val:
        val_of[l].append(p)
    eligible = sorted(l for l, ps in images.items() if len(ps) >= PER_IDENTITY)
    for r in range(1, a.repeats + 1):
        rng = random.Random(2000 + r)
        order = eligible.copy()
        rng.shuffle(order)
        chosen = {l: rng.sample(images[l], PER_IDENTITY) for l in order[:max(COUNTS)]}
        for n in COUNTS:
            ids = sorted(order[:n])
            new = {l: i for i, l in enumerate(ids)}
            dest = a.out / f"repeat_{r:02d}" / f"{n}ids"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "classes.txt").write_text("".join(f"{names[l]}\n" for l in ids))
            write(dest / "train.txt", [(p, new[l]) for l in ids for p in chosen[l]])
            write(dest / "val.txt", [(p, new[l]) for l in ids for p in val_of[l]])
            print(f"{dest}: {n} identities, {n * PER_IDENTITY} training images, "
                  f"{sum(len(val_of[l]) for l in ids)} validation images")


if __name__ == "__main__":
    main()
