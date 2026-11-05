import argparse
import random
from collections import defaultdict
from pathlib import Path

from fer_rsa.analysis import variants

SIZES = {name: int(name.removesuffix("k")) * 1000 for name in variants.SIZES}
COUNTS = tuple(int(ids.removesuffix("ids")) for ids in variants.IDENTITIES)
PER_IDENTITY = 100
DEFAULTS = {"sizes": ("vggface_train/indexes_repeated", variants.N_RUNS),
            "ids": ("vggface_train/indexes_identities", variants.N_DESIGN_REPEATS)}


def read(path):
    return [(p, int(l)) for p, l in (line.split("\t") for line in path.read_text().splitlines() if line)]


def write(path, rows):
    path.write_text("".join(f"{p}\t{l}\n" for p, l in rows))


def sizes(train, val, source, out, repeats):
    classes = (source / "classes.txt").read_text()
    for r in range(1, repeats + 1):
        shuffled = train.copy()
        random.Random(1000 + r).shuffle(shuffled)
        for name, size in SIZES.items():
            selected = shuffled[:size]
            labels = {l for _, l in selected}
            dest = out / f"repeat_{r:02d}" / name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "classes.txt").write_text(classes)
            write(dest / "train.txt", selected)
            write(dest / "val.txt", [row for row in val if row[1] in labels])
            print(f"{dest}: {len(labels)} identities")


def identities(train, val, source, out, repeats):
    names = [line for line in (source / "classes.txt").read_text().splitlines() if line.strip()]
    images, val_of = defaultdict(list), defaultdict(list)
    for p, l in train:
        images[l].append(p)
    for p, l in val:
        val_of[l].append(p)
    eligible = sorted(l for l, ps in images.items() if len(ps) >= PER_IDENTITY)
    for r in range(1, repeats + 1):
        rng = random.Random(2000 + r)
        order = eligible.copy()
        rng.shuffle(order)
        chosen = {l: rng.sample(images[l], PER_IDENTITY) for l in order[:max(COUNTS)]}
        for n in COUNTS:
            ids = sorted(order[:n])
            new = {l: i for i, l in enumerate(ids)}
            dest = out / f"repeat_{r:02d}" / f"{n}ids"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "classes.txt").write_text("".join(f"{names[l]}\n" for l in ids))
            write(dest / "train.txt", [(p, new[l]) for l in ids for p in chosen[l]])
            write(dest / "val.txt", [(p, new[l]) for l in ids for p in val_of[l]])
            print(f"{dest}: {n} identities, {n * PER_IDENTITY} training images, "
                  f"{sum(len(val_of[l]) for l in ids)} validation images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["sizes", "ids"], default="sizes")
    ap.add_argument("--source", type=Path, default=Path("vggface_train/index"))
    ap.add_argument("--out", type=Path, default=None, help="default: indexes_repeated (sizes), indexes_identities (ids)")
    ap.add_argument("--repeats", type=int, default=None, help="default: 10 (sizes), 3 (ids)")
    a = ap.parse_args()
    out = a.out or Path(DEFAULTS[a.design][0])
    repeats = a.repeats or DEFAULTS[a.design][1]
    train, val = read(a.source / "train.txt"), read(a.source / "val.txt")
    (sizes if a.design == "sizes" else identities)(train, val, a.source, out, repeats)


if __name__ == "__main__":
    main()
