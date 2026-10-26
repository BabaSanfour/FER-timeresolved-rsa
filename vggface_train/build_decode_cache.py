#!/usr/bin/env python3
"""Decode every listed image once, exactly as train.py does (128px), into one uint8 array.

train.py --cache-dir then skips JPEG decoding; the augmentation is unchanged.
726,502 images -> ~36 GB.
"""
import argparse
import os
import random
from multiprocessing import Pool

import numpy as np
from PIL import Image

_ROOT = _SIZE = None


def _init(root, size):
    global _ROOT, _SIZE
    _ROOT, _SIZE = root, size


def _decode(path):
    # same calls as train.py ListDataset.__getitem__
    with Image.open(os.path.join(_ROOT, path)) as img:
        img.draft("RGB", (_SIZE, _SIZE))
        img = img.convert("RGB")
        if img.size != (_SIZE, _SIZE):
            raise ValueError(f"{path}: decoded to {img.size}, expected {_SIZE}x{_SIZE}")
        return np.asarray(img, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--decode-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    paths = []
    for name in ("train", "val"):
        with open(os.path.join(a.index_dir, f"{name}.txt")) as f:
            paths += [line.split("\t")[0] for line in f if line.strip()]
    if len(set(paths)) != len(paths):
        raise ValueError("train and val lists overlap")
    os.makedirs(a.out_dir, exist_ok=True)
    s = a.decode_size
    tmp = os.path.join(a.out_dir, f"images_u8_{s}.npy.tmp")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), s, s, 3))

    with Pool(a.workers, initializer=_init, initargs=(a.data_root, s)) as pool:
        for i, img in enumerate(pool.imap(_decode, paths, chunksize=256)):
            arr[i] = img
            if (i + 1) % 50000 == 0:
                print(f"{i + 1}/{len(paths)}", flush=True)
    arr.flush()
    del arr
    os.replace(tmp, os.path.join(a.out_dir, f"images_u8_{s}.npy"))
    with open(os.path.join(a.out_dir, "paths.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")

    # spot-check: re-decode a sample and compare byte for byte
    arr = np.load(os.path.join(a.out_dir, f"images_u8_{s}.npy"), mmap_mode="r")
    _init(a.data_root, s)
    for i in random.Random(0).sample(range(len(paths)), 200):
        if not np.array_equal(arr[i], _decode(paths[i])):
            raise SystemExit(f"cache mismatch at {paths[i]}")
    print(f"done: {len(paths)} images, spot-check of 200 identical", flush=True)


if __name__ == "__main__":
    main()
