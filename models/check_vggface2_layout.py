"""Check an extracted VGGFace2-HQ copy: identity folders, image count, a sample of images.

    python models/check_vggface2_layout.py <root>
"""
import argparse
import collections
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--num-identities", type=int, default=5346)
    p.add_argument("--num-images", type=int, default=726_502)
    a = p.parse_args()

    folders = sorted(d for d in a.root.iterdir() if d.is_dir())
    counts = {d.name: sum(f.suffix.lower() in IMG_EXT for f in d.iterdir()) for d in folders}
    total, empty = sum(counts.values()), [k for k, v in counts.items() if v == 0]
    print(f"{len(folders)} identities, {total:,} images, {len(empty)} empty folders")

    try:
        from PIL import Image
        sample = collections.Counter()
        for d in folders[::max(1, len(folders) // 12)]:
            with Image.open(next(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)) as im:
                sample[(im.mode, im.size)] += 1
        print(f"sampled images (mode, size): {dict(sample)}")
    except ImportError:
        pass

    ok = len(folders) == a.num_identities and total == a.num_images and not empty
    print("PASS" if ok else f"FAIL: expected {a.num_identities} identities and {a.num_images:,} images")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
