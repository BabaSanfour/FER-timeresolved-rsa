import argparse
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./vggface_dataset/VGGface2_None_norm_512_true_bygfpgan")
    p.add_argument("--out", default="./vggface_train/index")
    p.add_argument("--val-per-class", type=int, default=2, help="images held out per identity")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    classes = sorted(e.name for e in os.scandir(args.root) if e.is_dir())
    print(f"{len(classes)} identities", flush=True)

    train, val, skipped = [], [], []
    for label, cls in enumerate(classes):
        imgs = sorted(
            e.name for e in os.scandir(os.path.join(args.root, cls))
            if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not imgs:
            skipped.append(cls)
            continue
        rng.shuffle(imgs)
        # hold out at most half of an identity's images
        k = min(args.val_per_class, len(imgs) // 2)
        for name in imgs[:k]:
            val.append((f"{cls}/{name}", label))
        for name in imgs[k:]:
            train.append((f"{cls}/{name}", label))
        if label % 500 == 0:
            print(f"  {label}/{len(classes)}", flush=True)

    rng.shuffle(train)

    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            f.writelines(f"{path}\t{label}\n" for path, label in rows)

    print(f"train={len(train)} val={len(val)} total={len(train) + len(val)}")
    if skipped:
        print(f"WARNING: {len(skipped)} identities had no images: {skipped[:10]}")


if __name__ == "__main__":
    main()
