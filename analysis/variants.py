"""Every model variant of the paper and the RDMs behind it.

A variant has one RDM (a released model: ImageNet, VGG-Face, Kinetics-400, DFEW,
Qwen3-VL) or ten: random initialisation seeds 42, 0-8 (saved weights), or training
runs r01-r10 (training seeds 1001-1010). rho_max is computed for each RDM and then
averaged, mean +/- SD over the ten. The first RDM of a variant (seed 42, run 1)
represents it in the cluster tests and the layer x time maps.

RDMs: $FER_ROOT/rdms/set{18,54}/<name>/<name>.npy (trillium/extract_tasks.py).
The size curve and the step-matched check (FaceNet) exist for 18 stimuli only.
"""
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
SIZES = ("10k", "50k", "100k", "200k", "300k", "500k")
FULL_POOL = "715k"           # the full VGGFace2-HQ training pool, 715,810 images


def seeds(prefix):
    return [f"{prefix}_random_s{s}" for s in SEEDS]


def runs(prefix):
    return [f"{prefix}_r{r:02d}" for r in range(1, 11)]


# family -> [(variant, RDM names)], in the order of Table 1
MAIN = {
    "ResNet18": [("Random", seeds("resnet18")), ("ImageNet", ["resnet18_imagenet"]),
                 ("VGGFace2", runs("resnet18_vggface2")), ("AffectNet FT", runs("resnet18_affectnet"))],
    "ResNet50": [("Random", seeds("resnet50")), ("ImageNet", ["resnet50_imagenet"]),
                 ("VGGFace2", runs("resnet50_vggface2")), ("AffectNet FT", runs("resnet50_affectnet"))],
    "FaceNet": [("Random", seeds("facenet")), ("ImageNet", ["facenet_imagenet"]),
                ("VGGFace2", runs("facenet_vggface2")), ("AffectNet FT", runs("facenet_affectnet"))],
    "VGG16": [("Random", seeds("vgg16")), ("ImageNet", ["vgg16_imagenet"]),
              ("VGG-Face", ["vgg16_vggface"]), ("AffectNet FT", runs("vgg16_affectnet"))],
    "M3DFEL": [("Random", seeds("m3dfel")), ("Kinetics-400", ["m3dfel_kinetics"]),
               ("DFEW FT", ["m3dfel_dfew"])],
    "Qwen3-VL": [("Random", seeds("qwen")), ("Pretrained", ["qwen_released"])],
}
# FaceNet trained on nested VGGFace2-HQ subsets for 40 epochs (size curve) or for the
# full pool's step count (step-matched check); the full pool closes the size curve.
SIZE_CURVE = [(size, runs(f"facenet_size{size}")) for size in SIZES] + [
    (FULL_POOL, runs("facenet_vggface2"))]
STEP_MATCHED = [(size, runs(f"facenet_step{size}")) for size in SIZES[:-1]]


def variants(stimuli):
    """-> [(group, family, variant, RDM names)]; groups: main, size, step."""
    out = [("main", family, variant, tags) for family, rows in MAIN.items() for variant, tags in rows]
    if stimuli == "18":
        out += [("size", "FaceNet", size, tags) for size, tags in SIZE_CURVE]
        out += [("step", "FaceNet", size, tags) for size, tags in STEP_MATCHED]
    return out


def all_rdms(stimuli):
    """Every RDM once (the full-pool FaceNet runs are in the main table and the size curve)."""
    return list(dict.fromkeys(tag for *_, tags in variants(stimuli) for tag in tags))


def representatives(stimuli, group="main"):
    """The first RDM of each variant (seed 42 or run 1), for the cluster tests."""
    return [tags[0] for g, *_, tags in variants(stimuli) if g == group]
