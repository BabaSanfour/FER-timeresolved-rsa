"""Every model variant of the paper and the RDMs behind it.

A variant has one RDM (a released model: ImageNet, VGG-Face, Kinetics-400, DFEW,
Qwen3-VL) or ten: random initialisation seeds 42, 0-8 (saved weights), or training
runs r01-r10 (training seeds 1001-1010). rho_max is computed for each RDM and then
averaged, mean +/- SD over the ten; the cluster tests and the layer x time maps
use the variant's grid averaged over its RDMs (rsa_core.variant_grid).

RDMs: $FER_ROOT/rdms/set{18,54}/<name>/<name>.npy (trillium/extract_tasks.py).
The FaceNet learning designs (size curve, step-matched check, checkpoints along training, identity
counts) exist for the 54 stimuli only. Their constants are defined here once; trillium/make_tasks.py and
extract_tasks.py import them.
The nuisance controls (analysis/control_rdms.py) are one-layer RDMs in the same
place, group "control"; occlusion exists for 54 stimuli only.
"""
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)     # random initialisations (saved weights)
N_RUNS = 10                                  # training runs r01-r10 (seeds 1001-1010)
SIZES = ("10k", "50k", "100k", "200k", "300k", "500k")
FULL_POOL = "715k"           # the full VGGFace2-HQ training pool, 715,810 images


def seeds(prefix):
    return [f"{prefix}_random_s{s}" for s in SEEDS]


def runs(prefix):
    return [f"{prefix}_r{r:02d}" for r in range(1, N_RUNS + 1)]


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
# FaceNet learning designs (trillium/make_tasks.py, three runs each): the weights at nine steps
# of full-pool training (seeds of runs r01-r03), and 50 to 3,500 identities at 100 images each,
# every model trained for the full pool's steps and so able to learn its task.
CHECKPOINT_STEPS = (0, 30, 100, 300, 1000, 3000, 10000, 30000, 55920)    # 55,920 = the full pool's steps
IDENTITIES = ("50ids", "300ids", "1500ids", "3500ids")
N_DESIGN_REPEATS = 3
CHECKPOINTS = [(str(step), [f"facenet_ckpt{step:05d}_r{r:02d}" for r in range(1, N_DESIGN_REPEATS + 1)])
               for step in CHECKPOINT_STEPS]
IDENTITY_COUNTS = [(ids, [f"facenet_{ids}_r{r:02d}" for r in range(1, N_DESIGN_REPEATS + 1)]) for ids in IDENTITIES]
CONTROLS = [("Identity", "identity"), ("Occlusion", "occlusion"), ("Pixel", "pixel"),
            ("Optical flow", "optical_flow"), ("Motion energy", "motion_energy")]


def controls(stimuli):
    """-> [(label, control name)] for this stimulus set."""
    return [(label, name) for label, name in CONTROLS if stimuli == "54" or name != "occlusion"]


def variants(stimuli):
    """-> [(group, family, variant, RDM names)]; groups: main, size, step, ckpt, ids, control."""
    out = [("main", family, variant, tags) for family, rows in MAIN.items() for variant, tags in rows]
    if stimuli == "54":                                  # the FaceNet learning designs
        out += [("size", "FaceNet", size, tags) for size, tags in SIZE_CURVE]
        out += [("step", "FaceNet", size, tags) for size, tags in STEP_MATCHED]
        out += [("ckpt", "FaceNet", step, tags) for step, tags in CHECKPOINTS]
        out += [("ids", "FaceNet", ids, tags) for ids, tags in IDENTITY_COUNTS]
    out += [("control", "Control", label, [f"control_{name}"]) for label, name in controls(stimuli)]
    return out


def all_rdms(stimuli):
    """Every RDM once (the full-pool FaceNet runs are in the main table and the size curve)."""
    return list(dict.fromkeys(tag for *_, tags in variants(stimuli) for tag in tags))
