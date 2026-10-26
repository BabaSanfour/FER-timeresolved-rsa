#!/bin/bash
# Inputs for the RDM extraction (run once, from the repo root, on a Trillium login node):
#   - 16 frames per video for both stimulus sets (one sampler) + Qwen video lists
#   - plain-VGG16 random seeds 0-8, drawn on CPU like the existing seed-42 file
set -euo pipefail
R=${FER_ROOT:-$SCRATCH/fer_revision}
D=$R/data
module load StdEnv/2023 python/3.12 opencv/4.11.0
source "$R/trainenv/bin/activate"

for s in 18 54; do
    python models/build_stimuli_h5_16frames.py --video-dir "$D/stimuli/set$s" --out-dir "$D/stim_frames/set$s" | tail -1
    python - "$D/stimuli/set$s" "$D/stim_frames/set$s" <<'PY'
import csv, json, sys
videos, frames = sys.argv[1:]
rows = json.load(open(f"{frames}/stimulus_manifest.json"))
with open(f"{frames}/qwen_videos.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["video_path", "stimulus_id", "frame_indices"])
    for r in rows:
        w.writerow([f"{videos}/{r['file']}", r["stimulus"], " ".join(map(str, r["sampled_frame_indices"]))])
PY
done

mkdir -p "$D/random_weights"
CUDA_VISIBLE_DEVICES= python - "$D" <<'PY'
import sys, torch
sys.path.insert(0, "models")
from extract_model_rdms import ARCHS, randomize
D = sys.argv[1]
def draw(seed):
    torch.manual_seed(seed)
    # 1000-class output layer, as in the existing seed-42 file (it is excluded from the RDMs)
    return {k: v.clone() for k, v in randomize(ARCHS["vgg16"](None), seed).state_dict().items()}
her = torch.load(f"{D}/drive/random_init_weights/54stim_seed42_cpu/vgg16_random_seed42_cpu.pt", weights_only=True)
same = all(torch.equal(her[k], v) for k, v in draw(42).items())
print(f"seed 42 redrawn on CPU equals the existing file: {same}")
if not same:
    sys.exit("ABORT: the random-weight code no longer reproduces the existing seed")
for seed in range(9):
    torch.save(draw(seed), f"{D}/random_weights/vgg16_random_seed{seed}.pt")
print("vgg16 random seeds 0-8 saved")
PY

# M3DFEL random seeds: redraw on CPU; must equal every downloaded seed; draw the missing ones
M3DFEL_R3D=$D/drive/M3DFEL/pretrained/r3d_18-b3b3357e.pth CUDA_VISIBLE_DEVICES= python - "$D" <<'PY'
import os, sys, torch
from types import SimpleNamespace
sys.path[:0] = ["models", "M3DFEL"]
from m3dfel_extract_rdms import randomize
from models.M3DFEL import M3DFEL
D = sys.argv[1]
def draw(seed):
    torch.manual_seed(seed)
    m = M3DFEL(SimpleNamespace(num_frames=64, instance_length=4, num_classes=7, gpu_ids=[],
                               r3d_weights=os.environ["M3DFEL_R3D"]))
    return {k: v.clone() for k, v in randomize(m, seed).state_dict().items()}
for seed in [42] + list(range(9)):
    her = f"{D}/drive/random_init_weights/m3dfel/m3dfel_random_seed{seed}.pt"
    new = draw(seed)
    if os.path.exists(her):
        old = torch.load(her, weights_only=True)
        if not all(torch.equal(old[k], v) for k, v in new.items()):
            sys.exit(f"ABORT: redrawn M3DFEL seed {seed} differs from the downloaded one")
        print(f"m3dfel seed {seed}: redraw equals the downloaded file")
    else:
        torch.save(new, f"{D}/random_weights/m3dfel_random_seed{seed}.pt")
        print(f"m3dfel seed {seed}: drawn")
PY
