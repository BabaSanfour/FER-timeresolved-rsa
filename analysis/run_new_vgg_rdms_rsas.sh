#!/bin/bash
#SBATCH --job-name=new-vgg-rdm-rsa
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --output=./analysis/logs/%x-%j.out
#SBATCH --error=./analysis/logs/%x-%j.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
EXTRACT="$ROOT/models/extract_model_rdms.py"
mkdir -p "$ROOT/analysis/logs"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

run_model() {
    local gpu=$1 arch=$2 checkpoint=$3 tag=$4
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXTRACT" \
        --arch "$arch" --checkpoint "$checkpoint" --preproc imagenet \
        --out-name "$tag" --layers weights --exclude-head \
        --h5-dir "$ROOT/stim_frames" \
        --out-dir "$ROOT/analysis/model_rdms_canonical"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXTRACT" \
        --arch "$arch" --checkpoint "$checkpoint" --preproc imagenet \
        --out-name "$tag" --layers weights --exclude-head \
        --h5-dir "$ROOT/stim_frames_54" \
        --out-dir "$ROOT/analysis/model_rdms_54"
}

TAG_VGG16=vgg16_imagenet_16f_weights
TAG_VGG16BN=vgg16bn_imagenet_affectnet_3class_16f_weights
run_model 0 vgg16 "$ROOT/net_weights/vgg16_imagenet" "$TAG_VGG16" &
PID0=$!
run_model 1 vgg16_bn \
    "$ROOT/affectnet_3class_train/results/18023262/vgg16_bn_imagenet/best_weights.pt" \
    "$TAG_VGG16BN" &
PID1=$!
wait "$PID0"
wait "$PID1"

"$PY" "$ROOT/models/build_emotion_rdms_54.py" \
    --dir "$ROOT/analysis/model_rdms_54" \
    --models "$TAG_VGG16" "$TAG_VGG16BN"

"$PY" "$ROOT/analysis/rsa_full_grid.py" \
    --models "$TAG_VGG16" "$TAG_VGG16BN"
"$PY" "$ROOT/analysis/rsa_full_grid_54.py" \
    --models "$TAG_VGG16" "$TAG_VGG16BN"

echo "[done] RDMs and RSAs for $TAG_VGG16 and $TAG_VGG16BN"
