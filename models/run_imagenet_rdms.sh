#!/bin/bash
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
# Extract activations + RDMs for the four ImageNet-pretrained checkpoints over
# the 16 stimulus frames. Runs on one GPU; intended for the idle 4th GPU of an
# existing allocation (srun --jobid=... --overlap).
set -uo pipefail

MODELS_DIR=$ROOT/models
WEIGHTS=$ROOT/net_weights
STIM=$ROOT/stim_frames
OUT=$ROOT/analysis/model_rdms_canonical

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$MODELS_DIR"
mkdir -p "$OUT"

# name:arch:checkpoint
RUNS=(
    "resnet18_imagenet_16f${SUF:-}:resnet18:resnet18_imagenet"
    "resnet50_imagenet_16f${SUF:-}:resnet50:resnet50_imagenet"
    "vgg16bn_imagenet_16f${SUF:-}:vgg16_bn:vgg16_bn_imagenet"
    "facenet_imagenet_16f${SUF:-}:facenet:FaceNet_imagenet"
)

STATUS=0
for r in "${RUNS[@]}"; do
    IFS=: read -r name arch ckpt <<< "$r"
    echo "================ $name  ($arch <- $ckpt)"
    if python extract_model_rdms.py \
            --arch "$arch" \
            --checkpoint "$WEIGHTS/$ckpt" \
            --preproc imagenet \
            --n-slides 16 \
            --h5-dir "$STIM" \
            --out-dir "$OUT" \
            --layers "${LAYERS:-trainable}" --head-classes "${HEAD:-3}" --exclude-head \
            --out-name "$name"; then
        echo "OK $name"
    else
        echo "FAILED $name" >&2
        STATUS=1
    fi
done

echo "=== done at $(date +%T)"
ls -la "$OUT"
exit $STATUS
