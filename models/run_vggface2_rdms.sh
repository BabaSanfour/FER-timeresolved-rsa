#!/bin/bash
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
# RDMs for the three VGGFace2 models trained in job 17733342, plus the
# VGGFace-pretrained vgg16_bn. Runs on one GPU (srun --jobid=... --overlap).
#
# Preprocessing is per-model, matching how each was trained:
#   - the three VGGFace2 models: (x/255-0.5)/0.5, stimuli resized 224 -> 112
#   - vgg16_bn_1D_VGGFace3:      1-channel grayscale, mean .3612 / std .3056
#                                (Stimuliloader convention), native 224
set -uo pipefail

MODELS_DIR=$ROOT/models
WEIGHTS=$ROOT/net_weights
TRAINED=$ROOT/vggface_train/results/17733342
STIM=$ROOT/stim_frames
OUT=$ROOT/analysis/model_rdms_canonical
LAYERS=${LAYERS:-weights}

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$MODELS_DIR"
mkdir -p "$OUT"

# name|arch|checkpoint|preproc|resize
RUNS=(
    "resnet50_vggface2_16f|resnet50|$TRAINED/resnet50/best.pt|half|112"
    "resnet18_vggface2_16f|resnet18|$TRAINED/resnet18/best.pt|half|112"
    "facenet_vggface2_16f|facenet|$TRAINED/facenet/best.pt|half|112"
    "vgg16bn_vggface_1D_16f|vgg16_bn|$WEIGHTS/vgg16_bn_1D_VGGFace3|gray_fer|0|1"
)

STATUS=0
for r in "${RUNS[@]}"; do
    IFS='|' read -r name arch ckpt pp rs ic <<< "$r"
    echo "================ $name  ($arch <- $(basename "$ckpt"), preproc=$pp resize=${rs:-native} ch=${ic:-3})"
    EXTRA=(--in-channels "${ic:-3}")
    [ "$rs" != "0" ] && EXTRA+=(--resize "$rs")
    if python extract_model_rdms.py \
            --arch "$arch" \
            --checkpoint "$ckpt" \
            --preproc "$pp" \
            --n-slides 16 \
            --h5-dir "$STIM" \
            --out-dir "$OUT" \
            --layers "$LAYERS" --head-classes "${HEAD:-3}" --exclude-head \
            "${EXTRA[@]}" \
            --out-name "${name}_${LAYERS}"; then
        echo "OK $name"
    else
        echo "FAILED $name" >&2
        STATUS=1
    fi
done

echo "=== done at $(date +%T)"
exit $STATUS
