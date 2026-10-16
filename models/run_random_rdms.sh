#!/bin/bash
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
# Random-weight (Xavier) control RDMs, one matched to every real model already
# extracted: same architecture, head size, preprocessing, input resolution and
# input channels. Only the weights differ, so any RSA difference is attributable
# to training rather than to architecture or input handling.
# Run on a full H100 GPU: the seed-42 weights in analysis/model_rdms_canonical/
# reproduce bit-for-bit there (weights are drawn on the GPU, and a MIG slice or
# the CPU gives different weights; see README).
set -uo pipefail

MODELS_DIR=$ROOT/models
STIM=$ROOT/stim_frames
OUT=$ROOT/analysis/model_rdms_canonical
LAYERS=${LAYERS:-weights}
SEED=${SEED:-42}

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd "$MODELS_DIR"
mkdir -p "$OUT"

# name|arch|num_classes|preproc|resize|in_channels
# Head size 3: these are the pre-fine-tuning baseline for the 3-class emotion
# task (fear / happy / neutral), so the head matches the FER models rather than
# the identity/object head of whatever the matched real model was trained on.
NC=${NC:-3}
# One random control per architecture. The earlier per-condition variants
# (224/imagenet vs 112/half preprocessing) correlated at mean r = 0.97, so they
# were redundant; the 224px ImageNet-preprocessing config is the canonical one.
# vgg16bn_random also serves as the control for the plain-vgg16 VGG-Face model:
# randomize() sets BN to an identity transform, so a randomised vgg16_bn and a
# randomised vgg16 are near-identical (measured r = 0.94).
RUNS=(
    "resnet18_random_16f|resnet18|$NC|imagenet|0|3"
    "resnet50_random_16f|resnet50|$NC|imagenet|0|3"
    "vgg16bn_random_16f|vgg16_bn|$NC|imagenet|0|3"
    "facenet_random_16f|facenet|$NC|imagenet|0|3"
)

STATUS=0
for r in "${RUNS[@]}"; do
    IFS='|' read -r name arch nc pp rs ic <<< "$r"
    echo "================ $name  ($arch random, nc=$nc preproc=$pp resize=${rs} ch=$ic)"
    EXTRA=(--in-channels "$ic")
    [ "$rs" != "0" ] && EXTRA+=(--resize "$rs")
    if python extract_model_rdms.py \
            --arch "$arch" --random --seed "$SEED" --num-classes "$nc" \
            --preproc "$pp" --n-slides 16 --exclude-head \
            --h5-dir "$STIM" --out-dir "$OUT" --layers "$LAYERS" \
            "${EXTRA[@]}" --out-name "${name}_${LAYERS}"; then
        echo "OK $name"
    else
        echo "FAILED $name" >&2
        STATUS=1
    fi
done
echo "=== done at $(date +%T)"
exit $STATUS
