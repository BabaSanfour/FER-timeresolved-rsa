#!/bin/bash
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
STIM="$ROOT/stim_frames_54"
OUT="$ROOT/analysis/model_rdms_54"
EXTRACT="$ROOT/models/extract_model_rdms.py"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16
mkdir -p "$OUT"

run_random() {
  local tag=$1 arch=$2 pp=$3 resize=${4:-0}
  local extra=()
  [ "$resize" != 0 ] && extra+=(--resize "$resize")
  "$PY" "$EXTRACT" --arch "$arch" --random --seed 42 --preproc "$pp" \
    --n-slides 16 --h5-dir "$STIM" --out-dir "$OUT" --layers weights \
    --exclude-head --out-name "$tag" "${extra[@]}"
}

run_ckpt() {
  local tag=$1 arch=$2 ckpt=$3 pp=$4 resize=${5:-0}
  local extra=()
  [ "$resize" != 0 ] && extra+=(--resize "$resize")
  "$PY" "$EXTRACT" --arch "$arch" --checkpoint "$ckpt" --preproc "$pp" \
    --n-slides 16 --h5-dir "$STIM" --out-dir "$OUT" --layers weights \
    --head-classes 3 --exclude-head --out-name "$tag" "${extra[@]}"
}

# Random controls
[ -f "$OUT/resnet18_random_16f_weights.npy" ] || run_random resnet18_random_16f_weights resnet18 imagenet
run_random resnet50_random_16f_weights resnet50 imagenet
run_random facenet_random_16f_weights facenet imagenet
run_random vgg16_random_16f_weights vgg16 imagenet

# ImageNet / face-identity pretraining
run_ckpt resnet18_imagenet_16f_weights resnet18 "$ROOT/net_weights/resnet18_imagenet" imagenet
run_ckpt resnet50_imagenet_16f_weights resnet50 "$ROOT/net_weights/resnet50_imagenet" imagenet
run_ckpt facenet_imagenet_16f_weights facenet "$ROOT/net_weights/FaceNet_imagenet" imagenet
run_ckpt resnet18_vggface2_16f_weights resnet18 "$ROOT/vggface_train/results/17733342/resnet18/best.pt" half 112
run_ckpt resnet50_vggface2_16f_weights resnet50 "$ROOT/vggface_train/results/17733342/resnet50/best.pt" half 112
run_ckpt facenet_vggface2_16f_weights facenet "$ROOT/vggface_train/results/17733342/facenet/best.pt" half 112
run_ckpt vgg16_vggface_16f_weights vgg16 "$ROOT/net_weights/vgg16_vggface" vggface

# VGGFace2 subset-size FaceNets
run_ckpt facenet_vggface2_300k_16f_weights facenet "$ROOT/vggface_train/results/17798746/300k/best_weights.pt" half 112
run_ckpt facenet_vggface2_500k_16f_weights facenet "$ROOT/vggface_train/results/17798746/500k/best_weights.pt" half 112

# AffectNet 3-class fine-tuning
run_ckpt resnet18_vggface2_affectnet_3class_16f_weights resnet18 "$ROOT/affectnet_3class_train/results/17810319/resnet18/best_weights.pt" half 112
run_ckpt resnet50_vggface2_affectnet_3class_16f_weights resnet50 "$ROOT/affectnet_3class_train/results/17810319/resnet50/best_weights.pt" half 112
run_ckpt facenet_vggface2_affectnet_3class_16f_weights facenet "$ROOT/affectnet_3class_train/results/17810319/facenet/best_weights.pt" half 112
run_ckpt vgg16_vggface_affectnet_3class_16f_weights vgg16 "$ROOT/affectnet_3class_train/results/17810319/vgg16_vggface/best_weights.pt" vggface_rgb

"$PY" "$ROOT/models/build_emotion_rdms_54.py" --dir "$OUT"
