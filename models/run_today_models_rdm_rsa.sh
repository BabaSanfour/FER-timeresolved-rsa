#!/bin/bash
#SBATCH --job-name=today-rdm-rsa
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=./models/logs/%x-%j.out
#SBATCH --error=./models/logs/%x-%j.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
MODELS="$ROOT/models"
OUT="$ROOT/analysis/model_rdms_canonical"
RSA_OUT="$ROOT/analysis/rsa_grid_today"
VGG_RUN="$ROOT/vggface_train/results/17798746"
AFF_RUN="$ROOT/affectnet_3class_train/results/17810319"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "$OUT" "$RSA_OUT" "$MODELS/logs"
cd "$MODELS"

# name|architecture|checkpoint|preprocessing|input size
RUNS=(
  "facenet_vggface2_300k_16f_weights|facenet|$VGG_RUN/300k/best_weights.pt|half|112"
  "facenet_vggface2_500k_16f_weights|facenet|$VGG_RUN/500k/best_weights.pt|half|112"
  "facenet_vggface2_affectnet_3class_16f_weights|facenet|$AFF_RUN/facenet/best_weights.pt|half|112"
  "vgg16_vggface_affectnet_3class_16f_weights|vgg16|$AFF_RUN/vgg16_vggface/best_weights.pt|vggface_rgb|224"
  "resnet18_vggface2_affectnet_3class_16f_weights|resnet18|$AFF_RUN/resnet18/best_weights.pt|half|112"
  "resnet50_vggface2_affectnet_3class_16f_weights|resnet50|$AFF_RUN/resnet50/best_weights.pt|half|112"
)

TAGS=()
for run in "${RUNS[@]}"; do
  IFS='|' read -r tag arch checkpoint preproc input_size <<< "$run"
  TAGS+=("$tag")
  echo "=== extracting $tag"
  python extract_model_rdms.py \
    --arch "$arch" \
    --checkpoint "$checkpoint" \
    --preproc "$preproc" \
    --resize "$input_size" \
    --layers weights \
    --exclude-head \
    --out-name "$tag"
done

echo "=== validating/splitting model RDMs by emotion"
python build_emotion_rdms.py --dir "$OUT" --self-test

echo "=== ROI + whole-scalp EEG RSA"
cd "$ROOT"
python analysis/rsa_full_grid.py \
  --models "${TAGS[@]}" \
  --out-dir "$RSA_OUT"

echo "=== completed at $(date -Is)"
