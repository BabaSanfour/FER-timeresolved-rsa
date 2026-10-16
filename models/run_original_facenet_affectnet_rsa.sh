#!/bin/bash
#SBATCH --job-name=facenet-original-rsa
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=./models/logs/%x-%j.out
#SBATCH --error=./models/logs/%x-%j.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PROJECT="$ROOT/models"
CHECKPOINT="$ROOT/net_weights/best_inceptionresnetv1_fer.pt"
RDM_DIR="$ROOT/analysis/model_rdms_canonical"
RSA_DIR="$ROOT/analysis/rsa_grid_facenet_original_vggface2_affectnet_16win"
PRE_TAG=vggface_affectnet_16f_pretrained
FT_TAG=vggface_affectnet_16f_finetuned

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export FER_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

mkdir -p "$PROJECT/logs" "$RSA_DIR"
cd "$PROJECT"

python image/extract_image_models_16frames.py \
  --model vggface \
  --finetuned "$CHECKPOINT" \
  --hdf5-dir "$ROOT/stim_frames" \
  --n-slides 16 \
  --out-name "$FT_TAG"

python "$ROOT/models/build_emotion_rdms.py" --dir "$RDM_DIR" --self-test

cd "$ROOT"
python analysis/rsa_full_grid.py \
  --models "$PRE_TAG" "$FT_TAG" \
  --out-dir "$RSA_DIR"
