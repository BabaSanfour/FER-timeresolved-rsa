#!/bin/bash
#SBATCH --job-name=m3dfel-f3-rdm-rsa
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
TAG=m3dfel_dfew_3class_fold3_16f_weights
CHECKPOINT="$ROOT/M3DFEL/results/17812056/selected_fold3/best_weights.pt"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT/models"
python m3dfel_extract_rdms.py \
  --checkpoint "$CHECKPOINT" \
  --out-name "$TAG"

python build_emotion_rdms.py \
  --dir "$ROOT/analysis/model_rdms_canonical" \
  --self-test

cd "$ROOT"
python analysis/rsa_full_grid.py \
  --models "$TAG" \
  --out-dir "$ROOT/analysis/rsa_grid_today"

echo "=== completed at $(date -Is)"
