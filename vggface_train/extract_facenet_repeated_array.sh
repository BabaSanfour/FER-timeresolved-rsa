#!/bin/bash
#SBATCH --job-name=facenet-repeat-rdm
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --array=0-29%10
#SBATCH --output=./vggface_train/logs/%x-%A_%a.out
#SBATCH --error=./vggface_train/logs/%x-%A_%a.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
REPEAT=$((SLURM_ARRAY_TASK_ID / 3 + 1))
COND_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CONDITIONS=(300k 500k 800k_bootstrap)
CONDITION=${CONDITIONS[$COND_INDEX]}
REP=$(printf '%02d' "$REPEAT")
TAG="facenet_vggface2_repeat${REP}_${CONDITION}_16f_weights"
CKPT="$ROOT/vggface_train/results/repeated_subsets/repeat_${REP}/${CONDITION}/best_weights.pt"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1

python "$ROOT/models/extract_model_rdms.py" \
    --arch facenet --checkpoint "$CKPT" --preproc half --resize 112 \
    --layers weights --exclude-head --out-name "$TAG" \
    --h5-dir "$ROOT/stim_frames" \
    --out-dir "$ROOT/analysis/model_rdms_facenet_size_repeats"
echo "[done] $TAG"
