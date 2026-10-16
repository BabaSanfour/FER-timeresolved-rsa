#!/bin/bash
#SBATCH --job-name=facenet-low-rdm
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --array=0-39%10
#SBATCH --output=./vggface_train/logs/%x-%A_%a.out
#SBATCH --error=./vggface_train/logs/%x-%A_%a.out
set -euo pipefail
ROOT=$(realpath "${FER_ROOT:-.}")
REPEAT=$((SLURM_ARRAY_TASK_ID / 4 + 1))
COND_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
CONDITIONS=(10k 50k 100k 200k)
CONDITION=${CONDITIONS[$COND_INDEX]}
REP=$(printf '%02d' "$REPEAT")
TAG="facenet_vggface2_repeat${REP}_${CONDITION}_16f_weights"
module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
python "$ROOT/models/extract_model_rdms.py" --arch facenet \
    --checkpoint "$ROOT/vggface_train/results/repeated_subsets/repeat_${REP}/${CONDITION}/best_weights.pt" \
    --preproc half --resize 112 --layers weights --exclude-head --out-name "$TAG" \
    --h5-dir "$ROOT/stim_frames" --out-dir "$ROOT/analysis/model_rdms_facenet_size_repeats"
