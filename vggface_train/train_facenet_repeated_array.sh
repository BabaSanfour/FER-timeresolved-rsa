#!/bin/bash
#SBATCH --job-name=facenet-repeat
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --array=0-29%10
#SBATCH --output=./vggface_train/logs/%x-%A_%a.out
#SBATCH --error=./vggface_train/logs/%x-%A_%a.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PROJECT="$ROOT/vggface_train"
DATA="$ROOT/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan"
REPEAT=$((SLURM_ARRAY_TASK_ID / 3 + 1))
COND_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CONDITIONS=(300k 500k 800k_bootstrap)
CONDITION=${CONDITIONS[$COND_INDEX]}
SEED=$((1000 + REPEAT))
OUT="$PROJECT/results/repeated_subsets/repeat_$(printf '%02d' "$REPEAT")/$CONDITION"
INDEX="$PROJECT/indexes_repeated/repeat_$(printf '%02d' "$REPEAT")/$CONDITION"
mkdir -p "$OUT"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
nvidia-smi -L

python "$PROJECT/train.py" \
    --arch facenet --data-root "$DATA" --index-dir "$INDEX" --out-dir "$OUT" \
    --epochs 40 --batch-size 512 --workers 14 --max-hours 3.8 --seed "$SEED" \
    >"$OUT/train.log" 2>&1

echo "[done] repeat=$REPEAT condition=$CONDITION seed=$SEED output=$OUT"
