#!/bin/bash
#SBATCH --job-name=facenet-low-repeat
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=1:30:00
#SBATCH --array=0-39%10
#SBATCH --output=./vggface_train/logs/%x-%A_%a.out
#SBATCH --error=./vggface_train/logs/%x-%A_%a.out
set -euo pipefail
ROOT=$(realpath "${FER_ROOT:-.}")
PROJECT="$ROOT/vggface_train"
REPEAT=$((SLURM_ARRAY_TASK_ID / 4 + 1))
COND_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
CONDITIONS=(10k 50k 100k 200k)
CONDITION=${CONDITIONS[$COND_INDEX]}
SEED=$((1000 + REPEAT))
REP=$(printf '%02d' "$REPEAT")
OUT="$PROJECT/results/repeated_subsets/repeat_${REP}/$CONDITION"
mkdir -p "$OUT"
module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python "$PROJECT/train.py" --arch facenet \
    --data-root "$ROOT/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan" \
    --index-dir "$PROJECT/indexes_repeated/repeat_${REP}/$CONDITION" \
    --out-dir "$OUT" --epochs 40 --batch-size 512 --workers 10 \
    --max-hours 1.4 --seed "$SEED" >"$OUT/train.log" 2>&1
echo "[done] repeat=$REPEAT condition=$CONDITION"
