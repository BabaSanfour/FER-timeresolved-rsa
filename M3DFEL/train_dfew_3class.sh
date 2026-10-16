#!/bin/bash
#SBATCH --job-name=m3dfel-dfew-3class
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --array=1-5
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=./M3DFEL/logs/%x-%A_%a.out
#SBATCH --error=./M3DFEL/logs/%x-%A_%a.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
set -euo pipefail

PROJECT=$ROOT/M3DFEL
ARCHIVE="${DFEW_DIR:-$ROOT/DFEW}/prepared_16f.tar"
DATA="$SLURM_TMPDIR/DFEW"
STAGE="$SLURM_TMPDIR/m3dfel-output/fold_${SLURM_ARRAY_TASK_ID}"
FINAL="$PROJECT/results/$SLURM_ARRAY_JOB_ID/fold_${SLURM_ARRAY_TASK_ID}"
R3D="$PROJECT/pretrained/r3d_18-b3b3357e.pth"

mkdir -p "$DATA" "$STAGE" "$FINAL" "$PROJECT/logs"
tar -C "$DATA" -xf "$ARCHIVE"

sync_back() { rsync -a "$STAGE/" "$FINAL/" 2>/dev/null || true; }
( while true; do sleep 300; sync_back; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_back; echo "results=$FINAL"' EXIT

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$PROJECT"
python main.py \
  --data_root "$DATA" \
  --output_path "$STAGE" \
  --r3d_weights "$R3D" \
  --gpu_ids 0,1,2,3 \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --num_classes 3 \
  --epochs "${EPOCHS:-300}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --workers "${WORKERS:-48}" \
  --max_hours "${MAX_HOURS:-5.7}"
