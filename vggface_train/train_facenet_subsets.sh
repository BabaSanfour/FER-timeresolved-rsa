#!/bin/bash
#SBATCH --job-name=facenet-subsets
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=./vggface_train/logs/%x-%j.out
#SBATCH --error=./vggface_train/logs/%x-%j.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT

# Train FaceNet from scratch on the deterministic 300k and 500k VGGFace2
# subsets concurrently, one H100 per run.
set -euo pipefail

PROJECT_DIR=$ROOT/vggface_train
DATA_ROOT=$ROOT/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan
VENV=${VENV:-$HOME/envs/fer}
OUTPUT_DIR="$PROJECT_DIR/results/$SLURM_JOB_ID"
WORK_DIR="$SLURM_TMPDIR/facenet-subsets"
TMP_OUTPUT="$WORK_DIR/output"

mkdir -p "$TMP_OUTPUT" "$OUTPUT_DIR"

echo "node=$(hostname) job=$SLURM_JOB_ID tmp=$SLURM_TMPDIR"
nvidia-smi -L

# Mirror checkpoints to persistent storage throughout the run.
sync_back() { rsync -a "$TMP_OUTPUT/" "$OUTPUT_DIR/" 2>/dev/null || true; }
( while true; do sleep 300; sync_back; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_back; echo "results=$OUTPUT_DIR"' EXIT

echo "data=$DATA_ROOT (already extracted; training directly from Lustre)"

module load StdEnv/2023 python/3.12
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-512}
WORKERS=${WORKERS:-22}
MAX_HOURS=${MAX_HOURS:-1.7}

sizes=(300k 500k)
pids=()
for gpu in "${!sizes[@]}"; do
    size=${sizes[$gpu]}
    mkdir -p "$TMP_OUTPUT/$size"
    CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT_DIR/train.py" \
        --arch facenet \
        --data-root "$DATA_ROOT" \
        --index-dir "$PROJECT_DIR/indexes/$size" \
        --out-dir "$TMP_OUTPUT/$size" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --max-hours "$MAX_HOURS" \
        > "$TMP_OUTPUT/$size/train.log" 2>&1 &
    pids+=("$!")
    echo "launched FaceNet-$size on GPU $gpu pid=${pids[$gpu]}"
done

status=0
for i in "${!sizes[@]}"; do
    if wait "${pids[$i]}"; then
        echo "FaceNet-${sizes[$i]}: completed"
    else
        echo "FaceNet-${sizes[$i]}: failed" >&2
        status=1
    fi
done

sync_back
for size in "${sizes[@]}"; do
    tail -3 "$TMP_OUTPUT/$size/train.log" || true
done
exit "$status"
