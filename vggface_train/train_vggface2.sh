#!/bin/bash
#SBATCH --job-name=vggface2-3x
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=490G
#SBATCH --time=3:00:00
#SBATCH --output=./vggface_train/logs/%x-%j.out
#SBATCH --error=./vggface_train/logs/%x-%j.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT

# Train resnet50, resnet18 and facenet on VGGFace2 identity classification,
# one model per GPU, all three concurrently on a single node.
set -euo pipefail

PROJECT_DIR=$ROOT/vggface_train
ZIP_DIR=$ROOT/vggface_dataset
DATASET_SUBDIR=VGGface2_None_norm_512_true_bygfpgan
VENV=${VENV:-$HOME/envs/fer}

# Persistent output
OUTPUT_DIR="$PROJECT_DIR/results/$SLURM_JOB_ID"

# Node-local work area
WORK_DIR="$SLURM_TMPDIR/work"
TMP_INPUT="$WORK_DIR/input"
TMP_OUTPUT="$WORK_DIR/output"

mkdir -p "$TMP_INPUT" "$TMP_OUTPUT" "$OUTPUT_DIR"

echo "Compute node: $(hostname)"
echo "Temporary directory: $SLURM_TMPDIR"
df -h "$SLURM_TMPDIR"
nvidia-smi -L

# Results survive even if the job is killed mid-run: mirror them back every
# 5 minutes, and once more on exit.
sync_back() { rsync -a "$TMP_OUTPUT/" "$OUTPUT_DIR/" 2>/dev/null || true; }
( while true; do sleep 300; sync_back; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_back; echo "results in $OUTPUT_DIR"' EXIT

# ---------------------------------------------------------------- 1. stage input
# Unzip straight from Lustre to node-local disk: the archives are 6 large
# sequential reads, which Lustre serves well, and this avoids writing 60 GB to
# local disk twice. Pre-creating the shared top-level directory prevents the
# racing-mkdir failure that parallel unzips otherwise hit.
echo "=== staging dataset at $(date +%T)"
STAGE_START=$SECONDS
mkdir -p "$TMP_INPUT/$DATASET_SUBDIR"
UNZIP_PIDS=()
for z in "$ZIP_DIR"/VGGFac0*.zip; do
    unzip -o -q "$z" -d "$TMP_INPUT/" &
    UNZIP_PIDS+=($!)
done
# Wait on the unzip PIDs specifically: a bare `wait` would also block on the
# immortal 5-minute sync loop started above, and never return.
wait "${UNZIP_PIDS[@]}"

N_STAGED=$(find "$TMP_INPUT/$DATASET_SUBDIR" -type f -name '*.jpg' | wc -l)
N_INDEX=$(wc -l < "$PROJECT_DIR/index/train.txt")
N_INDEX=$((N_INDEX + $(wc -l < "$PROJECT_DIR/index/val.txt")))
echo "=== staged $N_STAGED images in $((SECONDS - STAGE_START))s (index expects $N_INDEX)"
if [ "$N_STAGED" -ne "$N_INDEX" ]; then
    echo "FATAL: staged image count does not match the index" >&2
    exit 1
fi

# ---------------------------------------------------------------- 2. train local
cd "$WORK_DIR"
module load StdEnv/2023 python/3.12
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

DATA_ROOT="$TMP_INPUT/$DATASET_SUBDIR"
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-512}
# Whole node: 64 CPUs split across 3 models, leaving a few cores for the main
# processes. Dataloading is the bottleneck, so this is the number that matters.
WORKERS=${WORKERS:-20}
# Leave room for staging already spent plus the final sync.
MAX_HOURS=${MAX_HOURS:-2.3}

ARCHS=(resnet50 resnet18 facenet)
PIDS=()
for i in "${!ARCHS[@]}"; do
    arch=${ARCHS[$i]}
    mkdir -p "$TMP_OUTPUT/$arch"
    CUDA_VISIBLE_DEVICES=$i python "$PROJECT_DIR/train.py" \
        --arch "$arch" \
        --data-root "$DATA_ROOT" \
        --index-dir "$PROJECT_DIR/index" \
        --out-dir "$TMP_OUTPUT/$arch" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --max-hours "$MAX_HOURS" \
        > "$TMP_OUTPUT/$arch/train.log" 2>&1 &
    PIDS+=($!)
    echo "launched $arch on GPU $i (pid ${PIDS[$i]})"
done

STATUS=0
for i in "${!ARCHS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "${ARCHS[$i]}: ok"
    else
        echo "${ARCHS[$i]}: FAILED (see $OUTPUT_DIR/${ARCHS[$i]}/train.log)" >&2
        STATUS=1
    fi
done

# ---------------------------------------------------------------- 3. copy back
echo "=== summary"
for arch in "${ARCHS[@]}"; do
    tail -2 "$TMP_OUTPUT/$arch/train.log" 2>/dev/null | sed "s/^/  /"
done

exit $STATUS
