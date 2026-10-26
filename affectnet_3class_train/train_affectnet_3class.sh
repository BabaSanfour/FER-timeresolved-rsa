#!/bin/bash
#SBATCH --job-name=affectnet-3class
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --output=./affectnet_3class_train/logs/%x-%j.out
#SBATCH --error=./affectnet_3class_train/logs/%x-%j.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
set -euo pipefail

PROJECT=$ROOT/affectnet_3class_train
DATA="${AFFECTNET_DIR:?set AFFECTNET_DIR to the 3-class folder built by data/get_affectnet_3class.sh}"
FINAL="$PROJECT/results/$SLURM_JOB_ID"
STAGE="$SLURM_TMPDIR/affectnet-output"
mkdir -p "$STAGE" "$FINAL"

sync_back() { rsync -a "$STAGE/" "$FINAL/" 2>/dev/null || true; }
( while true; do sleep 300; sync_back; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_back; echo "results=$FINAL"' EXIT

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
nvidia-smi -L

declare -a names=(facenet vgg16_vggface resnet18 resnet50)
declare -a weights=(
    "$ROOT/vggface_train/results/17733342/facenet/best.pt"
    "$ROOT/net_weights/vgg16_vggface"
    "$ROOT/vggface_train/results/17733342/resnet18/best.pt"
    "$ROOT/vggface_train/results/17733342/resnet50/best.pt"
)
declare -a pids=()
for gpu in 0 1 2 3; do
    name=${names[$gpu]}
    mkdir -p "$STAGE/$name"
    batch_size=256
    [[ "$name" == vgg16_vggface ]] && batch_size=64
    CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/train_affectnet.py" \
        --arch "$name" --data-root "$DATA" --out-dir "$STAGE/$name" \
        --weights "${weights[$gpu]}" \
        --epochs "${EPOCHS:-30}" --batch-size "$batch_size" \
        --workers "${WORKERS:-10}" --max-hours "${MAX_HOURS:-3.7}" \
        >"$STAGE/$name/train.log" 2>&1 &
    pids+=("$!")
    echo "launched $name on GPU $gpu pid=${pids[$gpu]}"
done

status=0
for gpu in 0 1 2 3; do
    wait "${pids[$gpu]}" || status=1
done
sync_back
exit "$status"
