#!/bin/bash
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
# Launch the three trainings inside the already-running allocation 17733342,
# whose batch script is wedged on a bare `wait` but whose dataset is fully
# staged on node-local disk. Outputs go straight to Lustre so they do not
# depend on the wedged script's sync loop.
set -uo pipefail

PROJECT_DIR=$ROOT/vggface_train
OUT="$PROJECT_DIR/results/17733342"
TMPD=${SLURM_TMPDIR:?run inside the train_vggface2.sh allocation}
DATA_ROOT="$TMPD/work/input/VGGface2_None_norm_512_true_bygfpgan"

echo "node=$(hostname) data_root=$DATA_ROOT"
ls "$DATA_ROOT" | head -2
nvidia-smi -L

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

ARCHS=(resnet50 resnet18 facenet)
PIDS=()
for i in "${!ARCHS[@]}"; do
    arch=${ARCHS[$i]}
    mkdir -p "$OUT/$arch"
    CUDA_VISIBLE_DEVICES=$i python "$PROJECT_DIR/train.py" \
        --arch "$arch" \
        --data-root "$DATA_ROOT" \
        --index-dir "$PROJECT_DIR/index" \
        --out-dir "$OUT/$arch" \
        --epochs 18 \
        --batch-size 512 \
        --workers 20 \
        --max-hours 2.05 \
        > "$OUT/$arch/train.log" 2>&1 &
    PIDS+=($!)
    echo "launched $arch on GPU $i (pid ${PIDS[$i]})"
done

wait "${PIDS[@]}"
echo "all three finished at $(date +%T)"
