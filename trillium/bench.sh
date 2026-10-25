#!/bin/bash
#SBATCH --job-name=fer-bench
#SBATCH --account=rrg-kjerbi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Build the decoded-image cache once, then measure training throughput and GPU
# use per model and per packing (1-3 trainings sharing one H100), with and
# without torch.compile. Submit from the repo root on trillium-gpu.
set -uo pipefail
R=$SCRATCH/fer_revision
REPO=$R/repo
DATA=$R/data/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan
CACHE=$R/data/vggface_cache
OUT=$R/bench/$SLURM_JOB_ID
mkdir -p "$OUT"
module load StdEnv/2023 python/3.12
source "$R/trainenv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
NCPU=${SLURM_CPUS_PER_TASK:-$(OMP_NUM_THREADS= nproc)}  # not plain nproc: it reports OMP_NUM_THREADS
echo "node=$(hostname) cpus=$NCPU"; nvidia-smi -L; free -g | head -2

if [ ! -f "$CACHE/images_u8_128.npy" ]; then
    t=$SECONDS
    python "$REPO/vggface_train/build_decode_cache.py" --data-root "$DATA" \
        --index-dir "$REPO/vggface_train/index" --out-dir "$CACHE" --workers "$NCPU"
    echo "CACHE built in $((SECONDS - t)) s"
fi
ls -l "$CACHE"

run() {  # name arch workers extra...
    local name=$1 arch=$2 w=$3; shift 3
    python "$REPO/vggface_train/train.py" --arch "$arch" --data-root "$DATA" \
        --index-dir "$REPO/vggface_train/index" --out-dir "$OUT/$name" \
        --steps 400 --batch-size 512 --workers "$w" --resume none "$@" >"$OUT/$name.log" 2>&1
}
measure() {  # label, then the runs to launch concurrently as "name arch workers extra..."
    local label=$1; shift
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 >"$OUT/$label.smi" &
    local smi=$! pids=()
    for spec in "$@"; do run $spec & pids+=($!); done
    wait "${pids[@]}"; kill $smi
    local rates=""
    for spec in "$@"; do
        set -- $spec
        # mean img/s over the logged points after the first 100 steps (warm-up)
        rates+=" $(grep -o '[0-9]* img/s' "$OUT/$1.log" | awk 'NR>2{s+=$1;n++} END{printf "%.0f", n?s/n:0}')"
    done
    util=$(awk -F, 'NR>10{u+=$1;m=($2>m?$2:m);n++} END{printf "%.0f%% max_mem=%dMiB", n?u/n:0, m}' "$OUT/$label.smi")
    echo "RESULT $label img/s:$rates gpu_util=$util"
}

measure facenet_nocache_1x          "fn0 facenet $((NCPU-2))"
measure facenet_1x                  "fn1 facenet $((NCPU-2)) --cache-dir $CACHE"
measure facenet_compile_1x          "fnc facenet $((NCPU-2)) --cache-dir $CACHE --compile"
measure facenet_2x                  "fn2a facenet $((NCPU/2-1)) --cache-dir $CACHE" "fn2b facenet $((NCPU/2-1)) --cache-dir $CACHE"
measure facenet_3x                  "fn3a facenet $((NCPU/3-1)) --cache-dir $CACHE" "fn3b facenet $((NCPU/3-1)) --cache-dir $CACHE" "fn3c facenet $((NCPU/3-1)) --cache-dir $CACHE"
measure resnet50_1x                 "r50 resnet50 $((NCPU-2)) --cache-dir $CACHE"
measure resnet50_compile_1x         "r50c resnet50 $((NCPU-2)) --cache-dir $CACHE --compile"
measure resnet18_1x                 "r18 resnet18 $((NCPU-2)) --cache-dir $CACHE"
measure resnet18_2x                 "r18a resnet18 $((NCPU/2-1)) --cache-dir $CACHE" "r18b resnet18 $((NCPU/2-1)) --cache-dir $CACHE"
echo "bench done"
