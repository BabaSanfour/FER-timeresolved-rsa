#!/bin/bash
#SBATCH --job-name=fer-bench
#SBATCH --account=rrg-kjerbi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Throughput per model and packing (1-4 runs per H100), plus smoke tests of the
# step-matched options and the AffectNet validation split. Builds the cache if missing.
set -uo pipefail
R=$SCRATCH/fer_revision
REPO=$R/repo
DATA=$R/data/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan
OUT=$R/bench/$SLURM_JOB_ID
mkdir -p "$OUT"
module load StdEnv/2023 python/3.12
source "$R/trainenv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
# home is read-only on compute nodes: compiler and library caches go to scratch
export TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor XDG_CACHE_HOME=$R/.cache
NCPU=${SLURM_CPUS_PER_TASK:-$(OMP_NUM_THREADS= nproc)}  # not plain nproc: it reports OMP_NUM_THREADS
echo "node=$(hostname) cpus=$NCPU"; nvidia-smi -L

CACHE=/dev/shm/fer_cache_$SLURM_JOB_ID
trap 'rm -rf "$CACHE"' EXIT
mkdir -p "$CACHE"; t=$SECONDS
cp "$R/data/vggface_cache/images_u8_128.npy" "$R/data/vggface_cache/paths.txt" "$CACHE/"
echo "cache copied to memory in $((SECONDS - t)) s"

run() {  # name arch workers extra...
    local name=$1 arch=$2 w=$3; shift 3
    python "$REPO/vggface_train/train.py" --arch "$arch" --data-root "$DATA" \
        --index-dir "$REPO/vggface_train/index" --out-dir "$OUT/$name" --cache-dir "$CACHE" \
        --steps 800 --batch-size 512 --workers "$w" --resume none "$@" >"$OUT/$name.log" 2>&1
}
measure() {  # label, then the runs to launch concurrently as "name arch workers extra..."
    local label=$1; shift
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 >"$OUT/$label.smi" &
    local smi=$! pids=() rates="" total=0
    for spec in "$@"; do run $spec & pids+=($!); done
    wait "${pids[@]}"; kill $smi
    for spec in "$@"; do
        set -- $spec
        # the log prints a running average since the epoch began; turn it into the
        # rate over the second half (steps 400-800), past start-up and warm-up
        r=$(grep -oE '[0-9]+/[0-9]+ .* [0-9]+ img/s' "$OUT/$1.log" | awk '{
              split($1, s, "/"); n = s[1]; rate = $(NF-1); t = n * 512 / rate
              if (n == 400) { n0 = n; t0 = t } if (n == 800) { n1 = n; t1 = t } }
              END { if (t1 > t0 && n0) printf "%.0f", (n1 - n0) * 512 / (t1 - t0); else print 0 }')
        rates+=" $r"; total=$(( total + r ))
    done
    util=$(awk -F, 'NR>15{u+=$1;m=($2>m?$2:m);n++} END{printf "%.0f%% max_mem=%dMiB", n?u/n:0, m}' "$OUT/$label.smi")
    echo "RESULT $label img/s:$rates total=$total gpu_util=$util"
}

w() { echo $(( NCPU / $1 - 1 )); }
measure facenet_1x          "fn1 facenet $(w 1)"
measure facenet_compile_1x  "fnc facenet $(w 1) --compile"
measure facenet_2x          "fn2a facenet $(w 2)" "fn2b facenet $(w 2)"
measure facenet_3x          "fn3a facenet $(w 3)" "fn3b facenet $(w 3)" "fn3c facenet $(w 3)"
measure facenet_4x          "fn4a facenet $(w 4)" "fn4b facenet $(w 4)" "fn4c facenet $(w 4)" "fn4d facenet $(w 4)"
measure resnet50_1x         "r50 resnet50 $(w 1)"
measure resnet50_2x         "r50a resnet50 $(w 2)" "r50b resnet50 $(w 2)"
measure resnet18_3x         "r18a resnet18 $(w 3)" "r18b resnet18 $(w 3)" "r18c resnet18 $(w 3)"

echo "== smoke: step-matched options (10k subset, 120 steps, eval every 40)"
python "$REPO/vggface_train/train.py" --arch facenet --data-root "$DATA" \
    --index-dir "$REPO/vggface_train/indexes_repeated/repeat_01/10k" --out-dir "$OUT/smoke_steps" \
    --cache-dir "$CACHE" --steps 120 --eval-every-steps 40 --batch-size 512 --workers 8 --resume none \
    2>&1 | grep -E "classes=|epoch .* done|FINISHED"
echo "== smoke: AffectNet validation split + single Test evaluation (VGG-Face, 2 epochs)"
python "$REPO/affectnet_3class_train/train_affectnet.py" --arch vgg16_vggface \
    --data-root "$R/data/affectnet_kaggle/affectnet_3class" --out-dir "$OUT/smoke_ft" \
    --weights "$R/data/net_weights/vgg_face_dag.pth" --epochs 2 --batch-size 64 --workers 8 \
    --channels-last --resume none 2>&1 | tail -4
head -8 "$OUT/smoke_ft/test_metrics.json" 2>/dev/null
echo "bench done"
