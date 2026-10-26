#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x-%A_%a.out
# One array element = one H100: runs CHUNK lines of a task list, PER_GPU at a time.
# Finished runs are skipped and unfinished ones resume, so resubmitting completes a group.
# Called by submit_all.sh.
set -uo pipefail
: "${TASKS:?}" "${PER_GPU:=1}" "${CHUNK:=1}" "${WALL_H:?set WALL_H to the --time in hours}"
R=$SCRATCH/fer_revision
module load StdEnv/2023 python/3.12
source "$R/trainenv/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
# home is read-only on compute nodes: compiler and library caches go to scratch
export TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor XDG_CACHE_HOME=$R/.cache

NCPU=${SLURM_CPUS_PER_TASK:-$(OMP_NUM_THREADS= nproc)}  # not plain nproc: it reports OMP_NUM_THREADS
W=$(( NCPU / PER_GPU - 1 ))
START=$SECONDS
FIRST=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK + 1 ))
echo "node=$(hostname) cpus=$NCPU per_gpu=$PER_GPU workers=$W lines $FIRST..$((FIRST + CHUNK - 1)) of $TASKS"
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 &
SMI=$!

# random reads from network scratch are slow: copy the cache to node memory once
C=$R/data/vggface_cache
if sed -n "${FIRST},$(( FIRST + CHUNK - 1 ))p" "$TASKS" | grep -q '{C}'; then
    SHM=/dev/shm/fer_cache_${SLURM_JOB_ID:-$$}
    trap 'rm -rf "$SHM"' EXIT
    mkdir -p "$SHM"
    t=$SECONDS
    cp "$C/images_u8_128.npy" "$C/paths.txt" "$SHM/" && C=$SHM
    echo "cache in $C ($(( SECONDS - t )) s)"
fi

PIDS=()
while IFS=$'\t' read -r out done_re cmd; do
    [ -n "$out" ] || continue
    if grep -qE "$done_re" "$out/train.log" 2>/dev/null; then
        echo "skip (done) $out"; continue
    fi
    # wait for a free slot on the GPU
    while [ "$(jobs -rp | grep -vx "$SMI" | wc -l)" -ge "$PER_GPU" ]; do sleep 20; done
    left=$(awk -v w="$WALL_H" -v s=$(( SECONDS - START )) 'BEGIN{printf "%.2f", w - s/3600 - 0.15}')
    if awk -v l="$left" 'BEGIN{exit !(l < 0.25)}'; then echo "no time left for $out"; break; fi
    mkdir -p "$out"
    cmd=${cmd//\{W\}/$W}; cmd=${cmd//\{H\}/$left}; cmd=${cmd//\{C\}/$C}
    echo "$(date +%T) start $out (max ${left} h)"
    ( bash -c "$cmd" >>"$out/train.log" 2>&1; echo "$(date +%T) exit $? $out" ) &
    PIDS+=($!)
done < <(sed -n "${FIRST},$(( FIRST + CHUNK - 1 ))p" "$TASKS")

[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
kill $SMI 2>/dev/null
echo "chunk finished in $(( (SECONDS - START) / 60 )) min"
