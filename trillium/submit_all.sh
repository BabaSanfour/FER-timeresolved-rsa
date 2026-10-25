#!/bin/bash
# Submit all retraining on Trillium at once (run from the repo root on trillium-gpu).
# Each array element takes one H100 and packs PER_GPU runs on it (see run_tasks.sh);
# the packing values come from trillium/bench.sh. Fine-tuning waits for pretraining.
# Safe to re-run: finished runs are skipped and unfinished ones resume.
set -euo pipefail
mkdir -p logs
python trillium/make_tasks.py --out trillium/tasks

submit() {  # name tasks per_gpu chunk hours [extra sbatch args]
    local name=$1 tasks=$2 per=$3 chunk=$4 hours=$5; shift 5
    local n; n=$(grep -c . "$tasks")
    local last=$(( (n + chunk - 1) / chunk - 1 ))
    TASKS=$tasks PER_GPU=$per CHUNK=$chunk WALL_H=$hours sbatch --parsable \
        --job-name="fer-$name" --time="$hours:00:00" --array="0-$last" "$@" \
        --export=ALL trillium/run_tasks.sh
}

# Packing (runs per GPU, runs per array element, walltime): set from the benchmark.
PRE=$(submit pretrain    trillium/tasks/pretrain.txt    "${PRE_PER:-2}"  "${PRE_CHUNK:-2}"  "${PRE_H:-6}")
SIZ=$(submit sizecurve   trillium/tasks/sizecurve.txt   "${SIZ_PER:-3}"  "${SIZ_CHUNK:-6}"  "${SIZ_H:-6}")
STP=$(submit stepmatched trillium/tasks/stepmatched.txt "${STP_PER:-3}"  "${STP_CHUNK:-3}"  "${STP_H:-8}")
FT=$(submit finetune     trillium/tasks/finetune.txt    "${FT_PER:-4}"   "${FT_CHUNK:-4}"   "${FT_H:-3}" \
     --dependency="afterany:$PRE")
echo "pretrain=$PRE sizecurve=$SIZ stepmatched=$STP finetune=$FT (after pretrain)"
squeue -u "$USER" -o "%.12i %.14j %.2t %.10M %.6D %R"
