#!/bin/bash
# Submit all retraining on Trillium (from the repo root on trillium-gpu). Safe to re-run.
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

# runs per GPU, runs per array element, walltime (h); from bench.sh: FaceNet 9.1k img/s
# alone, 16.1k with 3 runs per H100
PRE=$(submit pretrain    trillium/tasks/pretrain.txt    "${PRE_PER:-3}"  "${PRE_CHUNK:-3}"  "${PRE_H:-4}")
SIZ=$(submit sizecurve   trillium/tasks/sizecurve.txt   "${SIZ_PER:-3}"  "${SIZ_CHUNK:-6}"  "${SIZ_H:-3}")
STP=$(submit stepmatched trillium/tasks/stepmatched.txt "${STP_PER:-3}"  "${STP_CHUNK:-3}"  "${STP_H:-4}")
FT=$(submit finetune     trillium/tasks/finetune.txt    "${FT_PER:-4}"   "${FT_CHUNK:-4}"   "${FT_H:-2}" \
     --dependency="afterany:$PRE")
echo "pretrain=$PRE sizecurve=$SIZ stepmatched=$STP finetune=$FT (after pretrain)"
squeue -u "$USER" -o "%.12i %.14j %.2t %.10M %.6D %R"
