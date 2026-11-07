#!/bin/bash
# Submit all RDM extractions on Trillium (after extract_prep.sh); safe to re-run.
set -euo pipefail
mkdir -p logs
python -m fer_rsa.tasks.extract_tasks --out scripts/trillium/tasks ${FEATURES:+--features}

submit() {  # name tasks per_gpu chunk hours
    local n; n=$(grep -c . "$2")
    TASKS=$2 PER_GPU=$3 CHUNK=$4 WALL_H=$5 LOG=extract.log sbatch --parsable --job-name="fer-$1" \
        --time="$5:00:00" --array="0-$(( (n + $4 - 1) / $4 - 1 ))" --export=ALL scripts/trillium/run_tasks.sh
}
# FEATURES=1: the linear probe's final-layer features (features/set*/), 4 per H100
if [ -n "${FEATURES:-}" ]; then echo "features=$(submit features scripts/trillium/tasks/extract_features.txt 4 12 2)"; exit; fi
# CNN/M3DFEL: 4 per H100; Qwen3-VL in float32 (~35 GB): 1 per H100
echo "cnn=$(submit extract-cnn scripts/trillium/tasks/extract_cnn.txt 4 12 2) qwen=$(submit extract-qwen scripts/trillium/tasks/extract_qwen.txt 1 2 2)"
