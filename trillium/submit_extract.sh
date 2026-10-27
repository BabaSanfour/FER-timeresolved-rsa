#!/bin/bash
# Submit all RDM extractions on Trillium (from the repo root on trillium-gpu; run
# trillium/extract_prep.sh once first). Safe to re-run: finished extractions are skipped.
set -euo pipefail
mkdir -p logs
python trillium/extract_tasks.py --out trillium/tasks

submit() {  # name tasks per_gpu chunk hours
    local n; n=$(grep -c . "$2")
    TASKS=$2 PER_GPU=$3 CHUNK=$4 WALL_H=$5 LOG=extract.log sbatch --parsable --job-name="fer-$1" \
        --time="$5:00:00" --array="0-$(( (n + $4 - 1) / $4 - 1 ))" --export=ALL trillium/run_tasks.sh
}
# CNN/M3DFEL: 4 per H100; Qwen3-VL in float32 (~35 GB): 1 per H100
echo "cnn=$(submit extract-cnn trillium/tasks/extract_cnn.txt 4 12 2) qwen=$(submit extract-qwen trillium/tasks/extract_qwen.txt 1 2 2)"
