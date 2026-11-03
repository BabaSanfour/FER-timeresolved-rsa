#!/bin/bash
#SBATCH --job-name=clusterperm
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:55:00
#SBATCH --output=./analysis/logs/%x-%A_%a.out
#SBATCH --error=./analysis/logs/%x-%A_%a.out
# One-sample grid cluster permutation tests, one array task per stimulus set.
#
#   sbatch submit_cluster_permutation.sh
#
# One test per variant (21 per study, analysis/variants.py), on its grid averaged
# over its RDMs; 20,000 sign flips. Serial: about 10-30 min per study.
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
mkdir -p "$ROOT/analysis/logs"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STUDY=$([ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && echo 54 || echo 18)
echo "[task $SLURM_ARRAY_TASK_ID] study $STUDY"
"$PY" "$(dirname "$0")/cluster_permutation.py" --study "$STUDY"
echo "[done] study $STUDY"
