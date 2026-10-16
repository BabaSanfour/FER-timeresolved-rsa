#!/bin/bash
#SBATCH --job-name=clusterperm
#SBATCH --account=YOUR_ALLOCATION
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
# The script is serial (~6 s per scope x emotion grid, 28 grids per model), so
# it runs for roughly an hour per study -- too long for the login node, whose
# user slice is cgroup-capped at 6 CPUs.
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
mkdir -p "$ROOT/analysis/logs"

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STUDY=$([ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && echo 54 || echo 18)
echo "[task $SLURM_ARRAY_TASK_ID] study $STUDY"
"$PY" "$ROOT/analysis/cluster_permutation_rsa_54.py" --study "$STUDY"
echo "[done] study $STUDY"
