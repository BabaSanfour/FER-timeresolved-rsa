#!/bin/bash
#SBATCH --job-name=m3dfel-rdm
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=./models/vlm/logs/%x-%j.out
#SBATCH --error=./models/vlm/logs/%x-%j.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
set -uo pipefail
module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 TORCH_HOME=$ROOT/.torch
cd $ROOT/models
nvidia-smi -L
STATUS=0
python m3dfel_extract_rdms.py --out-name m3dfel_pretrained_16f_weights || STATUS=1
python m3dfel_extract_rdms.py --random --seed 42 --out-name m3dfel_random_16f_weights || STATUS=1
echo "=== done at $(date +%T)"
exit $STATUS
