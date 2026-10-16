#!/bin/bash
#SBATCH --job-name=facenet-repeat-rsa
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --output=./vggface_train/logs/%x-%j.out
#SBATCH --error=./vggface_train/logs/%x-%j.out
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export FER_ROOT="$ROOT"
export OMP_NUM_THREADS=32

tags=()
for repeat in $(seq -w 1 10); do
    for condition in 300k 500k 800k_bootstrap; do
        tags+=("facenet_vggface2_repeat${repeat}_${condition}_16f_weights")
    done
done

python "$ROOT/analysis/rsa_full_grid.py" --models "${tags[@]}" \
    --model-dir "$ROOT/analysis/model_rdms_facenet_size_repeats" \
    --out-dir "$ROOT/analysis/rsa_grid_facenet_size_repeats"
python "$ROOT/analysis/summarize_facenet_size_repeats.py"
echo "[done] repeated FaceNet RSA summary"
