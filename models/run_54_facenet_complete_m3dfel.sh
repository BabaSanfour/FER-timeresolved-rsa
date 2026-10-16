#!/bin/bash
#SBATCH --job-name=rdm54-facenet-m3dfel
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=./models/logs/%x-%j.out
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
# 54-stimulus RDMs for the variants not covered by run_image_models_54_cpu.sh:
# FaceNet "Complete VGGFace2" and the three M3DFEL variants.
# No driver survived for these runs; the arguments below are reconstructed from
# each RDM's .layers.json provenance and were checked by regenerating the
# published RDMs (see README).
set -euo pipefail

PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
STIM=$ROOT/stim_frames_54
OUT=${OUT:-$ROOT/analysis/model_rdms_54}
module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export TORCH_HOME=$ROOT/torch_cache   # holds checkpoints/20180402-114759-vggface2.pt
mkdir -p "$OUT"
cd "$ROOT/models"

# FaceNet, Complete VGGFace2 (facenet-pytorch weights; 133 Conv/Linear hooks,
# the last being the random 3-unit head that the RSA step drops)
$PY image/extract_image_models_16frames.py --model vggface --layers weights \
    --hdf5-dir "$STIM" --manifest "$STIM/stimulus_manifest.json" --n-slides 16 \
    --out-dir "$OUT" --out-name vggface_complete_16f_pretrained

# M3DFEL: random (seed 42), Kinetics-400 backbone, DFEW 3-class fold 3
SAVE=()
[ -n "${SAVE_WEIGHTS_DIR:-}" ] && SAVE=(--save-weights "$SAVE_WEIGHTS_DIR/m3dfel_kinetics_seed42_head.pt")
$PY m3dfel_extract_rdms.py --random --seed 42 --frames-dir "$STIM" \
    --out-dir "$OUT" --out-name m3dfel_random_16f_weights
$PY m3dfel_extract_rdms.py --frames-dir "$STIM" "${SAVE[@]}" \
    --out-dir "$OUT" --out-name m3dfel_pretrained_16f_weights
$PY m3dfel_extract_rdms.py --checkpoint "$ROOT/M3DFEL/results/17812056/fold_3/best_weights.pt" \
    --frames-dir "$STIM" --out-dir "$OUT" --out-name m3dfel_dfew_3class_fold3_16f_weights

$PY build_emotion_rdms_54.py --dir "$OUT"
