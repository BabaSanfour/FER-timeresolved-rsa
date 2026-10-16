#!/bin/bash
#SBATCH --job-name=seedvar
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1   # STAGE=cnn: the paper's seeds were drawn on this MIG slice
#SBATCH --time=2:00:00
#SBATCH --output=./analysis/seed_variability/logs/%x-%j.out
#SBATCH --error=./analysis/seed_variability/logs/%x-%j.out
# Random-initialization seed variability for the 18-stimulus Table 1.
#
# CNN weights are drawn on the GPU, and CUDA's random kernels depend on the GPU
# partition: the paper's CNN seeds were drawn on an H100 MIG 2g.20gb slice and are
# reproduced bit-for-bit only there (a full H100 or CPU gives other weights).
# STAGE=qwen ran on a full H100 (Qwen and M3DFEL draw their weights on CPU).
# The exact weights are on Drive under random_init_weights/.
#
# The weights are drawn by randomize() AFTER the model is moved to the device,
# so the seed -> weights mapping is device dependent: the published seed-42
# controls were produced on a GPU and a CPU rerun of the same seed gives
# different weights. Everything here therefore runs on the GPU, and the seed-42
# outputs are checked bit-for-bit against the published RDMs before the rest of
# the campaign is trusted.
set -uo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
export FER_ROOT=$ROOT
OUT=$ROOT/analysis/seed_variability/rdms
STIM=$ROOT/stim_frames
VLM=$ROOT/models/vlm
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}

STAGE=${STAGE:?set STAGE=cnn or STAGE=qwen}
CNN_SEEDS="42 0 1 2 3 4 5 6 7 8"
M3DFEL_SEEDS="42 0 1 2 3 4 5 6 7 8"
QWEN_SEEDS="42 0 1 2 3"          # 8B video model: 5 seeds, per plan

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_HOME=$ROOT/.torch
export HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
nvidia-smi -L

STATUS=0
cd "$ROOT/models"

if [ "$STAGE" = cnn ]; then
# ---- image CNNs -----------------------------------------------------------
# name|arch  (num_classes=3, imagenet preproc, native 224px, weights layers,
# head excluded -- identical to models/run_random_rdms.sh)
for spec in "resnet18|resnet18" "resnet50|resnet50" "vgg16bn|vgg16_bn" "facenet|facenet"; do
    IFS='|' read -r name arch <<< "$spec"
    for s in $CNN_SEEDS; do
        tag="${name}_seed${s}_16f_weights"
        [ -f "$OUT/$tag.npy" ] && { echo "skip $tag (exists)"; continue; }
        echo "=========== $tag"
        $PY extract_model_rdms.py --arch "$arch" --random --seed "$s" \
            --num-classes 3 --preproc imagenet --n-slides 16 --exclude-head \
            --in-channels 3 --layers weights \
            --h5-dir "$STIM" --out-dir "$OUT" --out-name "$tag" \
            || { echo "FAILED $tag" >&2; STATUS=1; }
    done
done

# ---- M3DFEL ---------------------------------------------------------------
for s in $M3DFEL_SEEDS; do
    tag="m3dfel_seed${s}_16f_weights"
    [ -f "$OUT/$tag.npy" ] && { echo "skip $tag (exists)"; continue; }
    echo "=========== $tag"
    $PY m3dfel_extract_rdms.py --random --seed "$s" \
        --out-dir "$OUT" --frames-dir "$STIM" --out-name "$tag" \
        || { echo "FAILED $tag" >&2; STATUS=1; }
done

fi

if [ "$STAGE" = qwen ]; then
# ---- Qwen3-VL -------------------------------------------------------------
for s in $QWEN_SEEDS; do
    tag="qwen3vl_seed${s}_16f_weights"
    [ -f "$OUT/$tag.npy" ] && { echo "skip $tag (exists)"; continue; }
    run="$ROOT/analysis/seed_variability/qwen_runs/seed$s"
    echo "=========== $tag"
    if $PY "$VLM/vlm_extract_video_mode.py" \
            --metadata_csv "$VLM/stimuli_metadata_canon.csv" \
            --output_dir "$run" --model_id Qwen/Qwen3-VL-8B-Instruct \
            --nframes 16 --frame-list --dtype bfloat16 --device cuda \
            --random --seed "$s"; then
        $PY "$VLM/compute_rdms_from_vlm_activations.py" --run-dir "$run" \
            --out-dir "$OUT" --out-name "$tag" \
            || { echo "FAILED rdm $tag" >&2; STATUS=1; }
        rm -rf "$run/activations"          # ~175 MB per seed
    else
        echo "FAILED $tag" >&2; STATUS=1
    fi
done

fi

# ---- validate seed 42 against the published controls ----------------------
echo "=========== seed-42 reproduction check"
ROOT="$ROOT" $PY - <<'PYCHK'
import numpy as np, pathlib, os
OUT = pathlib.Path(os.environ["ROOT"], "analysis/seed_variability/rdms")
CAN = pathlib.Path(os.environ["ROOT"], "analysis/model_rdms_canonical")
pairs = [("resnet18_seed42_16f_weights", "resnet18_random_16f_weights"),
         ("resnet50_seed42_16f_weights", "resnet50_random_16f_weights"),
         ("vgg16bn_seed42_16f_weights",  "vgg16bn_random_16f_weights"),
         ("facenet_seed42_16f_weights",  "facenet_random_16f_weights"),
         ("m3dfel_seed42_16f_weights",   "m3dfel_random_16f_weights"),
         ("qwen3vl_seed42_16f_weights",  "qwen3vl_random_16f_weights")]
for new, old in pairs:
    pn, po = OUT / f"{new}.npy", CAN / f"{old}.npy"
    if not pn.exists() or not po.exists():
        print(f"  {new:34s} MISSING"); continue
    a, b = np.load(po), np.load(pn)
    if a.shape != b.shape:
        print(f"  {new:34s} SHAPE {b.shape} vs published {a.shape}"); continue
    d = float(np.abs(a - b).max())
    print(f"  {new:34s} {'IDENTICAL' if d == 0 else f'max|diff|={d:.2e}'}")
PYCHK

echo "=== done at $(date +%T) status=$STATUS"
exit $STATUS
