#!/bin/bash
#SBATCH --job-name=layermatch
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=./analysis/layer_matched/logs/%x-%j.out
#SBATCH --error=./analysis/layer_matched/logs/%x-%j.out
# Matched layer count for the FaceNet block of the 18-stimulus Table 1.
#
# rho_max is a max over layers x 16 windows, so a variant hooked at more layers
# wins value by selection alone. In the published table the FaceNet block mixes
# three layer sets: 133 (--layers weights, Conv+Linear) for random/ImageNet/
# 300k/500k, 246 for facenet_vggface3_16f ("VGGFace2 800k"), and 379 (the
# `trainable` set, which also hooks BatchNorm) for the two Complete VGGFace2
# variants. This re-extracts the three odd ones out at the weights selection so
# every FaceNet row is a max over the same layer axis. Nothing else changes:
# each model keeps its own checkpoint, preprocessing and input size.
set -uo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
OUT=$ROOT/analysis/layer_matched/rdms
STIM=$ROOT/stim_frames
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}

module load StdEnv/2023 python/3.12
source "${VENV:-$HOME/envs/fer}/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_HOME=$ROOT/torch_cache   # holds checkpoints/20180402-114759-vggface2.pt
mkdir -p "$OUT"
nvidia-smi -L

STATUS=0
cd "$ROOT/models"

# ---- VGGFace2 800k: BLOCKED ----------------------------------------------
# facenet_vggface3_16f (246 layers) cannot be re-extracted: its checkpoint,
# net_weights/FaceNet_weights_3D_input_VGGFace3, is a 0-byte file and the path
# its RDM provenance records (an older project folder) no
# longer exists. The 800k_bootstrap weights under vggface_train/results/
# repeated_subsets/ are NOT a substitute: those are num_classes=3, preproc
# half, resize 112, whereas this variant is a 4605-way identity model at the
# facenet preprocessing. It is not a Table 1 row, so the matched-layer table does not depend on
# it. Restore the checkpoint to re-enable.
echo "SKIP facenet_vggface3_16f_weights (checkpoint missing; see comment)"

# ---- Complete VGGFace2, pretrained + AffectNet FT: 379 -> 133 --------------
# This family is built by a separate script (InceptionResnetV1_Custom, whose
# last_linear/last_bn are Identity), so `weights` gives 132 convs + the 3-unit
# `logits` head = 133 hooks. logits is a randomly initialised projection and is
# already excluded from every contrast; it is dropped at the RSA step below.
tag=vggface_affectnet_16f_pretrained_weights
if [ -f "$OUT/$tag.npy" ]; then echo "skip $tag (exists)"; else
echo "=========== $tag"
$PY image/extract_image_models_16frames.py --model vggface \
    --hdf5-dir "$STIM" --n-slides 16 --layers weights \
    --out-dir "$OUT" --out-name "$tag" \
    || { echo "FAILED $tag" >&2; STATUS=1; }
fi

tag=vggface_affectnet_16f_finetuned_weights
if [ -f "$OUT/$tag.npy" ]; then echo "skip $tag (exists)"; else
echo "=========== $tag"
$PY image/extract_image_models_16frames.py --model vggface \
    --finetuned "$ROOT/net_weights/best_inceptionresnetv1_fer.pt" \
    --hdf5-dir "$STIM" --n-slides 16 --layers weights \
    --out-dir "$OUT" --out-name "$tag" \
    || { echo "FAILED $tag" >&2; STATUS=1; }
fi

# ---- RSA grids, both cohorts ----------------------------------------------
cd "$ROOT"
TAGS="vggface_affectnet_16f_pretrained_weights vggface_affectnet_16f_finetuned_weights"
for cohort in with-ceg no-ceg; do
    echo "=========== rsa $cohort"
    $PY analysis/seed_variability/rsa_grid_seeds.py --cohort "$cohort" \
        --model-dir "$OUT" --out-dir "$ROOT/analysis/layer_matched/rsa_$cohort" \
        --models $TAGS \
        || { echo "FAILED rsa $cohort" >&2; STATUS=1; }
done

echo "EXIT STATUS $STATUS"
exit $STATUS
