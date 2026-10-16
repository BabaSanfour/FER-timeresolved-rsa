#!/bin/bash
ROOT=$(realpath "${FER_ROOT:-.}")   # data/repo root; submit from the repository root or set FER_ROOT
set -uo pipefail
cd $ROOT/models
export LAYERS=weights HEAD=3
SUF=_weights bash run_imagenet_rdms.sh
bash run_vggface2_rdms.sh
bash run_random_rdms.sh
module load StdEnv/2023 python/3.12 >/dev/null 2>&1
source "${VENV:-$HOME/envs/fer}/bin/activate"
python extract_model_rdms.py --arch vgg16 \
    --checkpoint $ROOT/net_weights/vgg16_vggface \
    --preproc vggface --n-slides 16 --layers weights --head-classes 3 --exclude-head \
    --h5-dir $ROOT/stim_frames \
    --out-dir $ROOT/analysis/model_rdms_canonical \
    --out-name vgg16_vggface_16f_weights && echo "OK vgg16_vggface_16f"
echo "=== ALL FINAL DONE at $(date +%T)"
