#!/bin/bash
# Training and extraction environment on Trillium (from the repository root).
set -euo pipefail
ENV=${ENV:-${FER_ROOT:-$SCRATCH/fer_revision}/trainenv}
module load StdEnv/2023 python/3.12
python3.12 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install -e ".[models]"
pip install --no-deps facenet-pytorch==2.6.0
python -c "import torch, torchvision, facenet_pytorch, transformers, timm, einops, h5py; print(torch.__version__, transformers.__version__)"
