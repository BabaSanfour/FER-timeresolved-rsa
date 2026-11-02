#!/bin/bash
# Training and extraction environment (Trillium, Python 3.12), from the repository root.
# Dependencies: pyproject.toml (analysis + extra "models"); exact versions in trainenv_freeze.txt.
# facenet-pytorch without dependencies: it pins an old torch, but only its model class is used.
set -euo pipefail
ENV=${ENV:-$SCRATCH/fer_revision/trainenv}
module load StdEnv/2023 python/3.12
python3.12 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install ".[models]"
pip install --no-deps facenet-pytorch==2.6.0
python -c "import torch, torchvision, facenet_pytorch, transformers, timm, einops, h5py; print(torch.__version__, transformers.__version__)"
# OpenCV is the cluster module: module load opencv/4.11.0 (see extract_prep.sh)
