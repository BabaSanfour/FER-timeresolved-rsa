#!/bin/bash
# Training environment (Trillium, Python 3.12; versions in trainenv_freeze.txt).
# facenet-pytorch without dependencies: it pins an old torch, but only its model class is used.
set -euo pipefail
ENV=${ENV:-$SCRATCH/fer_revision/trainenv}
module load StdEnv/2023 python/3.12
python3.12 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch torchvision triton numpy pillow requests tqdm
pip install --no-deps facenet-pytorch==2.6.0
python -c "import torch, torchvision, facenet_pytorch; print(torch.__version__, torchvision.__version__)"
