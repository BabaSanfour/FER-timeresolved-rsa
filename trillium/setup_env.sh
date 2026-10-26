#!/bin/bash
# Build the training environment used for the paper's models (Trillium,
# StdEnv/2023, Python 3.12, Alliance wheelhouse). Exact versions: trainenv_freeze.txt.
# facenet-pytorch is installed without its dependencies: it pins torch<2.3, but only
# its model definition (InceptionResnetV1) is used, which runs on current torch.
# Run on a login node (it needs internet for facenet-pytorch).
set -euo pipefail
ENV=${ENV:-$SCRATCH/fer_revision/trainenv}
module load StdEnv/2023 python/3.12
python3.12 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch torchvision triton numpy pillow requests tqdm
pip install --no-deps facenet-pytorch==2.6.0
python -c "import torch, torchvision, facenet_pytorch; print(torch.__version__, torchvision.__version__)"
