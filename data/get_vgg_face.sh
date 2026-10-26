#!/bin/bash
# Fetch VGG-Face (Parkhi et al., 2015) as the PyTorch conversion by S. Albanie
# (pytorch-mcn), the starting point of the VGG16 AffectNet fine-tune and the
# "VGG-Face" variant. Layer names conv1_1 ... fc8 (2,622 identities); inputs are
# RGB in [0, 255] minus the mean (129.186, 104.762, 93.594), which is what
# train_affectnet.py's vgg16_vggface preprocessing applies. 580 MB.
set -euo pipefail

DEST=${DEST:-${FER_ROOT:-.}/data/net_weights}
F="$DEST/vgg_face_dag.pth"
SHA=f070b241e3faf17a08c78f8700d334413230bac5de7fbcfd01b3cf05ce10de1a
mkdir -p "$DEST"
[ -f "$F" ] || curl -sS -L -o "$F" https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
echo "$SHA  $F" | sha256sum -c - || { echo "FATAL: checksum differs"; exit 1; }
echo "VGG-Face weights: $F"
