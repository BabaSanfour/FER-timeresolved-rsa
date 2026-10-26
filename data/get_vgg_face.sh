#!/bin/bash
# VGG-Face (Parkhi et al., 2015), PyTorch conversion by S. Albanie: the VGG16 face model.
set -euo pipefail

DEST=${DEST:-${FER_ROOT:-.}/data/net_weights}
F="$DEST/vgg_face_dag.pth"
SHA=f070b241e3faf17a08c78f8700d334413230bac5de7fbcfd01b3cf05ce10de1a
mkdir -p "$DEST"
[ -f "$F" ] || curl -sS -L -o "$F" https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
echo "$SHA  $F" | sha256sum -c - || { echo "FATAL: checksum differs"; exit 1; }
echo "VGG-Face weights: $F"
