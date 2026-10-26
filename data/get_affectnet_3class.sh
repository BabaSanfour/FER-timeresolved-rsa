#!/bin/bash
# AffectNet for the 3-class fine-tuning, as $DEST/affectnet_3class/{Train,Test}/{fear,happy,neutral}.
# Kaggle sofiajch07/affectnet v1 (96x96 images): the first models' copy
# (mstjebashazida/affectnet) is gone; this one has the same per-class counts.
# AffectNet's academic terms apply. Needs a Kaggle token in ~/.kaggle.
set -euo pipefail

DEST=${DEST:-${FER_ROOT:-.}/data/affectnet_kaggle}
ZIP="$DEST/sofiajch07_affectnet_v1.zip"
SHA=52f201ebd6c98aca9390667df4e204b0cf3c68a06734d6767fd0703629ffcecf
URL="https://www.kaggle.com/api/v1/datasets/download/sofiajch07/affectnet?datasetVersionNumber=1"
mkdir -p "$DEST"

if [ ! -f "$ZIP" ]; then
  if [ -f ~/.kaggle/access_token ]; then
    curl -sS -L -o "$ZIP" -H "Authorization: Bearer $(cat ~/.kaggle/access_token)" "$URL"
  elif [ -f ~/.kaggle/kaggle.json ]; then
    user=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
    key=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")
    curl -sS -L -o "$ZIP" -u "$user:$key" "$URL"
  else
    echo "FATAL: no Kaggle token in ~/.kaggle (access_token or kaggle.json)"; exit 1
  fi
fi
echo "$SHA  $ZIP" | sha256sum -c - || { echo "FATAL: archive checksum differs"; exit 1; }

unzip -q -n "$ZIP" -d "$DEST/raw"
SRC="$DEST/raw/archive (3)"
OUT="$DEST/affectnet_3class"
for split in Train Test; do
  mkdir -p "$OUT/$split"
  for cls in fear happy neutral; do
    [ -d "$OUT/$split/$cls" ] || cp -al "$SRC/$split/$cls" "$OUT/$split/$cls"
  done
done

declare -A WANT=([Train/fear]=1512 [Train/happy]=2340 [Train/neutral]=2758
                 [Test/fear]=1664 [Test/happy]=2704 [Test/neutral]=2368)
for k in "${!WANT[@]}"; do
  have=$(find "$OUT/$k" -type f | wc -l)
  [ "$have" = "${WANT[$k]}" ] || { echo "FATAL: $k has $have images, expected ${WANT[$k]}"; exit 1; }
done
echo "AFFECTNET_DIR=$OUT  (Train 6,610 / Test 6,736 images)"
