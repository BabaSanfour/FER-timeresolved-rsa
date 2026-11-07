#!/bin/bash
# 3-class DFEW clips and fold labels as one archive (DFEW_PASSWORD='...' bash M3DFEL/prepare_dfew_3class.sh).
ROOT=$(realpath "${FER_ROOT:-.}")
set -euo pipefail

: "${DFEW_PASSWORD:?Set DFEW_PASSWORD without putting it in this script}"
SOURCE="${DFEW_DIR:-$ROOT/DFEW}"
WORK=$(mktemp -d /tmp/dfew-prepare.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
PREPARED="$WORK/prepared_16f"

mkdir -p "$PREPARED/Clip" "$PREPARED/EmoLabel_DataSplit"

unzip -q -P "$DFEW_PASSWORD" \
  "$SOURCE/folder_1/train(single-labeled).zip" \
  -d "$PREPARED/EmoLabel_DataSplit"
unzip -q -P "$DFEW_PASSWORD" \
  "$SOURCE/folder_1/test(single-labeled).zip" \
  -d "$PREPARED/EmoLabel_DataSplit"

# Keep only original DFEW labels 1=happy, 3=neutral, and 7=fear.
keep_ids="$WORK/keep_ids.txt"
find "$PREPARED/EmoLabel_DataSplit" -type f -name '*.csv' -print0 |
  xargs -0 awk -F, 'FNR > 1 && ($2 == 1 || $2 == 3 || $2 == 7) {
    printf "%05d\n", $1
  }' | sort -u > "$keep_ids"

# Filter the persisted split CSVs too, retaining their headers.
while IFS= read -r csv_file; do
  filtered="$csv_file.filtered"
  awk -F, 'FNR == 1 || $2 == 1 || $2 == 3 || $2 == 7' \
    "$csv_file" > "$filtered"
  mv "$filtered" "$csv_file"
done < <(find "$PREPARED/EmoLabel_DataSplit" -type f -name '*.csv')

unzip -q -P "$DFEW_PASSWORD" \
  "$SOURCE/folder_2/clip_224x224_16f.zip" -d "$WORK/clips"
mkdir -p "$PREPARED/Clip/clip_224x224"
while IFS= read -r clip_id; do
  source_clip="$WORK/clips/clip_224x224_16f/$clip_id"
  if [[ -d "$source_clip" ]]; then
    mv "$source_clip" "$PREPARED/Clip/clip_224x224/"
  fi
done < "$keep_ids"

# one archive stages to a compute node much faster than many small files
tar_tmp="$SOURCE/prepared_16f.tar.tmp"
tar -C "$PREPARED" -cf "$tar_tmp" .
mv "$tar_tmp" "$SOURCE/prepared_16f.tar"

echo "Staging archive: $SOURCE/prepared_16f.tar"
