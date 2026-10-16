#!/bin/bash
# Fetch VGGFace2-HQ train zips 01-06 from the HuggingFace mirror and extract.
#
# Why only 01-06: the zips are ordered by identity (zip06 starts at n004754).
# The paper's VGGFace2 models were trained on these six zips (5,346 identities,
# 715,810 training images after vggface_train/build_index.py). Set ZIPS="01 .. 10" to get all.
# vggface_train/train_vggface2.sh reads the zips from $FER_ROOT/vggface_dataset/.
#
# Why not Kaggle: kagglehub throttled to ~276 kB/s single-stream (~50h for the
# 48.3 GB repackage). HF sustains ~1.8 MB/s per stream and parallelises.
#
# RUN ON A LOGIN NODE, DETACHED so it survives your session ending:
#   nohup bash data/get_vggface2_hq.sh > vggface2_dl.log 2>&1 &
#   tail -f vggface2_dl.log
#
# Safe to re-run: curl -C - resumes partial files, unzip -n never overwrites.
set -uo pipefail

BASE="https://huggingface.co/datasets/ProgramComputer/VGGFace2-HQ/resolve/main/train"
DEST=${DEST:-${FER_ROOT:-.}/vggface_dataset}
ZIPDIR="$DEST"            # train_vggface2.sh reads $DEST/VGGFac0*.zip
TRAIN="$DEST"             # unpacks to $DEST/VGGface2_None_norm_512_true_bygfpgan/
ZIPS=${ZIPS:-"01 02 03 04 05 06"}
mkdir -p "$ZIPDIR" "$TRAIN"

# exact sizes from the HF API -- a truncated download is otherwise silent
declare -A SIZE=(
  [01]=9994763942 [02]=9994086229 [03]=9994151915
  [04]=9994056356 [05]=9994088404 [06]=9994042064
  [07]=9994142344 [08]=9994101971 [09]=9994089009 [10]=5763723321
)

echo "[$(date +%H:%M:%S)] fetching zips: $ZIPS -> $ZIPDIR"
for i in $ZIPS; do
  (
    f="$ZIPDIR/VGGFac$i.zip"
    for attempt in 1 2 3 4 5; do
      curl -sS -L -C - --retry 5 --retry-delay 10 --retry-all-errors \
           -o "$f" "$BASE/VGGFac$i.zip" && break
      echo "[$(date +%H:%M:%S)] zip$i attempt $attempt failed, retrying"
      sleep 15
    done
    have=$(stat -c%s "$f" 2>/dev/null || echo 0)
    want=${SIZE[$i]}
    if [ "$have" = "$want" ]; then
      echo "[$(date +%H:%M:%S)] zip$i OK ($have bytes)"
    else
      echo "[$(date +%H:%M:%S)] zip$i SIZE MISMATCH have=$have want=$want -- re-run to resume"
    fi
  ) &
done
wait
echo "[$(date +%H:%M:%S)] downloads finished"

echo "[$(date +%H:%M:%S)] extracting into $TRAIN"
for i in $ZIPS; do
  f="$ZIPDIR/VGGFac$i.zip"
  have=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$have" != "${SIZE[$i]}" ]; then
    echo "  skipping zip$i -- incomplete"
    continue
  fi
  # -n: never overwrite, so re-runs are cheap and idempotent
  unzip -q -n "$f" -d "$TRAIN" && echo "  zip$i extracted"
done

echo "[$(date +%H:%M:%S)] identity folders now present: $(find "$TRAIN/VGGface2_None_norm_512_true_bygfpgan" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "[$(date +%H:%M:%S)] verifying layout"
python "$(dirname "$0")/../models/check_vggface2_layout.py" "$TRAIN/VGGface2_None_norm_512_true_bygfpgan" \
    --num-identities 5346 --max-images 800000 || true

echo
echo "[done] next: python vggface_train/build_index.py, then sbatch vggface_train/train_vggface2.sh"
