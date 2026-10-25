#!/bin/bash
# Fetch VGGFace2-HQ train zips 01-06 from the HuggingFace mirror and extract.
#
# Why only 01-06: the zips are ordered by identity (zip06 starts at n004754).
# The paper's VGGFace2 models were trained on these six zips (5,346 identities,
# 715,810 training images after vggface_train/build_index.py). Set ZIPS="01 .. 10" to get all.
# vggface_train/train_vggface2.sh reads the zips from $FER_ROOT/vggface_dataset/.
#
# Why hf download: the repo is stored with Xet, and `hf download` with hf_xet
# fetches chunks in parallel. On the Trillium data mover a 17 GB repo took ~7 s;
# plain curl from the same mirror sustains ~1.8 MB/s per stream (~1.5-2 h here).
# Why not Kaggle: kagglehub throttled to ~276 kB/s single-stream (~50h for the
# 48.3 GB repackage).
#
# ONE-TIME SETUP:
#   pip install "huggingface_hub[hf_xet]"   # provides the `hf` CLI
#   hf auth login                           # a read token
#   accept the terms at https://huggingface.co/datasets/ProgramComputer/VGGFace2-HQ
#   (gated: access is granted automatically once accepted on the same account)
#
# RUN ON A DATA TRANSFER NODE, DETACHED so it survives your session ending.
# Login nodes cap CPU time per process (1 h on Trillium), which kills a long
# download or unzip; on Trillium use the data mover (ssh tri-dm1):
#   nohup bash data/get_vggface2_hq.sh > vggface2_dl.log 2>&1 &
#   tail -f vggface2_dl.log
#
# Safe to re-run: hf download skips complete files, unzip -n never overwrites.
set -uo pipefail

REPO="ProgramComputer/VGGFace2-HQ"
DEST=${DEST:-${FER_ROOT:-.}/vggface_dataset}
ZIPDIR="$DEST"            # train_vggface2.sh reads $DEST/VGGFac0*.zip
TRAIN="$DEST"             # unpacks to $DEST/VGGface2_None_norm_512_true_bygfpgan/
ZIPS=${ZIPS:-"01 02 03 04 05 06"}
mkdir -p "$ZIPDIR" "$TRAIN"
export HF_XET_HIGH_PERFORMANCE=1

# exact sizes from the HF API -- a truncated download is otherwise silent
declare -A SIZE=(
  [01]=9994763942 [02]=9994086229 [03]=9994151915
  [04]=9994056356 [05]=9994088404 [06]=9994042064
  [07]=9994142344 [08]=9994101971 [09]=9994089009 [10]=5763723321
)

command -v hf >/dev/null || { echo "FATAL: no hf CLI; pip install 'huggingface_hub[hf_xet]'"; exit 1; }
python -c "import hf_xet" 2>/dev/null || echo "WARNING: hf_xet not installed; download falls back to plain HTTP"

# fail early with a clear message instead of a 403 halfway through
python - "$REPO" <<'EOF' || exit 1
import sys
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
try:
    HfApi().auth_check(sys.argv[1], repo_type="dataset")
except GatedRepoError:
    sys.exit(f"FATAL: no access yet -- accept the terms at https://huggingface.co/datasets/{sys.argv[1]}")
except HfHubHTTPError as e:
    sys.exit(f"FATAL: {e} -- run `hf auth login` first")
EOF

echo "[$(date +%H:%M:%S)] fetching zips: $ZIPS -> $ZIPDIR"
INCLUDE=()
for i in $ZIPS; do INCLUDE+=(--include "train/VGGFac$i.zip"); done
# hf mirrors the repo layout (train/...) under --local-dir; keep that in a
# side folder and move the zips up to where train_vggface2.sh expects them
hf download "$REPO" --repo-type dataset "${INCLUDE[@]}" --local-dir "$ZIPDIR/.hf_download" || exit 1
for i in $ZIPS; do
  src="$ZIPDIR/.hf_download/train/VGGFac$i.zip"
  [ -f "$src" ] && mv -f "$src" "$ZIPDIR/VGGFac$i.zip"
done
echo "[$(date +%H:%M:%S)] downloads finished"

for i in $ZIPS; do
  have=$(stat -c%s "$ZIPDIR/VGGFac$i.zip" 2>/dev/null || echo 0)
  if [ "$have" = "${SIZE[$i]}" ]; then
    echo "  zip$i OK ($have bytes)"
  else
    echo "  zip$i SIZE MISMATCH have=$have want=${SIZE[$i]} -- re-run to resume"
  fi
done

echo "[$(date +%H:%M:%S)] extracting into $TRAIN"
# Pre-creating the shared top-level directory prevents the racing-mkdir
# failure that parallel unzips otherwise hit (as in train_vggface2.sh).
mkdir -p "$TRAIN/VGGface2_None_norm_512_true_bygfpgan"
UNZIP_PIDS=()
for i in $ZIPS; do
  f="$ZIPDIR/VGGFac$i.zip"
  have=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$have" != "${SIZE[$i]}" ]; then
    echo "  skipping zip$i -- incomplete"
    continue
  fi
  # -n: never overwrite, so re-runs are cheap and idempotent
  ( unzip -q -n "$f" -d "$TRAIN" && echo "  zip$i extracted" ) &
  UNZIP_PIDS+=($!)
done
[ ${#UNZIP_PIDS[@]} -gt 0 ] && wait "${UNZIP_PIDS[@]}"

echo "[$(date +%H:%M:%S)] identity folders now present: $(find "$TRAIN/VGGface2_None_norm_512_true_bygfpgan" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "[$(date +%H:%M:%S)] verifying layout"
python "$(dirname "$0")/../models/check_vggface2_layout.py" "$TRAIN/VGGface2_None_norm_512_true_bygfpgan" \
    --num-identities 5346 --max-images 800000 || true

echo
echo "[done] next: python vggface_train/build_index.py, then sbatch vggface_train/train_vggface2.sh"
