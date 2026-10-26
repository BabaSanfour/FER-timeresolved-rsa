#!/bin/bash
# VGGFace2-HQ training archives 01-06 (5,346 identities, 726,502 images), the
# face-pretraining data. Needs `pip install "huggingface_hub[hf_xet]"`, `hf auth login`
# and the terms accepted at https://huggingface.co/datasets/ProgramComputer/VGGFace2-HQ.
# Run on a data-transfer node (login nodes cap CPU time). Safe to re-run.
set -uo pipefail
REPO=ProgramComputer/VGGFace2-HQ
DEST=${DEST:-${FER_ROOT:-.}/data/vggface_dataset}
declare -A SIZE=([01]=9994763942 [02]=9994086229 [03]=9994151915
                 [04]=9994056356 [05]=9994088404 [06]=9994042064)
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$DEST"

python - "$REPO" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
try:
    HfApi().auth_check(sys.argv[1], repo_type="dataset")
except GatedRepoError:
    sys.exit(f"FATAL: accept the terms at https://huggingface.co/datasets/{sys.argv[1]}")
except HfHubHTTPError as e:
    sys.exit(f"FATAL: {e} -- run `hf auth login` first")
PY

inc=(); for z in "${!SIZE[@]}"; do inc+=(--include "train/VGGFac$z.zip"); done
hf download "$REPO" --repo-type dataset "${inc[@]}" --local-dir "$DEST/zips" || exit 1
for z in "${!SIZE[@]}"; do
  have=$(stat -c%s "$DEST/zips/train/VGGFac$z.zip")
  [ "$have" = "${SIZE[$z]}" ] || { echo "FATAL: zip$z has $have bytes, re-run to resume"; exit 1; }
done

mkdir -p "$DEST/VGGface2_None_norm_512_true_bygfpgan"  # pre-created: parallel unzips race on it
for z in "${!SIZE[@]}"; do unzip -q -n "$DEST/zips/train/VGGFac$z.zip" -d "$DEST" & done
wait
python "$(dirname "$0")/../models/check_vggface2_layout.py" \
    "$DEST/VGGface2_None_norm_512_true_bygfpgan" --num-identities 5346
