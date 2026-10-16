#!/bin/bash
set -euo pipefail

ROOT=$(realpath "${FER_ROOT:-.}")
PY=${PY:-${VENV:-$HOME/envs/fer}/bin/python}
RDM_DIR="$ROOT/analysis/model_rdms_facenet_size_repeats"
GRID_DIR="$ROOT/analysis/rsa_grid_facenet_size_repeats"
mkdir -p "$RDM_DIR" "$GRID_DIR"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

run_one() {
    local repeat=$1 condition=$2
    local rep tag ckpt
    rep=$(printf '%02d' "$repeat")
    tag="facenet_vggface2_repeat${rep}_${condition}_16f_weights"
    ckpt="$ROOT/vggface_train/results/repeated_subsets/repeat_${rep}/${condition}/best_weights.pt"
    if [[ -s "$RDM_DIR/$tag.npy" ]]; then
        echo "[skip] $tag"
        return
    fi
    "$PY" "$ROOT/models/extract_model_rdms.py" \
        --arch facenet --checkpoint "$ckpt" --preproc half --resize 112 \
        --layers weights --exclude-head --out-name "$tag" \
        --h5-dir "$ROOT/stim_frames" --out-dir "$RDM_DIR"
}

# Two workers avoid oversubscribing memory and CPU on the interactive host.
running=0
for repeat in $(seq 1 10); do
    for condition in 300k 500k 800k_bootstrap; do
        run_one "$repeat" "$condition" &
        running=$((running + 1))
        if (( running % 2 == 0 )); then
            wait
        fi
    done
done
wait

tags=()
for repeat in $(seq -w 1 10); do
    for condition in 300k 500k 800k_bootstrap; do
        tags+=("facenet_vggface2_repeat${repeat}_${condition}_16f_weights")
    done
done
"$PY" "$ROOT/analysis/rsa_full_grid.py" --models "${tags[@]}" \
    --model-dir "$RDM_DIR" --out-dir "$GRID_DIR"
"$PY" "$ROOT/analysis/summarize_facenet_size_repeats.py"
echo "[done] CPU fallback RDM, RSA, and plot"
