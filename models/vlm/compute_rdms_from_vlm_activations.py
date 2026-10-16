#!/usr/bin/env python3
"""Turn Qwen video-mode activations into canonical model RDMs.

Input : <run_dir>/activations/{Label}_{ActorXX}.npy, each (n_timepoints, n_layers, hidden):
        decoder hidden states (embedding output + language decoder blocks), and
        <run_dir>/activations/{Label}_{ActorXX}_vision.npy, (n_timepoints, n_blocks, vision_hidden):
        vision-encoder block outputs
Output: <out_dir>/<name>.npy  (n_layers, n_timepoints, 18, 18) float32; layers are ordered
        vision blocks 1..27, embedding output, decoder blocks 1..36 (64 for Qwen3-VL-8B)
        <out_dir>/<name>.layers.json

Stimuli are stacked in the canonical EMOTION-MAJOR order (fear 0-5, happy 6-11,
neutral 12-17; actors 01, 02, 04, 07, 08, 11) so the result drops straight into
analysis/rsa_full_grid.py alongside the CNN RDMs. The video files spell happy as
"Hapiness", which is mapped here.

Distance is correlation distance over the hidden dimension, computed separately
per (layer, timepoint) across the 18 stimuli - the same estimator as
models/extract_model_rdms.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ACTORS = [1, 2, 4, 7, 8, 11]
EMO_FILE = {"fear": "Fear", "happy": "Hapiness", "neutral": "Neutral"}
STIMS = [f"{e}_Actor{a:02d}" for e in EMO_FILE for a in ACTORS]


def corr_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def layer_rdms(acts):
    """(n_stim, T, L, H) activations -> (L, T, n_stim, n_stim) correlation-distance RDMs."""
    n_stim, T, L, _ = acts.shape
    rdms = np.zeros((L, T, n_stim, n_stim), dtype=np.float32)
    for l in range(L):
        for t in range(T):
            rdms[l, t] = corr_distance_rdm(acts[:, t, l, :].astype(np.float64))
    return rdms


def load_stack(act_dir, suffix=""):
    acts = []
    for emo, fpre in EMO_FILE.items():
        for a in ACTORS:
            p = act_dir / f"{fpre}_Actor{a:02d}{suffix}.npy"
            if not p.exists():
                raise SystemExit(f"ABORT: missing {p}")
            acts.append(np.load(p))                       # (T, L, H)
    shapes = {x.shape for x in acts}
    if len(shapes) != 1:
        raise SystemExit(f"ABORT: activation shapes differ across stimuli: {shapes}")
    return np.stack(acts, 0)                              # (18, T, L, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="output_dir used by the extractor")
    ap.add_argument("--out-dir",
                    default="./analysis/model_rdms_canonical")
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()

    act_dir = Path(args.run_dir) / "activations"
    A = load_stack(act_dir)
    V = load_stack(act_dir, "_vision")
    n_stim, T, L_dec, H = A.shape
    L_vis = V.shape[2]
    print(f"[in] {n_stim} stimuli x {T} timepoints: {L_vis} vision blocks x {V.shape[3]} + "
          f"{L_dec} decoder states x {H}")
    rdms = np.concatenate([layer_rdms(V), layer_rdms(A)], axis=0)
    L = rdms.shape[0]
    names = ([f"vision_block_{i}" for i in range(1, L_vis + 1)] + ["embedding"] +
             [f"decoder_block_{i}" for i in range(1, L_dec)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.out_name}.npy", rdms)

    summary = {}
    sp = Path(args.run_dir) / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
    (out_dir / f"{args.out_name}.layers.json").write_text(json.dumps({
        "layer_order": names,
        "shape": list(rdms.shape), "arch": "Qwen3-VL (video mode)",
        "n_timepoints": T, "hidden_size": H, "stim_order": STIMS,
        "source_run": str(args.run_dir), **summary}, indent=2))

    off = ~np.eye(n_stim, dtype=bool)
    v = rdms[:, :, off]
    degen = int((rdms.reshape(L * T, -1).std(1) < 1e-9).sum())
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} "
          f"range=[{v.min():.3f},{v.max():.3f}] degenerate_layer_frames={degen}/{L * T}")
    print(f"[ok] -> {out_dir / (args.out_name + '.npy')}")


if __name__ == "__main__":
    main()
