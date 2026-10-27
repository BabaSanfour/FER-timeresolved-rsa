#!/usr/bin/env python3
"""RDMs from Qwen3-VL video-mode activations.

Stacks the stimuli in stimulus-manifest order and writes <out-name>.npy (64, 16, n_stim, n_stim):
vision blocks 1..27, embedding output, decoder blocks 1..36; correlation distance per layer and
time point, as in extract_model_rdms.py.
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


def load_stack(act_dir, names, suffix=""):
    acts = []
    for name in names:
        p = act_dir / f"{name}{suffix}.npy"
        if not p.exists():
            raise SystemExit(f"ABORT: missing {p}")
        acts.append(np.load(p))                           # (T, L, H)
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
    ap.add_argument("--manifest", default=None,
                    help="stimulus_manifest.json; activations are named by its 'stimulus' field")
    args = ap.parse_args()

    act_dir = Path(args.run_dir) / "activations"
    if args.manifest:
        stims = [r["stimulus"] for r in json.loads(Path(args.manifest).read_text())]
        files = stims
    else:
        stims = STIMS
        files = [f"{fpre}_Actor{a:02d}" for fpre in EMO_FILE.values() for a in ACTORS]
    A = load_stack(act_dir, files)
    V = load_stack(act_dir, files, "_vision")
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
        "n_timepoints": T, "hidden_size": H, "stim_order": stims,
        "stimulus_manifest": args.manifest,
        "source_run": str(args.run_dir), **summary}, indent=2))

    off = ~np.eye(n_stim, dtype=bool)
    v = rdms[:, :, off]
    degen = int((rdms.reshape(L * T, -1).std(1) < 1e-9).sum())
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} "
          f"range=[{v.min():.3f},{v.max():.3f}] degenerate_layer_frames={degen}/{L * T}")
    print(f"[ok] -> {out_dir / (args.out_name + '.npy')}")


if __name__ == "__main__":
    main()
