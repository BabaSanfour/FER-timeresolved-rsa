#!/usr/bin/env python3
import argparse
import json

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve

import rsa_core
import variants

N_FRAMES = 16
GREY_SIZE = 128
DESCRIPTIONS = {
    "identity": "0 same actor, 1 otherwise (static)",
    "occlusion": "0 same occlusion (original, glasses, masked), 1 otherwise (static)",
    "pixel": "correlation distance of the RGB pixels of each frame (time-resolved)",
    "optical_flow": "Lucas-Kanade flow histograms between consecutive frames, grey 128 px (static)",
    "motion_energy": "log mean energy of 48 spatio-temporal Gabor filters, grey 128 px (static)",
}


def corr_distance_rdm(features):
    """(n, d) features -> (n, n) correlation-distance RDM."""
    x = features.astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    rdm = 1.0 - (x @ x.T) / np.maximum(norms * norms.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def load_frames(frames_dir):
    """-> ((n, 16, 224, 224, 3) uint8 frames, stimulus names)."""
    names = [row["stimulus"] for row in json.loads((frames_dir / "stimulus_manifest.json").read_text())]
    frames = []
    for k in range(1, N_FRAMES + 1):
        with h5py.File(frames_dir / f"Slide{k}.h5", "r") as f:
            frames.append(np.asarray(f["images"]))
    return np.stack(frames, axis=1), names


def grey(frames):
    """(n, T, H, W, 3) uint8 -> (n, T, 128, 128) ITU-R 601 luminance in [0, 1]."""
    lum = frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
    out = np.empty(frames.shape[:2] + (GREY_SIZE, GREY_SIZE), np.float32)
    for i in range(frames.shape[0]):
        for t in range(frames.shape[1]):
            im = Image.fromarray(lum[i, t].astype(np.float32), mode="F")
            out[i, t] = np.asarray(im.resize((GREY_SIZE, GREY_SIZE), Image.BILINEAR))
    return out / 255.0


# --- optical flow ---------------------------------------------------------------------
MAG_BINS, DIR_BINS, WINDOW_PX, SIGMA_PRE = 16, 8, 7, 1.0


def lucas_kanade_flow(I1, I2):
    """Two grey frames -> dense Lucas-Kanade flow (u, v)."""
    I1s, I2s = gaussian_filter(I1, SIGMA_PRE), gaussian_filter(I2, SIGMA_PRE)
    Ix = 0.5 * (np.roll(I1s, -1, axis=1) - np.roll(I1s, 1, axis=1))
    Iy = 0.5 * (np.roll(I1s, -1, axis=0) - np.roll(I1s, 1, axis=0))
    It = I2s - I1s
    a, b, d = (uniform_filter(Ix * Ix, WINDOW_PX), uniform_filter(Ix * Iy, WINDOW_PX),
               uniform_filter(Iy * Iy, WINDOW_PX))
    e, f = uniform_filter(Ix * It, WINDOW_PX), uniform_filter(Iy * It, WINDOW_PX)
    det = a * d - b * b
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    return (-d * e + b * f) / det, (b * e - a * f) / det


def flow_histograms(u, v):
    """Flow -> magnitude and direction histograms (24 values)."""
    mag, ang = np.sqrt(u ** 2 + v ** 2), np.arctan2(v, u)
    valid = mag > np.percentile(mag, 5)                  # near-zero flow is uninformative
    mag_v, ang_v = mag[valid], ang[valid]
    if len(mag_v) == 0:
        return np.zeros(MAG_BINS + DIR_BINS, dtype=np.float64)
    mag_max = np.percentile(mag_v, 99) + 1e-6
    h_mag, _ = np.histogram(np.clip(mag_v, 0, mag_max), bins=MAG_BINS, range=(0, mag_max), density=True)
    h_dir, _ = np.histogram(ang_v, bins=DIR_BINS, range=(-np.pi, np.pi), density=True)
    return np.concatenate([h_mag, h_dir]).astype(np.float64)


def flow_features(video):
    """(T, H, W) grey clip -> ((T - 1) * 24,) features."""
    return np.concatenate([flow_histograms(*lucas_kanade_flow(video[t], video[t + 1]))
                           for t in range(video.shape[0] - 1)])


# --- motion energy (Adelson & Bergen, 1985) -------------------------------------------
def spatio_temporal_gabor(sf, tf, theta, ksz_s=11, ksz_t=5, sigma_s=2.5, sigma_t=1.2):
    """Zero-mean complex 3-D Gabor (T, Y, X); sf cycles/pixel, tf cycles/frame (sign = direction)."""
    t, y, x = (np.arange(ksz_t) - ksz_t // 2, np.arange(ksz_s) - ksz_s // 2, np.arange(ksz_s) - ksz_s // 2)
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    xt = xx * np.cos(theta) + yy * np.sin(theta)
    env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_s ** 2) - tt ** 2 / (2 * sigma_t ** 2))
    g = env * np.exp(2j * np.pi * (sf * xt + tf * tt))
    return (g - g.mean()).astype(np.complex128)


def filter_bank():
    return [spatio_temporal_gabor(sf, tf, th) for sf in (0.05, 0.10, 0.20)
            for tf in (-0.30, -0.15, 0.15, 0.30) for th in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)]


def motion_energy(video, bank):
    """(T, H, W) grey clip -> mean quadrature energy per filter."""
    return np.array([(fftconvolve(video, k.real, mode="same") ** 2
                      + fftconvolve(video, k.imag, mode="same") ** 2).mean() for k in bank])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    args = ap.parse_args()
    frames_dir = rsa_core.ROOT / f"data/stim_frames/set{args.stimuli}"
    rdm_dir = rsa_core.RDM_DIRS[args.stimuli]
    frames, names = load_frames(frames_dir)
    if names != rsa_core.MODEL_STIMULI[args.stimuli]:
        raise SystemExit(f"ABORT: {frames_dir} stimulus order differs from the model order")
    n = len(names)
    print(f"[frames] {frames.shape} from {frames_dir}")

    rdms = {}
    actor = np.array([nm.rsplit("_Actor", 1)[1] for nm in names])
    rdms["identity"] = (actor[:, None] != actor[None, :]).astype(np.float32)
    if args.stimuli == "54":
        occlusion = np.array([nm.split("_")[1] for nm in names])
        rdms["occlusion"] = (occlusion[:, None] != occlusion[None, :]).astype(np.float32)
    rdms["pixel"] = np.stack([corr_distance_rdm(frames[:, k].reshape(n, -1)) for k in range(N_FRAMES)])
    g = grey(frames)
    rdms["optical_flow"] = corr_distance_rdm(np.stack([flow_features(v) for v in g]))
    print("[ok] static controls, pixel, optical flow", flush=True)
    bank = filter_bank()
    rdms["motion_energy"] = corr_distance_rdm(np.log1p(np.stack([motion_energy(v, bank) for v in g])))

    for _, name in variants.controls(args.stimuli):
        rdm = rdms[name]
        rdm = rdm if rdm.ndim == 3 else np.broadcast_to(rdm, (N_FRAMES, n, n))
        if not np.isfinite(rdm).all():
            raise SystemExit(f"ABORT {name}: non-finite distances")
        tag = f"control_{name}"
        out = rdm_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{tag}.npy", np.ascontiguousarray(rdm[None], dtype=np.float32))
        (out / f"{tag}.layers.json").write_text(json.dumps({
            "layer_order": [name], "shape": [1, N_FRAMES, n, n], "control": DESCRIPTIONS[name],
            "time_resolved": name == "pixel", "stimuli": names,
            "frames": str(frames_dir)}, indent=2))
        print(f"[ok] {tag}: {rdm.shape}")


if __name__ == "__main__":
    main()
