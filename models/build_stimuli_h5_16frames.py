#!/usr/bin/env python3
"""Sample 16 frames per stimulus video, the same way for every model and both stimulus sets.

Frames: idx = linspace(0, n - 1, 16).astype(int) over the container's frame count,
read by seeking, resized to 224x224 (INTER_AREA).

Order: 18 videos -> emotion (fear, happy, neutral), actor;
       54 videos -> emotion, occlusion (original, glasses, masked), actor.

Writes into --out-dir:
    Slide{1..16}.h5          "images" = (n_stim, 224, 224, 3) uint8 RGB, one file per frame
    stimulus_manifest.json   per stimulus: index, stimulus, emotion, condition, actor,
                             file, total_frames, sampled_frame_indices
"""
import argparse
import json
import re
from pathlib import Path

import h5py  # before cv2: the cluster OpenCV module ships an older HDF5 library
import cv2
import numpy as np

EMOTIONS = ("fear", "happy", "neutral")
CONDITIONS = ("original", "glasses", "masked")


def describe(path):
    s = path.stem.lower()
    emotion = "fear" if s.startswith("fear") else "happy" if s.startswith("hap") else \
        "neutral" if s.startswith("neutral") else None   # "Hapiness" and "Happiness"
    if emotion is None:
        raise ValueError(f"unrecognised emotion in {path.name}")
    condition = "glasses" if "glasses" in s else "masked" if "masked" in s else "original"
    actor = int(re.search(r"actor0*(\d+)", s).group(1))
    return emotion, condition, actor


def sample_frames(path, n_frames, size):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        raise RuntimeError(f"{path.name}: {total} frames, need {n_frames}")
    idx = np.linspace(0, total - 1, n_frames, dtype=int)
    out = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {int(i)} of {path.name}")
        out[k] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size),
                            interpolation=cv2.INTER_AREA)
    cap.release()
    return out, idx.tolist(), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    vids = sorted(a.video_dir.glob("*.mp4"))
    if len(vids) not in (18, 54):
        raise ValueError(f"expected 18 or 54 videos in {a.video_dir}, found {len(vids)}")
    info = {v: describe(v) for v in vids}
    vids.sort(key=lambda v: (EMOTIONS.index(info[v][0]), CONDITIONS.index(info[v][1]), info[v][2]))

    clips, manifest = [], []
    for i, v in enumerate(vids):
        emotion, condition, actor = info[v]
        frames, idx, total = sample_frames(v, a.n_frames, a.size)
        clips.append(frames)
        name = f"{emotion}_Actor{actor:02d}" if len(vids) == 18 else f"{emotion}_{condition}_Actor{actor:02d}"
        manifest.append({"index": i, "stimulus": name, "emotion": emotion, "condition": condition,
                         "actor": f"{actor:02d}", "file": v.name, "total_frames": total,
                         "sampled_frame_indices": idx})
        print(f"[{i:2d}] {name:<28s} {v.name}: {total} frames -> {idx}", flush=True)

    clips = np.stack(clips)                                   # (n_stim, 16, size, size, 3)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(a.n_frames):
        with h5py.File(a.out_dir / f"Slide{t + 1}.h5", "w") as f:
            f.create_dataset("images", data=clips[:, t])
    (a.out_dir / "stimulus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] {clips.shape} -> {a.out_dir}")


if __name__ == "__main__":
    main()
