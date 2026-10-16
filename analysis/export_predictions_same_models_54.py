#!/usr/bin/env python3
"""Export per-stimulus predictions for the 18 models in performance_18_unmasked.csv."""
import csv
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("FER_ROOT", "."))
SOURCE = ROOT / "analysis/model_performance_54/complete_54/performance_18_unmasked.csv"
RDM_DIR = ROOT / "analysis/model_rdms_54"
MANIFEST = ROOT / "stim_frames_54/stimulus_manifest.json"
OUT_DIR = ROOT / "analysis/model_performance_54/complete_54"
SPECIAL_VGG = "vgg16_vggface_16f_weights_affectnet"
LABELS = ["fear", "happy", "neutral"]


def source_models():
    rows = []
    for line in SOURCE.read_text().splitlines()[1:]:
        fields = line.split(",", 4)
        if len(fields) >= 4:
            rows.append((fields[0], fields[1], fields[3]))
    return rows


def rdm_predictions(tag, y):
    rdm = np.load(RDM_DIR / f"{tag}.npy")
    x = np.nan_to_num(np.nanmean(rdm[-1], axis=0))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, random_state=42))
    return clf.fit(x, y).predict(x), "linear probe"


def vgg_native_predictions():
    import torch
    sys.path.insert(0, str(ROOT / "models"))
    from extract_model_rdms import ARCHS, load_checkpoint, remap_state, _pp_vggface_rgb

    checkpoint = ROOT / "affectnet_3class_train/results/17810319/vgg16_vggface/best_weights.pt"
    state = remap_state("vgg16", load_checkpoint(checkpoint), "auto")
    model = ARCHS["vgg16"](3)
    model.load_state_dict(state, strict=True)
    model.eval()
    frames = np.load(ROOT / "stim_frames_54/all_video_frames_16.npy", mmap_mode="r")
    predictions = []
    with torch.inference_mode():
        for stimulus_frames in frames:
            logits = []
            for start in range(0, len(stimulus_frames), 4):
                x = _pp_vggface_rgb(np.asarray(stimulus_frames[start:start + 4]))
                logits.append(model(x).cpu())
            predictions.append(int(torch.cat(logits).mean(0).argmax()))
    return np.asarray(predictions), "native"


def main():
    manifest = json.loads(MANIFEST.read_text())
    y = np.asarray([LABELS.index(row["emotion"]) for row in manifest])
    detailed, summary = [], []
    for tag, display, requested_method in source_models():
        if tag == SPECIAL_VGG:
            pred, method = vgg_native_predictions()
        else:
            pred, method = rdm_predictions(tag, y)
        cm = np.zeros((3, 3), dtype=int)
        for truth, guess in zip(y, pred):
            cm[truth, guess] += 1
        for row, guess in zip(manifest, pred):
            detailed.append({
                "model": tag, "display_name": display, "method": method,
                "stimulus_index": row["index"], "stimulus": row["stimulus"],
                "emotion": row["emotion"], "condition": row["condition"],
                "actor": row["actor"], "predicted_emotion": LABELS[int(guess)],
                "correct": int(int(guess) == LABELS.index(row["emotion"])),
            })
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        summary.append({
            "model": tag, "display_name": display, "method": method, "n": 54,
            "accuracy": float(np.trace(cm) / cm.sum()),
            "balanced_accuracy": float(np.diag(norm).mean()),
            "cm": json.dumps(cm.tolist()),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "predictions_54_same_models.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed[0].keys())
        writer.writeheader()
        writer.writerows(detailed)
    with (OUT_DIR / "performance_54_same_models.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"[ok] {len(summary)} models, {len(detailed)} predictions")


if __name__ == "__main__":
    main()
