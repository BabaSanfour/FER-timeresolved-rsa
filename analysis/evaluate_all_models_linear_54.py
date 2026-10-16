#!/usr/bin/env python3
"""In-sample 3-class linear probes and confusion matrices for all 54 stimuli."""
import csv
import json
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("FER_ROOT", "."))
RDM_DIR = ROOT / "analysis/model_rdms_54"
MANIFEST = ROOT / "stim_frames_54/stimulus_manifest.json"
OUT = ROOT / "analysis/model_performance_54/complete_54"
CLASSES = ["fear", "happy", "neutral"]
MODELS = [
    ("ResNet18 Random", "resnet18_random_16f_weights"),
    ("ResNet18 ImageNet", "resnet18_imagenet_16f_weights"),
    ("ResNet18 VGGFace2", "resnet18_vggface2_16f_weights"),
    ("ResNet18 AffectNet FT", "resnet18_vggface2_affectnet_3class_16f_weights"),
    ("ResNet50 Random", "resnet50_random_16f_weights"),
    ("ResNet50 ImageNet", "resnet50_imagenet_16f_weights"),
    ("ResNet50 VGGFace2", "resnet50_vggface2_16f_weights"),
    ("ResNet50 AffectNet FT", "resnet50_vggface2_affectnet_3class_16f_weights"),
    ("FaceNet Random", "facenet_random_16f_weights"),
    ("FaceNet ImageNet", "facenet_imagenet_16f_weights"),
    ("FaceNet VGGFace2", "facenet_vggface2_16f_weights"),
    ("FaceNet VGGFace2 300k", "facenet_vggface2_300k_16f_weights"),
    ("FaceNet VGGFace2 500k", "facenet_vggface2_500k_16f_weights"),
    ("FaceNet Complete VGGFace2", "vggface_complete_16f_pretrained"),
    ("FaceNet AffectNet FT", "facenet_vggface2_affectnet_3class_16f_weights"),
    ("VGG16 Random", "vgg16_random_16f_weights"),
    ("VGG16 VGGFace", "vgg16_vggface_16f_weights"),
    ("M3DFEL Random", "m3dfel_random_16f_weights"),
    ("M3DFEL Kinetics", "m3dfel_pretrained_16f_weights"),
    ("M3DFEL DFEW FT", "m3dfel_dfew_3class_fold3_16f_weights"),
]


def features_from_rdm(tag):
    rdm = np.load(RDM_DIR / f"{tag}.npy")
    # Final represented layer, averaged over the 16 time samples. Each stimulus
    # is represented by its distances to every stimulus.
    x = np.nanmean(rdm[-1], axis=0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def draw(ax, cm, title):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > .55 else "black")
    acc = np.trace(cm) / max(cm.sum(), 1)
    ax.set_title(f"{title}\nacc={acc:.1%}", fontsize=9)
    ax.set_xticks(range(3), ["F", "H", "N"])
    ax.set_yticks(range(3), ["F", "H", "N"])


def main():
    manifest = json.loads(MANIFEST.read_text())
    y = np.array([CLASSES.index(row["emotion"]) for row in manifest])
    conditions = np.array([row["condition"] for row in manifest])
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = {}

    for name, tag in MODELS:
        x = features_from_rdm(tag)
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42),
        )
        probe.fit(x, y)
        pred = probe.predict(x)
        predictions[tag] = pred
        for condition in ["all", "original", "glasses", "masked"]:
            keep = np.ones(len(y), dtype=bool) if condition == "all" else conditions == condition
            cm = confusion_matrix(y[keep], pred[keep], labels=range(3))
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            results.append({
                "model": tag, "display_name": name, "condition": condition,
                "n": int(keep.sum()), "accuracy": float(np.trace(cm) / cm.sum()),
                "balanced_accuracy": float(np.diag(norm).mean()),
                "cm": json.dumps(cm.tolist()),
            })

    with (OUT / "performance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    fig, axes = plt.subplots(5, 4, figsize=(13, 15), constrained_layout=True)
    for ax, (name, tag) in zip(axes.flat, MODELS):
        draw(ax, confusion_matrix(y, predictions[tag], labels=range(3)), name)
    fig.supxlabel("Predicted: F = fear, H = happy, N = neutral")
    fig.supylabel("True emotion")
    fig.suptitle("Three-class linear probes — all 54 stimuli", fontsize=17, fontweight="bold")
    fig.savefig(OUT / "all_models_confusion_matrix_54.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Separate 18-stimulus figure. Fit the linear probes only on the original
    # stimuli; use the manually updated native/fine-tuned matrices when present.
    original_keep = conditions == "original"
    y18 = y[original_keep]
    native_cm = {}
    native_path = ROOT / "analysis/model_performance_54/unmasked/performance.csv"
    if native_path.exists():
        for line in native_path.read_text().splitlines()[1:]:
            model, condition, tail = line.split(",", 2)
            match = re.search(r"(\[\[.*\]\])", tail)
            if condition == "original" and match:
                native_cm[model] = np.asarray(json.loads(match.group(1)), dtype=int)

    fig, axes = plt.subplots(5, 4, figsize=(13, 15), constrained_layout=True)
    rows18 = []
    for ax, (name, tag) in zip(axes.flat, MODELS):
        if tag in native_cm:
            cm = native_cm[tag]
            method = "native"
        else:
            x18 = features_from_rdm(tag)[original_keep]
            probe18 = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, random_state=42),
            )
            pred18 = probe18.fit(x18, y18).predict(x18)
            cm = confusion_matrix(y18, pred18, labels=range(3))
            method = "linear probe"
        draw(ax, cm, name)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        rows18.append({
            "model": tag, "display_name": name, "condition": "original",
            "method": method, "n": 18,
            "accuracy": float(np.trace(cm) / cm.sum()),
            "balanced_accuracy": float(np.diag(norm).mean()),
            "cm": json.dumps(cm.tolist()),
        })
    fig.supxlabel("Predicted: F = fear, H = happy, N = neutral")
    fig.supylabel("True emotion")
    fig.suptitle("Emotion classification — 18 stimuli",
                 fontsize=17, fontweight="bold")
    fig.savefig(OUT / "all_models_confusion_matrix_18_unmasked.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)
    with (OUT / "performance_18_unmasked.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows18[0].keys())
        writer.writeheader()
        writer.writerows(rows18)

    # One figure per architecture, showing all/original/glasses/masked.
    groups = {
        "resnet18": MODELS[0:4], "resnet50": MODELS[4:8],
        "facenet": MODELS[8:15], "vgg16": MODELS[15:17],
        "m3dfel": MODELS[17:20],
    }
    for group, models in groups.items():
        fig, axes = plt.subplots(len(models), 4, figsize=(12, 2.7 * len(models)),
                                 squeeze=False, constrained_layout=True)
        for row_ax, (name, tag) in zip(axes, models):
            pred = predictions[tag]
            for ax, condition in zip(row_ax, ["all", "original", "glasses", "masked"]):
                keep = np.ones(len(y), dtype=bool) if condition == "all" else conditions == condition
                cm = confusion_matrix(y[keep], pred[keep], labels=range(3))
                draw(ax, cm, f"{name} | {condition}")
        fig.supxlabel("Predicted: F = fear, H = happy, N = neutral")
        fig.supylabel("True emotion")
        fig.savefig(OUT / f"{group}_confusion_by_condition.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"[ok] {len(MODELS)} models, 4 conditions -> {OUT}")


if __name__ == "__main__":
    main()
