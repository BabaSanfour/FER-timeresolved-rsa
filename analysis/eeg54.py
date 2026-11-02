"""54-stimulus EEG RDMs in the model stimulus order, and missing stimuli.

Order. The pooled EEG files (`<subject>_all_rdms.npy`) are ordered happy, fear,
neutral (`stimulus_order.json`, "all_54"); the model and control RDMs are ordered
fear, happy, neutral (models/build_emotion_rdms_54.py). `load` reorders the pooled
EEG to the model order by stimulus name. Within an emotion both use unmasked
(original), glasses, masked, actors in the same order, so the per-emotion files
need no reordering.

Missing stimuli. Eleven participants lack one video (neutral, glasses, index 3);
their RDMs hold NaN in that row and column. `pair_groups` groups participants by
their set of available stimulus pairs, so each participant is analysed on the
pairs they have (17 of 18 neutral stimuli, 53 of 54 pooled) instead of being
dropped. Rank-based statistics must re-rank both vectors on that subset.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FER_ROOT", "."))
EEG = ROOT / "eeg_rdms_complete"
MODEL_EMOTION_ORDER = ("fear", "happy", "neutral")
CONDITIONS = ("unmasked", "glasses", "masked")


def upper(x):
    iu = np.triu_indices(x.shape[-1], 1)
    return x[..., iu[0], iu[1]]


def model_order(eeg_dir=None):
    """Indices that put the pooled 54-stimulus EEG RDM into the model order."""
    names = json.loads(((eeg_dir or EEG) / "stimulus_order.json").read_text())["all_54"]
    return np.array([names.index(f"{e}_{c}_{i}") for e in MODEL_EMOTION_ORDER
                     for c in CONDITIONS for i in range(6)])


def subjects(eeg_dir=None):
    base = eeg_dir or EEG
    roi = {p.name.removesuffix("_all_rdms.npy") for p in (base / "perroi").glob("*_all_rdms.npy")}
    ws = {p.name.removesuffix("_all_rdms.npy") for p in (base / "wholescalp").glob("*_all_rdms.npy")}
    if roi != ws:
        raise RuntimeError(f"subject mismatch: ROI-only={roi - ws}, WS-only={ws - roi}")
    return sorted(roi)


def load(subs, emotion, eeg_dir=None):
    """-> (S, 7, W, n, n): six ROIs then whole scalp, in the model stimulus order."""
    base = eeg_dir or EEG
    roi = np.stack([np.load(base / "perroi" / f"{s}_{emotion}_rdms.npy") for s in subs])
    ws = np.stack([np.load(base / "wholescalp" / f"{s}_{emotion}_rdms.npy") for s in subs])
    eeg = np.concatenate([roi, ws[:, None]], axis=1)
    if emotion == "all":
        order = model_order(base)
        eeg = eeg[..., order, :][..., order]
    return eeg


def pair_groups(vectors):
    """vectors (S, ..., P) -> [(subject indices, bool pair mask)], most pairs first.

    A pair must be finite in every scope and window of a participant or missing
    in all of them; anything else is an error.
    """
    finite = np.isfinite(vectors).reshape(len(vectors), -1, vectors.shape[-1])
    keep = finite.all(axis=1)
    if not np.array_equal(keep, finite.any(axis=1)):
        raise RuntimeError("missing EEG values differ across scopes or windows")
    if (keep.sum(axis=1) < 3).any():
        raise RuntimeError("a participant has fewer than three stimulus pairs")
    patterns, inverse = np.unique(keep, axis=0, return_inverse=True)
    groups = [(np.flatnonzero(inverse.ravel() == g), patterns[g]) for g in range(len(patterns))]
    return sorted(groups, key=lambda g: -g[1].sum())
