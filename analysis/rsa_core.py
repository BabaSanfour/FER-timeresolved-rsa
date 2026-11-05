import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("FER_ROOT", "."))
EEG_DIR = ROOT / "eeg_rdms"                # eeg/build_eeg_rdms.py: <subject>.npy, meta.json
MODEL_EMOTION_ORDER = ("fear", "happy", "neutral")
ACTORS = ("01", "02", "04", "07", "08", "11")
# stimulus names in the model RDMs' .layers.json (build_stimuli_h5_16frames.py)
MODEL_STIMULI = {
    "18": [f"{e}_Actor{a}" for e in MODEL_EMOTION_ORDER for a in ACTORS],
    "54": [f"{e}_{c}_Actor{a}" for e in MODEL_EMOTION_ORDER
           for c in ("original", "glasses", "masked") for a in ACTORS],
}
# rows of the 54-video EEG RDM that form each stimulus set
EEG_ROWS = {"54": np.arange(54),
            "18": np.array([MODEL_STIMULI["54"].index(n.replace("_Actor", "_original_Actor")) for n in MODEL_STIMULI["18"]])}
EMOTIONS = {s: {"all": np.arange(len(names)),
                **{e: np.array([i for i, n in enumerate(names) if n.startswith(e + "_")])
                   for e in MODEL_EMOTION_ORDER}}
            for s, names in MODEL_STIMULI.items()}
RDM_DIRS = {s: ROOT / f"rdms/set{s}" for s in ("18", "54")}       # <name>/<name>.npy
GRID_DIRS = {"18": ROOT / "analysis/rsa_grid", "54": ROOT / "analysis/rsa_grid_54"}
PARTIAL_DIRS = {"18": ROOT / "analysis/partial_rsa", "54": ROOT / "analysis/partial_rsa_54"}   # <control set>/
STATS_DIRS = {"18": ROOT / "analysis/paper_stats", "54": ROOT / "analysis/paper_stats_54"}
N_ROI = 11                   # scopes 0-10 are ROIs, 11 is the whole scalp


def eeg_meta():
    return json.loads((EEG_DIR / "meta.json").read_text())


def scopes():
    return eeg_meta()["scopes"]


def window_bounds():
    """(13, 2) window bounds in ms from stimulus onset."""
    return np.asarray(eeg_meta()["window_bounds_ms"])


def load_eeg(stimuli):
    """-> (eeg (S, 12, W, n, n), subject ids, scope names)."""
    meta = eeg_meta()
    if meta["stimuli"] != MODEL_STIMULI["54"] or len(meta["scopes"]) != N_ROI + 1:
        raise SystemExit("ABORT: eeg_rdms/meta.json does not match rsa_core (stimuli or scopes)")
    rows = EEG_ROWS[stimuli]
    eeg = np.stack([np.load(EEG_DIR / f"{s}.npy") for s in meta["subjects"]])[..., rows, :][..., rows]
    if not np.isfinite(eeg).all():
        raise SystemExit(f"ABORT: missing EEG values in the {stimuli}-stimulus RDMs")
    return eeg, meta["subjects"], meta["scopes"]


def upper(r):
    """(..., k, k) -> (..., k*(k-1)/2) upper triangle."""
    iu = np.triu_indices(r.shape[-1], 1)
    return r[..., iu[0], iu[1]]


def ranked_centered(x):
    """Rank and centre along the last axis."""
    ranks = rankdata(x, axis=-1)
    return ranks - ranks.mean(axis=-1, keepdims=True)


def unit(v):
    """Unit-normalise along the last axis -> (vectors, where the norm vanishes)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.clip(n, 1e-12, None)), n[..., 0] < 1e-12


def rank_normalise(v):
    """Rank, centre and unit-normalise along the last axis (a dot product is Spearman's rho)."""
    return unit(ranked_centered(v))


def grid_path(stimuli, tag, partial=None):
    folder = GRID_DIRS[stimuli] if partial is None else PARTIAL_DIRS[stimuli] / partial
    return folder / f"{tag}_rsa_grid.npz"


def variant_grid(stimuli, tags, partial=None):
    """Mean of the RDMs' participant grids -> (rsa, npz of the first)."""
    first = np.load(grid_path(stimuli, tags[0], partial), allow_pickle=True)
    rsa = np.mean([np.load(grid_path(stimuli, t, partial))["rsa"] for t in tags], axis=0)
    return rsa.astype(np.float32), first


def slug(family, variant):
    return f"{family}_{variant}".lower().replace(" ", "_").replace("-", "")


def load_model_rdm(stimuli, tag, rdm_dir=None):
    """(L, frames, n, n) model RDM; checks its stimulus order first."""
    folder = Path(rdm_dir or RDM_DIRS[stimuli]) / tag
    meta = json.loads((folder / f"{tag}.layers.json").read_text())
    stimuli_order = meta.get("stimuli") or meta.get("stim_order")      # stim_order: M3DFEL and Qwen3-VL extractors
    if stimuli_order != MODEL_STIMULI[stimuli]:
        raise SystemExit(f"ABORT {tag}: stimulus order differs from the EEG's")
    return np.load(folder / f"{tag}.npy")


def pair_permutation(perm):
    """Stimulus permutation -> index into the upper-triangle pairs of the permuted RDM."""
    n = len(perm)
    iu = np.triu_indices(n, 1)
    pos = np.full((n, n), -1)
    pos[iu] = np.arange(len(iu[0]))
    pos = np.maximum(pos, pos.T)
    return pos[perm[iu[0]], perm[iu[1]]]
