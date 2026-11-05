"""Shared pieces of the RSA analyses: EEG loading for both stimulus sets, the
rank-correlation helpers, the RSA grid paths and the stimulus permutation of the nulls.

EEG, model and control RDMs are returned as (..., n, n) over the analysed
stimuli `STIMULI[stimuli]`, in the model order: emotion (fear, happy, neutral),
then for 54 stimuli occlusion (original, glasses, masked), then actor (01 02 04
07 08 11). `EMOTIONS[stimuli][emotion]` indexes all of them.

EEG RDMs come from eeg/build_eeg_rdms.py: one file per participant over the 54
videos (12 scopes: 11 ROIs + whole scalp; 13 windows from stimulus onset). The 18
stimuli are the 18 unoccluded videos of the same recordings, so both sets use the
same 25 participants.
"""
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
STIMULI = MODEL_STIMULI
# rows of the 54-video EEG RDM that form each stimulus set
EEG_ROWS = {"54": np.arange(54),
            "18": np.array([MODEL_STIMULI["54"].index(n.replace("_Actor", "_original_Actor")) for n in MODEL_STIMULI["18"]])}
EMOTIONS = {s: {"all": np.arange(len(names)),
                **{e: np.array([i for i, n in enumerate(names) if n.startswith(e + "_")])
                   for e in MODEL_EMOTION_ORDER}}
            for s, names in STIMULI.items()}
RDM_DIRS = {s: ROOT / f"rdms/set{s}" for s in ("18", "54")}       # <name>/<name>.npy
GRID_DIRS = {"18": ROOT / "analysis/rsa_grid", "54": ROOT / "analysis/rsa_grid_54"}
# partial-RSA grids (partial_rsa.py), one folder per control set
PARTIAL_DIRS = {"18": ROOT / "analysis/partial_rsa", "54": ROOT / "analysis/partial_rsa_54"}
N_ROI = 11                   # scopes 0-10 are ROIs, 11 is the whole scalp


def eeg_meta():
    return json.loads((EEG_DIR / "meta.json").read_text())


def subjects(stimuli=None):
    return eeg_meta()["subjects"]


def scopes(stimuli=None):
    return eeg_meta()["scopes"]


def window_bounds(stimuli=None):
    """(13, 2) window bounds in ms from stimulus onset."""
    return np.asarray(eeg_meta()["window_bounds_ms"])


def load_eeg(stimuli, emotion="all", subs=None):
    """-> (eeg (S, 12, W, n, n) over STIMULI, subject ids, scope names)."""
    meta = eeg_meta()
    if meta["stimuli"] != MODEL_STIMULI["54"] or len(meta["scopes"]) != N_ROI + 1:
        raise SystemExit("ABORT: eeg_rdms/meta.json does not match rsa_core (stimuli or scopes)")
    subs = meta["subjects"] if subs is None else subs
    rows = EEG_ROWS[stimuli]
    eeg = np.stack([np.load(EEG_DIR / f"{s}.npy") for s in subs])[..., rows, :][..., rows]
    if not np.isfinite(eeg).all():
        raise SystemExit(f"ABORT: missing EEG values in the {stimuli}-stimulus RDMs")
    idx = EMOTIONS[stimuli][emotion]
    if len(idx) != eeg.shape[-1]:
        eeg = eeg[..., idx, :][..., idx]
    return eeg, subs, scopes(stimuli)


def upper(r):
    """(..., k, k) -> (..., k*(k-1)/2) upper triangle."""
    iu = np.triu_indices(r.shape[-1], 1)
    return r[..., iu[0], iu[1]]


def rank_normalise(v):
    """Rank -> mean-centre -> unit norm along the last axis, so a dot product IS
    Spearman's rho. Also returns where the vector is constant (rho undefined)."""
    r = np.apply_along_axis(rankdata, -1, v)
    rc = r - r.mean(-1, keepdims=True)
    n = np.linalg.norm(rc, axis=-1, keepdims=True)
    return np.divide(rc, np.clip(n, 1e-12, None)), (n[..., 0] < 1e-12)


def grid_path(stimuli, tag, partial=None):
    """Grid of an RDM; partial = a control set of partial_rsa.py."""
    folder = GRID_DIRS[stimuli] if partial is None else PARTIAL_DIRS[stimuli] / partial
    return folder / f"{tag}_rsa_grid.npz"


def variant_grid(stimuli, tags, partial=None):
    """Participant-level grid of a variant: the mean over its RDMs' grids (its ten
    seeds or training runs; a released model has one) -> (rsa, npz of the first)."""
    first = np.load(grid_path(stimuli, tags[0], partial), allow_pickle=True)
    rsa = np.mean([np.load(grid_path(stimuli, t, partial))["rsa"] for t in tags], axis=0)
    return rsa.astype(np.float32), first


def slug(family, variant):
    return f"{family}_{variant}".lower().replace(" ", "_").replace("-", "")


def load_model_rdm(stimuli, tag, rdm_dir=None):
    """(L, frames, n, n) model RDM over STIMULI; checks its stimulus order first."""
    folder = Path(rdm_dir or RDM_DIRS[stimuli]) / tag
    meta = json.loads((folder / f"{tag}.layers.json").read_text())
    stimuli_order = meta.get("stimuli") or meta.get("stim_order")
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
