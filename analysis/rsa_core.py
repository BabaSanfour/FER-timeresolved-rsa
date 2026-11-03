"""Shared pieces of the RSA analyses: EEG loading for both stimulus sets, the
rank-correlation helpers, the noise ceilings, and the rho_max summary of a grid.

EEG RDMs are returned as (subjects, 7 scopes, windows, n, n): six ROIs in
`roi_order.npy` order, then the whole scalp, stimuli in the model order

    18 stimuli: fear 0-5, happy 6-11, neutral 12-17 (actors 01 02 04 07 08 11)
    54 stimuli: fear 0-17, happy 18-35, neutral 36-53, each block unmasked
                (original), glasses, masked, actors as above

so `EMOTIONS[stimuli][emotion]` indexes both the EEG and the model RDMs.

18 stimuli: all 26 participants of the 18-stimulus folders.
54 stimuli: 25 participants (CEG has no 54-stimulus data). The pooled files
(`<subject>_all_rdms.npy`) are ordered happy, fear, neutral (`stimulus_order.json`)
and are reordered here by stimulus name; the per-emotion files are blocks of
the pooled file and are not needed. Eleven participants lack one video
(neutral, glasses, index 3), stored as a NaN row and column: `pair_groups`
groups participants by their available stimulus pairs, so each is analysed on
the pairs they have (17 of 18 neutral stimuli, 53 of 54 pooled), and every
rank statistic re-ranks both vectors on that subset.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("FER_ROOT", "."))
EEG_DIRS = {
    "18": (ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full", "_windowed_rdms.npy",
           ROOT / "eeg_data/Unmasked_avg_wholescalp_16win", "_wholescalp_rdms.npy"),
    "54": (ROOT / "eeg_rdms_complete/perroi", "_all_rdms.npy",
           ROOT / "eeg_rdms_complete/wholescalp", "_all_rdms.npy"),
}
META_DIRS = {"18": ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full",
             "54": ROOT / "eeg_rdms_complete"}
EMOTIONS = {
    "18": {"all": np.arange(18), "fear": np.arange(0, 6),
           "happy": np.arange(6, 12), "neutral": np.arange(12, 18)},
    "54": {"all": np.arange(54), "fear": np.arange(0, 18),
           "happy": np.arange(18, 36), "neutral": np.arange(36, 54)},
}
MODEL_EMOTION_ORDER = ("fear", "happy", "neutral")
ACTORS = ("01", "02", "04", "07", "08", "11")
# stimulus names in the model RDMs' .layers.json (build_stimuli_h5_16frames.py)
MODEL_STIMULI = {
    "18": [f"{e}_Actor{a}" for e in MODEL_EMOTION_ORDER for a in ACTORS],
    "54": [f"{e}_{c}_Actor{a}" for e in MODEL_EMOTION_ORDER
           for c in ("original", "glasses", "masked") for a in ACTORS],
}
RDM_DIRS = {s: ROOT / f"rdms/set{s}" for s in ("18", "54")}       # <name>/<name>.npy
GRID_DIRS = {"18": ROOT / "analysis/rsa_grid", "54": ROOT / "analysis/rsa_grid_54"}
N_ROI = 6                    # scopes 0-5 are ROIs, 6 is the whole scalp
EMO_SPECIFIC = (1, 2, 3)     # emotions are ordered all, fear, happy, neutral
ROI_AVG, EMO_AVG = "ROI-avg", "emo-avg"
CONDITIONS_54 = ("unmasked", "glasses", "masked")


def subjects(stimuli):
    roi_dir, roi_suffix, ws_dir, ws_suffix = EEG_DIRS[stimuli]
    roi = {p.name.removesuffix(roi_suffix) for p in roi_dir.glob(f"*{roi_suffix}")
           if not p.name.startswith("group")}
    ws = {p.name.removesuffix(ws_suffix) for p in ws_dir.glob(f"*{ws_suffix}")}
    if roi != ws:
        raise SystemExit(f"ABORT: subject mismatch: ROI-only={sorted(roi - ws)}, "
                         f"whole-scalp-only={sorted(ws - roi)}")
    return sorted(roi)


def scopes(stimuli):
    rois = np.load(META_DIRS[stimuli] / "roi_order.npy", allow_pickle=True)
    return [str(x) for x in rois] + ["wholescalp"]


def window_bounds(stimuli):
    return np.load(META_DIRS[stimuli] / "window_bounds.npy")


def model_order_54():
    """Indices that put the pooled 54-stimulus EEG RDM into the model order."""
    names = json.loads((META_DIRS["54"] / "stimulus_order.json").read_text())["all_54"]
    return np.array([names.index(f"{e}_{c}_{i}") for e in MODEL_EMOTION_ORDER
                     for c in CONDITIONS_54 for i in range(6)])


def load_eeg(stimuli, emotion="all", subs=None):
    """-> (eeg (S, 7, W, n, n) in the model order, subject ids, scope names)."""
    subs = subjects(stimuli) if subs is None else subs
    roi_dir, roi_suffix, ws_dir, ws_suffix = EEG_DIRS[stimuli]
    roi = np.stack([np.load(roi_dir / f"{s}{roi_suffix}") for s in subs])
    ws = np.stack([np.load(ws_dir / f"{s}{ws_suffix}") for s in subs])
    if roi.ndim != 5 or ws.ndim != 4:
        raise SystemExit(f"ABORT: unexpected EEG shapes {roi.shape} / {ws.shape}")
    eeg = np.concatenate([roi, ws[:, None]], axis=1)
    if stimuli == "54":
        order = model_order_54()
        eeg = eeg[..., order, :][..., order]
    idx = EMOTIONS[stimuli][emotion]
    if len(idx) != eeg.shape[-1]:
        eeg = eeg[..., idx, :][..., idx]
    return eeg, subs, scopes(stimuli)


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


def spearman(x, y):
    """Spearman's rho on the pairs finite in both; NaN if either is constant."""
    keep = np.isfinite(x) & np.isfinite(y)
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def row_spearman(x, y):
    """Row-wise Spearman of two (rows, pairs) arrays without missing values."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    denom = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry, axis=1)
    return np.divide(np.sum(rx * ry, axis=1), denom,
                     out=np.full(len(x), np.nan), where=denom > 0)


def noise_ceilings(eeg, stimuli):
    """eeg (S, O, W, n, n) -> lower, upper, each (O, emotions, W).

    Rank-based ceilings for Spearman RSA (Nili et al., 2014): each participant's
    RDM is rank-transformed (ranks standardised per participant, so participants
    with fewer stimulus pairs weigh the same; on complete data this equals plain
    ranks), then per scope, emotion and window the mean over participants of
      upper: Spearman(participant, mean rank RDM including the participant)
      lower: Spearman(participant, leave-one-out mean rank RDM),
    each on the stimulus pairs the participant has; a pair's mean uses the
    participants who have that pair.
    """
    emotions = EMOTIONS[stimuli]
    O, W = eeg.shape[1:3]
    lower = np.full((O, len(emotions), W), np.nan)
    upper_nc = np.full((O, len(emotions), W), np.nan)
    for ei, idx in enumerate(emotions.values()):
        vec = upper(eeg[..., idx, :][..., idx])               # (S, O, W, P)
        groups = pair_groups(vec)
        ranked = np.full(vec.shape, np.nan)
        for sidx, keep in groups:                             # ranks over each participant's pairs
            r = rankdata(vec[sidx][..., keep], axis=-1)
            r = (r - r.mean(-1, keepdims=True)) / r.std(-1, keepdims=True)
            sub = ranked[sidx]
            sub[..., keep] = r
            ranked[sidx] = sub
        for oi in range(O):
            for wi in range(W):
                x = ranked[:, oi, wi]
                total = np.nansum(x, 0)
                count = np.isfinite(x).sum(0).astype(x.dtype)
                mean = total / count
                ups, los = [], []
                for sidx, keep in groups:
                    xs = x[sidx][:, keep]
                    ups.append(row_spearman(xs, np.broadcast_to(mean[keep], xs.shape)))
                    los.append(row_spearman(xs, (total[keep][None] - xs) / (count[keep][None] - 1)))
                upper_nc[oi, ei, wi] = np.nanmean(np.concatenate(ups))
                lower[oi, ei, wi] = np.nanmean(np.concatenate(los))
    return lower, upper_nc


def grid_path(stimuli, tag):
    return GRID_DIRS[stimuli] / f"{tag}_rsa_grid.npz"


def load_model_rdm(stimuli, tag, rdm_dir=None):
    """(L, frames, n, n) model RDM; checks its stimulus order against the EEG's."""
    folder = Path(rdm_dir or RDM_DIRS[stimuli]) / tag
    meta = json.loads((folder / f"{tag}.layers.json").read_text())
    stimuli_order = meta.get("stimuli") or meta.get("stim_order")
    if stimuli_order != MODEL_STIMULI[stimuli]:
        raise SystemExit(f"ABORT {tag}: stimulus order differs from the EEG's")
    return np.load(folder / f"{tag}.npy")


def cell_maxima(rsa, nc_lower=None, nc_upper=None):
    """rsa (S, O, L, E, W) -> {(scope, emotion): (rho_max, nc_lower, nc_upper, layer, window)}.

    The grid is averaged over participants first; the maximum over layer x window
    is then taken separately in each scope x emotion cell (never after averaging
    cells), and the ceilings (O, E, W) are read at that cell's window.
    """
    gm = np.nanmean(rsa, axis=0)                                   # (O, L, E, W)
    cells = {}
    for oi in range(gm.shape[0]):
        for ei in range(gm.shape[2]):
            block = gm[oi, :, ei, :]
            if np.all(np.isnan(block)):
                continue
            li, wi = np.unravel_index(np.nanargmax(block), block.shape)
            cells[oi, ei] = (float(block[li, wi]),
                             np.nan if nc_lower is None else float(nc_lower[oi, ei, wi]),
                             np.nan if nc_upper is None else float(nc_upper[oi, ei, wi]),
                             int(li), int(wi))
    return cells


def averaged_cells(cells):
    """Add the averaged scope (index 7, ROI-avg: the six ROIs) and emotion (index 4,
    emo-avg: fear, happy, neutral); ROI-avg x emo-avg averages the 18 cells.
    Values and ceilings are averaged; layer and window become -1."""
    out = {}
    for oi in range(N_ROI + 2):
        for ei in range(len(EMO_SPECIFIC) + 2):
            if oi <= N_ROI and ei <= len(EMO_SPECIFIC):
                out[oi, ei] = cells[oi, ei]
                continue
            rois = range(N_ROI) if oi == N_ROI + 1 else [oi]
            emos = EMO_SPECIFIC if ei == len(EMO_SPECIFIC) + 1 else [ei]
            group = [cells[r, e] for r in rois for e in emos]
            out[oi, ei] = tuple(float(np.mean([c[j] for c in group])) for j in range(3)) + (-1, -1)
    return out
