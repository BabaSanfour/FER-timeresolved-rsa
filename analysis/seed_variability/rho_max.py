#!/usr/bin/env python3
"""rho_max for the 18-stimulus Table 1, per RSA grid.

Definition, copied from analysis/build_maxr_table.py (which produced the
published table): average the grid over subjects FIRST, take the maximum over
layer x window separately within each ROI x emotion cell, then average the six
ROI values and the three emotion-specific values. The maximum is therefore
taken per cell, never after averaging cells -- which is what keeps the
selection bias identical between a single pretrained model and a single
random seed.
"""
import numpy as np

N_ROI = 6                    # scopes 0-5; index 6 is wholescalp
EMO_SPECIFIC = (1, 2, 3)     # emotions are ordered all, fear, happy, neutral


def cell_maxima(rsa):
    """rsa (S, O, L, E, W) -> dict[(roi, emotion)] = max over layer x window."""
    gm = np.nanmean(rsa, axis=0)                       # (O, L, E, W)
    out = {}
    for oi in range(gm.shape[0]):
        for ei in range(gm.shape[2]):
            block = gm[oi, :, ei, :]
            if np.all(np.isnan(block)):
                continue
            out[(oi, ei)] = float(np.nanmax(block))
    return out


def roi_avg_emo_avg(rsa):
    """The Table 1 'ROI avg.' number: mean over 6 ROIs x 3 emotions."""
    c = cell_maxima(rsa)
    return float(np.mean([c[(oi, ei)] for oi in range(N_ROI) for ei in EMO_SPECIFIC]))


def per_roi_emo_avg(rsa):
    """The six per-ROI Table 1 columns (each already emotion-averaged)."""
    c = cell_maxima(rsa)
    return [float(np.mean([c[(oi, ei)] for ei in EMO_SPECIFIC])) for oi in range(N_ROI)]


def wholescalp_emo_avg(rsa):
    c = cell_maxima(rsa)
    return float(np.mean([c[(6, ei)] for ei in EMO_SPECIFIC]))


def from_file(path):
    return roi_avg_emo_avg(np.load(path, allow_pickle=True)["rsa"])
