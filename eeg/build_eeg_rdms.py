#!/usr/bin/env python3
"""EEG RDMs of every participant, from the segmented recordings.

Input ($FER_ROOT/eeg_raw/):
  <subject>_EmoDyn_Bloc<b>_<scheme>_Segmentation all_y.{vhdr,dat}   Brain Vision Analyzer
      export after re-referencing, filtering, ICA ocular correction and segmentation:
      ASCII, 64 EEG channels (+ EXG, Status), 1024 Hz, 1024-sample segments from -200 ms,
      stimulus onset at sample 206.
  <subject>_EmoDyn_Bloc<b>_<scheme>_Segmentation all.Markers        trigger code per segment
  behavior_RT_perf_identity_occlusion.csv                           video shown on every trial

Trial labels. The trigger codes (S1-S15) name a group of videos, not a video; the behaviour
file names the video of every trial in presentation order. Each block's behaviour sequence,
mapped to codes (CODES), must match the block's code sequence except for trials missing at
either end of the recording. A block without a Markers file is labelled by order only if it
has as many segments as the behaviour file has trials; otherwise it is dropped.

Per participant:
  1. baseline correction on the 205 pre-stimulus samples (-200 to 0 ms);
  2. bad channels (above +-BAD_UV in more than BAD_FRACTION of trials) are interpolated
     (spherical splines, standard 10-05 positions);
  3. trials above +-REJECT_UV on any channel are rejected;
  4. the remaining trials are averaged per video (54 videos);
  5. RDMs: correlation distance (1 - Pearson) between the video averages, the features being
     the channels of a scope x the samples of a window. Scopes: the 11 ROIs of
     roi_definitions.json, then the whole scalp (64 channels). Windows: 13 from onset, 64
     samples (62.5 ms) each; the last holds the 51 samples left (750-800 ms).
Participants with fewer than half of the 972 trials recorded are excluded.

Output ($FER_ROOT/eeg_rdms/): <subject>.npy (12, 13, 54, 54) float32 over rsa_core.MODEL_STIMULI["54"],
and meta.json (scopes, channels, window bounds in ms, subjects, exclusions, trial counts).

  python eeg/build_eeg_rdms.py [--subjects AB AF ...]
"""
import argparse
import difflib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FER_ROOT", "."))
RAW = ROOT / "eeg_raw"
OUT = ROOT / "eeg_rdms"
ROIS = json.loads((Path(__file__).parent / "roi_definitions.json").read_text())
ROIS.pop("_note")
N_EEG, SEG, FS, ONSET = 64, 1024, 1024.0, 205
WIN, N_WIN = 64, 13
BAD_UV, BAD_FRACTION, REJECT_UV = 100.0, 0.2, 150.0
TRIALS_EXPECTED, MIN_FRACTION = 972, 0.5
EMOTIONS = {"Fear": "fear", "Hapiness": "happy", "Neutral": "neutral"}
OCCLUSIONS = {"": "original", "Glasses": "glasses", "Masked": "masked"}
ACTORS = ("01", "02", "04", "07", "08", "11")
STIMULI = [f"{e}_{c}_Actor{a}" for e in ("fear", "happy", "neutral")
           for c in ("original", "glasses", "masked") for a in ACTORS]
# trigger code of each video group: neutral by occlusion; fear and happiness by occlusion x actor group
GROUP_A = {"Fear": ("01", "02", "11"), "Hapiness": ("02", "04", "08")}
CODES = {("Neutral", ""): 1, ("Neutral", "Glasses"): 2, ("Neutral", "Masked"): 3}
for k, (emo, occ) in enumerate([(e, o) for e in ("Fear", "Hapiness") for o in ("", "Glasses", "Masked")]):
    CODES[emo, occ, "A"], CODES[emo, occ, "B"] = 4 + 2 * k, 5 + 2 * k


def parse_condition(cond):
    """'S07_Hapiness_Glasses' -> (model stimulus name, trigger code)."""
    actor, emo, *occ = cond.split("_")
    actor, occ = actor[1:], occ[0] if occ else ""
    name = f"{EMOTIONS[emo]}_{OCCLUSIONS[occ]}_Actor{actor}"
    code = CODES[emo, occ] if emo == "Neutral" else CODES[emo, occ, "A" if actor in GROUP_A[emo] else "B"]
    return name, code


def read_block(vhdr):
    """-> segments (n, 64, 1024) in uV, channel names, trigger codes (None without a Markers file)."""
    head = vhdr.read_text(encoding="utf-8", errors="ignore")
    channels = [m.group(1) for m in re.finditer(r"^Ch\d+=([^,]+),", head, re.M)][:N_EEG]
    assert int(re.search(r"SegmentDataPoints=(\d+)", head).group(1)) == SEG
    assert abs(1e6 / float(re.search(r"SamplingInterval=([\d.]+)", head).group(1)) - FS) < 1e-6
    with open(vhdr.with_suffix(".dat")) as f:
        x = np.stack([np.array(line.replace(",", ".").split()[1:], dtype=np.float32)
                      for _, line in zip(range(N_EEG), f)])
    n = x.shape[1] // SEG
    segments = x[:, :n * SEG].reshape(N_EEG, n, SEG).transpose(1, 0, 2)
    markers = Path(str(vhdr).replace("_y.vhdr", ".Markers"))
    codes = None
    if markers.exists():
        codes = [int(p.split(",")[1].strip().lstrip("S"))
                 for p in markers.read_text(encoding="utf-8", errors="ignore").splitlines() if p.startswith("Stimulus,")]
        assert len(codes) == n, f"{vhdr.name}: {len(codes)} codes for {n} segments"
    return segments, channels, codes


def label_block(codes, n_segments, conditions):
    """Behaviour trials of a block -> (segment indices, stimulus names) or None if unalignable."""
    names, beh = zip(*(parse_condition(c) for c in conditions))
    if codes is None:
        return (np.arange(n_segments), list(names), np.arange(len(names))) if n_segments == len(names) else None
    sm = difflib.SequenceMatcher(None, codes, list(beh), autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if len(blocks) != 1 or blocks[0].size != min(len(codes), len(beh)):
        return None                                   # anything but a cut at either end
    b = blocks[0]
    return np.arange(b.a, b.a + b.size), list(names[b.b:b.b + b.size]), np.arange(b.b, b.b + b.size)


def interpolate(x, channels, bad):
    """Spherical-spline interpolation of the bad channels (MNE, standard 10-05 positions)."""
    import mne
    info = mne.create_info(channels, FS, "eeg")
    names = mne.channels.get_builtin_montages()
    info.set_montage(mne.channels.make_standard_montage("colin27_1005" if "colin27_1005" in names else "standard_1005"))
    info["bads"] = [channels[i] for i in np.flatnonzero(bad)]
    ep = mne.EpochsArray(x * 1e-6, info, verbose="error")
    return ep.interpolate_bads(reset_bads=True, verbose="error").get_data(copy=False) * 1e6


def window_bounds():
    """Sample bounds of the 13 windows, relative to the segment."""
    return [(ONSET + w * WIN, min(ONSET + (w + 1) * WIN, SEG)) for w in range(N_WIN)]


def rdms(avgs, channels):
    """(54, 64, 1024) video averages -> (12, 13, 54, 54) correlation-distance RDMs."""
    scopes = [[channels.index(c) for c in chans] for chans in ROIS.values()] + [list(range(N_EEG))]
    out = np.zeros((len(scopes), N_WIN, len(avgs), len(avgs)), np.float32)
    for oi, idx in enumerate(scopes):
        for wi, (a, b) in enumerate(window_bounds()):
            f = avgs[:, idx, a:b].reshape(len(avgs), -1).astype(np.float64)
            f -= f.mean(1, keepdims=True)
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            d = 1 - f @ f.T
            np.fill_diagonal(d, 0)
            out[oi, wi] = d
    return out


CORRECT_ONLY = {"on": False}


def build(subject, behaviour):
    log = {"blocks": {}, "dropped_blocks": []}
    xs, labels, correct, channels = [], [], [], None
    for vhdr in sorted(RAW.glob(f"{subject}_EmoDyn_Bloc*_y.vhdr")):
        block = int(re.search(r"Bloc(\d)", vhdr.name).group(1))
        seg, ch, codes = read_block(vhdr)
        channels = channels or ch
        assert ch == channels, f"{vhdr.name}: channel order differs"
        rows = behaviour[(behaviour.Subject == subject) & (behaviour.Block == block)]
        cond = rows.Condition.tolist()
        got = label_block(codes, len(seg), cond)
        if got is None:
            log["dropped_blocks"].append(block)
            print(f"  [drop] {subject} block {block}: {len(seg)} segments, {len(cond)} trials, "
                  f"{'no markers' if codes is None else 'codes do not align'}")
            continue
        keep, names, trial = got
        log["blocks"][block] = {"segments": len(seg), "behaviour_trials": len(cond), "labelled": len(keep),
                                "markers": codes is not None}
        xs.append(seg[keep]); labels += names
        correct += list(rows.Performance.to_numpy()[trial] == 1)
    n = sum(len(x) for x in xs)
    log["trials_recorded"] = n
    if n < MIN_FRACTION * TRIALS_EXPECTED:
        log["excluded"] = f"{n} of {TRIALS_EXPECTED} trials recorded"
        return None, channels, log
    x = np.concatenate(xs).astype(np.float64)
    x -= x[:, :, :ONSET].mean(2, keepdims=True)
    labels = np.array(labels)
    bad = (np.abs(x).max(2) > BAD_UV).mean(0) > BAD_FRACTION
    if bad.any():
        x = interpolate(x, channels, bad)
    clean = np.abs(x).max((1, 2)) <= REJECT_UV
    if CORRECT_ONLY["on"]:                         # only trials the participant classified correctly
        ok = clean & np.array(correct)
        # a video never classified correctly keeps all its clean trials (logged)
        fallback = [s for s in STIMULI if not np.any(ok & (labels == s)) and np.any(clean & (labels == s))]
        for s in fallback:
            ok |= clean & (labels == s)
        log["trials_incorrect"] = int(np.sum(clean & ~np.array(correct)))
        log["videos_all_trials"] = fallback
        clean = ok
    counts = {s: int(np.sum(clean & (labels == s))) for s in STIMULI}
    log.update(bad_channels=[channels[i] for i in np.flatnonzero(bad)], trials_rejected=int((~clean).sum()),
               trials_kept=int(clean.sum()), trials_per_video_min=min(counts.values()),
               trials_per_video_max=max(counts.values()))
    if min(counts.values()) == 0:
        raise SystemExit(f"ABORT {subject}: no clean trial for {[s for s, c in counts.items() if not c]}")
    avgs = np.stack([x[clean & (labels == s)].mean(0) for s in STIMULI])
    return rdms(avgs, channels), channels, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--correct-only", action="store_true",
                    help="average only the trials each participant classified correctly (output eeg_rdms_correct/)")
    args = ap.parse_args()
    global OUT
    if args.correct_only:
        CORRECT_ONLY["on"] = True
        OUT = ROOT / "eeg_rdms_correct"
    behaviour = pd.read_csv(RAW / "behavior_RT_perf_identity_occlusion.csv")
    subjects = args.subjects or sorted({p.name.split("_")[0] for p in RAW.glob("*_EmoDyn_Bloc*_y.vhdr")})
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"participants": {}}
    for s in subjects:
        r, channels, log = build(s, behaviour)
        meta["participants"][s] = log
        if r is None:
            (OUT / f"{s}.npy").unlink(missing_ok=True)
            print(f"[excluded] {s}: {log['excluded']}")
            continue
        np.save(OUT / f"{s}.npy", r)
        print(f"[ok] {s}: kept {log['trials_kept']} of {log['trials_recorded']} trials, "
              f"{len(log['bad_channels'])} channels interpolated, {log['trials_per_video_min']}-"
              f"{log['trials_per_video_max']} trials per video", flush=True)
    bounds = window_bounds()
    meta.update(scopes=list(ROIS) + ["wholescalp"], roi_channels=ROIS, stimuli=STIMULI,
                window_bounds_ms=[[(a - ONSET) * 1000 / FS, (b - ONSET) * 1000 / FS] for a, b in bounds],
                window_bounds_samples=[[a, b] for a, b in bounds],
                subjects=sorted(k for k, v in meta["participants"].items() if "excluded" not in v),
                rules={"baseline_ms": [-200, 0], "bad_channel_uv": BAD_UV, "bad_channel_fraction": BAD_FRACTION,
                       "reject_uv": REJECT_UV, "min_fraction_trials": MIN_FRACTION, "distance": "correlation"})
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[ok] -> {OUT} ({len(meta['subjects'])} participants)")


if __name__ == "__main__":
    main()
