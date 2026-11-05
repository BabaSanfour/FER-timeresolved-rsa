import argparse
import csv

import numpy as np

from fer_rsa.analysis import rsa_core
from fer_rsa.analysis import variants
from fer_rsa.analysis.partial_rsa import control_sets, residual_normalise, window_bases
from fer_rsa.analysis.rsa_core import pair_permutation, rank_normalise, ranked_centered, upper

EMOTIONS = rsa_core.MODEL_EMOTION_ORDER        # --emotion keeps one
N_PERM = 1000
N_BOOT = 10000
SEED = 42
EMOTION_CELLS = (1, 2, 3)       # fear, happy, neutral on the grid's emotion axis (all, fear, happy, neutral)
GENERIC = {"ResNet18": "ImageNet", "ResNet50": "ImageNet", "FaceNet": "ImageNet", "VGG16": "ImageNet",
           "M3DFEL": "Kinetics-400", "Qwen3-VL": "Pretrained"}
FACE = {"ResNet18": "VGGFace2", "ResNet50": "VGGFace2", "FaceNet": "VGGFace2", "VGG16": "VGG-Face"}
FINE_TUNED = {"ResNet18": ("AffectNet FT", "VGGFace2"), "ResNet50": ("AffectNet FT", "VGGFace2"),
              "FaceNet": ("AffectNet FT", "VGGFace2"), "VGG16": ("AffectNet FT", "VGG-Face"),
              "M3DFEL": ("DFEW FT", "Kinetics-400")}
VIDEO = [("M3DFEL", "Kinetics-400"), ("Qwen3-VL", "Pretrained")]
IMAGE_PRETRAINED = [(f, v) for f in FACE for v in (GENERIC[f], FACE[f])]
PER_RDM = {"on": False}                                            # --per-rdm


def hypotheses():
    """{hypothesis: [(label, A, B)]}: the contrasts each pooled test averages."""
    return {
        "H1 pretrained vs random": [(f, (f, v), (f, "Random")) for f, v in GENERIC.items()],
        "H2 face vs ImageNet": [(f, (f, v), (f, "ImageNet")) for f, v in FACE.items()],
        "H3 fine-tuned vs start": [(f, (f, ft), (f, start)) for f, (ft, start) in FINE_TUNED.items()],
        "H4 video vs image": [(f"{' '.join(v)} - {' '.join(i)}", v, i) for v in VIDEO for i in IMAGE_PRETRAINED],
        "H5 face vs random": [(f, (f, v), (f, "Random")) for f, v in FACE.items()],
    }


TIME_HYPOTHESES = tuple(hypotheses())          # also resolved in time


def design_hypotheses(design):
    """{hypothesis: [(label, A, B)]}: each design model against the random FaceNet and the full pool."""
    if design == "sizes":
        levels = [("size", z) for z in variants.SIZES] + [("step", z) for z in variants.SIZES[:-1]]
    else:
        levels = [(design, z) for z in (variants.CHECKPOINT_STEPS if design == "ckpt" else variants.IDENTITIES)]
    out = {}
    for group, level in levels:
        a = ("FaceNet", f"{group} {level}")
        out[f"{group} {level} vs random"] = [("FaceNet", a, ("FaceNet", "Random"))]
        out[f"{group} {level} vs full pool"] = [("FaceNet", a, ("FaceNet", "VGGFace2"))]
    return out


def pairs_of(ta, tb):
    return list(zip(ta, tb)) if len(ta) == len(tb) else [(x, y) for x in ta for y in tb]


def cell_values(grid, select, by_window=False):
    """grid (..., O, L, 3, W) at the peaks of select (O, L, 3, W), mean over emotions -> (..., O) or (..., O, W)."""
    O, L, E, W = select.shape
    filled = np.where(np.isnan(select), -np.inf, select)
    if by_window:
        li = filled.argmax(1)                                               # (O, E, W)
        o, e, w = np.meshgrid(np.arange(O), np.arange(E), np.arange(W), indexing="ij")
        return grid[..., o, li, e, w].mean(-2)
    flat = filled.transpose(0, 2, 1, 3).reshape(O, E, L * W).argmax(-1)    # (O, E)
    li, wi = np.unravel_index(flat, (L, W))
    o, e = np.meshgrid(np.arange(O), np.arange(E), indexing="ij")
    return grid[..., o, li, e, wi].mean(-1)


def pair_difference(ga, gb, same_family, by_window=False):
    """Grids ([S,] O, L, 3, W) of A and B -> A - B, selected on (A + B) / 2 within a family."""
    mean = (lambda g: np.nanmean(g, 0)) if ga.ndim == 5 else (lambda g: g)
    if same_family:
        sel = mean((ga + gb) / 2)
        return cell_values(ga, sel, by_window) - cell_values(gb, sel, by_window)
    return cell_values(ga, mean(ga), by_window) - cell_values(gb, mean(gb), by_window)


def contrast(grid, tags, a, b, by_window=False):
    """grid(tag) -> A - B, mean over pairs of RDMs."""
    return np.mean([pair_difference(grid(x), grid(y), a[0] == b[0], by_window)
                    for x, y in pairs_of(tags[a], tags[b])], axis=0)


def own_peak(g):
    """Grid read at its own participant-mean peak."""
    return cell_values(g, np.nanmean(g, 0) if g.ndim == 5 else g)


def all_contrasts(grid, tags, hyp):
    """-> {key: value} of pooled tests, contrasts, variants at their own peak and time courses."""
    out = {}
    for key in sorted({x for items in hyp.values() for _, a, b in items for x in (a, b)}):
        out["variant", " ".join(key)] = np.mean([own_peak(grid(t)) for t in tags[key]], axis=0)
        if PER_RDM["on"]:
            for t in tags[key]:
                out["rdm", t] = own_peak(grid(t))
    for h, items in hyp.items():
        vals = [contrast(grid, tags, a, b) for _, a, b in items]
        out[h] = np.mean(vals, axis=0)
        if h.startswith("H4"):
            for v in VIDEO:
                out[h, " ".join(v)] = np.mean([x for x, (_, a, _) in zip(vals, items) if a == v], axis=0)
        else:
            for (label, _, _), x in zip(items, vals):
                out[h, label] = x
    for h in (x for x in TIME_HYPOTHESES if x in hyp):
        time = [contrast(grid, tags, a, b, by_window=True) for _, a, b in hyp[h]]
        out["time", h, "pooled"] = np.mean(time, axis=0)
        if h.startswith("H4"):
            for v in VIDEO:
                out["time", h, " ".join(v)] = np.mean([x for x, (_, a, _) in zip(time, hyp[h]) if a == v], axis=0)
        else:
            for (label, _, _), x in zip(hyp[h], time):
                out["time", h, label] = x
    return out


def upper_ceiling(z):
    """(S, ..., P) unit-norm EEG vectors -> upper noise ceiling (...) (Nili et al., 2014)."""
    unit = rsa_core.unit(np.nansum(z, 0))[0]
    return np.nanmean(np.sum(z * unit[None], -1), 0)


def permutation_nulls(s, tags, hyp, W, partial, normalize=False, observed=None):
    """-> ({key: (N_PERM, O[, W])} under within-emotion label permutation, EEG ceilings (O, 3, W))."""
    eeg, _, _ = rsa_core.load_eeg(s)
    O = eeg.shape[1]
    rng = np.random.default_rng(SEED)
    members = control_sets(s)[partial] if partial else []
    controls = {c: rsa_core.load_model_rdm(s, f"control_{c}")[0, :W] for c in members}
    eeg_mean, perms, bases, ceiling = {}, {}, {}, {}
    for e in EMOTIONS:
        idx = rsa_core.EMOTIONS[s][e]
        if partial:
            bases[e] = window_bases(controls, members, idx, W)
            ev = ranked_centered(upper(eeg[..., idx, :][..., idx]))[:, :, :W]            # (S, O, W, P)
            z = np.stack([residual_normalise(ev[:, :, k], bases[e][k])[0] for k in range(W)], 2)
        else:
            z = rank_normalise(upper(eeg[..., idx, :][..., idx]))[0][:, :, :W]            # (S, O, W, P)
        eeg_mean[e] = z.mean(0)
        ceiling[e] = upper_ceiling(z)                                                    # (O, W)
        if observed is not None:
            observed.setdefault("z", {})[e] = z
        perms[e] = [pair_permutation(rng.permutation(len(idx))) for _ in range(N_PERM)]
    del eeg
    needed = sorted({t for items in hyp.values() for _, a, b in items for t in tags[a] + tags[b]})
    model = {}
    for tag in needed:
        m = rsa_core.load_model_rdm(s, tag)[:, :W]
        model[tag] = [(ranked_centered if partial else lambda v: rank_normalise(v)[0])(
            upper(m[..., rsa_core.EMOTIONS[s][e], :][..., rsa_core.EMOTIONS[s][e]])) for e in EMOTIONS]
    if observed is not None:                       # participant grids recomputed (--drop-actor, --eeg-dir)
        for tag in needed:
            g = []
            for ei, e in enumerate(EMOTIONS):
                mv, z = model[tag][ei], observed["z"][e]                                 # (L, W, P), (S, O, W, P)
                if partial:
                    x = np.empty(z.shape[:2] + (mv.shape[0], W))
                    for w in range(W):
                        mz, bad = residual_normalise(mv[:, w], bases[e][w])
                        x[:, :, :, w] = np.einsum("sop,lp->sol", z[:, :, w], mz, optimize=True)
                        x[:, :, bad, w] = np.nan
                else:
                    flat = np.linalg.norm(mv, axis=-1) < 1e-12
                    x = np.einsum("sowp,lwp->solw", z, mv, optimize=True)
                    x[:, :, flat] = np.nan
                g.append(x)
            observed[tag] = np.stack(g, 3)                                                # (S, O, L, 3, W)
        del observed["z"]
    null = None
    for k in range(N_PERM):
        grids = {}
        for tag in needed:
            g = []
            for ei, e in enumerate(EMOTIONS):
                mv = model[tag][ei][..., perms[e][k]]                                  # (L, W, P)
                if partial:
                    x = np.empty((O, mv.shape[0], W))
                    for w in range(W):
                        mz, bad = residual_normalise(mv[:, w], bases[e][w])
                        x[:, :, w] = eeg_mean[e][:, w] @ mz.T
                        x[:, bad, w] = np.nan
                else:
                    flat = np.linalg.norm(mv, axis=-1) < 1e-12
                    x = np.einsum("owp,lwp->olw", eeg_mean[e], mv, optimize=True)
                    x[:, flat] = np.nan
                g.append(x / ceiling[e][:, None, :] if normalize else x)
            grids[tag] = np.stack(g, 2)                                                # (O, L, 3, W)
        vals = all_contrasts(grids.get, tags, hyp)
        if null is None:
            null = {key: np.zeros((N_PERM,) + v.shape) for key, v in vals.items()}
        for key, v in vals.items():
            null[key][k] = v
        if (k + 1) % 100 == 0:
            print(f"  [null] {k + 1}/{N_PERM} permutations", flush=True)
    return null, np.stack([ceiling[e] for e in EMOTIONS], 1)                              # (O, 3, W)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(run, 1.)
    return out


def tests(obs, null, rng):
    """obs (S, O, ...), null (N, O, ...) -> per-cell stats; maxT over the ROI cells and the whole-scalp cells."""
    centre, spread = null.mean(0), null.std(0)
    z = (obs.mean(0) - centre) / spread
    zn = np.abs((null - centre) / spread)
    p = (1 + (zn >= np.abs(z)).sum(0)) / (N_PERM + 1)
    p_max = np.empty_like(p)
    for part in (slice(0, rsa_core.N_ROI), slice(rsa_core.N_ROI, rsa_core.N_ROI + 1)):
        m = zn[:, part].reshape(N_PERM, -1).max(1)
        p_max[part] = (1 + (m[:, None] >= np.abs(z[part]).ravel()[None]).sum(0)).reshape(z[part].shape) / (N_PERM + 1)
    boot = obs[rng.integers(0, len(obs), size=(N_BOOT, len(obs)))].mean(1)
    return {"observed": obs.mean(0), "null_mean": centre, "corrected": obs.mean(0) - centre,
            "low": np.percentile(boot, 2.5, axis=0) - centre, "high": np.percentile(boot, 97.5, axis=0) - centre,
            "n_above": (obs > centre).sum(0), "p": p, "p_maxT": p_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", choices=["18", "54"], required=True)
    ap.add_argument("--partial", default=None, help="a control set of partial_rsa.py (e.g. joint)")
    ap.add_argument("--design", choices=["sizes", "ckpt", "ids"], default=None,
                    help="a FaceNet learning design instead of H1-H5")
    ap.add_argument("--eeg-dir", default=None, help="EEG RDMs of another build (e.g. eeg_rdms_correct)")
    ap.add_argument("--emotion", choices=list(EMOTIONS), default=None, help="one emotion instead of the mean")
    ap.add_argument("--drop-actor", type=int, default=None, help="leave out the k-th actor (0-5) of every emotion")
    ap.add_argument("--per-rdm", action="store_true", help="also each RDM at its own peak (hypothesis 'rdm')")
    ap.add_argument("--normalize", action="store_true", help="divide every cell by its upper noise ceiling (*_nc)")
    args = ap.parse_args()
    PER_RDM["on"] = args.per_rdm
    if args.emotion:
        globals()["EMOTION_CELLS"] = (1 + EMOTIONS.index(args.emotion),)
        globals()["EMOTIONS"] = (args.emotion,)
    if args.drop_actor is not None:
        names = rsa_core.MODEL_STIMULI[args.stimuli]
        for e, idx in list(rsa_core.EMOTIONS[args.stimuli].items()):
            if e == "all":
                continue
            actors = sorted({names[i].rsplit("_", 1)[1] for i in idx})
            gone = actors[args.drop_actor]
            rsa_core.EMOTIONS[args.stimuli][e] = np.array([i for i in idx if not names[i].endswith("_" + gone)])
    s = args.stimuli
    tags = {(f, v if g == "main" else f"{g} {v}"): t for g, f, v, t in variants.variants(s) if g in ("main", "size", "step", "ckpt", "ids")}
    cache, scale = {}, {}
    if args.eeg_dir:
        rsa_core.EEG_DIR = rsa_core.ROOT / args.eeg_dir
    observed = {} if (args.drop_actor is not None or args.eeg_dir) else None

    def grid(tag):
        if observed is not None and tag in observed:
            g = observed[tag]
            return g / scale["ceiling"][None, :, None, :, :g.shape[-1]] if "ceiling" in scale else g
        if tag not in cache:
            g = np.load(rsa_core.grid_path(s, tag, args.partial))["rsa"].astype(np.float64)[:, :, :, EMOTION_CELLS, :]
            cache[tag] = g / scale["ceiling"][None, :, None, :, :g.shape[-1]] if "ceiling" in scale else g
        return cache[tag]

    hyp = design_hypotheses(args.design) if args.design else hypotheses()
    W = grid(tags["ResNet18", "Random"][0]).shape[-1]
    scopes = rsa_core.scopes()
    bounds = rsa_core.window_bounds()[:W]
    null, ceiling = permutation_nulls(s, tags, hyp, W, args.partial, args.normalize, observed)
    if args.normalize:
        scale["ceiling"] = ceiling; cache.clear()
    obs = all_contrasts(grid, tags, hyp)
    rng = np.random.default_rng(SEED)
    n_pairs = {h: len(items) for h, items in hyp.items()}
    rows, time_rows = [], []
    for key, value in obs.items():
        r = tests(value, null[key], rng)
        if key[0] == "time":
            _, th, label = key
            for oi, sc in enumerate(scopes):
                for wi in range(W):
                    time_rows.append({"hypothesis": th, "contrast": label, "scope": sc,
                                      "window_start_ms": bounds[wi, 0], "window_end_ms": bounds[wi, 1],
                                      **{k: round(float(v[oi, wi]), 5) if k not in ("p", "p_maxT", "n_above") else
                                         (int(v[oi, wi]) if k == "n_above" else float(v[oi, wi])) for k, v in r.items()}})
            continue
        h, label = (key, "pooled") if isinstance(key, str) else key
        for oi, sc in enumerate(scopes):
            rows.append({"hypothesis": h, "contrast": label, "n_pairs": n_pairs[h] if label == "pooled" else "",
                         "scope": sc, "n": len(value),
                         **{k: round(float(v[oi]), 5) if k not in ("p", "p_maxT", "n_above") else
                            (int(v[oi]) if k == "n_above" else float(v[oi])) for k, v in r.items()}})
    whole = [r for r in rows if r["contrast"] == "pooled" and r["scope"] == "wholescalp"]
    for r, ph in zip(whole, holm([r["p"] for r in whole])):     # Holm across the whole-scalp contrasts
        r["p_holm"] = float(ph)
    suffix = (f"_{args.design}" if args.design else "") + \
        (f"_{args.partial}" if args.partial else "") + ("_nc" if args.normalize else "") + \
        (f"_dropactor{args.drop_actor}" if args.drop_actor is not None else "") + \
        (f"_{args.emotion}" if args.emotion else "") + (f"_{args.eeg_dir}" if args.eeg_dir else "")
    out = rsa_core.STATS_DIRS[s]
    out.mkdir(parents=True, exist_ok=True)
    for name, table in ((f"contrasts{suffix}.csv", rows), (f"contrasts_time{suffix}.csv", time_rows)):
        if not table:
            continue
        fields = list(dict.fromkeys(k for r in table for k in r))
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(table)
    for r in rows:
        if r["contrast"] == "pooled" or r["scope"] == "wholescalp":
            extra = f"  Holm={r['p_holm']:.4f}" if "p_holm" in r else ""
            print(f"{r['hypothesis']:24s} {r['contrast']:22s} {r['scope']:19s} corrected {r['corrected']:+.4f} "
                  f"[{r['low']:+.4f}, {r['high']:+.4f}] p={r['p']:.4f} maxT={r['p_maxT']:.4f}{extra}")
    print(f"[ok] -> {out / ('contrasts' + suffix + '.csv')}, contrasts_time{suffix}.csv")


if __name__ == "__main__":
    main()
