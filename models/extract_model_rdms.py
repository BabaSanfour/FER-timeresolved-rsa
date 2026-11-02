"""Layer-wise RDMs for ResNet18, ResNet50, VGG16 and FaceNet.

Loads a checkpoint (refusing partial loads; random variants load their saved
random weights), hooks every layer with learned parameters except the output
layer, runs the 16 frames of every stimulus (Slide{1..16}.h5 +
stimulus_manifest.json from build_stimuli_h5_16frames.py) and writes
<out-name>.npy (n_layers, 16, n_stim, n_stim) correlation-distance RDMs,
.layers.json (provenance) and .predictions.csv. Example:

    python models/extract_model_rdms.py --arch resnet50 --checkpoint <weights> --preproc imagenet \
        --h5-dir <frames> --out-dir <dir> --out-name resnet50_imagenet
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, "image"))   # FaceNet.py lives in models/image/

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Architecture registry: name -> builder(num_classes) -> nn.Module
#    Head is replaced to `num_classes` so the checkpoint's task head fits.
# ---------------------------------------------------------------------------
def _resnet(fn, head_dropout=False):
    def build(nc):
        m = fn(weights=None)
        if nc:
            # The net_weights/ checkpoints carry a plain Linear head named
            # `classifier`; wrapping it in Sequential(Dropout, Linear) would
            # rename the keys to fc.1.* and silently fail to load.
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, nc)) \
                if head_dropout else nn.Linear(m.fc.in_features, nc)
        return m
    return build

def _vgg(fn):
    def build(nc):
        m = fn(weights=None)
        if nc:
            m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc)
        return m
    return build

def _facenet(nc):
    from FaceNet import InceptionResnetV1 as FaceNetIRV1
    return FaceNetIRV1(num_classes=nc or 2622, n_input_channels=3)

ARCHS = {
    "resnet18": _resnet(tvm.resnet18),
    "resnet50": _resnet(tvm.resnet50),
    "vgg16": _vgg(tvm.vgg16),
    "facenet": _facenet,
}

# ---------------------------------------------------------------------------
# 2. Preprocessing registry: name -> (uint8 NHWC RGB) -> torch NCHW
# ---------------------------------------------------------------------------
def _pp_imagenet(x):
    x = x.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_vggface_rgb(x):
    """RGB mean subtraction used by the AffectNet VGG-Face fine-tuning run."""
    x = x.astype(np.float32) - np.array([129.1863, 104.7624, 93.5940], np.float32)
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_half(x):                          # (x/255 - 0.5)/0.5, as train.py normalises
    x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

PREPROCS = {"imagenet": _pp_imagenet, "vggface_rgb": _pp_vggface_rgb, "half": _pp_half}


def resize_batch(x, size):
    """Resize a (N,H,W,3) uint8 batch; models trained at 112 should see 112."""
    from PIL import Image
    return np.stack([np.array(Image.fromarray(im).resize((size, size), Image.BILINEAR))
                     for im in x])

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def infer_num_classes(state):
    """Guess head size = out_features of the deepest 2-D Linear/fc weight in the ckpt."""
    best = None
    for k, v in state.items():
        if hasattr(v, "ndim") and v.ndim == 2 and any(t in k.lower() for t in ("fc", "classifier", "logits", "head", "linear")):
            best = v.shape[0]
    # fallback: any 2-D weight
    if best is None:
        for k, v in state.items():
            if hasattr(v, "ndim") and v.ndim == 2:
                best = v.shape[0]
    return best

def load_checkpoint(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            state = obj[key]; break
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

# ---------------------------------------------------------------------------
# Checkpoint key layout. The net_weights/ checkpoints wrap the torchvision
# backbone in a `model.` attribute, rename ResNet layer{N} -> block{N}, and name
# the head `classifier`. Without remapping, every key is "unexpected", and
# strict=False leaves a randomly initialised network whose RDMs look plausible
# but mean nothing.
# ---------------------------------------------------------------------------
def _repo_to_torchvision(arch, k):
    if arch.startswith("resnet"):
        if k.startswith("classifier."):
            return k.replace("classifier.", "fc.", 1)
        k = re.sub(r"^model\.", "", k)
        return re.sub(r"^block(\d)\.", lambda m: f"layer{m.group(1)}.", k)
    if arch.startswith("vgg"):
        return re.sub(r"^model\.", "features.", k)
    return k                                  # facenet checkpoints are native


_VGGFACE = dict(zip(
    ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
     "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3", "fc6", "fc7", "fc8"],
    [f"features.{i}" for i in (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)] +
    [f"classifier.{i}" for i in (0, 3, 6)]))


def remap_state(arch, state):
    """Detect the checkpoint's key layout and rename it to torchvision's."""
    if arch == "vgg16" and "conv1_1.weight" in state:          # VGG-Face (Albanie's conversion)
        return {f"{_VGGFACE[k.rsplit('.', 1)[0]]}.{k.rsplit('.', 1)[1]}": v for k, v in state.items()}
    if not any(k.startswith("model.") for k in state):
        return state
    return {_repo_to_torchvision(arch, k): v for k, v in state.items()}


def randomize(model, seed):
    """Xavier weights, zero biases, reset norms: the random-initialisation rule
    (used by trillium/extract_prep.sh to draw the saved random weights)."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_(); m.running_var.fill_(1.0)
    return model

def trainable_layers(model):
    """Leaf modules that carry learnable parameters (Conv/Linear/Norm/...)."""
    out = []
    for name, mod in model.named_modules():
        if len(list(mod.children())):           # not a leaf
            continue
        if any(p.requires_grad for p in mod.parameters(recurse=False)) or \
           len(list(mod.parameters(recurse=False))) > 0:
            out.append((name, mod))
    return out

def corr_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)

def load_slide(h5_dir, s):
    with h5py.File(h5_dir / f"Slide{s}.h5", "r") as f:
        return np.array(f["images"])            # (18,224,224,3) uint8

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS), help="architecture to build")
    ap.add_argument("--checkpoint", required=True,
                    help="weights file (state_dict); random variants: their saved random weights")
    ap.add_argument("--preproc", required=True, choices=sorted(PREPROCS), help="input preprocessing")
    ap.add_argument("--resize", type=int, default=None,
                    help="resize stimuli to NxN before preprocessing (e.g. 112 for models "
                         "trained at 112; default: keep the native 224)")
    ap.add_argument("--h5-dir", required=True, help="frames from build_stimuli_h5_16frames.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    h5_dir, out_dir = Path(args.h5_dir), Path(args.out_dir)
    manifest_path = h5_dir / "stimulus_manifest.json"
    stim_names = [row["stimulus"] for row in json.loads(manifest_path.read_text())]

    state = remap_state(args.arch, load_checkpoint(args.checkpoint))
    nc = infer_num_classes(state)
    print(f"[head] inferred num_classes={nc} from checkpoint")
    model = ARCHS[args.arch](nc).to(DEVICE)
    miss, unexp = model.load_state_dict(state, strict=False)
    n_ref = len(model.state_dict())
    print(f"[{args.arch}] loaded {args.checkpoint}  "
          f"(matched {n_ref - len(miss)}/{n_ref}, missing {len(miss)}, unexpected {len(unexp)})")
    # A partial load yields a partly random network, whose RDMs are meaningless
    # but look perfectly well-formed.
    if miss or unexp:
        raise SystemExit(
            f"ABORT: checkpoint did not fully match the {args.arch} architecture "
            f"({len(miss)} missing, {len(unexp)} unexpected; e.g. {list(miss)[:3]} {list(unexp)[:3]})")
    model.eval()

    # --- hooks: every leaf with parameters, except the classification head (the
    # last one: a low-rank readout not comparable to the other layers) ---
    layers = trainable_layers(model)
    dropped_head_layer = layers[-1][0]
    layers = layers[:-1]
    print(f"[hooks] excluding head layer '{dropped_head_layer}' from the layer-wise RDMs")
    print(f"[hooks] {len(layers)} trainable layers")
    acts = {}
    handles = []
    for name, mod in layers:
        def mk(k):
            def hook(_m, _i, o):
                o = o[0] if isinstance(o, (tuple, list)) else o
                acts[k] = o.detach().cpu().float().numpy().reshape(o.size(0), -1)
            return hook
        handles.append(mod.register_forward_hook(mk(name)))

    pp = PREPROCS[args.preproc]
    per_slide, order, preds = [], None, []
    with torch.no_grad():
        for s in range(1, 17):
            arr = load_slide(h5_dir, s)
            if args.resize:
                arr = resize_batch(arr, args.resize)
            x = pp(arr).to(DEVICE)
            acts.clear()
            out = model(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
            if order is None:
                order = list(acts.keys())
            per_slide.append(np.stack([corr_distance_rdm(acts[k]) for k in order], 0))
            pr = out.argmax(1).cpu().numpy() if out.ndim == 2 else np.full(len(x), -1)
            if len(stim_names) != len(pr):
                raise RuntimeError(f"manifest has {len(stim_names)} stimuli but slide has {len(pr)}")
            for i, st in enumerate(stim_names):
                preds.append((st, s, int(pr[i])))
            print(f"  slide {s}: {len(order)} layers", flush=True)
    for h in handles:
        h.remove()

    rdms = np.stack(per_slide, 1)               # (L,16,18,18)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.out_name}.npy"
    np.save(out, rdms)
    (out_dir / f"{args.out_name}.layers.json").write_text(json.dumps(
        {"layer_order": order, "shape": list(rdms.shape), "arch": args.arch,
         "checkpoint": args.checkpoint, "preproc": args.preproc, "num_classes": nc,
         "resize": args.resize, "excluded_head_layer": dropped_head_layer,
         "stimuli": stim_names, "stimulus_manifest": str(manifest_path)}, indent=2))
    with (out_dir / f"{args.out_name}.predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["stim", "frame", "pred_idx"]); w.writerows(preds)
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} -> {out}")
    print(f"[ok] predictions -> {args.out_name}.predictions.csv")


if __name__ == "__main__":
    main()
