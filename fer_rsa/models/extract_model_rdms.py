import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES = 16


def _resnet(fn):
    def build(nc):
        m = fn(weights=None)
        if nc:
            m.fc = nn.Linear(m.fc.in_features, nc)
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
    from fer_rsa.models.facenet import InceptionResnetV1 as FaceNetIRV1
    return FaceNetIRV1(num_classes=nc or 2622, n_input_channels=3)

ARCHS = {
    "resnet18": _resnet(tvm.resnet18),
    "resnet50": _resnet(tvm.resnet50),
    "vgg16": _vgg(tvm.vgg16),
    "facenet": _facenet,
}


def _to_nchw(x):
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_imagenet(x):
    x = x.astype(np.float32) / 255.0
    return _to_nchw((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])

def _pp_vggface_rgb(x):
    """VGG-Face RGB mean subtraction on [0, 255] pixels."""
    return _to_nchw(x.astype(np.float32) - np.array([129.1863, 104.7624, 93.5940], np.float32))

def _pp_half(x):                          # as training/train_vggface2.py
    return _to_nchw((x.astype(np.float32) / 255.0 - 0.5) / 0.5)

PREPROCS = {"imagenet": _pp_imagenet, "vggface_rgb": _pp_vggface_rgb, "half": _pp_half}


def resize_batch(x, size):
    from PIL import Image
    return np.stack([np.array(Image.fromarray(im).resize((size, size), Image.BILINEAR))
                     for im in x])


def infer_num_classes(state):
    """Head size: out_features of the last 2-D head weight (else of any 2-D weight)."""
    best = None
    for k, v in state.items():
        if hasattr(v, "ndim") and v.ndim == 2 and any(t in k.lower() for t in ("fc", "classifier", "logits", "head", "linear")):
            best = v.shape[0]
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


# net_weights/ layout: backbone under `model.`, ResNet layerN named blockN, head named `classifier`
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
    """Rename checkpoint keys to torchvision's."""
    if arch == "vgg16" and "conv1_1.weight" in state:          # VGG-Face (Albanie's conversion)
        return {f"{_VGGFACE[k.rsplit('.', 1)[0]]}.{k.rsplit('.', 1)[1]}": v for k, v in state.items()}
    if not any(k.startswith("model.") for k in state):
        return state
    return {_repo_to_torchvision(arch, k): v for k, v in state.items()}


def randomize(model, seed):
    """Random-initialisation rule of the saved random weights (scripts/trillium/extract_prep.sh)."""
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
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
    return model

def trainable_layers(model):
    """Leaf modules with parameters."""
    return [(name, mod) for name, mod in model.named_modules()
            if not len(list(mod.children())) and len(list(mod.parameters(recurse=False))) > 0]

def corr_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)

def load_slide(h5_dir, s):
    with h5py.File(h5_dir / f"Slide{s}.h5", "r") as f:
        return np.array(f["images"])            # (n_stim, 224, 224, 3) uint8

def stimulus_names(manifest_path):
    return [row["stimulus"] for row in json.loads(Path(manifest_path).read_text())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--checkpoint", required=True, help="state dict (random variants: the saved random weights)")
    ap.add_argument("--preproc", required=True, choices=sorted(PREPROCS))
    ap.add_argument("--resize", type=int, default=None, help="resize frames to NxN (default: 224)")
    ap.add_argument("--h5-dir", required=True, help="frames from build_stimuli_h5_16frames.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--features", action="store_true", help="also save the input to the output layer")
    args = ap.parse_args()
    h5_dir, out_dir = Path(args.h5_dir), Path(args.out_dir)
    manifest_path = h5_dir / "stimulus_manifest.json"
    stim_names = stimulus_names(manifest_path)

    state = remap_state(args.arch, load_checkpoint(args.checkpoint))
    nc = infer_num_classes(state)
    print(f"[head] inferred num_classes={nc} from checkpoint")
    model = ARCHS[args.arch](nc).to(DEVICE)
    miss, unexp = model.load_state_dict(state, strict=False)
    n_ref = len(model.state_dict())
    print(f"[{args.arch}] loaded {args.checkpoint}  "
          f"(matched {n_ref - len(miss)}/{n_ref}, missing {len(miss)}, unexpected {len(unexp)})")
    # a partial load leaves random weights that still give well-formed RDMs
    if miss or unexp:
        raise SystemExit(
            f"ABORT: checkpoint did not fully match the {args.arch} architecture "
            f"({len(miss)} missing, {len(unexp)} unexpected; e.g. {list(miss)[:3]} {list(unexp)[:3]})")
    model.eval()

    # every leaf with parameters except the output layer (the last one)
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
    final, per_slide_features = {}, []
    if args.features:
        head = dict(model.named_modules())[dropped_head_layer]
        handles.append(head.register_forward_pre_hook(
            lambda _m, i: final.__setitem__("x", i[0].detach().cpu().float().numpy().reshape(i[0].size(0), -1))))

    pp = PREPROCS[args.preproc]
    per_slide, order, preds = [], None, []
    with torch.no_grad():
        for s in range(1, N_FRAMES + 1):
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
            if args.features:
                per_slide_features.append(final["x"])
            pr = out.argmax(1).cpu().numpy() if out.ndim == 2 else np.full(len(x), -1)
            if len(stim_names) != len(pr):
                raise RuntimeError(f"manifest has {len(stim_names)} stimuli but slide has {len(pr)}")
            for i, st in enumerate(stim_names):
                preds.append((st, s, int(pr[i])))
            print(f"  slide {s}: {len(order)} layers", flush=True)
    for h in handles:
        h.remove()

    rdms = np.stack(per_slide, 1)               # (L, 16, n_stim, n_stim)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.out_name}.npy"
    np.save(out, rdms)
    if args.features:
        np.save(out_dir / f"{args.out_name}.features.npy", np.stack(per_slide_features, 1).astype(np.float32))
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
