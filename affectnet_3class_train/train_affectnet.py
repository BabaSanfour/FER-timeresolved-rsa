#!/usr/bin/env python3
"""Fine-tune the face-pretrained image models on 3-class AffectNet (fear, happy, neutral).

    facenet        InceptionResnetV1, our VGGFace2 pretraining
    resnet18/50    torchvision ResNets, our VGGFace2 pretraining
    vgg16_vggface  VGG16 (Parkhi et al., 2015)
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


CLASSES = ["fear", "happy", "neutral"]


class VGG16(nn.Module):
    """VGG16 network."""
    def __init__(self, num_classes=2622):
        super().__init__()
        layers = []
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M"]
        channels = 3
        for value in cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(channels, value, 3, padding=1), nn.ReLU(True)]
                channels = value
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def remap_vgg16_state(state):
    """Map either torchvision-style or MatConvNet-conversion names."""
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(k.startswith("features.") for k in state):
        return state
    conv_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1",
                  "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3",
                  "conv5_1", "conv5_2", "conv5_3"]
    conv_indexes = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    mapped = {}
    for name, idx in zip(conv_names, conv_indexes):
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state:
                mapped[f"features.{idx}.{suffix}"] = state[key]
    for name, idx in (("fc6", 0), ("fc7", 3), ("fc8", 6)):
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state:
                mapped[f"classifier.{idx}.{suffix}"] = state[key]
    return mapped


def checkpoint_state(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    # weights_only: tensors and plain containers only, so a downloaded file
    # (e.g. the public VGG-Face conversion) cannot run code when loaded
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj.get("model", obj.get("state_dict", obj))


def build_model(arch, weights_path):
    state = checkpoint_state(weights_path)
    if arch == "facenet":
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=5346)
        model.load_state_dict(state)
        model.logits = nn.Linear(model.logits.in_features, len(CLASSES))
        return model
    if arch in ("resnet18", "resnet50"):
        model = getattr(models, arch)(weights=None, num_classes=5346)
        model.load_state_dict(state)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    model = VGG16(2622)
    state = remap_vgg16_state(state)
    model.load_state_dict(state)
    model.classifier[6] = nn.Linear(4096, len(CLASSES))
    return model


def make_transforms(arch):
    if arch == "vgg16_vggface":
        # Original VGG16 RGB mean subtraction on [0,255] pixels.
        mean = [129.1863 / 255, 104.7624 / 255, 93.5940 / 255]
        std = [1 / 255, 1 / 255, 1 / 255]
        size, resize = 224, 256
    else:
        # Matches the RGB preprocessing used for our VGGFace2 pretraining.
        mean, std = [0.5] * 3, [0.5] * 3
        size, resize = 112, 128
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(resize), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return train, val


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    loss_sum, total = 0.0, 0
    cm = torch.zeros(3, 3, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, target)
        pred = logits.argmax(1)
        loss_sum += loss.item() * target.numel()
        total += target.numel()
        cm += torch.bincount((target * 3 + pred).cpu(), minlength=9).reshape(3, 3)
    acc = cm.diag().sum().item() / max(total, 1)
    precision = cm.diag() / cm.sum(0).clamp_min(1)
    recall = cm.diag() / cm.sum(1).clamp_min(1)
    macro_f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).mean().item()
    return loss_sum / max(total, 1), acc, macro_f1, cm.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True,
                   choices=["facenet", "vgg16_vggface", "resnet18", "resnet50"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--weights", required=True,
                   help="VGGFace/VGGFace2 pretrained checkpoint for this architecture")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="auto")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--channels-last", action="store_true", help="faster convolutions on H100")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="share of each Train class held out for epoch selection")
    p.add_argument("--split-seed", type=int, default=0,
                   help="fixes the validation split (kept the same across models and seeds)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = make_transforms(args.arch)
    root = Path(args.data_root)
    full_train = datasets.ImageFolder(root / "Train")
    test_set = datasets.ImageFolder(root / "Test", val_tf)
    if full_train.classes != CLASSES or test_set.classes != CLASSES:
        raise ValueError(f"expected classes {CLASSES}; got train={full_train.classes}, test={test_set.classes}")
    # Validation split carved from Train, class-balanced and fixed by --split-seed,
    # so every model and seed sees the same split. The epoch is chosen on it; the
    # Test folder is evaluated once, at the end, for the chosen epoch only.
    rng = random.Random(args.split_seed)
    val_idx = []
    for c in range(len(CLASSES)):
        idx = [i for i, t in enumerate(full_train.targets) if t == c]
        rng.shuffle(idx)
        val_idx += idx[:round(len(idx) * args.val_fraction)]
    val_idx = sorted(val_idx)
    held = set(val_idx)
    train_idx = [i for i in range(len(full_train)) if i not in held]
    train_set = Subset(datasets.ImageFolder(root / "Train", train_tf), train_idx)
    val_set = Subset(datasets.ImageFolder(root / "Train", val_tf), val_idx)
    with open(out / "val_split.txt", "w") as f:
        f.write("\n".join(os.path.relpath(full_train.samples[i][0], root) for i in val_idx) + "\n")
    print(f"train={len(train_set)} val={len(val_set)} (from Train, split seed {args.split_seed}) "
          f"test={len(test_set)}", flush=True)
    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, args.batch_size * 2, shuffle=False, **common)
    model = build_model(args.arch, args.weights).to(device)
    mem_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=mem_format)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch, best_f1, best_epoch = 0, -1.0, -1
    resume = out / "last.pt" if args.resume == "auto" else Path(args.resume)
    if resume.is_file():
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_f1 = ckpt["epoch"] + 1, ckpt["best_macro_f1"]
        best_epoch = ckpt.get("best_epoch", -1)
    metrics_path = out / "metrics.csv"
    if start_epoch == 0:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"])
    started = time.time()
    completed = True
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, seen = 0.0, 0
        for images, target in train_loader:
            images = images.to(device, non_blocking=True, memory_format=mem_format)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = criterion(model(images), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * target.numel()
            seen += target.numel()
        val_loss, acc, macro_f1, cm = evaluate(model, val_loader, device, amp_dtype)
        scheduler.step()
        if macro_f1 > best_f1:
            best_f1, best_epoch = macro_f1, epoch
            atomic_save(model.state_dict(), out / "best_weights.pt")
        state = {"epoch": epoch, "arch": args.arch, "classes": CLASSES,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "best_macro_f1": best_f1, "best_epoch": best_epoch, "confusion_matrix": cm,
                 "args": vars(args)}
        atomic_save(state, out / "last.pt")
        atomic_save(model.state_dict(), out / "last_weights.pt")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss / seen, val_loss, acc, macro_f1,
                                    optimizer.param_groups[0]["lr"]])
        print(f"epoch={epoch+1}/{args.epochs} train_loss={train_loss/seen:.4f} "
              f"val_loss={val_loss:.4f} val_acc={acc:.4f} val_macro_f1={macro_f1:.4f}", flush=True)
        if args.max_hours and (time.time() - started) / 3600 >= args.max_hours and epoch + 1 < args.epochs:
            print("stopping cleanly at max-hours", flush=True)
            completed = False
            break

    if completed:
        # the one Test evaluation: the epoch chosen on validation
        model.load_state_dict(torch.load(out / "best_weights.pt", map_location="cpu", weights_only=True))
        model.to(device, memory_format=mem_format)
        test_loader = DataLoader(test_set, args.batch_size * 2, shuffle=False, **common)
        test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, device, amp_dtype)
        result = {"best_epoch": best_epoch + 1, "val_macro_f1": best_f1, "test_accuracy": test_acc,
                  "test_macro_f1": test_f1, "test_loss": test_loss, "test_confusion_matrix": test_cm,
                  "n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set)}
        with open(out / "test_metrics.json", "w") as f:
            json.dump(result, f, indent=1)
        print(f"TEST best_epoch={best_epoch + 1} accuracy={test_acc:.4f} macro_f1={test_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
