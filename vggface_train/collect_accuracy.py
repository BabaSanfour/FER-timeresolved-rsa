#!/usr/bin/env python3
"""Held-out identity accuracy of the FaceNet learning designs, collected from the training outputs.

  checkpoint_accuracy.csv       run,step,val_top1,val_top5 (%), one row per saved step of each full-pool run
                                (train.py --save-at-steps writes <run>/checkpoints/checkpoints.csv)
  identity_count_accuracy.csv   repeat_NN/<n>ids,best_val_top1 (%), the last best validation top-1 in each
                                identity-subset run's train.log (accuracy on that run's own identities)

Usage
  python vggface_train/collect_accuracy.py --results $SCRATCH/fer_revision/results --out $FER_ROOT/analysis
Expects <results>/checkpoints/repeat_0r/ and <results>/identities/repeat_0r/<n>ids/ (trillium/make_tasks.py).
"""
import argparse
import csv
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="training output root")
    ap.add_argument("--out", type=Path, required=True, help="folder for the two CSV files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted((args.results / "checkpoints").glob("repeat_*")):
        run = int(run_dir.name.split("_")[1])
        with (run_dir / "checkpoints/checkpoints.csv").open() as fh:
            rows += [[run, r["step"], r["val_top1"], r["val_top5"]] for r in csv.DictReader(fh)]
    with (args.out / "checkpoint_accuracy.csv").open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    ids = []
    for log in sorted((args.results / "identities").glob("repeat_*/*ids/train.log")):
        best = re.findall(r"best_val_top1=([0-9.]+)", log.read_text())
        if best:
            ids.append([f"{log.parent.parent.name}/{log.parent.name}", best[-1]])
    with (args.out / "identity_count_accuracy.csv").open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(ids)
    print(f"[ok] {len(rows)} checkpoint rows, {len(ids)} identity runs -> {args.out}")


if __name__ == "__main__":
    main()
