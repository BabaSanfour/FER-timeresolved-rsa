#!/usr/bin/env python3
"""Held-out face-identity accuracy of every face-trained run, from its training outputs.

  identity_accuracy.csv        rdm,accuracy (fraction): full-pool face pretraining
                               (results/pretrain/<arch>/repeat_NN/), the FaceNet size curve and step-matched
                               subsets (results/sizecurve|stepmatched/repeat_NN/<size>/), with the RDM names of
                               analysis/variants.py
  checkpoint_accuracy.csv      run,step,val_top1,val_top5 (%), every saved step of the full-pool runs kept
                               along training (results/checkpoints/repeat_NN/checkpoints/checkpoints.csv,
                               train.py --save-at-steps)
  identity_count_accuracy.csv  repeat_NN/<n>ids,best_val_top1 (%), the identity-subset runs
                               (results/identities/repeat_NN/<n>ids/), accuracy on each run's own identities

best_val_top1 is read from the FINISHED line of train.log: the top-1 accuracy on the validation images,
the criterion the checkpoint was chosen on. Outputs go to $FER_ROOT/analysis/.
"""
import csv
import re

import rsa_core

RESULTS = rsa_core.ROOT / "results"
BEST = re.compile(r"FINISHED \(completed\) best_val_top1=([\d.]+)%")


def best(log):
    m = BEST.findall(log.read_text(errors="ignore")) if log.exists() else []
    return float(m[-1]) / 100 if m else None


def main():
    out = rsa_core.ROOT / "analysis"
    rows = []
    for log in sorted(RESULTS.glob("pretrain/*/repeat_*/train.log")):
        arch, rep = log.parts[-3], int(log.parts[-2].split("_")[1])
        rows.append((f"{arch}_vggface2_r{rep:02d}", best(log)))
    for group, prefix in (("sizecurve", "size"), ("stepmatched", "step")):
        for log in sorted(RESULTS.glob(f"{group}/repeat_*/*/train.log")):
            rep, size = int(log.parts[-3].split("_")[1]), log.parts[-2]
            rows.append((f"facenet_{prefix}{size}_r{rep:02d}", best(log)))
    rows = [(r, a) for r, a in rows if a is not None]
    with (out / "identity_accuracy.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["rdm", "accuracy"]); w.writerows(rows)
    steps = []
    for run_dir in sorted(RESULTS.glob("checkpoints/repeat_*")):
        run = int(run_dir.name.split("_")[1])
        with (run_dir / "checkpoints/checkpoints.csv").open() as fh:
            steps += [[run, r["step"], r["val_top1"], r["val_top5"]] for r in csv.DictReader(fh)]
    with (out / "checkpoint_accuracy.csv").open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(steps)
    counts = []
    for log in sorted(RESULTS.glob("identities/repeat_*/*ids/train.log")):
        m = BEST.findall(log.read_text(errors="ignore"))
        if m:
            counts.append([f"{log.parent.parent.name}/{log.parent.name}", m[-1]])
    with (out / "identity_count_accuracy.csv").open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(counts)
    print(f"[ok] {len(rows)} runs, {len(steps)} checkpoints, {len(counts)} identity-subset runs -> {out}")


if __name__ == "__main__":
    main()
