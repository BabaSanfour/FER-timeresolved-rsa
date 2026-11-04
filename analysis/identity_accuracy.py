#!/usr/bin/env python3
"""Face-identity validation accuracy of every face-trained run, from its training log.

Reads $FER_ROOT/results/pretrain/<arch>/repeat_NN/train.log (full pool),
results/sizecurve/repeat_NN/<size>/train.log and results/stepmatched/repeat_NN/<size>/train.log
(FaceNet), and takes best_val_top1 from the FINISHED line: the top-1 identity accuracy on the
validation images, the criterion the checkpoint was chosen on.

Output: analysis/identity_accuracy.csv (rdm, accuracy), with the RDM names of analysis/variants.py.
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
    rows = []
    for log in sorted(RESULTS.glob("pretrain/*/repeat_*/train.log")):
        arch, rep = log.parts[-3], int(log.parts[-2].split("_")[1])
        rows.append((f"{arch}_vggface2_r{rep:02d}", best(log)))
    for group, prefix in (("sizecurve", "size"), ("stepmatched", "step")):
        for log in sorted(RESULTS.glob(f"{group}/repeat_*/*/train.log")):
            rep, size = int(log.parts[-3].split("_")[1]), log.parts[-2]
            rows.append((f"facenet_{prefix}{size}_r{rep:02d}", best(log)))
    rows = [(r, a) for r, a in rows if a is not None]
    out = rsa_core.ROOT / "analysis/identity_accuracy.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["rdm", "accuracy"]); w.writerows(rows)
    print(f"[ok] {len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
