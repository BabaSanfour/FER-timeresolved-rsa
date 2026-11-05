import csv
import re

from fer_rsa.analysis import rsa_core

RESULTS = rsa_core.ROOT / "results"
BEST = re.compile(r"FINISHED \(completed\) best_val_top1=([\d.]+)%")


def best(log):
    """Last best_val_top1 (%) of a train.log, as written, or None."""
    m = BEST.findall(log.read_text(errors="ignore")) if log.exists() else []
    return m[-1] if m else None


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
    rows = [(r, float(a) / 100) for r, a in rows if a is not None]
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
        b = best(log)
        if b:
            counts.append([f"{log.parent.parent.name}/{log.parent.name}", b])
    with (out / "identity_count_accuracy.csv").open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(counts)
    print(f"[ok] {len(rows)} runs, {len(steps)} checkpoints, {len(counts)} identity-subset runs -> {out}")


if __name__ == "__main__":
    main()
