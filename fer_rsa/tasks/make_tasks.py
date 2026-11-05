import argparse
import os

from fer_rsa.analysis.variants import CHECKPOINT_STEPS, IDENTITIES, N_DESIGN_REPEATS, N_RUNS, SIZES

STEPS_PER_EPOCH = 1398  # 715,810 // 512
FULL_POOL_STEPS = STEPS_PER_EPOCH * 40
assert CHECKPOINT_STEPS[-1] == FULL_POOL_STEPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("FER_ROOT", os.path.expandvars("$SCRATCH/fer_revision")))
    p.add_argument("--out", default="scripts/trillium/tasks")
    a = p.parse_args()
    repo, data = f"{a.root}/repo", f"{a.root}/data"
    vgg = f"{data}/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan"
    res = f"{a.root}/results"
    os.makedirs(a.out, exist_ok=True)

    def pretrain(out, arch, index, seed, extra=""):
        cmd = (f"python -m fer_rsa.training.train_vggface2 --arch {arch} --data-root {vgg} "
               f"--index-dir {index} --out-dir {out} --batch-size 512 --workers {{W}} "
               f"--seed {seed} --cache-dir {{C}} --max-hours {{H}} {extra}").strip()
        return f"{out}\tFINISHED \\(completed\\)\t{cmd}"

    groups = {"pretrain": [], "sizecurve": [], "stepmatched": [], "finetune": [], "checkpoints": [],
              "identities": []}
    for r in range(1, N_DESIGN_REPEATS + 1):
        rep = f"repeat_{r:02d}"
        # full pool, seed of pretraining run r, weights kept along training
        groups["checkpoints"].append(pretrain(
            f"{res}/checkpoints/{rep}", "facenet", f"{repo}/vggface_train/index", 1000 + r,
            f"--epochs 40 --save-at-steps {','.join(map(str, CHECKPOINT_STEPS))}"))
        # 100 images per identity, full pool's step count
        for ids in IDENTITIES:
            groups["identities"].append(pretrain(
                f"{res}/identities/{rep}/{ids}", "facenet",
                f"{repo}/vggface_train/indexes_identities/{rep}/{ids}", 1000 + r,
                f"--steps {FULL_POOL_STEPS} --eval-every-steps {STEPS_PER_EPOCH}"))
    for r in range(1, N_RUNS + 1):
        rep = f"repeat_{r:02d}"
        for arch in ["facenet", "resnet18", "resnet50"]:
            groups["pretrain"].append(pretrain(f"{res}/pretrain/{arch}/{rep}", arch,
                                               f"{repo}/vggface_train/index", 1000 + r, "--epochs 40"))
        for size in SIZES:
            groups["sizecurve"].append(pretrain(f"{res}/sizecurve/{rep}/{size}", "facenet",
                                                f"{repo}/vggface_train/indexes_repeated/{rep}/{size}",
                                                1000 + r, "--epochs 40"))
        for size in SIZES[:-1]:
            groups["stepmatched"].append(pretrain(
                f"{res}/stepmatched/{rep}/{size}", "facenet",
                f"{repo}/vggface_train/indexes_repeated/{rep}/{size}", 1000 + r,
                f"--steps {FULL_POOL_STEPS} --eval-every-steps {STEPS_PER_EPOCH}"))
        for arch in ["facenet", "resnet18", "resnet50", "vgg16_vggface"]:
            start = (f"{data}/net_weights/vgg_face_dag.pth" if arch == "vgg16_vggface"
                     else f"{res}/pretrain/{arch}/{rep}/best_weights.pt")
            bs = 64 if arch == "vgg16_vggface" else 256
            out = f"{res}/finetune/{arch}/{rep}"
            cmd = (f"python -m fer_rsa.training.train_affectnet --arch {arch} "
                   f"--data-root {data}/affectnet_kaggle/affectnet_3class --out-dir {out} "
                   f"--weights {start} --epochs 30 --batch-size {bs} --workers {{W}} "
                   f"--seed {1000 + r} --channels-last --max-hours {{H}}")
            if arch != "vgg16_vggface":
                # never fine-tune from a pretraining run that stopped early
                log = f"{res}/pretrain/{arch}/{rep}/train.log"
                cmd = (f"grep -q 'FINISHED (completed)' {log} || "
                       f"{{ echo 'pretraining not complete: {log}'; exit 1; }}; {cmd}")
            groups["finetune"].append(f"{out}\t^TEST \t{cmd}")

    for name, lines in groups.items():
        with open(f"{a.out}/{name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} runs")


if __name__ == "__main__":
    main()
