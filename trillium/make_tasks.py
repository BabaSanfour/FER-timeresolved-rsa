#!/usr/bin/env python3
"""Write the task lists for the revision's retraining on Trillium (one line per run).

Recipe changes from the original runs (decided 2026-10-25):
  - every VGGFace2-HQ model trains for 40 epochs (the full-pool models had 18, a
    time-budget choice in rescue_17733342.sh; the subsets had 40);
  - the size curve ends at the full 715,810-image pool trained the same way,
    10 repeats, instead of the public facenet-pytorch checkpoint; the
    "800k_bootstrap" condition (draws with replacement) is dropped;
  - step-matched check: subsets 10k-300k trained for the full pool's step count;
  - AffectNet fine-tunes keep the final epoch (no selection on the Test folder)
    and start from our own 40-epoch VGGFace2 models (VGG16 from VGG-Face);
  - 3 seeds for every trained variant (seed 1000 + repeat, as in the size curve).

Line format (tab-separated): out_dir, done-pattern for out_dir/train.log, command.
{W} in the command is replaced by the data-loader worker count at launch and
{H} by the hours left in the job.
"""
import argparse
import os

SIZES = ["10k", "50k", "100k", "200k", "300k", "500k"]
FULL_POOL_STEPS = 1398 * 40  # 715,810 // 512 steps per epoch x 40 epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expandvars("$SCRATCH/fer_revision"))
    p.add_argument("--out", default="trillium/tasks")
    a = p.parse_args()
    repo, data = f"{a.root}/repo", f"{a.root}/data"
    vgg = f"{data}/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan"
    cache = f"{data}/vggface_cache"
    res = f"{a.root}/results"
    os.makedirs(a.out, exist_ok=True)

    def pretrain(out, arch, index, seed, extra=""):
        cmd = (f"python {repo}/vggface_train/train.py --arch {arch} --data-root {vgg} "
               f"--index-dir {index} --out-dir {out} --batch-size 512 --workers {{W}} "
               f"--seed {seed} --cache-dir {cache} --max-hours {{H}} {extra}").strip()
        return f"{out}\tFINISHED \\(completed\\)\t{cmd}"

    groups = {"pretrain": [], "sizecurve": [], "stepmatched": [], "finetune": []}
    for r in range(1, 11):
        rep = f"repeat_{r:02d}"
        archs = ["facenet"] + (["resnet18", "resnet50"] if r <= 3 else [])
        for arch in archs:
            groups["pretrain"].append(pretrain(f"{res}/pretrain/{arch}/{rep}", arch,
                                               f"{repo}/vggface_train/index", 1000 + r, "--epochs 40"))
        for size in SIZES:
            groups["sizecurve"].append(pretrain(f"{res}/sizecurve/{rep}/{size}", "facenet",
                                                f"{repo}/vggface_train/indexes_repeated/{rep}/{size}",
                                                1000 + r, "--epochs 40"))
        if r <= 3:
            for size in SIZES[:-1]:
                groups["stepmatched"].append(pretrain(
                    f"{res}/stepmatched/{rep}/{size}", "facenet",
                    f"{repo}/vggface_train/indexes_repeated/{rep}/{size}", 1000 + r,
                    f"--steps {FULL_POOL_STEPS} --eval-every-steps 1398"))
            for arch in ["facenet", "resnet18", "resnet50", "vgg16_vggface"]:
                start = (f"{data}/net_weights/vgg_face_dag.pth" if arch == "vgg16_vggface"
                         else f"{res}/pretrain/{arch}/{rep}/last_weights.pt")
                bs = 64 if arch == "vgg16_vggface" else 256
                out = f"{res}/finetune/{arch}/{rep}"
                cmd = (f"python {repo}/affectnet_3class_train/train_affectnet.py --arch {arch} "
                       f"--data-root {data}/affectnet_kaggle/affectnet_3class --out-dir {out} "
                       f"--weights {start} --epochs 30 --batch-size {bs} --workers {{W}} "
                       f"--seed {1000 + r} --channels-last --max-hours {{H}}")
                if arch != "vgg16_vggface":
                    # never fine-tune from a pretraining run that stopped early
                    log = f"{res}/pretrain/{arch}/{rep}/train.log"
                    cmd = (f"grep -q 'FINISHED (completed)' {log} || "
                           f"{{ echo 'pretraining not complete: {log}'; exit 1; }}; {cmd}")
                groups["finetune"].append(f"{out}\tepoch=30/30\t{cmd}")

    for name, lines in groups.items():
        with open(f"{a.out}/{name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} runs")


if __name__ == "__main__":
    main()
