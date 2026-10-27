#!/usr/bin/env python3
"""Task lists for the RDM extraction on Trillium (same line format as make_tasks.py).

Every variant, both stimulus sets (the size curve and step-matched check: 18 videos only).
Layers: every layer with learned parameters except the output layer; M3DFEL: backbone
convolutions and batch norms; Qwen3-VL: its 64 blocks. Random variants load saved weights.
"""
import argparse
import os

SIZES = ["10k", "50k", "100k", "200k", "300k", "500k"]
SEEDS = [42] + list(range(9))
REPS = [f"repeat_{r:02d}" for r in range(1, 11)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expandvars("$SCRATCH/fer_revision"))
    p.add_argument("--out", default="trillium/tasks")
    a = p.parse_args()
    R, repo = a.root, f"{a.root}/repo"
    D, W, res = f"{R}/data", f"{R}/data/drive", f"{R}/results"
    # local snapshot at the pinned revision (compute nodes are offline)
    qwen_dir = f"{D}/.hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    os.makedirs(a.out, exist_ok=True)
    cnn, qwen = [], []

    for s in ("18", "54"):
        frames, rdms = f"{D}/stim_frames/set{s}", f"{R}/rdms/set{s}"

        def image(name, arch, ckpt, preproc):
            out = f"{rdms}/{name}"
            resize = " --resize 112" if preproc == "half" else ""
            cnn.append(f"{out}\t^\\[ok\\] \\(\tpython {repo}/models/extract_model_rdms.py --arch {arch} "
                       f"--checkpoint {ckpt} --preproc {preproc}{resize} --layers trainable --exclude-head "
                       f"--h5-dir {frames} --out-dir {out} --out-name {name}")

        def m3dfel(name, extra):
            out = f"{rdms}/{name}"
            cnn.append(f"{out}\t^\\[ok\\] \\(\tM3DFEL_R3D={W}/M3DFEL/pretrained/r3d_18-b3b3357e.pth "
                       f"python {repo}/models/m3dfel_extract_rdms.py {extra} --frames-dir {frames} "
                       f"--out-dir {out} --out-name {name}")

        for arch in ("resnet18", "resnet50", "facenet", "vgg16"):
            for seed in SEEDS:
                if arch == "vgg16":
                    ckpt = (f"{W}/random_init_weights/54stim_seed42_cpu/vgg16_random_seed42_cpu.pt" if seed == 42
                            else f"{D}/random_weights/vgg16_random_seed{seed}.pt")
                else:
                    ckpt = f"{W}/random_init_weights/18stim_seed_ensemble_mig2g20gb/{arch}_random_seed{seed}.pt"
                image(f"{arch}_random_s{seed}", arch, ckpt, "imagenet")
            imagenet = {"resnet18": f"{D}/net_weights/her/resnet18_imagenet",
                        "resnet50": f"{D}/net_weights/her/resnet50_imagenet",
                        "vgg16": f"{D}/net_weights/her/vgg16_imagenet",
                        "facenet": f"{W}/net_weights/FaceNet_imagenet"}[arch]
            image(f"{arch}_imagenet", arch, imagenet, "imagenet")
            if arch == "vgg16":
                image("vgg16_vggface", arch, f"{D}/net_weights/vgg_face_dag.pth", "vggface_rgb")
            for i, rep in enumerate(REPS, 1):
                if arch == "vgg16":
                    image(f"vgg16_affectnet_r{i:02d}", arch,
                          f"{res}/finetune/vgg16_vggface/{rep}/best_weights.pt", "vggface_rgb")
                else:
                    image(f"{arch}_vggface2_r{i:02d}", arch, f"{res}/pretrain/{arch}/{rep}/best_weights.pt", "half")
                    image(f"{arch}_affectnet_r{i:02d}", arch, f"{res}/finetune/{arch}/{rep}/best_weights.pt", "half")
                if arch == "facenet" and s == "18":
                    for size in SIZES:
                        image(f"facenet_size{size}_r{i:02d}", arch,
                              f"{res}/sizecurve/{rep}/{size}/best_weights.pt", "half")
                        if size != "500k":
                            image(f"facenet_step{size}_r{i:02d}", arch,
                                  f"{res}/stepmatched/{rep}/{size}/best_weights.pt", "half")

        for seed in SEEDS:
            state = f"{W}/random_init_weights/m3dfel/m3dfel_random_seed{seed}.pt"
            if not os.path.exists(state):              # not downloadable: redrawn by extract_prep.sh
                state = f"{D}/random_weights/m3dfel_random_seed{seed}.pt"
            m3dfel(f"m3dfel_random_s{seed}", f"--state {state}")
        m3dfel("m3dfel_kinetics", "")
        m3dfel("m3dfel_dfew", f"--checkpoint {W}/M3DFEL/results/17812056/fold_3/best_weights.pt")

        for name, extra in [("qwen_released", "")] + [(f"qwen_random_s{x}", f"--random --seed {x}") for x in SEEDS]:
            out = f"{rdms}/{name}"
            qwen.append(f"{out}\t^\\[ok\\] \\(\tpython {repo}/models/vlm/vlm_extract_video_mode.py "
                        f"--metadata_csv {frames}/qwen_videos.csv --output_dir {out}/run "
                        f"--model_id {qwen_dir} --dtype float32 --frame-list --nframes 16 {extra} "
                        f"&& python {repo}/models/vlm/compute_rdms_from_vlm_activations.py --run-dir {out}/run "
                        f"--out-dir {out} --out-name {name} --manifest {frames}/stimulus_manifest.json")

    for name, lines in (("extract_cnn", cnn), ("extract_qwen", qwen)):
        with open(f"{a.out}/{name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} extractions")


if __name__ == "__main__":
    main()
