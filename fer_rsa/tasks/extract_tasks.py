import argparse
import os

from fer_rsa.analysis.variants import CHECKPOINT_STEPS, IDENTITIES, N_DESIGN_REPEATS, N_RUNS, SEEDS, SIZES

REPS = [f"repeat_{r:02d}" for r in range(1, N_RUNS + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("FER_ROOT", os.path.expandvars("$SCRATCH/fer_revision")))
    p.add_argument("--out", default="scripts/trillium/tasks")
    p.add_argument("--features", action="store_true", help="final-layer features for the probe")
    a = p.parse_args()
    R = a.root
    D, W, res = f"{R}/data", f"{R}/data/drive", f"{R}/results"
    # pinned local snapshot (compute nodes are offline)
    qwen_dir = f"{D}/.hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    os.makedirs(a.out, exist_ok=True)
    cnn, qwen = [], []

    flag = " --features" if a.features else ""
    for s in ("18", "54"):
        frames = f"{D}/stim_frames/set{s}"
        rdms = f"{R}/features/set{s}" if a.features else f"{R}/rdms/set{s}"

        def image(name, arch, ckpt, preproc):
            out = f"{rdms}/{name}"
            resize = " --resize 112" if preproc == "half" else ""
            cnn.append(f"{out}\t^\\[ok\\] \\(\tpython -m fer_rsa.models.extract_model_rdms --arch {arch} "
                       f"--checkpoint {ckpt} --preproc {preproc}{resize} "
                       f"--h5-dir {frames} --out-dir {out} --out-name {name}{flag}")

        def m3dfel(name, extra):
            out = f"{rdms}/{name}"
            cnn.append(f"{out}\t^\\[ok\\] \\(\tM3DFEL_R3D={W}/M3DFEL/pretrained/r3d_18-b3b3357e.pth "
                       f"python -m fer_rsa.models.m3dfel_extract_rdms {extra} --frames-dir {frames} "
                       f"--out-dir {out} --out-name {name}{flag}")

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
                if arch == "facenet" and s == "54" and not a.features:    # size curve and step-matched check
                    for size in SIZES:
                        image(f"facenet_size{size}_r{i:02d}", arch,
                              f"{res}/sizecurve/{rep}/{size}/best_weights.pt", "half")
                        if size != SIZES[-1]:
                            image(f"facenet_step{size}_r{i:02d}", arch,
                                  f"{res}/stepmatched/{rep}/{size}/best_weights.pt", "half")

        # FaceNet learning designs: 54 videos only
        for i in range(1, N_DESIGN_REPEATS + 1) if s == "54" and not a.features else ():
            rep = f"repeat_{i:02d}"
            for step in CHECKPOINT_STEPS:
                image(f"facenet_ckpt{step:05d}_r{i:02d}", "facenet",
                      f"{res}/checkpoints/{rep}/checkpoints/step{step:06d}_weights.pt", "half")
            for ids in IDENTITIES:
                image(f"facenet_{ids}_r{i:02d}", "facenet", f"{res}/identities/{rep}/{ids}/best_weights.pt", "half")

        for seed in SEEDS:
            state = f"{W}/random_init_weights/m3dfel/m3dfel_random_seed{seed}.pt"
            if not os.path.exists(state):              # not downloadable: redrawn by extract_prep.sh
                state = f"{D}/random_weights/m3dfel_random_seed{seed}.pt"
            m3dfel(f"m3dfel_random_s{seed}", f"--state {state}")
        m3dfel("m3dfel_kinetics", "")
        m3dfel("m3dfel_dfew", f"--checkpoint {W}/M3DFEL/results/17812056/fold_3/best_weights.pt")

        for name, extra in [("qwen_released", "")] + [(f"qwen_random_s{x}", f"--random --seed {x}") for x in SEEDS]:
            out = f"{rdms}/{name}"
            if a.features:                            # from the RDM run's kept activations
                cnn.append(f"{out}\t^\\[ok\\] \\(\tpython -m fer_rsa.models.compute_rdms_from_vlm_activations "
                           f"--run-dir {R}/rdms/set{s}/{name}/run --out-dir {out} --out-name {name} "
                           f"--manifest {frames}/stimulus_manifest.json --features")
                continue
            qwen.append(f"{out}\t^\\[ok\\] \\(\tpython -m fer_rsa.models.vlm_extract_video_mode "
                        f"--metadata_csv {frames}/qwen_videos.csv --output_dir {out}/run "
                        f"--model_id {qwen_dir} --dtype float32 {extra} "
                        f"&& python -m fer_rsa.models.compute_rdms_from_vlm_activations --run-dir {out}/run "
                        f"--out-dir {out} --out-name {name} --manifest {frames}/stimulus_manifest.json")

    groups = (("extract_features", cnn),) if a.features else (("extract_cnn", cnn), ("extract_qwen", qwen))
    for name, lines in groups:
        with open(f"{a.out}/{name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} extractions")


if __name__ == "__main__":
    main()
