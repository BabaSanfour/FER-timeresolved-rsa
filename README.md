# Emotion-Trained Vision Models Do Not Necessarily Learn EEG-Aligned Facial Dynamics Beyond Pretraining

Code for the paper: model training, stimulus preparation, activation extraction and model
RDMs, EEG RDMs, time-locked EEG–model RSA, cluster permutation tests, nuisance controls,
partial RSA, linear probing, and every table and figure.

Data that cannot go on GitHub (model weights, VGGFace2 subset indexes, EEG and model RDMs,
RSA results) are in a Google Drive folder: **`https://drive.google.com/drive/folders/1w-IQJMyxpl0GaoB_b7Gx0Ho3YRmGENw_?usp=sharing`**. See [Data on Google Drive](#data-on-google-drive).

- [Setup](#setup)
- [Which script gives which result](#which-script-gives-which-result)
- [Full pipeline](#full-pipeline)
- [Data on Google Drive](#data-on-google-drive)
- [Steps without code](#steps-without-code)

## Setup

```bash
git clone https://github.com/MaryemBenslimane/FER-timeresolved-rsa.git
cd FER-timeresolved-rsa
python -m venv $HOME/envs/fer && source $HOME/envs/fer/bin/activate   # Python 3.12
pip install -r requirements.txt
```

Download the Drive folder and merge its contents into the repository root, so that
for example `net_weights/`, `eeg_data/` and `analysis/rsa_grid_54/` sit next to the code.
Its folders use the same relative paths as the scripts. Alternatively, keep the data elsewhere
and set `FER_ROOT` to a directory that contains both the code and the data.

- Run scripts from the repository root. Python scripts read `FER_ROOT` (default `.`).
- The `*.sh` files are SLURM drivers. Set `#SBATCH --account` (`YOUR_ALLOCATION`) and submit
  from the repository root. They activate `$HOME/envs/fer`; set `VENV=/path/to/env` to use
  another environment.
- Compute nodes without internet need the HuggingFace cache for Qwen3-VL-8B-Instruct
  (`HF_HOME=./.hf`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`).
- `torch_cache/checkpoints/` (on Drive) holds facenet-pytorch's VGGFace2 weights; the scripts
  that need it set `TORCH_HOME=$ROOT/torch_cache`.
- `M3DFEL/pretrained/r3d_18-b3b3357e.pth` (on Drive; torchvision's Kinetics-400 R3D-18) is the
  M3DFEL backbone.

**Random CNN weights depend on the hardware.** `models/extract_model_rdms.py` randomizes the
weights after moving the model to the device, and CUDA's random kernels depend on the GPU
partition, so one seed gives different weights on a CPU, a full GPU and a MIG slice. Each set
below reproduces bit-for-bit with this code on the hardware listed:

| Random CNN RDMs | Hardware |
|---|---|
| Table 1 seed ensemble (`analysis/seed_variability/rdms/`, `run_extract_seeds.sh STAGE=cnn`) | H100 MIG 2g.20gb slice |
| 18-stimulus seed 42 (`analysis/model_rdms_canonical/*_random_16f_weights`, `run_random_rdms.sh`) | full H100 |
| 54-stimulus seed 42 (`run_image_models_54_cpu.sh`) | CPU |

M3DFEL and Qwen3-VL draw their weights on the CPU, so their seeds reproduce on any machine. All of
these random models are saved on Drive under `random_init_weights/`.

## Which script gives which result

Paths are relative to the repository root. Every input and output listed here is on Drive, so
each row can be re-run from the downloaded RDMs and RSA grids without retraining or re-extracting.

### Main text

| Result in the paper | Script | Command | Reads | Writes |
|---|---|---|---|---|
| **Table 1** (`tab:main-results`), 18 stimuli, ρ<sub>max</sub> per scalp analysis | `analysis/build_paper_tables.py` | `python analysis/build_maxr_table.py` (trained rows), then `python analysis/build_paper_tables.py` | trained rows: `analysis/rsa_grid/`, `rsa_grid_today/`, `rsa_grid_facenet_original_vggface2_affectnet_16win/` via `analysis/paper_stats/timelocked_maxr_over_layers.csv`; Random rows: `analysis/seed_variability/results/no-ceg_summary.csv` | `analysis/paper_stats/table_main_results.tex` |
| ↳ Random rows (mean ± SD over seeds) | `analysis/seed_variability/` | RDMs: `STAGE=cnn sbatch analysis/seed_variability/run_extract_seeds.sh`, `STAGE=qwen sbatch …`; grids: `python analysis/seed_variability/rsa_grid_seeds.py --cohort no-ceg --model-dir analysis/seed_variability/rdms --out-dir analysis/seed_variability/rsa_no-ceg`; summary: `python analysis/seed_variability/aggregate_seeds.py --seed-grids analysis/seed_variability/rsa_no-ceg --pretrained-grids analysis/seed_variability/rsa_noceg_pretrained --cohort-label no-ceg --out-prefix analysis/seed_variability/results/no-ceg` | `stim_frames/`, EEG RDMs | `analysis/seed_variability/{rdms,rsa_no-ceg,results}/` |
| **Fig. 2**, matched training comparisons (`new_model_comparison.pdf`) | `analysis/seed_variability/plot_model_comparison_seeds.py` | `python analysis/seed_variability/plot_model_comparison_seeds.py --out analysis/seed_variability/results/new_model_comparison` | `analysis/seed_variability/rsa_no-ceg/`, `rsa_noceg_pretrained/`, `analysis/layer_matched/rsa_no-ceg/`; EEG RDMs (lower noise ceiling) | `analysis/seed_variability/results/new_model_comparison.{pdf,png}` |
| ROI-averaged ρ<sub>max</sub> in Sections 4.1 and 4.3 | as Table 1 and the seed table | | | |
| **Fig. 3**, FaceNet training-set size (`facenet_vggface2_size.pdf`) and its values in Section 4.2 | `analysis/summarize_facenet_low_size_repeats.py`, `analysis/plot_facenet_size_curve.py` | `python analysis/summarize_facenet_low_size_repeats.py && python analysis/plot_facenet_size_curve.py` | `analysis/rsa_grid_facenet_size_repeats/` (10 repeats per size; "800k" = `800k_bootstrap`) | `analysis/facenet_low_size_repeats/max_r_facenet_all_sizes.csv`, `facenet_vggface2_size_curve.pdf` |
| 54-stimulus ρ<sub>max</sub> (Section 4.4) | `analysis/build_maxr_table_54.py` | `python analysis/build_maxr_table_54.py` | `analysis/rsa_grid_54/` | `analysis/paper_stats_54/timelocked_maxr_54.csv` |
| Nuisance RDMs and their RSA, 18 stimuli (Section 4.4) | `analysis/compute_{identity,identity_graded,pixel_diff,optical_flow,motion_energy}_rdm.py` → `analysis/build_control_rsa.py` | run each `compute_*` script, then `python analysis/build_control_rsa.py` | `Stims_Videos/` (via `analysis/_stim_loader_v2.py`) | `analysis/motion_controls/{rdms,results}/` |
| Nuisance RDMs and their RSA, 54 stimuli | `analysis/build_control_rdms_54.py` → `analysis/build_control_rsa_54.py` | no arguments | stimulus videos, 54-stimulus EEG RDMs | `analysis/motion_controls_54/{rdms,results}/` |
| Partial RSA (single and joint nuisance controls) | `analysis/partial_rsa.py` | `python analysis/partial_rsa.py --dataset 54` (and `--dataset 18`) | model, EEG and nuisance RDMs | `analysis/partial_rsa_{18,54}/partial_rsa_maxima.csv` |
| Noise ceilings (grey bars in Fig. 2) | `analysis/build_control_rsa.py` | `python analysis/build_control_rsa.py` | 18-stimulus EEG RDMs | `analysis/paper_stats/noise_ceiling_16win.csv`; per window: `analysis/noise_ceiling.py` (`*_perwindow.*`) |
| Cluster-based permutation tests (Section 4.5), both stimulus sets | `analysis/cluster_permutation_rsa_54.py` | `sbatch analysis/submit_cluster_permutation.sh` (array task 0 = 54 stimuli, 1 = 18) | `analysis/rsa_grid*/` | `analysis/cluster_permutation_{18,54}/` (`grid_summary.csv`, `clusters.csv`, `model_survivors.csv`, `*_cluster_masks.npz`) |
| **Fig. 4**, layer × time maps of the Holm-surviving clusters (`layer_time_clusters.pdf`) | `analysis/plot_layer_time_clusters.py` | `python analysis/plot_layer_time_clusters.py` | `analysis/rsa_grid_54/`, `analysis/cluster_permutation_54/` | `analysis/cluster_permutation_54/plots/layer_time_clusters.pdf` |

### Appendix

| Result in the paper | Script | Command | Reads | Writes |
|---|---|---|---|---|
| Scalp ROI definitions | `eeg/roi_definitions.json` | | | |
| Model details: analyzed layers | the `layer_order` of each RDM's `.layers.json` (written by the extraction scripts) | | | |
| Model details: training recipes and accuracies | `vggface_train/train.py`, `affectnet_3class_train/train_affectnet.py`, `M3DFEL/main.py` (see [Training](#3-training)) | `bash trillium/submit_all.sh` | | `results/*/repeat_NN/train.log`, `metrics.csv`, `test_metrics.json`; M3DFEL: `log.txt` |
| Random-initialization variability table (`tab:seeds`) | `analysis/build_paper_tables.py` | as Table 1 | `analysis/seed_variability/results/no-ceg_{summary,per_seed}.csv` | `analysis/paper_stats/table_seeds.tex` |
| Linear probing (18 and 54 videos) | `analysis/evaluate_all_models_linear_54.py`, `analysis/export_predictions_same_models_54.py` | no arguments | `analysis/model_rdms_54/` | `analysis/model_performance_54/complete_54/` |
| Linear-probing figure (`model_accuracy_summary_54.pdf`) | `analysis/plot_accuracy_summary_54.py` | `python analysis/plot_accuracy_summary_54.py` | `analysis/model_performance_54/complete_54/performance_54_same_models.csv` | `analysis/model_performance_54/complete_54/model_accuracy_summary_54.pdf` |

## Full pipeline

### 1. EEG RDMs

| Step | Script | Notes |
|---|---|---|
| 18-stimulus per-ROI RDMs, `(6, 16, 18, 18)` per participant | `eeg/compute_windowed_eeg_rdms_16win.py` | Reads per-participant condition-averaged epochs `eeg_data/Unmasked_avg/*_avg.pkl` (not distributed); writes `eeg_data/Unmasked_avg_windowed_rdms_16win_full/` |
| 18-stimulus whole-scalp RDMs, 54-stimulus per-ROI and whole-scalp RDMs | none | Distributed as data (`eeg_data/Unmasked_avg_wholescalp_16win/`, `eeg_rdms_complete/`) |

The 18-stimulus folders contain 25 participants. The seed-ensemble analyses (Table 1 Random rows,
the seed table, Fig. 2) use the 25 shared with the 54-stimulus set (`--cohort no-ceg`).
`analysis/rsa_full_grid.py`, whose grids give Table 1's trained rows and the 18-stimulus cluster
tests, and `build_control_rsa.py` (noise ceilings) read all 25. 

### 2. Stimuli

| Step | Script |
|---|---|
| 18 videos → 16 frames per video (`stim_frames/Slide1..16.h5`) | `python models/build_stimuli_h5_16frames.py --out-dir stim_frames` |
| 54 videos (unoccluded, mask, sunglasses) → `stim_frames_54/` + `stimulus_manifest.json` | `python models/prepare_stimuli_54.py` |
| Qwen3-VL video list | `models/vlm/stimuli_metadata_canon.csv` (paths relative to the root) |

### 3. Training

All trained models in the paper come from one pipeline, run on Trillium (Digital Research
Alliance of Canada) H100 GPUs; `trillium/` holds the job scripts. Every run has ten seeds
(1001--1010; repeat *r* uses seed 1000+*r*). `FER_ROOT` below is the data root
(`$SCRATCH/fer_revision` on Trillium).

**1. Environment.** `bash trillium/setup_env.sh` (exact versions in `trillium/trainenv_freeze.txt`).

**2. Data** (on a data-transfer node; on Trillium `ssh tri-dm1`, since login nodes cap CPU time):

| Data | Command | Check |
|---|---|---|
| VGGFace2-HQ training archives 01--06 (726,502 images) | `DEST=$FER_ROOT/data/vggface_dataset bash data/get_vggface2_hq.sh` (accept the dataset terms on HuggingFace first) | sizes of the six zips |
| Image lists: full pool and validation | `python vggface_train/build_index.py --root $FER_ROOT/data/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan --out vggface_train/index` | 715,810 train, 10,692 val |
| Image lists: subsets | `python vggface_train/build_low_size_repeated_indexes.py` (10k--200k), `python vggface_train/build_repeated_subset_indexes.py` (300k, 500k; its `800k_bootstrap` lists are not used) | reproduce the lists on Drive byte for byte |
| Decoded-image cache (36 GB) | `python vggface_train/build_decode_cache.py --data-root ... --index-dir vggface_train/index --out-dir $FER_ROOT/data/vggface_cache` (also built by `trillium/bench.sh`) | 200-image spot check |
| AffectNet, 3 classes | `bash data/get_affectnet_3class.sh` (Kaggle token) | archive checksum and per-class counts |
| VGG-Face weights | `bash data/get_vgg_face.sh` | checksum |

**3. Train.** From the repository root on `trillium-gpu`: `bash trillium/submit_all.sh`. It writes the
task lists (`trillium/make_tasks.py`) and submits four job arrays; each array element takes one H100
and runs three trainings on it (four for fine-tuning), and fine-tuning starts after pretraining.
Re-running it skips finished runs and resumes unfinished ones. Outputs go to
`$FER_ROOT/results/<group>/.../repeat_NN/`; the model used downstream is `best_weights.pt`.

| Runs (10 seeds each) | Script and recipe | Output |
|---|---|---|
| Face pretraining: ResNet18, ResNet50, FaceNet on the full pool (715,810 images, 5,346 identities) | `vggface_train/train.py`, 40 epochs; checkpoint = best validation top-1 | `results/pretrain/<arch>/repeat_NN/` |
| Size curve: FaceNet on 10k--500k subsets; the full-pool FaceNets are its end point | as above | `results/sizecurve/repeat_NN/<size>/` |
| Step-matched check: FaceNet on 10k--300k subsets for the full pool's 55,920 steps | as above, `--steps 55920 --eval-every-steps 1398` | `results/stepmatched/repeat_NN/<size>/` |
| AffectNet fine-tuning: ResNet18, ResNet50, FaceNet (from their pretraining `best_weights.pt`) and VGG16 (from VGG-Face) | `affectnet_3class_train/train_affectnet.py`; epoch chosen on a fixed, class-balanced 10% of Train; Test evaluated once (`test_metrics.json`) | `results/finetune/<arch>/repeat_NN/` |
| M3DFEL on DFEW fold 3 (not retrained: DFEW requires a licence) | `M3DFEL/prepare_dfew_3class.sh`, `M3DFEL/train_dfew_3class.sh` (5 folds; fold 3 used) | `M3DFEL/results/17812056/fold_3/best_weights.pt` |

`trillium/bench.sh` measures throughput per model and packing (FaceNet: 9.1k images/s alone,
16.1k with three runs per H100). The 180 runs took about 180 GPU-hours.

The first version of the paper used single runs from `vggface_train/train_vggface2.sh` (completed by
`rescue_17733342.sh` at 18 epochs), `train_facenet_subsets.sh`, `train_facenet_low_size_array.sh`,
`train_facenet_repeated_array.sh` (with an `800k_bootstrap` condition) and
`affectnet_3class_train/train_affectnet_3class.sh` (epoch chosen on the Test folder). These scripts and
their checkpoints on Drive are kept for reference; they are not the models in the paper.

`M3DFEL/` is the Tencent TFace implementation (CVPR 2023) with our changes for 3-class DFEW.
`models/train_vggface2_rgb.py` did not produce any checkpoint used in the paper.

### 4. Model RDMs

Model RDMs are `(n_layers, 16, n_stim, n_stim)` correlation-distance RDMs, one per layer and frame;
`models/build_emotion_rdms.py` / `build_emotion_rdms_54.py` split them per emotion.
`models/extract_model_rdms.py`, `models/m3dfel_extract_rdms.py` and
`models/vlm/vlm_extract_video_mode.py` accept `--save-weights PATH` to save the network they used.

| Variants | 18 stimuli (`analysis/model_rdms_canonical/`) | 54 stimuli (`analysis/model_rdms_54/`) |
|---|---|---|
| Image CNNs: ImageNet, VGGFace2, VGGFace (BN), VGG-Face, random seed 42 | `models/run_all_final.sh` (→ `run_imagenet_rdms.sh`, `run_vggface2_rdms.sh`, `run_random_rdms.sh`) | `models/run_image_models_54_cpu.sh` |
| FaceNet 300k/500k, AffectNet fine-tunes | `models/run_today_models_rdm_rsa.sh` | `models/run_image_models_54_cpu.sh` |
| VGG16 ImageNet (non-BN) | `analysis/run_new_vgg_rdms_rsas.sh` | `analysis/run_new_vgg_rdms_rsas.sh` |
| FaceNet Complete VGGFace2 / AffectNet FT | 379 layers (Table 1, 18-stimulus cluster tests, size curve): `models/run_original_facenet_affectnet_rsa.sh`. 133 layers (Fig. 2): `analysis/layer_matched/run_layer_matched.sh` | `models/run_54_facenet_complete_m3dfel.sh` |
| M3DFEL random / Kinetics / DFEW fold 3 | `models/run_m3dfel.sh`, `models/run_m3dfel_fold3_rdm_rsa.sh` | `models/run_54_facenet_complete_m3dfel.sh` |
| Qwen3-VL-8B pretrained / random (64 layers: 27 vision-encoder blocks, embedding output, 36 decoder blocks) | `models/vlm/run_qwen3vl.sh` | `models/vlm/run_qwen3vl_54_gpu_parallel.sh` (`models/extract_qwen_rdms.py`) |
| Random-init seed ensemble (Table 1) | `analysis/seed_variability/run_extract_seeds.sh` → `analysis/seed_variability/rdms/` | — |
| FaceNet training-set-size repeats | `vggface_train/extract_facenet_low_size_array.sh`, `extract_facenet_repeated_array.sh` (+ `run_facenet_repeated_cpu_fallback.sh` for tasks re-run on CPU) → `analysis/model_rdms_facenet_size_repeats/` | — |


### 5. RSA grids

| | Script | Output |
|---|---|---|
| 18 stimuli, 26 participants | `analysis/rsa_full_grid.py --models <tags> [--model-dir] [--out-dir] [--cohort with-ceg]` (also called by the drivers above, and by `vggface_train/run_facenet_low_size_rsa.sh`, `run_facenet_repeated_rsa_summary.sh`) | `analysis/rsa_grid/`, `rsa_grid_today/`, `rsa_grid_facenet_original_vggface2_affectnet_16win/`, `rsa_grid_facenet_size_repeats/` |
| 18 stimuli, 25 participants (Table 1, Figure 1) | `analysis/seed_variability/rsa_grid_seeds.py --cohort no-ceg` | `analysis/seed_variability/rsa_no-ceg/`, `rsa_noceg_pretrained/`, `analysis/layer_matched/rsa_no-ceg/` |
| 54 stimuli | `analysis/rsa_full_grid_54.py --models <tags>` | `analysis/rsa_grid_54/` |

Each grid file holds the participant × scalp analysis (6 ROIs + whole scalp) × layer × condition
(all, fear, happy, neutral) × EEG window array of Spearman ρ. Statistics, controls and tables are
in [Which script gives which result](#which-script-gives-which-result).

## Data on Google Drive

Folder: **`<DRIVE_LINK>`**. `MANIFEST.tsv` at its root lists every file with its size and SHA-256.

| Folder | Content |
|---|---|
| `net_weights/` | ImageNet weights for ResNet18/50, VGG16, VGG16-BN and FaceNet; grayscale VGG16-BN trained on VGGFace (`vgg16_bn_1D_VGGFace3`); VGG-Face (`vgg16_vggface`); FaceNet Complete-VGGFace2 → AffectNet fine-tune used in Table 1 (`best_inceptionresnetv1_fer.pt`) |
| `vggface_train/results/` | VGGFace2-trained ResNet18/50/FaceNet; FaceNet 300k/500k; all 70 training-set-size checkpoints, with training logs and metrics |
| `vggface_train/index/`, `indexes/`, `indexes_repeated/` | VGGFace2 split (715,810 train / 10,692 val images, 5,346 identities) and every subset index (300k/500k; 10 repeats × 7 sizes, each with `sampling.json`) |
| `affectnet_3class_train/results/17810319/` | AffectNet fine-tuned ResNet18, ResNet50, FaceNet, VGG-Face |
| `M3DFEL/results/17812056/fold_3/`, `M3DFEL/pretrained/` | DFEW fine-tuned M3DFEL, and the Kinetics-400 R3D-18 backbone |
| `torch_cache/checkpoints/` | facenet-pytorch `20180402-114759-vggface2.pt` (FaceNet "Complete VGGFace2") |
| `random_init_weights/` | Every random-initialization model in the paper, each checked to reproduce the paper's RDMs exactly: Table 1 seed ensemble (ResNet18, ResNet50, VGG16-BN, FaceNet; seeds 42, 0–8), 18-stimulus seed-42 CNNs, 54-stimulus seed-42 CNNs, M3DFEL (seeds 42, 0–8), Qwen3-VL-8B (seeds 42, 0–3, HuggingFace format). See `random_init_weights/README_random.txt` |
| `eeg_data/`, `eeg_rdms_complete/` | EEG RDMs: 18 stimuli per ROI and whole scalp; 54 stimuli per ROI and whole scalp (with per-emotion 18 × 18 RDMs) |
| `analysis/model_rdms_*`, `analysis/layer_matched/rdms/`, `analysis/seed_variability/rdms/` | Model RDMs with `.layers.json` provenance |
| `analysis/rsa_grid*/`, `seed_variability/rsa_*/`, `layer_matched/rsa_*/` | RSA grids |
| `analysis/cluster_permutation_{18,54}/` | Cluster permutation tests |
| `analysis/paper_stats*/`, `motion_controls*/`, `partial_rsa_{18,54}/`, `facenet_low_size_repeats/`, `model_performance_54/`, `seed_variability/results/`, `layer_matched/results/` | Noise ceilings, control RDMs and RSA, partial RSA, summary tables, linear separability |

Qwen3-VL-8B-Instruct pretrained weights are not copied: download `Qwen/Qwen3-VL-8B-Instruct` at
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Stimulus videos and frames are not distributed.

## Steps without code

These steps were done by hand or with code that no longer exists. Their outputs are on Drive.

- **EEG preprocessing** in Brain Vision Analyzer 2.1 (re-referencing, 0.1–30 Hz filter, ICA, artifact
  review, epoching, baseline correction), the export to Python, and the trial averaging that wrote
  `*_avg.pkl`. The raw recordings and the `*_avg.pkl` files are not distributed.
- **Behavioral data and analysis** behind the per-stimulus accuracy and reaction-time table and the
  accuracy figures quoted in Section 3.1.
- **The channel selection** for the ROIs (the electrode lists are in `eeg/roi_definitions.json`), the
  **18-stimulus whole-scalp RDMs** and **all 54-stimulus EEG RDMs**.
  The 18-stimulus whole-scalp RDMs and the unoccluded part of the 54-stimulus RDMs agree exactly,
  and the per-ROI RDMs to within 5e-7, so all of them come from the same averaged data.
- **The AffectNet three-class subset** (the Kaggle archive was already reduced).
- **Four checkpoints obtained outside this code**: `net_weights/best_inceptionresnetv1_fer.pt`,
  `vgg16_bn_1D_VGGFace3`, `vgg16_vggface` (VGG-Face conversion) and `FaceNet_imagenet`.
- **The 18-stimulus "VGGFace2 800k" FaceNet** (`facenet_vggface3_16f`): its checkpoint is lost. Its
  RDMs and RSA grids are on Drive, but it cannot be re-extracted.
- **Two untrained heads**: the random 3-unit `logits` layer of FaceNet "Complete VGGFace2" and the
  untrained MIL head (`lstm`, `to_qkv`, `norm`) of Kinetics-pretrained M3DFEL. Their weights were not
  saved, and current code draws different ones (all other layers of these models reproduce exactly).
  The FaceNet `logits` layer is the last layer of the 379-layer grids (Table 1, 18-stimulus cluster
  tests) and of the 54-stimulus grids; the 133-layer grids of Fig. 2 exclude it. Both stimulus sets
  include the M3DFEL head layers. Their RDMs and RSA grids are on Drive.

## License

MIT, see `LICENSE`. `M3DFEL/` is adapted from Tencent TFace and remains under the Apache License 2.0 (`M3DFEL/LICENSE`).
