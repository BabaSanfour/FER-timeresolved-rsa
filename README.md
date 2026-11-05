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
pip install .                       # analysis (pyproject.toml)
# training and RDM extraction (GPU) also need:
pip install ".[models]" && pip install --no-deps facenet-pytorch==2.6.0
# and OpenCV: module load opencv/4.11.0 on Alliance clusters, else pip install opencv-python-headless
```

Download the Drive folder and merge its contents into the repository root, so that
for example `net_weights/`, `eeg_rdms/` and `analysis/rsa_grid_54/` sit next to the code.
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

**Random weights are loaded, not redrawn.** Random variants use the saved weights on Drive
(`random_init_weights/`: ResNet18, ResNet50, FaceNet and M3DFEL, seeds 42 and 0--8).
`trillium/extract_prep.sh` draws the plain-VGG16 seeds 0--8 and any missing M3DFEL seed on the CPU,
after checking that redrawing the existing seeds reproduces those files bit for bit. Random
Qwen3-VL models are drawn from its default initializer (seeds 42 and 0--8).

## Which script gives which result

Paths are relative to the repository root. Every input and output listed here is on Drive, so
each row can be re-run from the downloaded RDMs and RSA grids without retraining or re-extracting.

### Main text

| Result in the paper | Script | Command | Reads | Writes |
|---|---|---|---|---|
| **Fig. 1**, study design (`fig1_study_design.pdf`) | `analysis/figures/fig1_study_design.py` | `python analysis/figures/fig1_study_design.py --stimuli <dir with set18/ and set54/> --out <path>` | verified stimulus videos, `eeg/roi_definitions.json`; panels d–f are schematic (synthetic matrices) | `<path>.pdf`, `<path>.png` |
| **Table 1** (`tab:main-results`), 18 stimuli, ρ<sub>max</sub> per scalp analysis, mean ± SD over 10 seeds or training runs | `analysis/build_paper_tables.py` | `python analysis/rsa_grid.py --stimuli 18` (every RDM of `analysis/variants.py`), then `python analysis/maxr_table.py --stimuli 18`, then `python analysis/build_paper_tables.py` | `rdms/set18/`, EEG RDMs | `analysis/paper_stats/rho_max.csv`, `rho_max_per_rdm.csv`, `table_main_results.tex` |
| **Fig. 2**, what training contributes: the planned contrasts H1–H4 corrected for layer-selection bias (per family, pooled with Holm p), and held-out emotion decoding against rho_max above chance | `analysis/fig2_training.py` | after `contrasts.py`, `rho_max_null.py` and `probe.py` (54 videos): `python analysis/fig2_training.py` | `analysis/paper_stats_54/contrasts.csv`, `rho_max_null.csv`, `probe.csv` | `analysis/paper_stats_54/fig2_training.{pdf,png}` |
| ROI-averaged ρ<sub>max</sub> in Sections 4.1 and 4.3 | as Table 1 and the seed table | | | |
| 54-stimulus ρ<sub>max</sub> (Section 4.4) | `analysis/maxr_table.py` | `python analysis/rsa_grid.py --stimuli 54`, then `python analysis/maxr_table.py --stimuli 54` | `rdms/set54/`, EEG RDMs | `analysis/paper_stats_54/rho_max.csv`, `rho_max_per_rdm.csv` |
| Nuisance controls and their RSA (Section 4.4): identity, occlusion (54 videos), pixel (per frame), optical flow, motion energy | `analysis/control_rdms.py` → `analysis/rsa_grid.py` → `analysis/maxr_table.py` | `python analysis/control_rdms.py --stimuli 18` (and `54`), then as Table 1 (controls are one-layer RDMs) | `data/stim_frames/set18\|set54/` (the models' frames) | `rdms/set*/control_*/`; group `control` of `analysis/paper_stats{,_54}/rho_max.csv` |
| Partial RSA (each control alone, and jointly; window k uses the controls of frame k) | `analysis/partial_rsa.py` → `analysis/maxr_table.py --partial <set>` | `python analysis/partial_rsa.py --stimuli 18` (and `54`), then `python analysis/maxr_table.py --stimuli 18 --partial joint` | model, EEG and control RDMs | `analysis/partial_rsa{,_54}/<set>/`, `analysis/paper_stats{,_54}/rho_max_partial_<set>.csv` |
| Noise ceilings (rank-based, Nili et al., 2014), averaged over windows | `analysis/maxr_table.py` (`rsa_core.noise_ceilings`) | as Table 1 | EEG RDMs | `analysis/paper_stats{,_54}/noise_ceilings.csv` |
| Linear probe of the emotion label on final-layer features (Appendix): leave one actor out, within-actor permutation null; face-identity accuracy of the face-trained runs | `analysis/probe.py`, `analysis/identity_accuracy.py` | `FEATURES=1 trillium/submit_extract.sh`, then `python analysis/probe.py --stimuli 54 --jobs 8`; `python analysis/identity_accuracy.py` | `features/set*/`, `results/*/train.log` | `analysis/paper_stats{,_54}/probe.csv`, `probe_per_rdm.csv`; `analysis/identity_accuracy.csv` |
| rho_max against its own chance level (models vs nuisance controls): stimulus labels permuted within each emotion, 1,000 times; excess over the null mean and permutation p | `analysis/rho_max_null.py` | `python analysis/rho_max_null.py --stimuli 18` (and `54`) | `rdms/set*/`, EEG RDMs | `analysis/paper_stats{,_54}/rho_max_null.csv` |
| Comparisons between training regimes (Sections 4.1–4.3): four planned tests (pretrained vs random, face vs ImageNet, fine-tuned vs start, video vs image) and each family's contrast, in every ROI and on the whole scalp, each against its own permutation null (labels within emotion, 1,000 times; corrects the layer-selection bias); maxT across the 11 ROIs, Holm across the four whole-scalp tests; the H1 time course per ROI × window (maxT across cells); `--partial joint` for the controlled grids | `analysis/contrasts.py` | `python analysis/contrasts.py --stimuli 18` (and `54`; add `--partial <set>`) | `analysis/rsa_grid{,_54}/` or `partial_rsa{,_54}/<set>/`, `rdms/set*/`, EEG RDMs | `analysis/paper_stats{,_54}/contrasts[_<set>].csv`, `contrasts_time[_<set>].csv` |
| Cluster-based permutation tests (Section 4.5), both stimulus sets: one test per variant on its grid averaged over its 10 seeds or runs, 20,000 sign flips, corrected across all grids of all variants by the maximum statistic (each grid's maximum cluster mass standardised by its null) | `analysis/cluster_permutation.py` | `sbatch analysis/submit_cluster_permutation.sh` (array task 0 = 54 stimuli, 1 = 18); on the joint partial grids: `python analysis/cluster_permutation.py --study 54 --partial joint` | `analysis/rsa_grid{,_54}/` | `analysis/cluster_permutation_{18,54}/` (`grid_summary.csv`, `clusters.csv`, `model_survivors.csv`, `method.json`) |
| **Fig. 3**, where and when alignment survives the study-wide test: layer × time map of each surviving grid with its clusters, and the time course at its best layer against random initialisation | `analysis/fig3_time.py` | after `cluster_permutation.py --study 54`: `python analysis/fig3_time.py` | `analysis/rsa_grid_54/`, `analysis/cluster_permutation_54/` | `analysis/paper_stats_54/fig3_time.{pdf,png}` |

### Appendix

| Result in the paper | Script | Command | Reads | Writes |
|---|---|---|---|---|
| Scalp ROI definitions | `eeg/roi_definitions.json` | | | |
| Model details: analyzed layers | the `layer_order` of each RDM's `.layers.json` (written by the extraction scripts) | | | |
| Model details: training recipes and accuracies | `vggface_train/train.py`, `affectnet_3class_train/train_affectnet.py`, `M3DFEL/main.py` (see [Training](#3-training)) | `bash trillium/submit_all.sh` | | `results/*/repeat_NN/train.log`, `metrics.csv`, `test_metrics.json`; M3DFEL: `log.txt` |
| Seed and training-run variability table (`tab:seeds`): each regime of each family, mean ± SD [range] | `analysis/build_paper_tables.py` | as Table 1 | `analysis/paper_stats/rho_max.csv` | `analysis/paper_stats/table_seeds.tex` |
| 54-video counterpart of Table 1 (appendix) | `analysis/build_paper_tables.py` | as Table 1, with `--stimuli 54` for `rsa_grid.py` and `maxr_table.py` | `analysis/paper_stats_54/rho_max.csv` | `analysis/paper_stats_54/table_main_results_54.tex` |
| Nuisance-control table, rho_max of each control RDM per scalp analysis (appendix) | `analysis/build_paper_tables.py` | after `analysis/control_rdms.py`, as Table 1 | `analysis/paper_stats{,_54}/rho_max.csv` | `analysis/paper_stats{,_54}/table_controls{,_54}.tex` |
| Partial-RSA table, ROI- and emotion-averaged rho_max raw and after each control and all jointly (appendix) | `analysis/build_paper_tables.py` | `python analysis/partial_rsa.py --stimuli 18` (and `54`), `python analysis/maxr_table.py --stimuli 18 --partial <set>` for every set, then `python analysis/build_paper_tables.py` | `analysis/paper_stats{,_54}/rho_max_partial_*.csv` | `analysis/paper_stats{,_54}/table_partial{,_54}.tex` |

## Full pipeline

### 1. EEG RDMs

| Step | Script | Notes |
|---|---|---|
| Trial labels, artefact handling, video averages, RDMs `(12, 13, 54, 54)` per participant | `eeg/build_eeg_rdms.py [--subjects ...]` | Reads `$FER_ROOT/eeg_raw/` (Brain Vision Analyzer segment exports and `behavior_RT_perf_identity_occlusion.csv`); writes `eeg_rdms/<subject>.npy` and `eeg_rdms/meta.json` |

Each trial's video comes from the behaviour file; its sequence is checked against the block's trigger
codes (which name groups of videos), allowing only trials missing at either end of a recording. Per
participant: baseline -200 to 0 ms; channels above ±100 µV in more than 20% of trials are interpolated
(spherical splines); trials above ±150 µV are rejected; the rest are averaged per video. RDMs are
correlation distances between video averages over the channels of a scope × the samples of a window:
11 ROIs (`eeg/roi_definitions.json`) and the whole scalp; 13 windows of 64 samples (62.5 ms) from
onset, the last holding the 51 samples to 800 ms. Participants with fewer than half of the 972 trials
are excluded (CEG, one block). AA has no Markers file: its blocks are labelled by order, and block 6
(126 of 162 segments) is dropped. Both stimulus sets use the same 25 participants; the 18 stimuli are
the unoccluded videos (`rsa_core.EEG_ROWS`).

### 2. Stimuli

`bash trillium/extract_prep.sh` builds, for the 18- and 54-video sets (`data/stimuli/set18`, `set54`):
16 frames per video with one sampler (`models/build_stimuli_h5_16frames.py`: container frame count,
`linspace`, seek, 224 x 224), `stimulus_manifest.json` (order, file, sampled frame indices) and the
Qwen3-VL video list `qwen_videos.csv`, in `data/stim_frames/set18|set54/`.

### 3. Training

All trained models in the paper come from one pipeline, run on Trillium (Digital Research
Alliance of Canada) H100 GPUs; `trillium/` holds the job scripts. Every run has ten seeds
(1001--1010; repeat *r* uses seed 1000+*r*). `FER_ROOT` below is the data root
(`$SCRATCH/fer_revision` on Trillium).

**1. Environment.** `bash trillium/setup_env.sh` (dependencies in `pyproject.toml`, extra `models`;
exact versions in `trillium/trainenv_freeze.txt`).

**2. Data** (on a data-transfer node; on Trillium `ssh tri-dm1`, since login nodes cap CPU time):

| Data | Command | Check |
|---|---|---|
| VGGFace2-HQ training archives 01--06 (726,502 images) | `DEST=$FER_ROOT/data/vggface_dataset bash data/get_vggface2_hq.sh` (accept the dataset terms on HuggingFace first) | sizes of the six zips |
| Image lists: full pool and validation | `python vggface_train/build_index.py --root $FER_ROOT/data/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan --out vggface_train/index` | 715,810 train, 10,692 val |
| Image lists: subsets | `python vggface_train/build_subset_indexes.py --design sizes` (10k--500k, 10 repeats) and `--design ids` (50--3,500 identities x 100 images, 3 repeats) | reproduce the lists on Drive byte for byte |
| Decoded-image cache (36 GB) | `python vggface_train/build_decode_cache.py --data-root ... --index-dir vggface_train/index --out-dir $FER_ROOT/data/vggface_cache` | 200-image spot check |
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

Packing was chosen from a throughput benchmark (FaceNet: 9.1k images/s alone, 16.1k with three
runs per H100). The 180 runs took about 180 GPU-hours.

The first version of the paper used single training runs (18-epoch full-pool models, a bootstrap
"800k" subset, AffectNet epochs chosen on the Test folder); their scripts are in the git history and
their checkpoints on Drive, but they are not the models in the paper.

`M3DFEL/` is the Tencent TFace implementation (CVPR 2023) with our changes for 3-class DFEW.

### 4. Model RDMs

Model RDMs are `(n_layers, 16, n_stim, n_stim)` correlation-distance RDMs, one per layer and frame
(the RSA scripts slice the emotions themselves).
From the repository root on `trillium-gpu`: `bash trillium/extract_prep.sh` once, then
`bash trillium/submit_extract.sh` (386 extractions; re-running skips finished ones). Outputs:
`$FER_ROOT/rdms/set18|set54/<name>/<name>.npy` with `.layers.json` (layers, weights, preprocessing,
stimuli).

| Family (layers) | Variants | Extractor |
|---|---|---|
| ResNet18 (40), ResNet50 (106), FaceNet (245), VGG16 (15): every layer with parameters except the output layer | random (10 seeds), ImageNet, face (VGGFace2-HQ, 10 runs; VGG16: VGG-Face), AffectNet (10 runs); FaceNet learning designs: size curve, step-matched check, checkpoints along training, identity counts (54 videos) | `models/extract_model_rdms.py` |
| M3DFEL (40): R3D-18 backbone convolutions and batch norms | random (10 seeds), Kinetics-400, DFEW fold 3 | `models/m3dfel_extract_rdms.py` |
| Qwen3-VL-8B (64): 27 vision blocks, embedding output, 36 decoder blocks; float32 | released, random (10 seeds) | `models/vlm/vlm_extract_video_mode.py`, `compute_rdms_from_vlm_activations.py` |

### 5. RSA grids

| | Script | Output |
|---|---|---|
| 18 stimuli, 25 participants | `analysis/rsa_grid.py --stimuli 18 [--models <names>] [--rdm-dir] [--out-dir]` | `analysis/rsa_grid/` |
| 54 stimuli, 25 participants | `analysis/rsa_grid.py --stimuli 54 [--models <names>] [--rdm-dir] [--out-dir]` | `analysis/rsa_grid_54/` |

Models: every RDM listed in `analysis/variants.py` (21 variants; random seeds 42, 0–8 and training
runs r01–r10 have ten RDMs each), read from `rdms/set18|set54/<name>/<name>.npy`; each RDM's stimulus
order is checked against the EEG's. Each grid file holds the participant × scalp analysis (11 ROIs + whole scalp) × layer × condition
(all, fear, happy, neutral) × EEG window array of Spearman ρ. Statistics, controls and tables are
in [Which script gives which result](#which-script-gives-which-result).

## Data on Google Drive

Folder: **`<DRIVE_LINK>`** (to be published with the revision; planned layout below, as under
`$FER_ROOT`). `MANIFEST.tsv` at its root will list every file with its size and SHA-256.

| Folder | Content |
|---|---|
| `eeg_raw/` | Segmented EEG recordings (Brain Vision Analyzer exports) and the trial-level behaviour file |
| `eeg_rdms/` | EEG RDMs of the 25 participants and `meta.json` (`eeg/build_eeg_rdms.py`) |
| `rdms/set18/`, `rdms/set54/` | Model RDMs, one folder per RDM (`analysis/variants.py`), with `.layers.json` (layers, weights, preprocessing, stimulus order) |
| `results/{pretrain,sizecurve,stepmatched,finetune}/` | The retrained models (`best_weights.pt`), with training logs, metrics and `test_metrics.json` |
| `random_weights/` | The saved random-initialization weights the random variants load |
| `analysis/rsa_grid/`, `rsa_grid_54/` | RSA grids |
| `analysis/paper_stats/`, `paper_stats_54/` | rho_max tables (`rho_max.csv`, `rho_max_per_rdm.csv`), noise ceilings, paper tables and figures |
| `analysis/cluster_permutation_{18,54}*/`, `partial_rsa{,_54}/` | Cluster tests (also on the joint partial grids), partial RSA |

Qwen3-VL-8B-Instruct pretrained weights are not copied: download `Qwen/Qwen3-VL-8B-Instruct` at
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Stimulus videos and frames are not distributed.

## Steps without code

These steps were done by hand or with code that no longer exists. Their outputs are on Drive.

- **EEG preprocessing up to segmentation** in Brain Vision Analyzer 2.1 (re-referencing, filtering,
  ICA ocular correction, segmentation from -200 to +800 ms); `eeg/build_eeg_rdms.py` starts from its
  exports. The recordings are not distributed.
- **Behavioral analysis** behind the per-stimulus accuracy and reaction-time table and the
  accuracy figures quoted in Section 3.1.
- **The AffectNet three-class subset** (the Kaggle archive was already reduced).
- **Two ImageNet checkpoints from the authors' earlier study** (MFRS): FaceNet and ResNet50
  (`resnet50_None_imagenet`); the ResNet18 and VGG16 ImageNet weights are torchvision's.

## License

MIT, see `LICENSE`. `M3DFEL/` is adapted from Tencent TFace and remains under the Apache License 2.0 (`M3DFEL/LICENSE`).
