# Which Steps of Training Make Vision Models More Brain-Like? Evidence from EEG Responses to Dynamic Facial Expressions

Code for the paper. It compares 25 participants' EEG responses to 54 dynamic facial-expression videos with
vision networks at each step of training (random, generic pretraining, face training, emotion fine-tuning),
using time-locked representational similarity analysis (RSA).

Data: **https://drive.google.com/drive/folders/1w-IQJMyxpl0GaoB_b7Gx0Ho3YRmGENw_?usp=sharing**
(EEG and model RDMs, RSA results, trained weights; see [Data](#data-on-google-drive)).

- [Quick start: every figure and table](#quick-start-every-figure-and-table)
- [What each step makes](#what-each-step-makes)
- [Repository layout](#repository-layout)
- [Full pipeline from raw data](#full-pipeline-from-raw-data)
- [Data on Google Drive](#data-on-google-drive)
- [Steps without code](#steps-without-code)

## Quick start: every figure and table

Runs on a CPU from the Drive data; no GPU, training or extraction needed.

**1. Install** (Python 3.12):

```bash
git clone https://github.com/MaryemBenslimane/FER-timeresolved-rsa.git
cd FER-timeresolved-rsa
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**2. Get the data.** Download the Drive folder and merge it into the repository root, so that
`rdms/`, `eeg_rdms/` and `analysis/` sit next to `fer_rsa/`. To keep the data elsewhere, set
`FER_ROOT=/path/to/data`.

**3. Run.**

```bash
STAGES=figures bash scripts/run_paper.sh     # redraw the figures and tables from the results on Drive (minutes)
bash scripts/run_paper.sh                    # recompute everything: grids, statistics, cluster tests, probe, figures
```

`STAGES` picks any of `grids stats clusters probe figures` (default: all). Recomputing the statistics
takes hours: each comparison is tested against 1,000 permutations and each cluster test uses 20,000
sign flips. Outputs go to `analysis/paper_stats_54/` (18 videos: `analysis/paper_stats/`); the appendix
table rows go to `analysis/paper_stats_54/tables/`. Fig. 1 needs the stimulus videos, which are not
distributed (`python -m fer_rsa.figures.fig1_study_design --stimuli <dir with set18/ and set54/>`).

## What each step makes

Every step is a module run as `python -m fer_rsa.<part>.<script>` from the repository root; `--help`
lists its options. All values are fractions of the upper noise ceiling (`--normalize`); statistics use
the 54 videos unless stated.

### Statistics

| Step | Command | Writes |
|---|---|---|
| RSA grids: Spearman ρ per participant, scalp analysis (11 ROIs + whole scalp), layer, emotion and EEG window | `fer_rsa.analysis.rsa_grid --stimuli 54` (and `18`) | `analysis/rsa_grid{_54,}/` |
| Partial RSA: each nuisance RDM (identity, occlusion, pixel, optical flow, motion energy) removed alone and all jointly | `fer_rsa.analysis.partial_rsa --stimuli 54` | `analysis/partial_rsa_54/<set>/` |
| The five planned comparisons and every variant against its own permutation null, per ROI and whole scalp, with time courses | `fer_rsa.analysis.contrasts --stimuli 54 --normalize [--partial <set>]` | `analysis/paper_stats_54/contrasts[_<set>]_nc.csv`, `contrasts_time[_<set>]_nc.csv` |
| Each network at its own peak (probe, learning designs) | add `--per-rdm` | rows `rdm` of the same files |
| Robustness: one emotion, one actor left out, correct trials only | add `--emotion fear\|happy\|neutral`, `--drop-actor 0..5`, `--eeg-dir eeg_rdms_correct` | `contrasts*_nc_{emotion,dropactorK,eeg_rdms_correct}.csv` |
| FaceNet learning designs: training-set sizes, checkpoints, identity counts | add `--design sizes\|ckpt\|ids` | `contrasts_{sizes,ckpt,ids}[_joint]_nc.csv` |
| Cluster permutation tests, corrected study-wide by the maximum statistic | `fer_rsa.analysis.cluster_permutation --study 54 [--partial joint]` (and `--study 18`) | `analysis/cluster_permutation_{54,18,54_partial_joint}/` |
| Linear emotion probe on final-layer features | `fer_rsa.analysis.probe --stimuli 54 --jobs 8` | `analysis/paper_stats_54/probe.csv`, `probe_per_rdm.csv` |
| Identity accuracy of the face models and learning designs | `fer_rsa.analysis.identity_accuracy` | `analysis/{identity,checkpoint,identity_count}_accuracy.csv` |

### Figures and tables

| In the paper | Command | Reads |
|---|---|---|
| Fig. 1, study design | `fer_rsa.figures.fig1_study_design --stimuli <dir>` | stimulus videos, `fer_rsa/eeg/roi_definitions.json` |
| Fig. 2, what each training step contributes, before and after the controls | `fer_rsa.figures.fig2_training` | `contrasts{,_identity,_occlusion,_pixel,_optical_flow,_motion_energy,_joint}_nc.csv` |
| Fig. 3, the comparisons in time and at 187–250 ms | `fer_rsa.figures.fig3_time_course` | `contrasts_time_joint_nc.csv` |
| Appendix Figs. 4 and 6, remaining ROIs | `fer_rsa.figures.figS_regions` | as Figs. 2 and 3 |
| Appendix Fig. 5, the 18 unoccluded videos | `fer_rsa.figures.figS_18` | `analysis/paper_stats/contrasts{,_joint}_nc.csv` |
| Appendix Fig. 7, robustness | `fer_rsa.figures.figS_robustness` | the robustness tables, `behavior/recognition_frames.csv` |
| Appendix Fig. 8, task accuracy and face training | `fer_rsa.figures.figS_learning` | `probe_per_rdm.csv`, rows `rdm`, the learning-design tables, the accuracy files |
| Appendix Tables 5–7, peak alignment of every variant, cluster tests | `fer_rsa.figures.tables_supp --out <dir>` | `contrasts{,_joint}_nc.csv`, `cluster_permutation_*/model_survivors.csv` |
| Appendix Tables 2–4: ROIs, layers, training recipes | `fer_rsa/eeg/roi_definitions.json`; each RDM's `.layers.json`; the training scripts | |

## Repository layout

```
fer_rsa/                 the Python package
  eeg/                   EEG RDMs from the segmented recordings; ROI definitions
  models/                stimulus frames, activations, model RDMs (CNNs, M3DFEL, Qwen3-VL)
  training/              VGGFace2-HQ image lists and training, AffectNet fine-tuning
  analysis/              RSA grids, nuisance controls, partial RSA, comparisons, cluster tests, probe
  figures/               every figure and appendix table
  tasks/                 task lists of the cluster job arrays
scripts/                 run_paper.sh; SLURM drivers (scripts/trillium/, cluster tests); downloads (scripts/data/)
M3DFEL/                  M3DFEL (Wang et al., CVPR 2023) with the 3-class DFEW changes
```

`fer_rsa/analysis/variants.py` lists every model variant and its RDMs; `fer_rsa/analysis/rsa_core.py`
holds the data paths (all under `FER_ROOT`, default `.`) and the shared RSA helpers.

## Full pipeline from raw data

Optional: these steps rebuild the inputs that the quick start downloads. Training and extraction ran
on Trillium (Digital Research Alliance of Canada) H100 GPUs, where `FER_ROOT=$SCRATCH/fer_revision`.

**Environment.** `bash scripts/trillium/setup_env.sh` installs the package with the `models` extra
(PyTorch, transformers), `facenet-pytorch` (`--no-deps`) and checks the imports; exact versions are in
`scripts/trillium/trainenv_freeze.txt`. OpenCV comes from `module load opencv/4.11.0` (elsewhere:
`pip install opencv-python-headless`). In every `#SBATCH` script, set `--account` (`YOUR_ALLOCATION`)
and submit from the repository root.

### 1. EEG RDMs

```bash
python -m fer_rsa.eeg.build_eeg_rdms              # eeg_raw/ -> eeg_rdms/<subject>.npy (12, 13, 54, 54), meta.json
python -m fer_rsa.eeg.build_eeg_rdms --correct-only   # -> eeg_rdms_correct/ (trials classified as intended)
```

Each trial's video comes from the behaviour file and is checked against the block's trigger codes. Per
participant: baseline −200 to 0 ms; channels above ±100 µV in more than 20% of trials are interpolated;
trials above ±150 µV are rejected; the rest are averaged per video. RDMs are correlation distances between
video averages over the channels of a scope × the samples of a window: 11 ROIs and the whole scalp, 13
windows of 62.5 ms from onset (the last ends at 800 ms). Participants with fewer than half of the 972
trials are excluded (CEG); AA has no Markers file, so its blocks are labelled by order and block 6 is
dropped. Both stimulus sets use the same 25 participants; the 18 stimuli are the unoccluded videos.

### 2. Stimulus frames

`bash scripts/trillium/extract_prep.sh` (once) samples 16 frames per video (224 × 224) for both stimulus
sets into `data/stim_frames/set18|set54/`, writes the Qwen3-VL video lists, and checks or draws the random
weights. Random variants load the saved weights in `random_init_weights/` (seeds 42 and 0–8); redrawing
the existing seeds must reproduce those files bit for bit.

### 3. Training

Data (on a data-transfer node):

| Data | Command |
|---|---|
| VGGFace2-HQ archives 01–06 (726,502 images) | `DEST=$FER_ROOT/data/vggface_dataset bash scripts/data/get_vggface2_hq.sh` |
| Image lists: full pool (715,810 train, 10,692 val) | `python -m fer_rsa.training.build_index --root $FER_ROOT/data/vggface_dataset/VGGface2_None_norm_512_true_bygfpgan --out vggface_train/index` |
| Image lists: subsets (sizes 10k–500k; 50–3,500 identities) | `python -m fer_rsa.training.build_subset_indexes --design sizes` and `--design ids` |
| Decoded-image cache (36 GB) | `python -m fer_rsa.training.build_decode_cache --data-root ... --index-dir vggface_train/index --out-dir $FER_ROOT/data/vggface_cache` |
| AffectNet, 3 classes | `bash scripts/data/get_affectnet_3class.sh` |
| VGG-Face weights | `bash scripts/data/get_vgg_face.sh` |

Then `bash scripts/trillium/submit_all.sh` writes the task lists and submits the training arrays; re-running
skips finished runs. Seeds: repeat *r* uses 1000 + *r*. Outputs: `results/<group>/.../repeat_NN/best_weights.pt`.

| Runs | Recipe | Output |
|---|---|---|
| Face pretraining: ResNet18, ResNet50, FaceNet, 10 runs each | `fer_rsa.training.train_vggface2`, full pool, 40 epochs, best validation top-1 | `results/pretrain/<arch>/` |
| Size curve and step-matched check (FaceNet, 10 runs) | subsets of 10k–500k images; step-matched: 55,920 steps | `results/sizecurve/`, `results/stepmatched/` |
| Learning designs (FaceNet, 3 runs) | checkpoints along full-pool training; 50–3,500 identities × 100 images | `results/checkpoints/`, `results/identities/` |
| AffectNet fine-tuning: ResNet18, ResNet50, FaceNet, VGG16, 10 runs each | `fer_rsa.training.train_affectnet`; epoch chosen on 10% of Train, Test evaluated once | `results/finetune/<arch>/` |
| M3DFEL on DFEW fold 3 (not retrained: DFEW needs a licence) | `M3DFEL/prepare_dfew_3class.sh`, `M3DFEL/train_dfew_3class.sh` | `M3DFEL/results/17812056/fold_3/` |

### 4. Model RDMs and probe features

`bash scripts/trillium/submit_extract.sh` extracts every model RDM (425; re-running skips finished ones) into
`rdms/set18|set54/<name>/` (`<name>.npy`, `(layers, 16 frames, stimuli, stimuli)`, and `<name>.layers.json`
with layers, weights, preprocessing and stimulus order). `FEATURES=1 bash scripts/trillium/submit_extract.sh`
writes the final-layer features of the probe to `features/`.

| Family (layers) | Variants | Extractor |
|---|---|---|
| ResNet18 (40), ResNet50 (106), FaceNet (245), VGG16 (15) | random, ImageNet, face, AffectNet; FaceNet learning designs | `fer_rsa.models.extract_model_rdms` |
| M3DFEL (40) | random, Kinetics-400, DFEW | `fer_rsa.models.m3dfel_extract_rdms` |
| Qwen3-VL-8B (64), float32 | released, random | `fer_rsa.models.vlm_extract_video_mode`, `compute_rdms_from_vlm_activations` |

The nuisance-control RDMs come from the frames: `python -m fer_rsa.analysis.control_rdms --stimuli 54`
(and `18`). Then continue with the [quick start](#quick-start-every-figure-and-table).

## Data on Google Drive

The folder mirrors `FER_ROOT`.

| Folder | Content |
|---|---|
| `eeg_rdms/`, `eeg_rdms_correct/` | EEG RDMs of the 25 participants (all trials; correct trials only) and `meta.json` |
| `rdms/set18/`, `rdms/set54/` | Model and nuisance-control RDMs, one folder per RDM |
| `features/` | Final-layer features for the probe |
| `behavior/recognition_frames.csv` | Frame at which each participant recognised each video (no names) |
| `analysis/` | RSA and partial-RSA grids, comparison tables, cluster tests, probe and accuracy tables, figures |
| `results/` | Trained models (`best_weights.pt`), training logs and metrics |
| `random_init_weights/`, `net_weights/` | Saved random weights; ImageNet and VGG-Face weights |
| `vggface_train/` | VGGFace2-HQ image lists (full pool, subsets, identity subsets) |
| `M3DFEL/pretrained/`, `M3DFEL/results/` | R3D-18 Kinetics-400 backbone; the DFEW fold-3 model |

Not distributed: the raw EEG recordings, the stimulus videos and frames, and the Qwen3-VL-8B-Instruct
weights (download `Qwen/Qwen3-VL-8B-Instruct` at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`).

## Steps without code

- **EEG preprocessing up to segmentation** in Brain Vision Analyzer 2.1 (re-referencing, filtering, ICA
  ocular correction, segmentation from −200 to +800 ms); `fer_rsa.eeg.build_eeg_rdms` starts from its exports.
- **Behavioral analysis** behind the per-stimulus accuracy and reaction-time table and the recognition frames.
- **The AffectNet three-class subset** (the Kaggle archive was already reduced).
- **Two ImageNet checkpoints from the authors' earlier study** (FaceNet and ResNet50); the ResNet18 and
  VGG16 ImageNet weights are torchvision's.

The first version of the paper used single training runs; its scripts are in the git history.

## License

MIT (`LICENSE`). `M3DFEL/` is adapted from Tencent TFace and remains under the Apache License 2.0 (`M3DFEL/LICENSE`).
