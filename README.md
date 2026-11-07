# Training Vision Models to Recognize Faces and Emotions Does Not Consistently Improve Neural Alignment

Code for the NeurIPS 2026 paper by Maryem Benslimane, Hamza Abdelhedi, Franco Lepore, Vanessa Hadid and Karim Jerbi.

**Paper:** [OpenReview](https://openreview.net/forum?id=jMlzffZ5UW) · [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/loc/sydney/poster/150276)

Deep networks are increasingly evaluated by how closely their representations match brain activity. We
asked which steps of training (generic pretraining, face-identity training, emotion fine-tuning) change
that alignment. We compared the EEG of 25 participants watching 54 one-second videos of fearful, happy and
neutral faces (unoccluded, with sunglasses, or with a mask) with 21 variants of six model families: random,
generically pretrained, face-recognition-trained and emotion-fine-tuned versions of the same architectures,
spanning image, video and video-language models. Alignment was measured with time-resolved, within-emotion
representational similarity analysis (RSA).
Every comparison is tested against a permutation null that repeats the selection of layer and time window,
and against models of identity, occlusion, pixels, optical flow and motion energy.

## Quick start

```bash
git clone https://github.com/thecocolab/FER-timeresolved-rsa.git
cd FER-timeresolved-rsa
pip install -e .                                  # Python 3.12
export FER_ROOT=/path/to/data                     # EEG and model RDMs (default: the repository root)
STAGES=figures bash scripts/run_paper.sh          # every figure and table from the result tables
bash scripts/run_paper.sh                         # recompute grids, statistics, cluster tests, probe, then figures
```

`STAGES` is any of `grids stats clusters probe figures`. Outputs go to `$FER_ROOT/analysis/paper_stats_54/`
(18 videos: `paper_stats/`). Every step is a module run as `python -m fer_rsa.<part>.<script>`; `--help`
lists its options.

## Paper results

| Result | Command (`python -m ...`) |
|---|---|
| RSA grids; partial RSA (each nuisance model removed alone and jointly) | `fer_rsa.analysis.rsa_grid --stimuli 54`; `fer_rsa.analysis.partial_rsa --stimuli 54` |
| The five planned comparisons against their permutation nulls (`--partial <set>`: after the controls; `--per-rdm`: each network at its peak) | `fer_rsa.analysis.contrasts --stimuli 54 --normalize` |
| Robustness: one emotion, one actor left out, correct trials | `contrasts` with `--emotion`, `--drop-actor`, `--eeg-dir eeg_rdms_correct` |
| FaceNet learning designs: training-set size, checkpoints, identity counts | `contrasts` with `--design sizes\|ckpt\|ids` |
| Cluster permutation tests | `fer_rsa.analysis.cluster_permutation --study 54 [--partial joint]` |
| Emotion probe; identity accuracy | `fer_rsa.analysis.probe --stimuli 54`; `fer_rsa.analysis.identity_accuracy` |
| Fig. 1 (needs the stimulus videos) | `fer_rsa.figures.fig1_study_design --stimuli <dir>` |
| Figs. 2 and 3 | `fer_rsa.figures.fig2_training`; `fer_rsa.figures.fig3_time_course` |
| Appendix figures | `fer_rsa.figures.figS_regions`, `figS_18`, `figS_robustness`, `figS_learning` |
| Appendix tables | `fer_rsa.figures.tables_supp --out <dir>` |

`scripts/run_paper.sh` lists the exact flags behind every number.

## Layout

```
fer_rsa/
  eeg/         EEG RDMs from the segmented recordings; ROI definitions
  models/      stimulus frames, activations and model RDMs (CNNs, M3DFEL, Qwen3-VL)
  training/    VGGFace2-HQ and AffectNet training
  analysis/    RSA grids, nuisance controls, partial RSA, comparisons, cluster tests, probe
  figures/     figures and appendix tables
  tasks/       task lists of the cluster job arrays
scripts/       run_paper.sh; SLURM drivers (trillium/); downloads (data/)
M3DFEL/        M3DFEL (Wang et al., CVPR 2023) with our 3-class DFEW changes
```

`fer_rsa/analysis/variants.py` lists every model variant and its RDMs.

## From raw data

Training and extraction ran on H100 GPUs (Trillium, Digital Research Alliance of Canada). Install the GPU
dependencies with `bash scripts/trillium/setup_env.sh` (exact versions: `scripts/trillium/trainenv_freeze.txt`);
in every `#SBATCH` script, set `--account` and submit from the repository root.

1. **EEG RDMs:** `python -m fer_rsa.eeg.build_eeg_rdms [--correct-only]`, from the Brain Vision Analyzer
   exports in `eeg_raw/`.
2. **Stimulus frames and random weights:** `bash scripts/trillium/extract_prep.sh`.
3. **Training:** download with `scripts/data/*.sh`, build the image lists with `fer_rsa.training.build_index`
   and `build_subset_indexes`, then `bash scripts/trillium/submit_all.sh` (10 runs per model; M3DFEL:
   `M3DFEL/train_dfew_3class.sh`).
4. **Model RDMs and probe features:** `bash scripts/trillium/submit_extract.sh` (`FEATURES=1` for the
   features); nuisance-control RDMs: `python -m fer_rsa.analysis.control_rdms --stimuli 54`.

Then run the quick start.

## Citation

```bibtex
@inproceedings{benslimane2026training,
  title     = {Training Vision Models to Recognize Faces and Emotions Does Not Consistently Improve Neural Alignment},
  author    = {Benslimane, Maryem and Abdelhedi, Hamza and Lepore, Franco and Hadid, Vanessa and Jerbi, Karim},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=jMlzffZ5UW}
}
```

## License

MIT (`LICENSE`). `M3DFEL/` is adapted from Tencent TFace and remains under the Apache License 2.0.
