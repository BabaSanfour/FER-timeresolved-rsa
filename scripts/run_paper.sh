#!/bin/bash
# Every statistic, figure and table of the paper from the Drive data (CPU), from the repository root.
# Stages: STAGES="grids stats clusters probe figures" (default: all). JOBS sets the probe's processes.
set -euo pipefail
export FER_ROOT=${FER_ROOT:-.} MPLBACKEND=Agg PYTHONWARNINGS=ignore
STAGES=${STAGES:-grids stats clusters probe figures}
run() { echo "== python -m $*"; python -m "$@"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has grids; then                                   # RSA grids, raw and after each nuisance control
    for s in 54 18; do
        run fer_rsa.analysis.rsa_grid --stimuli $s
        run fer_rsa.analysis.partial_rsa --stimuli $s
    done
fi

if has stats; then                                   # comparisons against their permutation nulls
    C=fer_rsa.analysis.contrasts
    for p in "" identity occlusion pixel optical_flow motion_energy joint; do
        run $C --stimuli 54 --normalize --per-rdm ${p:+--partial $p}
    done
    for p in "" joint; do
        run $C --stimuli 18 --normalize ${p:+--partial $p}
        for e in fear happy neutral; do run $C --stimuli 54 --normalize --emotion $e ${p:+--partial $p}; done
        for k in 0 1 2 3 4 5; do run $C --stimuli 54 --normalize --drop-actor $k ${p:+--partial $p}; done
        run $C --stimuli 54 --normalize --eeg-dir eeg_rdms_correct ${p:+--partial $p}
        for d in sizes ckpt ids; do run $C --stimuli 54 --normalize --per-rdm --design $d ${p:+--partial $p}; done
    done
fi

if has clusters; then                                # cluster permutation tests
    run fer_rsa.analysis.cluster_permutation --study 54
    run fer_rsa.analysis.cluster_permutation --study 18
    run fer_rsa.analysis.cluster_permutation --study 54 --partial joint
fi

if has probe; then                                   # emotion probe and identity accuracies
    run fer_rsa.analysis.probe --stimuli 54 --jobs "${JOBS:-8}"
    run fer_rsa.analysis.identity_accuracy
fi

if has figures; then                                 # Figs. 2-3, appendix figures and tables
    for f in fig2_training fig3_time_course figS_regions figS_18 figS_robustness figS_learning; do
        run fer_rsa.figures.$f
    done
    run fer_rsa.figures.tables_supp --out "$FER_ROOT/analysis/paper_stats_54/tables"
fi
echo "[done] figures and tables in $FER_ROOT/analysis/paper_stats_54/ (18 videos: paper_stats/)"
