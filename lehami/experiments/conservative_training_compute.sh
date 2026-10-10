#!/usr/bin/env bash
# Accuracy vs training budget on the three-rod conservative pendulum. No training:
# it reuses the checkpoints conservative_multi_rod.sh already wrote.
#
#   1. evaluate the every-10-epochs checkpoints (epochepoch=009 ... 099.ckpt) of
#      a0m, b0m and b1m (config seed only) -> results/exp096_<arm>_ep<NNN>/
#   2. measure one training step of each arm (FLOPs and GPU time, CUDA only)
#      -> outputs/exp096/training_compute_<gpu>.csv
#
# Paper: the iso-compute table (appendix). Epochs whose results/ dir already holds
# a summary.json are skipped, so an interrupted run can simply be restarted.
#
# Needs: conservative_multi_rod.sh run first (default seed of a0m, b0m, b1m).
#
# Usage:
#   bash lehami/experiments/conservative_training_compute.sh
#   bash lehami/experiments/conservative_training_compute.sh --arms a0m --epochs "009 049 099"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json"
LOG_DIR="$ROOT/lehami/experiments/logs/conservative_training_compute"
HORIZON=70
ARMS="a0m,b0m,b1m"
EPOCHS="009 019 029 039 049 059 069 079 089 099"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)   ARMS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done

config() {
    case "$1" in
        a0m) echo phase2_lehami_sg_sep_conv_w1_multi_exp067_a0m ;;
        b0m) echo phase0_lewm_ms_conv_w1_multi_exp068_b0m ;;
        b1m) echo phase0_lewm_ms_conv_te_w3_multi_exp066_b1m ;;
        *) die "unknown arm '$1' (a0m | b0m | b1m)" ;;
    esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    cfg=$(config "$arm")
    best=$(ckpt_for_suffix "$(config_suffix "$cfg")")
    dir=$(dirname "$best")
    for ep in $EPOCHS; do
        name="exp096_${arm}_ep$ep"
        if [[ -f "$ROOT/lehami/experiments/results/$name/summary.json" ]]; then
            log "$name done already -- skipping"; continue
        fi
        evaluate "$LOG_DIR/eval_${arm}_ep$ep.log" "$dir/epochepoch=$ep.ckpt" "$MANIFEST" "$name" "$HORIZON"
    done
done

log "measuring training cost per step"
mkdir -p "$LOG_DIR"
"${PY[@]}" "$SRC/measure_training_compute.py" --arms "$ARMS" --n-warm 20 --n-time 100 \
    > "$LOG_DIR/measure_training_compute.log" 2>&1 \
    || die "measure_training_compute.py failed -- see $LOG_DIR/measure_training_compute.log"
log "done -- results in lehami/experiments/results/exp096_*, outputs/exp096/"
