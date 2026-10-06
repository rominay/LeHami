#!/usr/bin/env bash
# Conservative pendulum, SINGLE ROD (pendulum.h5): Hamiltonian flow vs LeWM.
#
#   arm    experiment config                              results/<name>/
#   hjepa  phase2_hjepa_sg_sep_sreg_joint_ms_d32_exp027   exp027_sg_ms_d32
#   lewm   phase0_lewm_ms_srnn_k100_exp030                exp030_lewm_ms_srnn_k100
#
# One seed each (the config's seed, 42). Split: 450 train / 50 test episodes.
# In the paper these checkpoints feed the single-rod latent figure (the encoder's
# (q, p) against the true (theta, omega)).
#
# Needs: pendulum.h5 (lehami/data/prepare_conservative_data.sh).
#
# Usage:
#   bash lehami/experiments/conservative_single_rod.sh                 # both arms
#   bash lehami/experiments/conservative_single_rod.sh --arms lewm
#   bash lehami/experiments/conservative_single_rod.sh --eval-only     # reuse checkpoints

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_episode_srnn_K100_seed42.json"
LOG_DIR="$ROOT/lehami/experiments/logs/conservative_single_rod"
ARMS="hjepa,lewm"
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)      ARMS="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
need_file "$STABLEWM_HOME/pendulum.h5" "run lehami/data/prepare_conservative_data.sh"

# arm -> config | results name | eval horizon
spec() {
    case "$1" in
        hjepa) echo "phase2_hjepa_sg_sep_sreg_joint_ms_d32_exp027|exp027_sg_ms_d32|30" ;;
        lewm)  echo "phase0_lewm_ms_srnn_k100_exp030|exp030_lewm_ms_srnn_k100|97" ;;
        *) die "unknown arm '$1' (hjepa | lewm)" ;;
    esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    line=$(spec "$arm")
    IFS="|" read -r cfg name horizon <<< "$line"
    suffix=$(config_suffix "$cfg")
    [[ "$EVAL_ONLY" == true ]] || train "$LOG_DIR/train_$arm.log" "$suffix" "$cfg" "$MANIFEST"
    ckpt=$(ckpt_for_suffix "$suffix")
    evaluate "$LOG_DIR/eval_$arm.log" "$ckpt" "$MANIFEST" "$name" "$horizon"
done
log "done -- results in lehami/experiments/results/, logs in $LOG_DIR"
