#!/usr/bin/env bash
# Conservative pendulum, single rod, with a FROZEN DINOv2 encoder (dinov2_vits14,
# CLS token) shared by both arms, so the predictor is the only difference.
#
#   arm  model                        experiment config                     results/<name>/
#   a0d  Hamiltonian flow             phase2_hjepa_sg_sep_dino_cls_exp098_a0d  exp098_a0d_hjepa_dino
#   b0d  DINO-WM ViT predictor        phase0_dino_wm_cls_exp098_b0d            exp098_b0d_dino_wm_cls
#
# Each arm is evaluated twice: at best.ckpt, and at the epoch-60 checkpoint
# (epochepoch=059.ckpt -> results/exp098_<arm>_ep059/), which is the matched-epoch
# comparison the paper reports (DINOv2 figure).
# Split: 450 train / 50 test episodes, rollout horizon 100.
#
# Needs: pendulum.h5 (lehami/data/prepare_conservative_data.sh) and the DINOv2 weights
# (download_dinov2.sh, with the same TORCH_HOME).
#
# Usage:
#   bash lehami/experiments/conservative_dino.sh               # both arms
#   bash lehami/experiments/conservative_dino.sh --arms a0d
#   bash lehami/experiments/conservative_dino.sh --eval-only

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_episode_srnn_K100_seed42.json"
LOG_DIR="$ROOT/lehami/experiments/logs/conservative_dino"
HORIZON=100
MATCHED_EPOCH=059
ARMS="a0d,b0d"
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)      ARMS="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
need_file "$STABLEWM_HOME/pendulum.h5" "run lehami/data/prepare_conservative_data.sh"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
[[ -d "$TORCH_HOME/hub/facebookresearch_dinov2_main" ]] \
    || die "DINOv2 cache missing under TORCH_HOME=$TORCH_HOME -- run download_dinov2.sh"

# arm -> config | results name
spec() {
    case "$1" in
        a0d) echo "phase2_hjepa_sg_sep_dino_cls_exp098_a0d|exp098_a0d_hjepa_dino" ;;
        b0d) echo "phase0_dino_wm_cls_exp098_b0d|exp098_b0d_dino_wm_cls" ;;
        *) die "unknown arm '$1' (a0d | b0d)" ;;
    esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    line=$(spec "$arm")
    IFS="|" read -r cfg name <<< "$line"
    suffix=$(config_suffix "$cfg")
    [[ "$EVAL_ONLY" == true ]] || train "$LOG_DIR/train_$arm.log" "$suffix" "$cfg" "$MANIFEST"
    ckpt=$(ckpt_for_suffix "$suffix")
    evaluate "$LOG_DIR/eval_$arm.log" "$ckpt" "$MANIFEST" "$name" "$HORIZON"
    evaluate "$LOG_DIR/eval_${arm}_ep$MATCHED_EPOCH.log" \
        "$(dirname "$ckpt")/epochepoch=$MATCHED_EPOCH.ckpt" "$MANIFEST" \
        "exp098_${arm}_ep$MATCHED_EPOCH" "$HORIZON"
done
log "done -- results in lehami/experiments/results/, logs in $LOG_DIR"
