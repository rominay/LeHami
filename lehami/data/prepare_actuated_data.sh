#!/usr/bin/env bash
# Generate the ACTUATED pendulum dataset (torque-driven, used for swing-up).
#
#   pendulum_swingup.h5   3 rod lengths x 3 bob masses x 100 episodes, 8 s each,
#                         dt = 0.1 s, torque |u| <= 4 held for 3 frames
#
# Lands in $STABLEWM_HOME. CPU only, a few seconds. An existing file is kept
# (delete it to regenerate).
#
# The split manifest used in the paper is shipped in lehami/data/. Pass
# --manifests to rebuild it with the exact recipe (overwrites the shipped copy).
#
# Usage:
#   bash lehami/data/prepare_actuated_data.sh [--manifests]

source "$(dirname "${BASH_SOURCE[0]}")/../experiments/_common.sh"

MANIFESTS=false
[[ "${1:-}" == "--manifests" ]] && MANIFESTS=true
DATASET="$STABLEWM_HOME/pendulum_swingup.h5"
mkdir -p "$STABLEWM_HOME"

if [[ -f "$DATASET" ]]; then
    log "pendulum_swingup.h5 exists -- skipping"
else
    log "generating pendulum_swingup.h5"
    "${PY[@]}" "$DATA/pendulum_swingup.py" \
        --lengths 0.7 1.0 1.4 --masses 0.5 1.0 2.0 --dt-mult 2 --n-per-cell 100 \
        --window 8.0 --u-max 4.0 --hold 3 --seed 42 --out "$DATASET"
fi

if [[ "$MANIFESTS" == true ]]; then
    log "rebuilding split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"
    "${PY[@]}" "$DATA/data_prep.py" --dataset "$DATASET" \
        --split-method episode_srnn --train-ratio 0.7 --val-ratio 0.15 \
        --train-frames 80 --num-steps 16 --seed 42
fi
log "done"
