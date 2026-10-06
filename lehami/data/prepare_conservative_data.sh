#!/usr/bin/env bash
# Generate the two CONSERVATIVE (unforced, frictionless) pendulum datasets.
#
#   pendulum.h5           single rod, 500 episodes x 100 frames, dt = 0.05 s
#   pendulum_multi_v2.h5  3 rod lengths x 4 frame rates, 1200 episodes x 8 s
#
# Both land in $STABLEWM_HOME. CPU only, about a minute. Existing files are kept
# (delete one to regenerate it).
#
# The train/val/test splits are NOT rebuilt: the manifests used in the paper are
# shipped in lehami/data/. Pass --manifests to rebuild them anyway with the
# exact recipe (this overwrites the shipped copies; the episode lists come out
# identical when the .h5 files are).
#
# Usage:
#   bash lehami/data/prepare_conservative_data.sh [--manifests]

source "$(dirname "${BASH_SOURCE[0]}")/../experiments/_common.sh"

MANIFESTS=false
[[ "${1:-}" == "--manifests" ]] && MANIFESTS=true
mkdir -p "$STABLEWM_HOME"

if [[ -f "$STABLEWM_HOME/pendulum.h5" ]]; then
    log "pendulum.h5 exists -- skipping"
else
    log "generating pendulum.h5"
    "${PY[@]}" "$DATA/generate_pendulum.py" --out-dir "$STABLEWM_HOME"
fi

if [[ -f "$STABLEWM_HOME/pendulum_multi_v2.h5" ]]; then
    log "pendulum_multi_v2.h5 exists -- skipping"
else
    log "generating pendulum_multi_v2.h5"
    "${PY[@]}" "$DATA/pendulum_multi.py" --train-dt-mults 2 3 4 --test-dt-mults 1 \
        --out "$STABLEWM_HOME/pendulum_multi_v2.h5"
fi

if [[ "$MANIFESTS" == true ]]; then
    log "rebuilding split_pendulum_episode_srnn_K100_seed42.json"
    "${PY[@]}" "$DATA/data_prep.py" --dataset "$STABLEWM_HOME/pendulum.h5" \
        --split-method episode_srnn --train-ratio 0.9 --train-frames 100 --num-steps 10 --seed 42
    log "rebuilding split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json"
    "${PY[@]}" "$DATA/data_prep.py" --dataset "$STABLEWM_HOME/pendulum_multi_v2.h5" \
        --split-method episode_srnn --train-ratio 0.70 --val-ratio 0.15 \
        --restrict-dt-mults 2 --train-frames 80 --num-steps 16 --seed 42
fi
log "done"
