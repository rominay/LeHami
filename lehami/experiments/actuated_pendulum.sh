#!/usr/bin/env bash
# Actuated pendulum (pendulum_swingup.h5): train, evaluate and probe every arm the
# paper reports, 3 seeds each (0, 1, 2).
#
#   arm  model                                  encoder   experiment config
#   a0q  Hamiltonian flow, SIGReg on q  (main)  W=1       phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q
#   a0p  Hamiltonian flow, SIGReg on q, p split W=1       phase2_lehami_sg_sep_conv_w1_swingup_exp080_a0p
#   a0s  Hamiltonian flow, SIGReg on [q,p]      W=1       phase2_lehami_sg_sep_conv_w1_swingup_exp071_a0s
#   a3q  Hamiltonian flow, SIGReg on q          W=3       phase2_lehami_sg_sep_conv_w3_swingup_exp081_a3q
#   a3p  Hamiltonian flow, SIGReg on q, p split W=3       phase2_lehami_sg_sep_conv_w3_swingup_exp082_a3p
#   a3s  Hamiltonian flow, SIGReg on [q,p]      W=3       phase2_lehami_sg_sep_conv_w3_swingup_exp075_a3s
#   b0s  LeWM                                   W=1       phase0_lewm_ms_conv_w1_swingup_exp072_b0s
#   b3s  LeWM                                   W=3       phase0_lewm_ms_conv_w3_swingup_exp076_b3s
#   b3f  LeWM, untied temporal window           W=3       phase0_lewm_ms_conv_w3_free_swingup_exp099_b3f
#
# W is the number of frames the encoder sees (W=3: a symmetric temporal window).
# Main comparison: a0q vs b0s vs b3s. The a*-rows with different SIGReg placement
# feed the canonicality figure; b3f is a planning control.
#
# Per arm and seed this writes
#   results/<exp>_<arm>_<model>_s<seed>/   rollout + probe metrics (run_eval.py, and
#                                          train_probe_physics.py for the joint-SIGReg
#                                          and LeWM arms, as in the paper)
#   and the probe-fit / rollout dumps of dump_probe_fit_scatter.py
# and, once the checkpoints they need exist,
#   results/latent_volume/jacobian_rollout_actuated.csv
#       per-step Jacobian log-volume of the one-step map, a0q vs b0s (seed 0)
#   results/capacity_probes.csv
#       ridge / 25-NN / MLP probes on the same features, a0q a0p a3p b0s b3s
# Split: 630 train / 135 val / 135 test episodes.
# Paper: also the actuated Jacobian-volume figure and the probe-capacity tables.
#
# Needs: pendulum_swingup.h5 (lehami/data/prepare_actuated_data.sh). Planning is separate:
# actuated_planning.sh.
#
# Usage:
#   bash lehami/experiments/actuated_pendulum.sh                          # all arms
#   bash lehami/experiments/actuated_pendulum.sh --arms a0q,b0s,b3s --seeds "0"
#   bash lehami/experiments/actuated_pendulum.sh --eval-only

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"
LOG_DIR="$ROOT/lehami/experiments/logs/actuated_pendulum"
ARMS="a0q,a0p,a0s,a3q,a3p,a3s,b0s,b3s,b3f"
SEEDS="0 1 2"
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)      ARMS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
need_file "$STABLEWM_HOME/pendulum_swingup.h5" "run lehami/data/prepare_actuated_data.sh"

# arm -> config | results name | horizon | evaluation protocol
#   probe:  run_eval.py on the test split, then train_probe_physics.py
#   single: run_eval.py alone (it fits the probes itself)
# The horizon is the longest rollout an 80-frame episode allows after the anchor.
spec() {
    case "$1" in
        a0q) echo "phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q|exp079_a0q_lehami|76|single" ;;
        a0p) echo "phase2_lehami_sg_sep_conv_w1_swingup_exp080_a0p|exp080_a0p_lehami|76|single" ;;
        a3q) echo "phase2_lehami_sg_sep_conv_w3_swingup_exp081_a3q|exp081_a3q_lehami|74|single" ;;
        a3p) echo "phase2_lehami_sg_sep_conv_w3_swingup_exp082_a3p|exp082_a3p_lehami|74|single" ;;
        a0s) echo "phase2_lehami_sg_sep_conv_w1_swingup_exp071_a0s|exp071_a0s_lehami|76|probe" ;;
        a3s) echo "phase2_lehami_sg_sep_conv_w3_swingup_exp075_a3s|exp075_a3s_lehami|76|probe" ;;
        b0s) echo "phase0_lewm_ms_conv_w1_swingup_exp072_b0s|exp072_b0s_lewm|76|probe" ;;
        b3s) echo "phase0_lewm_ms_conv_w3_swingup_exp076_b3s|exp076_b3s_lewm|76|probe" ;;
        b3f) echo "phase0_lewm_ms_conv_w3_free_swingup_exp099_b3f|exp099_b3f_lewm|76|probe" ;;
        *) die "unknown arm '$1' (a0q a0p a0s a3q a3p a3s b0s b3s b3f)" ;;
    esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    line=$(spec "$arm")
    IFS="|" read -r cfg name horizon protocol <<< "$line"
    for s in $SEEDS; do
        suffix=$(seed_suffix "$cfg" "$s")
        cell="${name}_s$s"
        [[ "$EVAL_ONLY" == true ]] \
            || train "$LOG_DIR/train_$cell.log" "$suffix" "$cfg" "$MANIFEST" \
                     seed="$s" run_suffix="$suffix"
        ckpt=$(ckpt_for_suffix "$suffix")
        if [[ "$protocol" == probe ]]; then
            evaluate "$LOG_DIR/eval_$cell.log" "$ckpt" "$MANIFEST" "$cell" "$horizon" --splits test
            log "probe  $cell"
            "${PY[@]}" "$SRC/train_probe_physics.py" --checkpoint "$ckpt" --manifest "$MANIFEST" \
                --exp-name "$cell" > "$LOG_DIR/probe_$cell.log" 2>&1 \
                || die "probe failed -- see $LOG_DIR/probe_$cell.log"
        else
            evaluate "$LOG_DIR/eval_$cell.log" "$ckpt" "$MANIFEST" "$cell" "$horizon"
        fi
        log "dump   $cell"
        "${PY[@]}" "$SRC/dump_probe_fit_scatter.py" --checkpoint "$ckpt" --manifest "$MANIFEST" \
            --exp-name "$cell" --dump-rollout --horizon "$horizon" \
            > "$LOG_DIR/scatter_$cell.log" 2>&1 \
            || die "dump failed -- see $LOG_DIR/scatter_$cell.log"
    done
done

# Jacobian log-volume of the one-step map along a 76-step rollout, at each
# episode's true torque (seed 0 of a0q and b0s).
if has_ckpt "$(seed_suffix phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q 0)" \
            "$(seed_suffix phase0_lewm_ms_conv_w1_swingup_exp072_b0s 0)"; then
    log "jacobian volume  a0q_s0, b0s_s0"
    "${PY[@]}" "$SRC/eval_jacobian_volume.py" --arms a0q,b0s --horizon 76 \
        --out "$ROOT/lehami/experiments/results/latent_volume/jacobian_rollout_actuated.csv" \
        > "$LOG_DIR/jacobian_actuated.log" 2>&1 \
        || die "eval_jacobian_volume.py failed -- see $LOG_DIR/jacobian_actuated.log"
else
    log "skipping the Jacobian volume: needs seed 0 of both a0q and b0s"
fi

# Probe capacity at the anchor: ridge vs 25-NN vs MLP on the same dumped features,
# for every arm/seed that has them (CPU only) -> results/capacity_probes.csv.
cap=()
for c in exp079_a0q_lehami exp080_a0p_lehami exp082_a3p_lehami exp072_b0s_lewm exp076_b3s_lewm; do
    for s in 0 1 2; do
        f="$ROOT/lehami/experiments/results/${c}_s$s/probe_features.npz"
        [[ -f "$f" ]] && cap+=("${c}_s$s")
    done
done
if [[ ${#cap[@]} -gt 0 ]]; then
    log "probe capacity  ${#cap[@]} runs"
    "${PY[@]}" "$SRC/fit_capacity_probes.py" --exps "$(IFS=,; echo "${cap[*]}")" \
        > "$LOG_DIR/capacity_probes.log" 2>&1 \
        || die "fit_capacity_probes.py failed -- see $LOG_DIR/capacity_probes.log"
else
    log "skipping probe capacity: no probe_features.npz yet"
fi
log "done -- results in lehami/experiments/results/, logs in $LOG_DIR"
