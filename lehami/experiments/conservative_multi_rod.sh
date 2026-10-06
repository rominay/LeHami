#!/usr/bin/env bash
# Conservative pendulum, THREE ROD LENGTHS (pendulum_multi_v2.h5, dt x2 slice):
# the main conservative comparison, Hamiltonian flow vs LeWM.
#
#   arm  model                     experiment config                                  results/<name>[_s1,_s2]/
#   a0m  Hamiltonian flow, W=1     phase2_hjepa_sg_sep_conv_w1_multi_exp067_a0m       exp067_a0m_hjepa_w1_multi
#   b0m  LeWM, W=1                 phase0_lewm_ms_conv_w1_multi_exp068_b0m            exp068_b0m_lewm_w1_multi
#   b1m  LeWM, 3-frame encoder     phase0_lewm_ms_conv_te_w3_multi_exp066_b1m         exp066_b1m_lewm_te_w3_multi
#
# a0m and b0m run three seeds: the config's seed (42, unsuffixed dirs) plus seeds 1
# and 2 (suffixed _s1, _s2). b1m runs one seed; it is only needed for its per-epoch
# checkpoints in conservative_training_compute.sh.
# Split: 210 train / 45 val / 45 test episodes, rollout horizon 70.
# Paper: rollout-error figure and the conservative-pendulum tables.
#
# After a0m's config-seed eval, eval_physics_invariants.py runs two tests on that
# checkpoint (Hamiltonian flow only; LeWM has no step size and no inverse), both
# decoded to physical (theta, omega) with the k=1 kNN probe and compared per episode
# against the true pendulum ODE (horizon 40):
#   physics_dt.csv        the same learned H integrated at finer step sizes -- does it
#                         trace the same continuous trajectory?
#   physics_reversal.csv  K steps forward then K back, reversed by dt -> -dt (exact
#                         for leapfrog: the control) and by p -> -p (exact only if the
#                         learned kinetic energy is even in p)
# Paper: the dt-and-reversal figure. Written to results/exp067_a0m_hjepa_w1_multi/.
#
# Once a0m and b0m (config seed) both exist, eval_jacobian_volume.py measures how the
# one-step latent map changes phase-space volume, (1/n) log|det J| (0 = preserved,
# the exact value for the true pendulum and, by construction, for a symplectic map):
#   latent_volume/jacobian_k0.csv       at the anchor, 45 test episodes
#   latent_volume/jacobian_rollout.csv  at every step of a 70-step rollout
#   latent_volume/epsball.csv           an independent finite-difference estimate
#                                       (eps-ball, 10 episodes) as a check on autograd
# Paper: the Jacobian log-volume figure and table. Written to results/latent_volume/.
#
# Needs: pendulum_multi_v2.h5 (lehami/data/prepare_conservative_data.sh).
#
# Usage:
#   bash lehami/experiments/conservative_multi_rod.sh                      # all arms
#   bash lehami/experiments/conservative_multi_rod.sh --arms a0m --seeds "1 2"
#   bash lehami/experiments/conservative_multi_rod.sh --eval-only

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json"
LOG_DIR="$ROOT/lehami/experiments/logs/conservative_multi_rod"
HORIZON=70
ARMS="a0m,b0m,b1m"
SEEDS=""        # empty: each arm's default seed list below
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)      ARMS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;   # space-separated; "default" = config seed
        --eval-only) EVAL_ONLY=true; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
need_file "$STABLEWM_HOME/pendulum_multi_v2.h5" "run lehami/data/prepare_conservative_data.sh"

# arm -> config | results name | default seeds
spec() {
    case "$1" in
        a0m) echo "phase2_hjepa_sg_sep_conv_w1_multi_exp067_a0m|exp067_a0m_hjepa_w1_multi|default 1 2" ;;
        b0m) echo "phase0_lewm_ms_conv_w1_multi_exp068_b0m|exp068_b0m_lewm_w1_multi|default 1 2" ;;
        b1m) echo "phase0_lewm_ms_conv_te_w3_multi_exp066_b1m|exp066_b1m_lewm_te_w3_multi|default" ;;
        *) die "unknown arm '$1' (a0m | b0m | b1m)" ;;
    esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
    line=$(spec "$arm")
    IFS="|" read -r cfg name default_seeds <<< "$line"
    base=$(config_suffix "$cfg")
    for s in ${SEEDS:-$default_seeds}; do
        if [[ "$s" == default ]]; then
            tag="" overrides=()
        else
            tag="_s$s" overrides=(seed="$s" run_suffix="${base}_s$s")
        fi
        [[ "$EVAL_ONLY" == true ]] \
            || train "$LOG_DIR/train_${arm}${tag}.log" "${base}${tag}" "$cfg" "$MANIFEST" "${overrides[@]}"
        ckpt=$(ckpt_for_suffix "${base}${tag}")
        evaluate "$LOG_DIR/eval_${arm}${tag}.log" "$ckpt" "$MANIFEST" "${name}${tag}" "$HORIZON"
        if [[ "$arm" == a0m && "$s" == default ]]; then
            for t in dt reversal; do
                log "physics $t  $name"
                "${PY[@]}" "$SRC/eval_physics_invariants.py" --exp "$name" --checkpoint "$ckpt" \
                    --manifest "$MANIFEST" --test "$t" --episodes 50 --horizon 40 \
                    > "$LOG_DIR/physics_${t}_$arm.log" 2>&1 \
                    || die "eval_physics_invariants.py --test $t failed -- see $LOG_DIR/physics_${t}_$arm.log"
            done
        fi
    done
done

if has_ckpt "$(config_suffix phase2_hjepa_sg_sep_conv_w1_multi_exp067_a0m)" \
            "$(config_suffix phase0_lewm_ms_conv_w1_multi_exp068_b0m)"; then
    for mode in anchor rollout epsball; do
        case "$mode" in
            anchor)  extra=() ;;
            rollout) extra=(--horizon 70) ;;
            epsball) extra=(--eps-ball --max-episodes 10 --eps 1e-2,1e-3,1e-4) ;;
        esac
        log "jacobian volume ($mode)  a0m, b0m"
        "${PY[@]}" "$SRC/eval_jacobian_volume.py" --arms a0m,b0m "${extra[@]}" \
            > "$LOG_DIR/jacobian_$mode.log" 2>&1 \
            || die "eval_jacobian_volume.py ($mode) failed -- see $LOG_DIR/jacobian_$mode.log"
    done
else
    log "skipping the Jacobian volume: needs the config-seed checkpoints of both a0m and b0m"
fi
log "done -- results in lehami/experiments/results/, logs in $LOG_DIR"
