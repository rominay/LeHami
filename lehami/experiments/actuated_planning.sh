#!/usr/bin/env bash
# Swing-up planning on the actuated pendulum: CEM in latent space towards a goal
# frame taken from the dataset, with the models trained by actuated_pendulum.sh.
#
# The ladder varies how often the plan is recomputed (the replanning interval R);
# every rung aims at the same goal 30 steps ahead:
#
#   rung                          horizon  replan every R  solves per episode
#   cell1_single_shot_H30_R30        30        30               1  (open loop)
#   cell2_goal_H30_R15               30        15               2
#   cell3_goal_H30_R9                30         9               4
#   cell6_goal_H30_R6                30         6               5
#   cell7_goal_H30_R3                30         3              10
#
# With replanning the horizon shrinks to the steps left before the goal. CEM uses
# 300 samples x 30 iterations, 30 elites; actions are held for 3 steps, |u| <= 4.
# Success: |theta error| < 0.1 rad and |omega error| < 0.5 rad/s, recorded both at
# the goal step and at any step. 50 test episodes x 3 planner seeds per model seed.
#
# Arms: a0q (Hamiltonian flow), b0s (LeWM W=1), b3s (LeWM W=3), b3f (LeWM W=3
# untied), each over model seeds 0, 1, 2.
#
# Stages:
#   0. model-free bounds per rung (no checkpoints): the floor (a null policy) and
#      the ceiling (CEM with the exact dynamics)  -> <rung>/preflight.txt
#   1. the planner                              -> <rung>/<arm>_s<seed>/swingup_episodes.csv
#   2. planning cost per solve (FLOPs, time; CUDA; model seed 0)
#                                               -> outputs/planning_compute/
# Everything is written under outputs/plan_replan_swingup/test/. Paper: the
# swing-up planning tables and the planning-compute table.
#
# Needs: pendulum_swingup.h5 and the checkpoints of actuated_pendulum.sh
# (arms a0q, b0s, b3s, b3f).
#
# Usage:
#   bash lehami/experiments/actuated_planning.sh                       # full ladder
#   bash lehami/experiments/actuated_planning.sh --rungs "1 3" --arms "a0q b0s" --seeds "0"
#   bash lehami/experiments/actuated_planning.sh --skip-compute

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="$DATA/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"
DATASET="$STABLEWM_HOME/pendulum_swingup.h5"
OUT="$ROOT/outputs/plan_replan_swingup/test"
LOG_DIR="$ROOT/lehami/experiments/logs/actuated_planning"
RUNGS="1 2 3 6 7"
ARMS="a0q b0s b3s b3f"
SEEDS="0 1 2"
COMPUTE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rungs)        RUNGS="$2"; shift 2 ;;
        --arms)         ARMS="$2"; shift 2 ;;
        --seeds)        SEEDS="$2"; shift 2 ;;
        --skip-compute) COMPUTE=false; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
need_file "$DATASET" "run lehami/data/prepare_actuated_data.sh"

GOAL_OFFSET=30; BLOCK=3; U_MAX=4.0; N_EVAL=50; PLANNER_SEEDS=3; IC_SEED=1234

# rung -> tag | horizon | replanning interval | extra planner flags
rung() {
    case "$1" in
        1) echo "cell1_single_shot_H30_R30|30|30|" ;;
        2) echo "cell2_goal_H30_R15|30|15|--replan" ;;
        3) echo "cell3_goal_H30_R9|30|9|--replan" ;;
        6) echo "cell6_goal_H30_R6|30|6|--replan" ;;
        7) echo "cell7_goal_H30_R3|30|3|--replan" ;;
        *) die "unknown rung '$1' (1 2 3 6 7)" ;;
    esac
}
config() {
    case "$1" in
        a0q) echo phase2_hjepa_sg_sep_conv_w1_swingup_exp079_a0q ;;
        b0s) echo phase0_lewm_ms_conv_w1_swingup_exp072_b0s ;;
        b3s) echo phase0_lewm_ms_conv_w3_swingup_exp076_b3s ;;
        b3f) echo phase0_lewm_ms_conv_w3_free_swingup_exp099_b3f ;;
        *) die "unknown arm '$1' (a0q b0s b3s b3f)" ;;
    esac
}

mkdir -p "$LOG_DIR"
for r in $RUNGS; do
    line=$(rung "$r")
    IFS="|" read -r tag horizon receding flags <<< "$line"
    mkdir -p "$OUT/$tag"
    if [[ ! -s "$OUT/$tag/preflight.txt" ]]; then
        log "$tag: model-free bounds"
        "${PY[@]}" "$SRC/preflight_dataset_goals.py" --n-eval "$N_EVAL" --split test \
            --goal-offset "$GOAL_OFFSET" --action-block "$BLOCK" --u-max "$U_MAX" \
            --ic-seed "$IC_SEED" --dataset "$DATASET" --manifest "$MANIFEST" --sweep \
            > "$OUT/$tag/preflight.txt" 2>&1 || die "preflight failed -- see $OUT/$tag/preflight.txt"
    fi
    for arm in $ARMS; do
        cfg=$(config "$arm")
        for s in $SEEDS; do
            ckpt=$(ckpt_for_suffix "$(seed_suffix "$cfg" "$s")")
            log "$tag: plan ${arm}_s$s"
            # $flags is deliberately unquoted: it is empty or one flag.
            "${PY[@]}" "$SRC/plan_pendulum_swingup.py" \
                --checkpoints "${arm}_s$s=$ckpt" --goal-mode dataset --split test \
                --goal-offset "$GOAL_OFFSET" --horizon "$horizon" --receding "$receding" \
                --budget "$GOAL_OFFSET" --action-block "$BLOCK" $flags \
                --n-eval "$N_EVAL" --planner-seeds "$PLANNER_SEEDS" --ic-seed "$IC_SEED" \
                --u-max "$U_MAX" --dataset "$DATASET" --manifest "$MANIFEST" \
                --cost terminal --success-at target \
                --cem-samples 300 --cem-iters 30 --cem-elites 30 \
                --theta-tol 0.1 --omega-tol 0.5 \
                --out "$OUT/$tag/${arm}_s$s" > "$LOG_DIR/${tag}_${arm}_s$s.log" 2>&1 \
                || die "planning failed -- see $LOG_DIR/${tag}_${arm}_s$s.log"
        done
    done
done

if [[ "$COMPUTE" == true ]]; then
    # One call at the reference budget (CEM 300 x 30) for all arms, then each arm at
    # the iteration count that matches the others' compute, then merge.
    for spec in "hjepa,lewm,lewm_w3:" "hjepa:48" "lewm:19" "lewm_w3:18"; do
        arms="${spec%%:*}" iters="${spec##*:}"
        log "planning compute: arms=$arms cem_iters=${iters:-30}"
        "${PY[@]}" "$SRC/measure_planning_compute.py" --arms "$arms" ${iters:+--cem-iters "$iters"} \
            >> "$LOG_DIR/planning_compute.log" 2>&1 \
            || die "measure_planning_compute.py failed -- see $LOG_DIR/planning_compute.log"
    done
    "${PY[@]}" "$SRC/measure_planning_compute.py" --build-grid >> "$LOG_DIR/planning_compute.log" 2>&1 \
        || die "measure_planning_compute.py --build-grid failed -- see $LOG_DIR/planning_compute.log"
fi
log "done -- outputs in $OUT and outputs/planning_compute/"
