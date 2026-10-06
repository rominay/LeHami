# Shared helpers, sourced by every runner in this directory. Not meant to be run.
#
# Environment variables honoured by every runner:
#   STABLEWM_HOME   where datasets are read and checkpoints written
#                   (default ~/.stable_worldmodel)
#   UV              uv binary (default: `uv` on PATH)
#   TRAIN_OVERRIDES extra Hydra overrides appended to every train.py call, e.g.
#                   TRAIN_OVERRIDES="trainer.max_epochs=1" for a quick smoke run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CKPT_ROOT="$STABLEWM_HOME/checkpoints"
DATA="$ROOT/lehami/data"
SRC="$ROOT/lehami/experiments/src"

UV="${UV:-uv}"
command -v "$UV" >/dev/null 2>&1 || { echo "ERROR: uv not found; install it or set UV=/path/to/uv" >&2; exit 1; }
PY=("$UV" run python)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# need_file <path> <hint>
need_file() { [[ -f "$1" ]] || die "missing $1 -- $2"; }

# train <logfile> <run_suffix> <experiment-config> <manifest> [hydra overrides...]
# Refuses to train into a checkpoint dir that already holds checkpoints: Lightning
# would write best-v1.ckpt beside the old best.ckpt, and evaluation reads best.ckpt.
train() {
    local logfile="$1" suffix="$2" exp="$3" manifest="$4"; shift 4
    local d
    for d in "$CKPT_ROOT"/*"$suffix"; do
        if [[ -d "$d" ]] && compgen -G "$d/*.ckpt" >/dev/null; then
            die "$d already holds checkpoints; move it away to retrain, or pass --eval-only"
        fi
    done
    mkdir -p "$(dirname "$logfile")"
    local extra=()
    [[ -n "${TRAIN_OVERRIDES:-}" ]] && read -ra extra <<< "$TRAIN_OVERRIDES"
    log "train  $exp $* ${extra[*]}  (log: $logfile)"
    "${PY[@]}" lehami/training/train.py +experiment="$exp" manifest="$manifest" "$@" \
        "${extra[@]}" > "$logfile" 2>&1 || die "training failed -- see $logfile"
}

# evaluate <logfile> <checkpoint> <manifest> <exp-name> <horizon> [run_eval.py args...]
# Writes lehami/experiments/results/<exp-name>/.
evaluate() {
    local logfile="$1" ckpt="$2" manifest="$3" name="$4" horizon="$5"; shift 5
    mkdir -p "$(dirname "$logfile")"
    need_file "$ckpt" "train this arm first"
    log "eval   $name  (horizon $horizon, log: $logfile)"
    "${PY[@]}" "$SRC/run_eval.py" --checkpoint "$ckpt" --manifest "$manifest" \
        --exp-name "$name" --horizon "$horizon" --rollout-seed 0 "$@" \
        > "$logfile" 2>&1 || die "eval failed -- see $logfile"
}

# ckpt_for_suffix <run_suffix>: the best.ckpt of the one checkpoint dir ending in it.
ckpt_for_suffix() {
    local dirs=( "$CKPT_ROOT"/*"$1" )
    [[ ${#dirs[@]} -eq 1 && -d "${dirs[0]}" ]] \
        || die "expected exactly one checkpoint dir matching *$1, found: ${dirs[*]}"
    echo "${dirs[0]}/best.ckpt"
}

# has_ckpt <run_suffix>...: true if every suffix has exactly one checkpoint dir with a best.ckpt.
has_ckpt() {
    local s dirs
    for s in "$@"; do
        dirs=( "$CKPT_ROOT"/*"$s" )
        [[ ${#dirs[@]} -eq 1 && -f "${dirs[0]}/best.ckpt" ]] || return 1
    done
}

# config_suffix <experiment-config>: the run_suffix set in that config.
config_suffix() {
    awk '/^run_suffix:/{print $2; exit}' "lehami/training/configs/experiment/$1.yaml"
}

# seed_suffix <experiment-config> <seed>: the run_suffix used for per-seed runs.
seed_suffix() { echo "_$(echo "$1" | sed 's/^phase[0-9]*_//')_s$2"; }
