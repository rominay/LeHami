"""What a probe is asked to decode, defined ONCE for every dataset.

WHY THIS MODULE EXISTS.  The rule "state -> probe targets" used to live in two
places: `train_probe_physics._targets` (one frame at a time) and
`run_eval._probe_targets_block` (a whole (T, D) block).  Same rule, two shapes,
two files -- and they had already drifted: the block version grew an `ep_length`
argument that appends the pendulum's rod length L as a 4th channel, and the
per-frame version never did, so the two produced target sets of different WIDTH
for the same dataset.  `_probe_has_length()` then had to recover the width by
sniffing the fitted estimator.

It could not simply be consolidated into either file, because the two are in an
import cycle: `train_probe_physics` imports ~20 names from `run_eval` at module
level, while `run_eval` imports `train_probe_physics` only deferred, inside
main() (see the comment at run_eval.py's top).  This module is therefore a LEAF:
it imports nothing from either, so both can import down into it.

THE RULE.  Dispatch on the state width, never on a caller-supplied flag.  Width
is ground truth; a flag can be wrong, and wrong is silent here -- a 12-D state
pushed through the pendulum branch does not raise, it takes the sine of a
Cartesian coordinate and writes plausible numbers.  That exact failure has now
been found three times in this pipeline.

    PENDULUM (state_dim 2)   [theta, omega] -> [sin(theta), cos(theta), omega]
        theta is periodic, so it is decoded through its sine and cosine rather
        than directly; a probe asked for theta itself would be penalised for the
        wrap at +/-pi.  When the dataset VARIES the rod length, L is appended as
        a 4th channel -- it is a PARAMETER, not a state, and the point of probing
        it is that a frozen probe on a predicted q_k must hold it FLAT.

    THREE-BODY (state_dim 12) the 12 numbers as they stand, (q in R^6, v in R^6).
        Nothing is transformed: there is no periodic coordinate to protect, the
        coordinates are already Cartesian and O(1) in the generator's units, and
        the six velocity components are the channel the comparison is about --
        they play the role omega plays on the pendulum.  There is NO length
        channel and passing `ep_length` here is an error, not a no-op.

        Deliberately NOT included: energy, orbital radius, speed.  They are
        nonlinear functions of the state, and a probe that decodes all 12 numbers
        determines them already; adding them would double-count the same
        information under a different weighting.  Compute them downstream, where
        the masses and the softening length are available (they are not here).

Any other width raises.  Falling through to a default is what produced the bugs
this module exists to prevent.
"""
from __future__ import annotations

import numpy as np

# ── the datasets, keyed by state width ────────────────────────────────────────

PENDULUM_STATE_DIM = 2
TWO_BODY_STATE_DIM = 8
THREE_BODY_STATE_DIM = 12

PENDULUM_TARGET_NAMES = ["sin_theta", "cos_theta", "omega"]
PENDULUM_LENGTH_NAME = "length"

# Body indices are 0-BASED, matching the generator's state layout
# (q_0x, q_0y, q_1x, q_1y, q_2x, q_2y, v_0x, ..., v_2y) and `ACTUATED_BODY = 2`
# in lehami/data/three_body.py.  Note that run_eval's per-body METRIC
# columns (r2_pos_body1 ...) number the same bodies from 1, so body i here is
# `body{i+1}` there, and the actuated body is `x_2, y_2` / `body3`.
THREE_BODY_TARGET_NAMES = (
    [f"{c}_{b}" for b in (0, 1, 2) for c in ("x", "y")]
    + [f"v{c}_{b}" for b in (0, 1, 2) for c in ("x", "y")]
)

# The ACTUATED TWO-BODY set (lehami/data/two_body.py): a massive centre, body 0,
# and the actuated orbiter, body 1.  Same layout convention as three-body -- all
# positions, then all velocities -- so the only difference is the width.  ACTUATED_BODY
# is 1 here and 2 there; nothing in this module depends on which body is actuated, but
# anything downstream that slices out "the actuated body" does, and hardcoding index 2
# is the off-by-one this comment exists to prevent.
TWO_BODY_TARGET_NAMES = (
    [f"{c}_{b}" for b in (0, 1) for c in ("x", "y")]
    + [f"v{c}_{b}" for b in (0, 1) for c in ("x", "y")]
)


def target_names(state_dim: int, with_length: bool = False) -> list[str]:
    """Column names for the target block, so downstream code never guesses.

    `with_length` applies to the pendulum only; asking for it on any other
    dataset raises rather than silently returning names for a column that the
    target builder will not produce.
    """
    if state_dim == PENDULUM_STATE_DIM:
        return PENDULUM_TARGET_NAMES + ([PENDULUM_LENGTH_NAME] if with_length else [])
    if state_dim == THREE_BODY_STATE_DIM:
        if with_length:
            raise ValueError(
                "with_length=True is pendulum-only: the three-body dataset has no "
                "rod-length parameter and probe_targets() will not emit that channel."
            )
        return list(THREE_BODY_TARGET_NAMES)
    if state_dim == TWO_BODY_STATE_DIM:
        if with_length:
            raise ValueError(
                "with_length=True is pendulum-only: the two-body dataset has no "
                "rod-length parameter and probe_targets() will not emit that channel."
            )
        return list(TWO_BODY_TARGET_NAMES)
    raise ValueError(f"no probe targets defined for state_dim={state_dim}")


def n_targets(state_dim: int, with_length: bool = False) -> int:
    """Width of the target block, without building it."""
    return len(target_names(state_dim, with_length))


def state_names(n_state: int) -> list[str]:
    """Names for a RAW state block, as dumped in `rollout_latents.npz`.

    Distinct from `target_names`: that names the probe's OUTPUT, this names the
    stored state. They coincide for three-body (the targets are the state) and
    differ for the pendulum, whose raw state is [theta, omega] while its targets
    are [sin, cos, omega].

    Widths: 2 = [theta, omega]; 3 = the same plus rod length L; 8 = two-body;
    12 = three-body.
    """
    if n_state in (PENDULUM_STATE_DIM, PENDULUM_STATE_DIM + 1):
        return ["theta", "omega", PENDULUM_LENGTH_NAME][:n_state]
    if n_state == THREE_BODY_STATE_DIM:
        return list(THREE_BODY_TARGET_NAMES)
    if n_state == TWO_BODY_STATE_DIM:
        return list(TWO_BODY_TARGET_NAMES)
    raise ValueError(f"no state-column names defined for width {n_state}")


def probe_targets(states: np.ndarray, ep_length: float | None = None) -> np.ndarray:
    """Raw state(s) -> probe targets.  Accepts one frame or a whole block.

    `states` is (D,) for a single frame or (T, D) for a block; the return has the
    matching rank, (n_targets,) or (T, n_targets).

    `ep_length` is the episode's rod length, appended as a 4th PENDULUM channel
    when the dataset varies it.  It is rejected outright for three-body: silently
    ignoring it there is how the two old implementations came to disagree about
    the width of their own output.

    Dtypes are preserved from the implementations this replaces, so the refactor
    does not perturb any fitted probe: the pendulum trig is computed and returned
    in float64 for a block, cast to float32 for a single frame, and the
    three-body path returns float32 (it is the stored state, unmodified).
    """
    arr = np.asarray(states)
    single = arr.ndim == 1
    block = arr.reshape(1, -1) if single else arr
    if block.ndim != 2:
        raise ValueError(f"expected (D,) or (T, D), got shape {arr.shape}")

    width = block.shape[1]

    if width in (THREE_BODY_STATE_DIM, TWO_BODY_STATE_DIM):
        if ep_length is not None:
            # The message is spelled per-width rather than generated, so the
            # three-body wording is byte-identical to what it was before two-body
            # was added. Tests and logs match on this string.
            raise ValueError(
                "ep_length is pendulum-only; the three-body state has no rod-length "
                "parameter. Pass None for 12-D states."
                if width == THREE_BODY_STATE_DIM else
                "ep_length is pendulum-only; the two-body state has no rod-length "
                "parameter. Pass None for 8-D states."
            )
        out = np.asarray(block, dtype=np.float32)
        return out[0] if single else out

    if width == PENDULUM_STATE_DIM:
        theta = block[:, 0].astype(np.float64)
        omega = block[:, 1].astype(np.float64)
        cols = [np.sin(theta), np.cos(theta), omega]
        if ep_length is not None:
            cols.append(np.full_like(theta, ep_length))
        out = np.stack(cols, axis=1)
        return out[0].astype(np.float32) if single else out

    raise ValueError(f"no probe targets defined for state_dim={width}")
