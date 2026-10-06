"""The pendulum / three-body target dispatch, end to end through the probe path.

Run with:
    uv run pytest lehami/tests/test_probe_targets_dispatch.py -v

WHAT THIS GUARDS.  A 12-D three-body state pushed through the pendulum branch does
not raise: it takes the sine of a Cartesian coordinate, keeps three of twelve
channels, and writes plausible-looking numbers under the headers `sin`, `cos`,
`omega`. That silent failure has been found three times in this pipeline -- in the
block collectors, in `dump_probe_fit_scatter`, and in the `y_cols` / `state_cols`
labels of the npz dumps. These tests assert the WIDTH and the NAMES at every layer
that consumes a state, so the next occurrence fails loudly instead.

No checkpoint, dataset or GPU: a synthetic latent is built to carry the state
linearly, which is enough to fit a real Ridge and drive the real scorers. The high
R2 that results is a property of the fixture, not evidence about any model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "experiments" / "src"))

from dump_probe_fit_scatter import _scatter_frame, _scatter_summary  # noqa: E402
from probe_targets import (  # noqa: E402
    PENDULUM_STATE_DIM,
    THREE_BODY_STATE_DIM,
    probe_targets,
    state_names,
    target_names,
)
from run_eval import (  # noqa: E402
    Probe,
    _apply_probes_standard,
    _apply_probes_three_body,
    _probe_targets_block,
)
from train_probe_physics import _eval_probe, _targets  # noqa: E402

N_SAMPLES = 400
LATENT_DIM = 16


def _synth(state_dim: int, seed: int = 0):
    """(states, Y, X) with a latent that carries the targets linearly."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(N_SAMPLES, state_dim)).astype(np.float32)
    Y = probe_targets(states)
    W = rng.normal(size=(Y.shape[1], 2 * LATENT_DIM))
    X = (Y @ W + 0.05 * rng.normal(size=(N_SAMPLES, 2 * LATENT_DIM))).astype(np.float32)
    return states, Y, X


def _fitted(state_dim: int):
    states, Y, X = _synth(state_dim)
    return states, Y, X, Ridge(alpha=1.0).fit(X, Y)


# ── the rule itself ───────────────────────────────────────────────────────────

def test_target_width_is_keyed_on_state_width():
    """3 channels for the pendulum, 12 for three-body — never 3 for a 12-D state."""
    assert probe_targets(np.zeros((7, PENDULUM_STATE_DIM))).shape == (7, 3)
    assert probe_targets(np.zeros((7, THREE_BODY_STATE_DIM))).shape == (7, 12)


def test_single_frame_and_block_agree():
    """The per-frame and per-block forms were separate implementations that drifted."""
    for dim in (PENDULUM_STATE_DIM, THREE_BODY_STATE_DIM):
        block = np.random.default_rng(1).normal(size=(5, dim)).astype(np.float32)
        per_frame = np.stack([probe_targets(row) for row in block])
        np.testing.assert_allclose(per_frame, probe_targets(block), rtol=1e-6)


def test_rod_length_is_pendulum_only():
    """L is a pendulum parameter; three-body has none and must not silently ignore it."""
    assert probe_targets(np.zeros((5, PENDULUM_STATE_DIM)), 0.85).shape == (5, 4)
    with pytest.raises(ValueError, match="pendulum-only"):
        probe_targets(np.zeros((5, THREE_BODY_STATE_DIM)), 0.85)
    with pytest.raises(ValueError, match="pendulum-only"):
        target_names(THREE_BODY_STATE_DIM, with_length=True)


def test_unknown_width_raises_rather_than_defaulting():
    """Falling through to a default is what produced every bug this file guards."""
    for bad in (3, 4, 7, 24):
        with pytest.raises(ValueError, match="no probe targets defined"):
            probe_targets(np.zeros((5, bad)))


def test_run_eval_wrapper_delegates_to_the_shared_rule():
    """`_probe_targets_block` is kept as a name for its importers; same rule beneath."""
    for dim in (PENDULUM_STATE_DIM, THREE_BODY_STATE_DIM):
        block = np.random.default_rng(2).normal(size=(6, dim)).astype(np.float32)
        np.testing.assert_array_equal(_probe_targets_block(block), probe_targets(block))
    assert _targets is probe_targets


# ── the names ─────────────────────────────────────────────────────────────────

def test_names_match_the_generator_layout():
    """(q_0x, q_0y, q_1x, ..., v_2y): body-major, x before y, 0-based bodies."""
    assert target_names(THREE_BODY_STATE_DIM) == [
        "x_0", "y_0", "x_1", "y_1", "x_2", "y_2",
        "vx_0", "vy_0", "vx_1", "vy_1", "vx_2", "vy_2",
    ]
    assert target_names(PENDULUM_STATE_DIM) == ["sin_theta", "cos_theta", "omega"]


def test_name_count_matches_target_width():
    """A short name list beside a wide array is exactly the npz `y_cols` bug."""
    for dim in (PENDULUM_STATE_DIM, THREE_BODY_STATE_DIM):
        assert len(target_names(dim)) == probe_targets(np.zeros((3, dim))).shape[1]
    assert len(target_names(PENDULUM_STATE_DIM, True)) == \
        probe_targets(np.zeros((3, PENDULUM_STATE_DIM)), 0.85).shape[1]


def test_state_names_name_the_raw_state_not_the_targets():
    """They coincide for three-body and differ for the pendulum."""
    assert state_names(PENDULUM_STATE_DIM) == ["theta", "omega"]
    assert state_names(PENDULUM_STATE_DIM + 1) == ["theta", "omega", "length"]
    assert state_names(THREE_BODY_STATE_DIM) == target_names(THREE_BODY_STATE_DIM)
    with pytest.raises(ValueError):
        state_names(7)


# ── the scorers ───────────────────────────────────────────────────────────────

def test_three_body_metrics_split_position_from_velocity():
    """The two clauses of the planning criterion are never pooled, and are per body."""
    states, _, X, ridge = _fitted(THREE_BODY_STATE_DIM)
    assert ridge.coef_.shape[0] == 12, "probe must predict all 12 channels"

    probes = {"ridge_gt": Probe(ridge=ridge, latent_dim=LATENT_DIM, uses_p=True)}
    row = _apply_probes_three_body(probes, X, states)

    for col in ("r2_pos", "r2_vel", "mae_pos", "mae_vel"):
        assert col in row
    for b in (1, 2, 3):
        assert f"r2_pos_body{b}" in row and f"r2_vel_body{b}" in row
    # no pendulum column may appear on this path
    assert not any("theta" in c or "omega" in c for c in row)


def test_both_scorer_entry_points_agree():
    """run_eval and train_probe_physics must not disagree about the same probe."""
    states, Y, X, ridge = _fitted(THREE_BODY_STATE_DIM)
    probes = {"ridge_gt": Probe(ridge=ridge, latent_dim=LATENT_DIM, uses_p=True)}
    a = _apply_probes_three_body(probes, X, states)
    b = _eval_probe(ridge, X, Y, None)
    for k in ("r2_pos", "r2_vel", "mae_pos", "mae_vel"):
        assert a[k] == pytest.approx(b[k], rel=1e-9)


def test_pendulum_scorer_still_reports_angles():
    """The control: the pendulum path is unchanged by the refactor."""
    states, _, X, ridge = _fitted(PENDULUM_STATE_DIM)
    probes = {"ridge_gt": Probe(ridge=ridge, latent_dim=LATENT_DIM, uses_p=True)}
    row = _apply_probes_standard(probes, X, states[:, 0], states[:, 1])
    for col in ("r2_theta", "r2_omega", "mae_theta", "mae_omega"):
        assert col in row
    assert not any(c.endswith("_pos") or c.endswith("_vel") for c in row)


# ── the scatter dump ──────────────────────────────────────────────────────────

def test_scatter_keeps_every_channel_including_the_actuated_body():
    """It used to keep 3 of 12 and drop both coordinates of the actuated body."""
    _, Y, X, ridge = _fitted(THREE_BODY_STATE_DIM)
    df = _scatter_frame("ridge_gt", 0, Y, ridge.predict(X))

    assert len(df.columns) == 2 + 2 * 12          # tier, k, then true/pred per channel
    for name in target_names(THREE_BODY_STATE_DIM):
        assert f"{name}_true" in df.columns and f"{name}_pred" in df.columns
    # ACTUATED_BODY = 2 in lehami/data/three_body.py — the body planning scores
    for col in ("x_2_true", "y_2_true", "vx_2_true", "vy_2_true"):
        assert col in df.columns
    # and no leftover pendulum column, in particular no atan2 of two coordinates
    assert not any(c.startswith(("theta", "sin", "cos", "omega")) for c in df.columns)
    assert "world units" in _scatter_summary(df)


def test_scatter_pendulum_columns_are_unchanged():
    _, Y, X, ridge = _fitted(PENDULUM_STATE_DIM)
    df = _scatter_frame("ridge_gt", 0, Y, ridge.predict(X))
    assert list(df.columns) == [
        "tier", "k", "sin_true", "sin_pred", "cos_true", "cos_pred",
        "omega_true", "omega_pred", "theta_true", "theta_pred",
    ]
    assert "MAE_theta" in _scatter_summary(df)
