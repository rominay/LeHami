"""The pre-flight defines the CEILING every planning number is read against.

`preflight_dataset_goals.py` decides whether an experiment is worth GPU-hours at
all: if its oracle (CEM on the true plant) is not well clear of best-of-300-random,
the instrument cannot resolve a model difference and the run should not happen. So
an error that makes the ORACLE too low is the worst kind here -- it widens the
apparent gap and makes a dead design point look usable. Nothing in the printed table
can reveal it, because the table IS the output.

Two such errors are pinned below. Both were live until 2026-09-01, and the script's
own docstring cited a `test_preflight_matches_plant_step` that had never been
written.

Run with:
    uv run pytest lehami/tests/test_preflight_dataset_goals.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "lehami" / "data",
           ROOT / "lehami" / "experiments" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plan_pendulum_swingup as P  # noqa: E402
import preflight_dataset_goals as PF  # noqa: E402


def _args(budget, action_block):
    return SimpleNamespace(
        budget=budget, action_block=action_block, dt_mult=1, u_max=4.0,
        cem_samples=6, cem_iters=2, cem_elites=2)


def _episode():
    return dict(theta0=0.3, omega0=-0.2, L=1.0, m=1.0,
                goal=dict(theta=float(np.pi), omega=0.0))


# ── 1. the oracle must span the whole budget ──────────────────────────────────


@pytest.mark.parametrize("budget,block", [(30, 3), (25, 3), (20, 3), (50, 5),
                                          (25, 5), (7, 4), (9, 2)])
def test_oracle_rolls_out_exactly_the_budget(monkeypatch, budget, block):
    """Every candidate the oracle scores must be `budget` env steps long.

    `n_blk = budget // block` (FLOOR) was the bug. When the block did not divide
    the budget it produced `floor(B/blk)*blk` steps, and the two criteria failed
    differently: `target` indexes step `at-1` and raised IndexError, while `any`
    reduces with `.any(1)`, never indexes, and so SILENTLY scored a short rollout
    -- an oracle handicapped by up to `block-1` steps, i.e. a ceiling too low.

    `cem_plan` uses ceiling division and truncates; this must match it, or the
    ceiling is not a bound on the thing it is supposed to bound.
    """
    widths = []
    real = PF.rollout_batch

    def spy(th0, om0, L, m, acts, dt):
        widths.append(acts.shape[1])
        return real(th0, om0, L, m, acts, dt)

    monkeypatch.setattr(PF, "rollout_batch", spy)
    PF.oracle_cem(_episode(), _args(budget, block), (0.1, 0.5), budget,
                  np.random.default_rng(0))

    assert widths, "the oracle never rolled anything out"
    assert set(widths) == {budget}, (
        f"oracle scored {sorted(set(widths))}-step rollouts on a {budget}-step "
        f"budget (block {block}); a short rollout understates the ceiling")


def test_oracle_block_count_matches_the_planners(monkeypatch):
    """The oracle's block count must equal `cem_plan`'s for the same horizon.

    The oracle is only a valid ceiling if it searches at least the space the
    planner searches. A different rounding rule on the two sides is exactly how
    that stops being true without anything failing.
    """
    # Bind the ORIGINAL once: re-reading PF.rollout_batch inside the loop would
    # pick up the previous iteration's spy and recurse forever.
    real = PF.rollout_batch
    for budget, block in [(30, 3), (25, 3), (20, 3), (7, 4)]:
        planner_blocks = -(-budget // block)      # cem_plan's arithmetic
        seen = []

        def spy(th0, om0, L, m, acts, dt, _seen=seen, _real=real):
            _seen.append(acts.shape[1])
            return _real(th0, om0, L, m, acts, dt)

        monkeypatch.setattr(PF, "rollout_batch", spy)
        PF.oracle_cem(_episode(), _args(budget, block), (0.1, 0.5), budget,
                      np.random.default_rng(0))
        assert seen[0] == budget
        assert -(-seen[0] // block) == planner_blocks


# ── 2. the batched plant must equal the planner's own ─────────────────────────


def test_rollout_batch_matches_plant_step():
    """The claim the module docstring makes, finally tested.

    `rollout_batch` re-implements `plant_step` batched over candidates so an oracle
    CEM is affordable. If the two integrators drift apart, the ceiling is measured
    on a different plant from the one the arms are scored on, and every comparison
    silently changes meaning.
    """
    rng = np.random.default_rng(4)
    L, m, dt = 1.1, 0.9, P.BASE_DT
    acts = rng.uniform(-4.0, 4.0, size=(3, 12))
    TH, OM = PF.rollout_batch(0.25, -0.4, L, m, acts, dt)

    for i in range(acts.shape[0]):
        th, om = 0.25, -0.4
        for t in range(acts.shape[1]):
            th, om = P.plant_step(th, om, L, m, acts[i, t], dt)
            assert TH[i, t] == pytest.approx(th, abs=1e-9), f"theta cand {i} step {t}"
            assert OM[i, t] == pytest.approx(om, abs=1e-9), f"omega cand {i} step {t}"
