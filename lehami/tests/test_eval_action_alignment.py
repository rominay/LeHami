"""The EVAL rollout must feed the same action training fed, at the same index.

Until 2026-08-18 every rollout path in `run_eval.py` drove the predictor with u = 0.
That was correct for every dataset before the swing-up one, all of which were
unactuated, and silently wrong afterwards: it scored a u = 0 rollout against a
torque-driven ground truth.  These tests pin the two things that have to hold for the
fix to mean anything.

1.  The latent->pixel offset used by eval is the one the ENCODER declares, not `W-1`.
    A symmetric window has lag = (W-1)/2, so assuming a causal stencil reads every
    action (W-1)/2 frames late -- on both arms, with no metric flagging it.  This is
    the same bug class that shipped once in `train_probe_physics._align_states`, where
    it was invisible at W = 1 and a CUDA gather assert at W = 3.

2.  The action driving the step INTO latent frame f is the action AT latent frame
    f-1, matching `builders._forward`, which pairs the transition into A+1+step with
    the action at A+step.  An off-by-one here is a half-step phase error.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "lehami" / "models" / "third_party" / "leworldmodel",
           ROOT / "lehami" / "models",
           ROOT / "lehami" / "experiments" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_eval import (  # noqa: E402
    _act_emb_at,
    _act_emb_window,
    _latent_to_pixel_offset,
)


class _Enc:
    def __init__(self, window, lag):
        self.window, self.lag = window, lag


class _Model:
    """Minimal stand-in: the action encoder is the identity, so the embedding a test
    gets back IS the action it was handed, and an index error cannot hide."""
    def __init__(self, window=1, lag=0.0):
        self.encoder = _Enc(window, lag)

    @staticmethod
    def action_encoder(a):
        return a


@pytest.mark.parametrize(("window", "lag", "off"), [
    (1, 0.0, 0),    # per-frame
    (3, 1.0, 1),    # symmetric: lag = (W-1)/2 = 1  ->  offset 1, NOT 2
    (5, 2.0, 2),    # symmetric, wider
    (2, 0.0, 1),    # the `diff` control arm: causal, lag 0
])
def test_offset_comes_from_the_encoder_not_from_the_window(window, lag, off):
    assert _latent_to_pixel_offset(_Model(window, lag)) == off


def test_symmetric_window_is_not_treated_as_causal():
    """W-1 would be 2 for a symmetric W=3 encoder; the right answer is 1."""
    assert _latent_to_pixel_offset(_Model(3, 1.0)) != 3 - 1


def test_non_integral_lag_is_refused():
    with pytest.raises(AssertionError):
        _latent_to_pixel_offset(_Model(4, 1.5))


def test_action_is_read_at_the_frame_being_stepped_out_of():
    actions = torch.arange(10, dtype=torch.float32).view(10, 1)
    zero = torch.zeros(1, 1)
    m = _Model(window=1, lag=0.0)
    # stepping out of latent frame 4 must use action 4, matching builders._forward
    assert float(_act_emb_at(m, actions, zero, 4, 0).reshape(-1)[0]) == 4.0


def test_action_index_shifts_with_the_encoder_offset():
    actions = torch.arange(10, dtype=torch.float32).view(10, 1)
    zero = torch.zeros(1, 1)
    m = _Model(window=3, lag=1.0)
    off = _latent_to_pixel_offset(m)            # 1
    assert float(_act_emb_at(m, actions, zero, 4, off).reshape(-1)[0]) == 5.0


def test_unactuated_dataset_still_gets_the_zero_token():
    """`actions is None` is how an unactuated dataset arrives; every pre-swing-up
    number must be reproducible, so this path has to stay exactly zero."""
    zero = torch.zeros(1, 1)
    out = _act_emb_at(_Model(), None, zero, 3, 0)
    assert torch.equal(out, zero)


def test_index_past_the_last_action_falls_back_rather_than_crashing():
    """The final action of an episode is a pad that drives nothing; a rollout that
    reaches it must not index past the end."""
    actions = torch.arange(4, dtype=torch.float32).view(4, 1)
    zero = torch.zeros(1, 1)
    assert torch.equal(_act_emb_at(_Model(), actions, zero, 99, 0), zero)


def test_lewm_window_matches_its_training_context():
    """LeWM consumes one action per context frame: predicting latent frame h+step is
    paired with latent frames [step, step+h-1]. Eval must use the same window or the
    baseline is handicapped and the head-to-head becomes meaningless."""
    actions = torch.arange(10, dtype=torch.float32).view(10, 1)
    zero_ctx = torch.zeros(1, 3, 1)
    m = _Model(window=1, lag=0.0)
    got = _act_emb_window(m, actions, zero_ctx, start_latent=2, n=3, off=0)
    assert got.shape == (1, 3, 1)
    assert [float(v) for v in got.reshape(-1)] == [2.0, 3.0, 4.0]


def test_lewm_window_clamps_at_the_episode_end():
    actions = torch.arange(4, dtype=torch.float32).view(4, 1)
    zero_ctx = torch.zeros(1, 3, 1)
    got = _act_emb_window(_Model(), actions, zero_ctx, start_latent=2, n=3, off=0)
    assert [float(v) for v in got.reshape(-1)] == [2.0, 3.0, 3.0]


def test_lewm_unactuated_falls_back_to_the_zero_context():
    zero_ctx = torch.zeros(1, 3, 1)
    got = _act_emb_window(_Model(), None, zero_ctx, start_latent=0, n=3, off=0)
    assert torch.equal(got, zero_ctx)
