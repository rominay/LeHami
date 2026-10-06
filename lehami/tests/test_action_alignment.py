"""The action-indexing test that `train_probe_physics.py` did not have.

`_action_emb` translates LATENT frame indices into PIXEL frame indices.  A
window-W encoder is causal and consumes W pixel frames per latent, so latent i
stands for pixel frame i + (W-1) and the action driving a transition out of
latent i is `action[i + W - 1]`.

The failure mode this guards is specific: at W = 1 the offset is zero, so a wrong
implementation is INDISTINGUISHABLE from a right one, runs clean, and only starts
returning misaligned actions once someone raises the window.  That is precisely
how the missing `_align_states` call survived in the probe script.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "models"))

from hjepa.builders import _action_emb  # noqa: E402


class _IdentityEncoder:
    """Mimics TemporalConvEncoder's two declared attributes.

    `lag` is the centroid lag w.r.t. the NEWEST frame in the window:
    (W-1)/2 for the symmetric `full`/`depthwise` variants, 0 for the causal
    `diff` variant (W=2), and 0 for the per-frame encoder.
    """

    def __init__(self, window, lag=None):
        self.window = window
        self.lag = (window - 1) / 2.0 if lag is None else lag


class _Model:
    """Action encoder is the identity, so the test reads the SELECTED indices."""

    def __init__(self, window, lag=None):
        self.encoder = _IdentityEncoder(window, lag)

    @staticmethod
    def action_encoder(a):
        return a


class _Cfg:
    class data:
        action_dim = 1


# (window, lag, expected offset).  The offset is (W-1) - lag, i.e. how far the
# latent's represented frame sits from the START of its window.
@pytest.mark.parametrize("window,lag,off", [
    (1, 0.0, 0),      # per-frame encoder
    (3, 1.0, 1),      # `full` / `depthwise`, symmetric odd window
    (5, 2.0, 2),      # ditto, wider
    (2, 0.0, 1),      # `diff`, the causal W=2 control arm
])
def test_action_index_uses_the_encoder_declared_lag(window, lag, off):
    T = 20
    # action[:, t] == t, so the returned values ARE the indices that were picked
    batch = {"action": torch.arange(T, dtype=torch.float32).view(1, T, 1)}
    for start in (0, 3, 7):
        n = 2
        out = _action_emb(batch, _Model(window, lag), _Cfg, torch.device("cpu"),
                          1, start, n)
        expected = torch.arange(start + off, start + off + n,
                                dtype=torch.float32).view(1, n, 1)
        assert torch.equal(out, expected), (
            f"W={window} lag={lag} start={start}: picked {out.flatten().tolist()}, "
            f"expected {expected.flatten().tolist()}")


def test_symmetric_window_is_not_treated_as_causal():
    """The specific regression: using W-1 instead of (W-1)-lag reads late."""
    batch = {"action": torch.arange(20, dtype=torch.float32).view(1, 20, 1)}
    out = _action_emb(batch, _Model(3, 1.0), _Cfg, torch.device("cpu"), 1, 0, 1)
    assert out.item() == 1.0, "symmetric W=3 must read action[1], not action[2]"


def test_missing_action_column_falls_back_to_zeros():
    """Unactuated datasets must stay bit-identical to the pre-actuation code."""
    out = _action_emb({}, _Model(1), _Cfg, torch.device("cpu"), 4, 0, 3)
    assert out.shape == (4, 3, 1)
    assert torch.count_nonzero(out) == 0


def test_out_of_range_index_raises_rather_than_silently_wrapping():
    """T must be T_eff + (W-1); if a config gets that wrong, fail loudly."""
    batch = {"action": torch.zeros(1, 5, 1)}
    with pytest.raises(AssertionError, match="exceeds the clip"):
        _action_emb(batch, _Model(3, 0.0), _Cfg, torch.device("cpu"), 1, 4, 2)
