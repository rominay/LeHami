"""Shape, parity and alignment tests for the temporal encoder (EXP arms A1/A2/A3/B1/B2).

Covers the checklist in paper/notes/temporal_encoder.tex §sec:te-impl item 5, plus
the two properties the note's argument actually rests on:

  * the temporal kernel is symmetric BY CONSTRUCTION, not by penalty — so the
    module commutes with time reversal of its input (§sec:te-parity); and
  * latent time is offset from pixel time by an exact integer, so eval can
    re-base ground truth with a slice and never an interpolation (§sec:te-lag).

The "diff" variant is the deliberate parity-violating control (arm A3): its tests
assert that it *fails* the reversal check, since an arm that quietly satisfied it
would not be a control at all.

"free" is the UNTIED baseline encoder (EXP-099 b3f, paper/notes/
asymmetric_baseline_encoder.tex).  It is asserted to *fail* the same two checks
for the same reason: its whole purpose is to have the odd component the tie
removes, and an arm that silently kept the symmetry would answer nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "models"))

from hjepa.builders import ConvEncoder, TemporalConvEncoder  # noqa: E402

EMBED = 16
IMG = 32


def _clip(B=2, T=7, img=IMG, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 3, img, img, generator=g)


# ── shapes ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant,window", [("full", 3), ("full", 5),
                                            ("depthwise", 3), ("diff", 2),
                                            ("free", 3), ("free", 5)])
def test_forward_sequence_shape(variant, window):
    enc = TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant).eval()
    x = _clip(B=2, T=9)
    out = enc.forward_sequence(x)
    assert out.shape == (2, 9 - window + 1, EMBED)


def test_window_frames_in_one_embedding_out():
    """W frames in → exactly one embedding out (the project's shape convention)."""
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3).eval()
    assert enc.forward_sequence(_clip(B=1, T=3)).shape == (1, 1, EMBED)


def test_clip_shorter_than_window_raises():
    enc = TemporalConvEncoder(embed_dim=EMBED, window=5).eval()
    with pytest.raises(ValueError, match="clip too short"):
        enc.forward_sequence(_clip(B=1, T=4))


def test_forward_rejects_per_frame_call():
    """A stray per-frame call must fail loudly, never silently encode one frame."""
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3).eval()
    with pytest.raises(RuntimeError, match="consumes a clip"):
        enc(torch.randn(2, 3, IMG, IMG))


@pytest.mark.parametrize("variant,window", [("full", 2), ("full", 4), ("depthwise", 1),
                                            ("diff", 3), ("free", 2), ("free", 4)])
def test_invalid_window_variant_combinations_raise(variant, window):
    with pytest.raises(ValueError):
        TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="full|depthwise|diff|free"):
        TemporalConvEncoder(embed_dim=EMBED, window=3, variant="centred")


@pytest.mark.parametrize("variant,window,lag", [("full", 3, 0), ("depthwise", 3, 0),
                                                ("diff", 2, 1), ("free", 3, 3),
                                                ("free", 3, -1)])
def test_invalid_lag_raises(variant, window, lag):
    """A tied kernel's lag is a theorem, so it may not be overridden; a free
    kernel's must name a tap that exists."""
    with pytest.raises(ValueError):
        TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant, lag=lag)


# ── the temporal-encoder contract (window / lag) ──────────────────────────────

def test_per_frame_encoder_declares_the_contract():
    """ConvEncoder must carry W=1/lag=0 so callers can branch on one attribute."""
    enc = ConvEncoder(embed_dim=EMBED)
    assert enc.window == 1 and enc.lag == 0.0


@pytest.mark.parametrize("variant,window,lag", [("full", 3, 1.0), ("full", 5, 2.0),
                                                ("depthwise", 3, 1.0), ("diff", 2, 0.0),
                                                ("free", 3, 1.0), ("free", 5, 2.0)])
def test_lag_is_the_centroid_lag(variant, window, lag):
    enc = TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant)
    assert enc.lag == lag


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_free_lag_is_declared_not_derived(lag):
    """Unlike the tied variants, "free" takes its lag from the caller — the
    centroid of an untied kernel is learned, so (W-1)/2 is not a theorem here."""
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3, variant="free", lag=lag)
    assert enc.lag == float(lag)
    # ...and the delta init follows it onto the tap the latent represents, so
    # the arm still starts as the per-frame encoder whichever lag is declared.
    w = enc.temporal_weight().detach()
    assert float(w[..., enc.window - 1 - lag].abs().sum()) == pytest.approx(32.0)
    assert float(w.abs().sum()) == pytest.approx(32.0)      # every other tap is 0


@pytest.mark.parametrize("variant,window", [("full", 3), ("full", 5), ("diff", 2),
                                            ("free", 3), ("free", 5)])
def test_latent_time_offset_is_an_integer(variant, window):
    """(W-1) - lag must be an exact integer or GT alignment needs interpolation."""
    enc = TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant)
    shift = (enc.window - 1) - enc.lag
    assert shift == pytest.approx(round(shift), abs=1e-12)


# ── symmetry / parity ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["full", "depthwise"])
@pytest.mark.parametrize("window", [3, 5])
def test_kernel_is_structurally_symmetric(variant, window):
    """Tying is (a, …, b, …, a) by construction — true at init AND after a step."""
    enc = TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant)
    for _ in range(2):
        w = enc.temporal_weight()
        assert w.shape[-1] == window
        torch.testing.assert_close(w, w.flip(-1))
        # perturb the free parameters: symmetry must survive optimisation
        with torch.no_grad():
            enc.w_half.add_(torch.randn_like(enc.w_half))
            enc.w_centre.add_(torch.randn_like(enc.w_centre))


@pytest.mark.parametrize("variant", ["full", "depthwise"])
def test_time_reversal_equivariance(variant):
    """Symmetric kernel ⇒ encoding a time-reversed clip reverses the embeddings.

    This is the check the note calls for: it detects a broken tying that a
    weights-only test would miss (e.g. an asymmetric bias or a flipped conv).
    """
    torch.manual_seed(0)
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3, variant=variant).eval()
    with torch.no_grad():           # move off the delta init so the test has teeth
        enc.w_half.add_(0.3 * torch.randn_like(enc.w_half))
    x = _clip(B=2, T=8, seed=1)
    with torch.no_grad():
        fwd = enc.forward_sequence(x)
        rev = enc.forward_sequence(x.flip(1))
    torch.testing.assert_close(fwd, rev.flip(1), rtol=1e-4, atol=1e-5)


def test_free_variant_is_not_tied():
    """The whole point of "free": the kernel must be able to leave w0 != w2.

    `full` materialises (a, b, a) from half of itself, so no gradient can ever
    separate the outer taps.  `free` stores the kernel outright, so a single
    optimiser step can.
    """
    torch.manual_seed(0)
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3, variant="free")
    torch.testing.assert_close(enc.temporal_weight(),
                               enc.temporal_weight().flip(-1))   # delta init IS even
    with torch.no_grad():
        enc.w_free.add_(torch.randn_like(enc.w_free))
    w = enc.temporal_weight()
    assert not torch.allclose(w, w.flip(-1), rtol=1e-3, atol=1e-3)


def test_free_variant_breaks_time_reversal():
    """With an odd component the encoder stops commuting with time reversal.

    This is the property the arm exists to have: a clip and its reverse share
    every angle but differ in the SIGN of the velocity, so an encoder that maps
    them to the same latents cannot express direction (Proposition 1 of
    paper/notes/asymmetric_baseline_encoder.tex).  `full` satisfies the identity
    below by construction; `free` must be able to violate it.
    """
    torch.manual_seed(0)
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3, variant="free").eval()
    with torch.no_grad():       # a centred difference: the purely odd kernel
        enc.w_free.zero_()
        for c in range(32):
            enc.w_free[c, c, 0] = -1.0
            enc.w_free[c, c, 2] = +1.0
    x = _clip(B=2, T=8, seed=1)
    with torch.no_grad():
        fwd = enc.forward_sequence(x)
        rev = enc.forward_sequence(x.flip(1))
    assert not torch.allclose(fwd, rev.flip(1), rtol=1e-3, atol=1e-3)


def test_odd_ratio_is_zero_for_tied_and_positive_for_free():
    """The training-time diagnostic must be its own control.

    `odd_ratio` is exactly 0 for a tied kernel at every step, so a non-zero
    reading on a `free` arm cannot be an artefact of the measurement.
    """
    from hjepa.builders import _encoder_diagnostics

    z = torch.randn(4, EMBED)
    torch.manual_seed(0)
    tied = TemporalConvEncoder(embed_dim=EMBED, window=3, variant="full")
    with torch.no_grad():
        tied.w_half.add_(torch.randn_like(tied.w_half))
    assert float(_encoder_diagnostics(tied, z)["odd_ratio"]) == pytest.approx(0.0, abs=1e-9)

    free = TemporalConvEncoder(embed_dim=EMBED, window=3, variant="free")
    with torch.no_grad():
        free.w_free.add_(torch.randn_like(free.w_free))
    assert float(_encoder_diagnostics(free, z)["odd_ratio"]) > 1e-3


def test_diff_variant_breaks_time_reversal():
    """Arm A3 is the parity-violating control; assert it really does violate it."""
    torch.manual_seed(0)
    enc = TemporalConvEncoder(embed_dim=EMBED, window=2, variant="diff").eval()
    x = _clip(B=2, T=8, seed=1)
    with torch.no_grad():
        fwd = enc.forward_sequence(x)
        rev = enc.forward_sequence(x.flip(1))
    assert not torch.allclose(fwd, rev.flip(1), rtol=1e-3, atol=1e-3)


# ── init: the module must start as the per-frame baseline ─────────────────────

@pytest.mark.parametrize("variant", ["full", "depthwise", "free"])
@pytest.mark.parametrize("window", [3, 5])
def test_delta_init_reproduces_the_per_frame_encoder(variant, window):
    """At init the temporal tap is (0,…,1,…,0), so A1 == A0 before any training.

    Deliberate deviation from the note, which suggested a (1,1,1)/3 box init:
    with the delta init any A1-vs-A0 gain is attributable to the window rather
    than to a different starting point.
    """
    torch.manual_seed(0)
    tmp = TemporalConvEncoder(embed_dim=EMBED, window=window, variant=variant).eval()
    if variant != "free":       # "free" stores the whole kernel, tied ones a half
        torch.testing.assert_close(tmp.w_half, torch.zeros_like(tmp.w_half))
    w = tmp.temporal_weight()
    centre = w[..., window // 2]
    assert (w.abs().sum() - centre.abs().sum()).abs() < 1e-12
    torch.testing.assert_close(tmp.t_bias, torch.zeros_like(tmp.t_bias))

    # ...and the whole module then equals the per-frame encoder on every frame.
    ref = ConvEncoder(embed_dim=EMBED).eval()
    ref.net[0].load_state_dict(tmp.stem[0].state_dict())
    ref.net[1].load_state_dict(tmp.stem[1].state_dict())
    for i, layer in enumerate(tmp.rest):
        if list(layer.parameters()):
            ref.net[4 + i].load_state_dict(layer.state_dict())
    # The delta tap sits at the kernel CENTRE, so at init the module reproduces
    # the per-frame encoder on the frame latent j *represents* — pixel frame
    # j + (W-1) - lag — not on the newest frame of the window.  That offset is
    # exactly what _align_states applies to the GT states.
    x = _clip(B=1, T=window + 2, seed=3)
    shift = int((tmp.window - 1) - tmp.lag)
    with torch.no_grad():
        got = tmp.forward_sequence(x)[0]
        want = ref(x[0])[shift : shift + got.shape[0]]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_parameter_cost_is_negligible():
    """The note's confound answer: the encoder delta must be a rounding error."""
    base = sum(p.numel() for p in ConvEncoder(embed_dim=192).parameters())
    temp = sum(p.numel() for p in
               TemporalConvEncoder(embed_dim=192, window=3).parameters())
    assert 0 < temp - base < 0.01 * base


# ── eval-side alignment (run_eval helpers) ────────────────────────────────────

def _run_eval():
    import importlib
    sys.path.insert(0, str(ROOT / "lehami" / "experiments" / "src"))
    try:
        return importlib.import_module("run_eval")
    except ImportError as e:      # stable_worldmodel / h5py absent
        pytest.skip(f"run_eval not importable in this environment: {e}")


class _FakeModel:
    def __init__(self, encoder):
        self.encoder = encoder


@pytest.mark.parametrize("enc,shift,window", [
    (ConvEncoder(embed_dim=EMBED), 0, 1),
    (TemporalConvEncoder(embed_dim=EMBED, window=3), 1, 3),
    (TemporalConvEncoder(embed_dim=EMBED, window=5), 2, 5),
    (TemporalConvEncoder(embed_dim=EMBED, window=2, variant="diff"), 1, 2),
])
def test_align_states_recentres_ground_truth(enc, shift, window):
    re = _run_eval()
    m = _FakeModel(enc)
    assert re._enc_window(m) == window
    assert re._enc_shift(m) == shift

    states = np.arange(20, dtype=np.float64).reshape(10, 2)
    out = re._align_states(m, states)
    assert out.shape == (10 - window + 1, 2)
    # latent j must be scored against the state at pixel frame j + shift
    assert out[0, 0] == states[shift, 0]
    assert out[-1, 0] == states[shift + out.shape[0] - 1, 0]


def test_align_states_is_identity_for_per_frame_encoders():
    """Guarantees every pre-existing number is reproduced bit-for-bit."""
    re = _run_eval()
    states = np.arange(20, dtype=np.float64).reshape(10, 2)
    assert re._align_states(_FakeModel(ConvEncoder(embed_dim=EMBED)), states) is states


def test_encode_episode_chunking_matches_a_single_pass():
    """Overlapping chunks must not cut the temporal window at a chunk boundary."""
    re = _run_eval()
    torch.manual_seed(0)
    enc = TemporalConvEncoder(embed_dim=EMBED, window=3).eval()
    with torch.no_grad():
        enc.w_half.add_(0.3 * torch.randn_like(enc.w_half))

    class M:
        def __init__(self, e):
            self.encoder = e

        @staticmethod
        def projector(c):
            return c

    m = M(enc)
    pix = torch.randn(11, 3, IMG, IMG, generator=torch.Generator().manual_seed(4))
    cls_chunked, _ = re._encode_episode(m, pix, enc_batch=3)
    cls_single, _ = re._encode_episode(m, pix, enc_batch=1000)
    assert cls_chunked.shape[0] == 11 - 3 + 1
    torch.testing.assert_close(cls_chunked, cls_single, rtol=1e-5, atol=1e-6)
