"""Per-episode integrator step h = dt_episode / dt_ref.

Until now the flow carried ONE scalar step, fixed at 1.0 = "one frame", so the
learned V absorbed the frame spacing: in frame units the pendulum obeys
theta'' = -(g/L) dt^2 sin(theta), and a V fitted at one dt is wrong by
(dt/dt')^2 at another. That made mixed-rate training ill posed rather than
merely hard. Passing h per episode divides the dt^2 factor out of the target so
a single V is valid across every arm.

The tests below pin, in order:

  1. the gauge property the ratio form rests on -- rescaling h and H inversely
     leaves the trajectory invariant, which is *why* only the ratio across
     episodes is identifiable and the absolute unit is free;
  2. h = None reproduces the scalar path bit-for-bit, so every existing
     checkpoint and single-rate run is untouched;
  3. per-sample h actually varies per sample (a broadcast bug here would be
     silent -- every episode would get sample 0's step and the run would look
     fine);
  4. the momentum rides the same clock, p = dq/dh, exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "models"))

from lehami.derivative import p_causal_last, p_interp_series  # noqa: E402

from lehami.ham_flow import SeparableHamiltonianFlow  # noqa: E402

D, B = 4, 6


def _flow(dt=1.0, seed=0):
    torch.manual_seed(seed)
    return SeparableHamiltonianFlow(latent_dim=D, action_dim=D, hidden_dim=16,
                                    n_layers=2, dt=dt).eval()


def _qpu(seed=1, b=B):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, D, generator=g),
            torch.randn(b, D, generator=g),
            torch.randn(b, D, generator=g))


# ── 1. the gauge freedom that makes the ratio the only identifiable thing ────

def test_scalar_dt_is_a_gauge_choice():
    """h -> h/λ with H -> λH leaves the step invariant.

    This is the whole argument for h = dt/dt_ref rather than h = dt in seconds:
    the absolute time unit is not identifiable from data, only the ratio across
    episodes is, so dt_ref merely picks where to put a free constant.
    """
    lam = 4.0
    f = _flow(dt=1.0)
    q, p, u = _qpu()
    q1, p1 = f(q, p, u)

    # Scale H by λ (both K and V) and the step by 1/λ.
    g = _flow(dt=1.0 / lam)
    g.load_state_dict(f.state_dict())
    with torch.no_grad():
        for net in (g.H_net.K_net, g.H_net.V_net):
            last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.weight.mul_(lam)
            if last.bias is not None:
                last.bias.mul_(lam)
    q2, p2 = g(q, p, u)
    torch.testing.assert_close(q1, q2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-5)


# ── 2. backward compatibility ───────────────────────────────────────────────

def test_dt_none_reproduces_the_scalar_path_bitwise():
    f = _flow(dt=1.0)
    q, p, u = _qpu()
    a = f(q, p, u)
    b = f(q, p, u, None)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_uniform_per_sample_dt_equals_the_scalar_of_the_same_value(dt):
    """A constant per-sample h must agree with constructing the flow at that h."""
    f = _flow(dt=dt)
    g = _flow(dt=1.0)
    g.load_state_dict(f.state_dict())
    q, p, u = _qpu()
    a = f(q, p, u)
    b = g(q, p, u, torch.full((B,), dt))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


# ── 3. per-sample really means per-sample ───────────────────────────────────

def test_each_sample_uses_its_own_step():
    """Guards the broadcast: a bug here silently gives everyone sample 0's h."""
    f = _flow(dt=1.0)
    q, p, u = _qpu()
    steps = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    q_mixed, p_mixed = f(q, p, u, steps)
    for i, h in enumerate(steps.tolist()):
        qi, pi = f(q[i:i + 1], p[i:i + 1], u[i:i + 1], torch.tensor([h]))
        torch.testing.assert_close(q_mixed[i:i + 1], qi, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(p_mixed[i:i + 1], pi, rtol=1e-5, atol=1e-6)


def test_column_shape_is_accepted_and_batch_mismatch_raises():
    f = _flow(dt=1.0)
    q, p, u = _qpu()
    flat = f(q, p, u, torch.full((B,), 0.7))
    col = f(q, p, u, torch.full((B, 1), 0.7))
    for x, y in zip(flat, col):
        torch.testing.assert_close(x, y)
    with pytest.raises(ValueError, match="per-sample dt has batch"):
        f(q, p, u, torch.full((B + 1,), 0.7))


# ── 4. the momentum rides the same clock ────────────────────────────────────

@pytest.mark.parametrize("h", [0.5, 2.0, 3.0])
def test_momentum_stencils_are_exactly_linear_in_one_over_h(h):
    """p at step h is the frame-unit p divided by h — so the builder's single
    division is exact, and no stencil needed changing."""
    g = torch.Generator().manual_seed(3)
    z = torch.randn(2, 12, D, generator=g)
    torch.testing.assert_close(p_interp_series(z, h), p_interp_series(z, 1.0) / h,
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(p_causal_last(z[:, :4], h), p_causal_last(z[:, :4], 1.0) / h,
                               rtol=1e-6, atol=1e-7)


# ── the builder-side helper ─────────────────────────────────────────────────

def _cfg(dt_ref=None):
    from omegaconf import OmegaConf
    return OmegaConf.create({"model": ({"dt_ref": dt_ref} if dt_ref is not None else {})})


def _episode_h():
    from lehami.builders import _episode_h as f
    return f


def test_episode_h_is_none_without_config_or_column():
    f = _episode_h()
    batch = {"dt_frame": torch.full((3, 5, 1), 0.2)}
    assert f(batch, _cfg(None)) is None          # no dt_ref -> scalar path
    assert f({"pixels": torch.zeros(3)}, _cfg(0.1)) is None   # no column -> scalar path


def test_episode_h_is_the_ratio_and_is_one_at_the_reference():
    f = _episode_h()
    dt = torch.tensor([0.1, 0.2, 0.3])[:, None, None].expand(3, 5, 1).clone()
    h = f({"dt_frame": dt}, _cfg(0.1))
    torch.testing.assert_close(h, torch.tensor([1.0, 2.0, 3.0]))


def test_episode_h_rejects_a_clip_that_straddles_episodes():
    """dt must be constant within a clip; if it is not, the clip crossed a
    boundary and every index in the loss is misaligned."""
    f = _episode_h()
    dt = torch.full((2, 5, 1), 0.1)
    dt[0, 3:] = 0.2
    with pytest.raises(ValueError, match="dt varies within a clip"):
        f({"dt_frame": dt}, _cfg(0.1))


def test_episode_h_rejects_a_nonpositive_reference():
    f = _episode_h()
    with pytest.raises(ValueError, match="dt_ref must be > 0"):
        f({"dt_frame": torch.full((2, 5, 1), 0.1)}, _cfg(0.0))


# ── eval side: it must agree with training, or the arm looks broken ─────────

def _run_eval():
    import importlib
    sys.path.insert(0, str(ROOT / "lehami" / "experiments" / "src"))
    try:
        return importlib.import_module("run_eval")
    except ImportError as e:
        pytest.skip(f"run_eval not importable: {e}")


DATASET = Path.home() / ".stable_worldmodel" / "pendulum_multi_v2.h5"


def test_cfg_get_survives_omegaconf_simplenamespace_and_dict():
    """OmegaConf raises KeyError for a missing key and SimpleNamespace is not an
    OmegaConf node at all; the lookup must tolerate every cfg shape in the repo."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    re = _run_eval()
    assert re._cfg_get(OmegaConf.create({"model": {"dt_ref": 0.1}}), "model.dt_ref") == 0.1
    assert re._cfg_get(OmegaConf.create({"model": {}}), "model.dt_ref") is None
    assert re._cfg_get(SimpleNamespace(model=SimpleNamespace()), "model.dt_ref") is None
    assert re._cfg_get(SimpleNamespace(model=SimpleNamespace(dt_ref=0.2)), "model.dt_ref") == 0.2
    assert re._cfg_get({"model": {"dt_ref": 0.3}}, "model.dt_ref") == 0.3
    assert re._cfg_get(None, "model.dt_ref", "fallback") == "fallback"


@pytest.mark.skipif(not DATASET.exists(), reason="pendulum_multi_v2.h5 not generated here")
def test_eval_h_matches_training_h():
    """Same ratio on both sides. If eval integrated at h=1 while training used
    h=2, every off-reference episode would be advanced half as far as trained
    and the arm would look broken for a reason that is not the model."""
    from omegaconf import OmegaConf
    re = _run_eval()
    cfg = OmegaConf.create({"model": {"dt_ref": 0.1}})
    h = re._episode_dt_ratio(DATASET, cfg)
    assert h is not None
    import h5py
    with h5py.File(DATASET, "r") as f:
        dt = f["dt"][:]
        dt_frame, off = f["dt_frame"][:], f["ep_offset"][:]
    # eval reads the per-EPISODE column, training the per-FRAME one: same values
    assert all(abs(float(dt_frame[off[i], 0]) - float(dt[i])) < 1e-9
               for i in range(0, len(dt), 53))
    # The file holds every arm, including the quarantined x1 one, so h spans
    # 0.5 (x1, held out) .. 2.0 (x4). h = 1 on the x2 reference arm is what
    # makes every single-rate run the exact special case, and h = 0.5 on the
    # OOD arm is precisely the step the model is never trained at.
    assert sorted(round(float(v), 4) for v in set(h.tolist())) == [0.5, 1.0, 1.5, 2.0]


@pytest.mark.skipif(not DATASET.exists(), reason="pendulum_multi_v2.h5 not generated here")
def test_eval_h_is_none_without_dt_ref():
    """No dt_ref -> scalar path, so every pre-existing run scores unchanged."""
    from omegaconf import OmegaConf
    re = _run_eval()
    assert re._episode_dt_ratio(DATASET, OmegaConf.create({"model": {}})) is None


def test_modes_that_ignore_dt_refuse_to_run():
    """Loud beats subtly wrong: the clip/multiclip paths are not dt-aware."""
    from omegaconf import OmegaConf
    re = _run_eval()
    with pytest.raises(NotImplementedError, match="per-episode integrator step"):
        re._assert_scalar_dt_ok(OmegaConf.create({"model": {"dt_ref": 0.1}}), "multiclip")
    re._assert_scalar_dt_ok(OmegaConf.create({"model": {}}), "multiclip")   # no raise
