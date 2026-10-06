"""Shape + symmetry tests for the even-kinetic-energy option (`even_kinetic`).

CLAUDE.md: "Every new model component needs a shape-assertion test."  The shape
assertions are here, and so are the two properties the option exists for:

  * K(-p) == K(p)  exactly, for every even mode;
  * the leapfrog map is EXACTLY time-reversible,
        R o Psi_h^K o R o Psi_h^K = Id      with R(q,p) = (q,-p),
    to floating-point roundoff and no further.

Also pinned here are the two things that must NOT change: the default
(`even_kinetic=False`) reproduces the historical unconstrained K bit for bit,
and the `symmetrise`/`square` modes keep the same state_dict keys and shapes so
existing checkpoints stay loadable.

Run with:
    uv run pytest lehami/tests/test_ham_flow_even.py -v

See paper/notes/even_kinetic_energy.tex for the derivation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "models"))

from hjepa.ham_flow import (  # noqa: E402
    HamiltonianPredictor,
    SeparableHamiltonianFlow,
    SeparableHamiltonianNet,
    momentum_parity,
    normalize_even_kinetic,
)

B, D, A = 8, 16, 3
HID, NL = 32, 3
SEED = 0

EVEN_MODES = ["symmetrise", "square", "quadratic"]
ALL_MODES = ["none"] + EVEN_MODES


def make_qpu(seed: int = SEED, scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, D, generator=g) * scale
    p = torch.randn(B, D, generator=g) * scale
    u = torch.randn(B, A, generator=g)
    return q, p, u


def make_flow(mode, seed: int = SEED, **kw) -> SeparableHamiltonianFlow:
    torch.manual_seed(seed)
    return SeparableHamiltonianFlow(
        latent_dim=D, action_dim=A, hidden_dim=HID, n_layers=NL,
        even_kinetic=mode, **kw,
    )


def roll(flow, q, p, u, n):
    for _ in range(n):
        q, p = flow(q, p, u)
    return q, p


# ── config parsing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spec,expected",
    [
        (False, "none"), (None, "none"), ("none", "none"), ("off", "none"),
        (True, "symmetrise"), ("symmetrise", "symmetrise"),
        ("Symmetrize", "symmetrise"), ("sym", "symmetrise"),
        ("square", "square"), ("p2", "square"), ("squared", "square"),
        ("quadratic", "quadratic"), ("quad", "quadratic"),
    ],
)
def test_normalize_even_kinetic(spec, expected):
    assert normalize_even_kinetic(spec) == expected


def test_normalize_even_kinetic_rejects_garbage():
    with pytest.raises(ValueError, match="even_kinetic must be one of"):
        normalize_even_kinetic("odd")


# ── shape assertions ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ALL_MODES)
def test_net_K_shape(mode):
    torch.manual_seed(SEED)
    net = SeparableHamiltonianNet(D, A, HID, NL, even_kinetic=mode)
    _, p, _ = make_qpu()
    K = net.K(p)
    assert K.shape == (B,), f"{mode}: expected ({B},), got {tuple(K.shape)}"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_net_H_shape(mode):
    torch.manual_seed(SEED)
    net = SeparableHamiltonianNet(D, A, HID, NL, even_kinetic=mode)
    q, p, u = make_qpu()
    assert net(q, p, u).shape == (B,)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_flow_step_shapes(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu()
    q1, p1 = flow(q, p, u)
    assert q1.shape == (B, D), f"{mode}: q_next {tuple(q1.shape)}"
    assert p1.shape == (B, D), f"{mode}: p_next {tuple(p1.shape)}"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_flow_dtype_preserved(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu()
    q1, p1 = flow(q, p, u)
    assert q1.dtype == q.dtype and p1.dtype == p.dtype


@pytest.mark.parametrize("mode", ALL_MODES)
def test_eval_H_and_symplectic_grads_shapes(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu()
    assert flow.eval_H(q, p, u).shape == (B,)
    dHdq, dHdp = flow.symplectic_grads(q, p, u, create_graph=False)
    assert dHdq.shape == (B, D) and dHdp.shape == (B, D)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_predictor_shapes(mode):
    torch.manual_seed(SEED)
    T = 5
    pred = HamiltonianPredictor(
        embed_dim=2 * D, latent_dim=D, action_emb_dim=A, hidden_dim=HID,
        n_layers=NL, phase_space=True, separable=True, even_kinetic=mode,
    )
    x = torch.randn(B, T, 2 * D)
    c = torch.randn(B, T, A)
    assert pred(x, c).shape == (B, T, 2 * D)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_flow_rejects_wrong_latent_dim(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu()
    with pytest.raises(AssertionError, match="expected latent_dim"):
        flow(q[:, :-1], p[:, :-1], u)


# ── the property: K is even ──────────────────────────────────────────────────

@pytest.mark.parametrize("mode", EVEN_MODES)
def test_K_is_even(mode):
    torch.manual_seed(SEED)
    net = SeparableHamiltonianNet(D, A, HID, NL, even_kinetic=mode)
    _, p, _ = make_qpu()
    err = (net.K(p) - net.K(-p)).abs().max().item()
    assert err < 1e-6, f"{mode}: K(p) != K(-p), max |diff| = {err:.3e}"


def test_unconstrained_K_is_not_even():
    """The control: the historical default breaks the symmetry (that is the bug)."""
    torch.manual_seed(SEED)
    net = SeparableHamiltonianNet(D, A, HID, NL, even_kinetic=False)
    _, p, _ = make_qpu()
    asym = (net.K(p) - net.K(-p)).abs().mean().item()
    span = (net.K(p) - net.K(torch.zeros_like(p))).abs().mean().item()
    assert asym > 1e-3 * span, "unconstrained K happened to be even — retune the test"


@pytest.mark.parametrize("mode", EVEN_MODES)
def test_dHdp_is_odd(mode):
    """K even <=> grad K odd. This is the form the integrator actually uses."""
    flow = make_flow(mode)
    q, p, u = make_qpu()
    _, g_pos = flow.symplectic_grads(q, p, u, create_graph=False)
    _, g_neg = flow.symplectic_grads(q, -p, u, create_graph=False)
    err = (g_pos + g_neg).abs().max().item()
    assert err < 1e-5, f"{mode}: dH/dp not odd, max |g(p)+g(-p)| = {err:.3e}"


@pytest.mark.parametrize("mode", EVEN_MODES)
def test_momentum_parity_reports_pass(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu()
    m = momentum_parity(flow, q, p, u)
    assert m["rho_odd"] > 1 - 1e-8, f"{mode}: rho_odd = {m['rho_odd']}"
    assert m["H_asym_rel"] < 1e-6, f"{mode}: H_asym_rel = {m['H_asym_rel']}"
    # non-degeneracy: an even-but-constant K would pass the two lines above
    assert m["drift_norm"] > 1e-4, f"{mode}: K collapsed, drift = {m['drift_norm']}"


def test_momentum_parity_reports_fail_for_unconstrained():
    flow = make_flow("none")
    q, p, u = make_qpu()
    m = momentum_parity(flow, q, p, u)
    assert m["rho_odd"] < 0.99, f"rho_odd = {m['rho_odd']} — expected a broken symmetry"


# ── the payoff: exact time reversal ──────────────────────────────────────────

@pytest.mark.parametrize("mode", EVEN_MODES)
@pytest.mark.parametrize("K", [1, 5, 25])
def test_p_flip_roundtrip_is_exact(mode, K):
    """R o Psi^K o R o Psi^K = Id, exactly, when grad K is odd."""
    flow = make_flow(mode)
    q0, p0, u = make_qpu()
    qf, pf = roll(flow, q0, p0, u, K)
    qb, pb = roll(flow, qf, -pf, u, K)
    qb, pb = qb, -pb
    rel = (torch.cat([qb - q0, pb - p0], -1).norm()
           / torch.cat([q0, p0], -1).norm()).item()
    assert rel < 1e-4, f"{mode} K={K}: p-flip round trip rel err {rel:.3e}"


@pytest.mark.parametrize("K", [5, 25])
def test_p_flip_roundtrip_fails_for_unconstrained(K):
    """The control that makes the previous test meaningful."""
    flow = make_flow("none")
    q0, p0, u = make_qpu()
    qf, pf = roll(flow, q0, p0, u, K)
    qb, pb = roll(flow, qf, -pf, u, K)
    rel = (torch.cat([qb - q0, -pb - p0], -1).norm()
           / torch.cat([q0, p0], -1).norm()).item()
    assert rel > 1e-2, f"unconstrained K recovered the state (rel err {rel:.3e})"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_dt_flip_roundtrip_is_exact_for_every_mode(mode):
    """Self-adjointness Psi_h^{-1} = Psi_{-h} holds for ANY K.

    This is the control that separates the integrator's symmetry (no learning
    content) from the model's time-reversal symmetry (the scientific claim).
    """
    flow = make_flow(mode)
    q0, p0, u = make_qpu()
    K = 25
    qf, pf = roll(flow, q0, p0, u, K)
    dt0 = flow.dt
    try:
        flow.dt = -dt0
        qb, pb = roll(flow, qf, pf, u, K)
    finally:
        flow.dt = dt0
    rel = (torch.cat([qb - q0, pb - p0], -1).norm()
           / torch.cat([q0, p0], -1).norm()).item()
    assert rel < 1e-4, f"{mode}: dt-flip round trip rel err {rel:.3e}"


@pytest.mark.parametrize("mode", EVEN_MODES)
def test_reversal_survives_p_scale(mode):
    """p -> -p commutes with division by a positive (per-dim) scale."""
    for kw in ({"p_scale": 0.37}, {"p_scale": 0.52, "learnable_scale": True}):
        flow = make_flow(mode, **kw)
        if flow.log_p_scale is not None:  # make the per-dim scales non-uniform
            with torch.no_grad():
                flow.log_p_scale.add_(torch.linspace(-0.5, 0.5, D))
        q0, p0, u = make_qpu()
        qf, pf = roll(flow, q0, p0, u, 10)
        qb, pb = roll(flow, qf, -pf, u, 10)
        rel = (torch.cat([qb - q0, -pb - p0], -1).norm()
               / torch.cat([q0, p0], -1).norm()).item()
        assert rel < 1e-4, f"{mode} {kw}: rel err {rel:.3e}"


# ── backwards compatibility ──────────────────────────────────────────────────

def test_default_is_off():
    flow = make_flow(False)
    assert flow.even_kinetic == "none"
    assert SeparableHamiltonianFlow(D, A, HID, NL).even_kinetic == "none"


def test_default_matches_pre_change_behaviour():
    """`even_kinetic` unset must reproduce the plain K_net path exactly."""
    flow = make_flow("none")
    _, p, _ = make_qpu()
    assert torch.equal(flow.H_net.K(p), flow.H_net.K_net(p).squeeze(-1))


@pytest.mark.parametrize("mode", ["symmetrise", "square"])
def test_mlp_modes_are_checkpoint_compatible(mode):
    """Same parameter shapes as `none`, so old checkpoints still load."""
    ref = make_flow("none").state_dict()
    new = make_flow(mode).state_dict()
    assert set(ref) == set(new)
    assert all(ref[k].shape == new[k].shape for k in ref)
    make_flow(mode).load_state_dict(ref)  # must not raise


def test_quadratic_has_its_own_parameterisation():
    sd = make_flow("quadratic").state_dict()
    assert "H_net.L_kin" in sd and sd["H_net.L_kin"].shape == (D, D)
    assert not any(k.startswith("H_net.K_net") for k in sd)


def test_quadratic_inits_to_unit_mass():
    """L = I at init, so K = ||p||^2 / 2 — the physical starting point."""
    flow = make_flow("quadratic")
    _, p, _ = make_qpu()
    assert torch.allclose(flow.H_net.K(p), 0.5 * p.pow(2).sum(-1), atol=1e-6)


def test_joint_H_rejects_even_kinetic():
    with pytest.raises(ValueError, match="even_kinetic requires separable=True"):
        HamiltonianPredictor(
            embed_dim=2 * D, latent_dim=D, action_emb_dim=A, hidden_dim=HID,
            n_layers=NL, phase_space=True, separable=False, even_kinetic=True,
        )


# ── training-path sanity (no training is run; just that gradients exist) ─────

@pytest.mark.parametrize("mode", ALL_MODES)
def test_gradients_reach_kinetic_parameters(mode):
    flow = make_flow(mode)
    flow.train()
    q, p, u = make_qpu()
    q1, p1 = flow(q, p, u)
    (q1.pow(2).sum() + p1.pow(2).sum()).backward()
    kin = [pr for n, pr in flow.named_parameters() if "K_net" in n or "L_kin" in n]
    assert kin, f"{mode}: no kinetic parameters found"
    assert any(pr.grad is not None and pr.grad.abs().sum() > 0 for pr in kin), (
        f"{mode}: no gradient reached K"
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_step_is_finite(mode):
    flow = make_flow(mode)
    q, p, u = make_qpu(scale=3.0)
    q1, p1 = roll(flow, q, p, u, 20)
    assert torch.isfinite(q1).all() and torch.isfinite(p1).all()
