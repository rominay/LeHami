"""Hamiltonian flow predictors over latent phase-space coordinates (q, p).

`HamiltonianPredictor` is the predictor the Hamiltonian models plug into the JEPA
in place of LeWorldModel's ARPredictor. It wraps one of two flows:

  SeparableHamiltonianFlow  H = K(p / s) + V(q, u), integrated with the leapfrog;
                            exactly symplectic. Used by every model in the paper.
  HamiltonianFlow           a single network H(q, p, u), integrated with explicit
                            Euler or Störmer-Verlet (`euler_step`,
                            `stormer_verlet_step`); a non-separable reference.

Caution:
  - torch.compile must NOT be used on this module (autograd through H is incompatible).
  - All gradient computations run in fp32 even under bf16 training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class HamiltonianNet(nn.Module):
    """MLP parameterizing the scalar Hamiltonian H_phi(q, p, u) -> R."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        in_dim = 2 * latent_dim + action_dim
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            is_output = i == len(dims) - 2
            # No bias on the output: a constant energy shift has no effect on
            # dH/dq or dH/dp, so the bias would be a permanently dead parameter.
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not is_output))
            if not is_output:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, q: Tensor, p: Tensor, u: Tensor) -> Tensor:
        """Returns H(q, p, u), shape (B,)."""
        return self.net(torch.cat([q, p, u], dim=-1)).squeeze(-1)


def _ensure_grad(t: Tensor) -> Tensor:
    """Return t with requires_grad=True, safe for both training and test tensors.

    During training, t is the encoder output (non-leaf with grad_fn) and already
    requires grad — returned as-is so the outer graph stays connected.
    In test code, t is a plain torch.randn leaf (no grad) — detached and marked
    so that autograd.grad can differentiate w.r.t. it.
    """
    if t.requires_grad or t.grad_fn is not None:
        return t
    return t.detach().requires_grad_(True)


def euler_step(
    H_fn: HamiltonianNet, q: Tensor, p: Tensor, u: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """One forward-Euler step of Hamilton's equations.

    Runs in fp32 with autocast disabled so that autograd.grad is safe under
    bf16-mixed training and inside torch.no_grad() (e.g., rollout eval).
    create_graph is only True when the outer context has gradients enabled
    (training); during inference it is False, which is ~3-5x faster.
    """
    create_graph = torch.is_grad_enabled()
    device_type = "cuda" if q.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        with torch.enable_grad():
            q32 = _ensure_grad(q.float())
            p32 = _ensure_grad(p.float())
            H = H_fn(q32, p32, u.float())
            dHdq, dHdp = torch.autograd.grad(H.sum(), [q32, p32], create_graph=create_graph)
    return (q32 + dt * dHdp).to(q.dtype), (p32 - dt * dHdq).to(q.dtype)


def stormer_verlet_step(
    H_fn: HamiltonianNet, q: Tensor, p: Tensor, u: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """One Störmer-Verlet (leapfrog) step — symplectic, lower drift than Euler.

    Action u is treated as constant across the three sub-steps.
    Costs three H_fn evaluations vs one for Euler.
    create_graph is only True when the outer context has gradients enabled
    (training); during inference it is False, which is ~3-5x faster.
    """
    create_graph = torch.is_grad_enabled()
    device_type = "cuda" if q.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        with torch.enable_grad():
            q32 = _ensure_grad(q.float())
            p32 = _ensure_grad(p.float())
            u32 = u.float()

            # Half momentum: p_{n+1/2} = p_n - (dt/2) * dH/dq(q_n, p_n)
            H1 = H_fn(q32, p32, u32)
            (dHdq1,) = torch.autograd.grad(H1.sum(), [q32], create_graph=create_graph, retain_graph=create_graph)
            p_half = p32 - 0.5 * dt * dHdq1

            # Full position: q_{n+1} = q_n + dt * dH/dp(q_n, p_{n+1/2})
            H2 = H_fn(q32, p_half, u32)
            (dHdp,) = torch.autograd.grad(H2.sum(), [p_half], create_graph=create_graph)
            q_next = q32 + dt * dHdp

            # Half momentum: p_{n+1} = p_{n+1/2} - (dt/2) * dH/dq(q_{n+1}, p_{n+1/2})
            H3 = H_fn(q_next, p_half, u32)
            (dHdq2,) = torch.autograd.grad(H3.sum(), [q_next], create_graph=create_graph)
            p_next = p_half - 0.5 * dt * dHdq2

    return q_next.to(q.dtype), p_next.to(q.dtype)


_INTEGRATORS = {"euler": euler_step, "stormer_verlet": stormer_verlet_step}


# ── Even kinetic energy / exact time-reversal symmetry ───────────────────────
#
# The time-reversal map is R(q, p) = (q, -p) together with t -> -t.  For a
# separable H = K(p) + V(q, u) the leapfrog map Psi_h is R-reversible,
#
#     R o Psi_h o R = Psi_h^{-1} = Psi_{-h},
#
# if and only if grad K is ODD, equivalently K is EVEN: K(-p) = K(p).  That is
# an EXACT statement about the discrete map, not an O(h^k) one — the round trip
# "roll forward K steps, flip p, roll forward K steps, flip p" returns the
# initial state up to floating-point roundoff only.
#
# This is a DIFFERENT property from symplecticity.  The separable leapfrog is
# exactly symplectic for ANY K (each sub-step is a shear with unit Jacobian);
# evenness buys nothing there.  Conversely an even K does not make energy
# exactly conserved (that is still only O(h^2)-bounded via the shadow
# Hamiltonian).
#
# Modes:
#   "none"        unconstrained MLP K(p) — the default, NOT even.
#   "symmetrise"  K(p) <- 1/2 [K_net(p) + K_net(-p)].  Orthogonal projection of
#                 the unconstrained class onto its even subspace: it loses no
#                 even function, keeps the exact same parameter shapes (so
#                 checkpoints stay loadable), and costs one extra K_net call.
#   "square"      K(p) = f(p . p) elementwise-squared features.  Even, but also
#                 invariant under the much larger sign group Z_2^d, which
#                 forbids cross terms p_i p_j — i.e. it forces a mass tensor
#                 that is diagonal in the encoder's arbitrary latent basis.
#   "quadratic"   K(p) = 1/2 ||L p||^2 with L a free d x d matrix, i.e.
#                 K = 1/2 p^T M^{-1} p with M^{-1} = L^T L positive
#                 semi-definite.  The physically exact form for a mechanical
#                 system in canonical coordinates.  Different parameter shapes
#                 from the MLP modes.
_EVEN_KINETIC_MODES = ("none", "symmetrise", "square", "quadratic")

_EVEN_KINETIC_ALIASES = {
    "false": "none",
    "off": "none",
    "true": "symmetrise",
    "on": "symmetrise",
    "sym": "symmetrise",
    "symmetrize": "symmetrise",
    "symmetrised": "symmetrise",
    "symmetrized": "symmetrise",
    "p2": "square",
    "squared": "square",
    "quad": "quadratic",
}


def normalize_even_kinetic(spec: bool | str | None) -> str:
    """Canonicalise the ``even_kinetic`` config value to one of _EVEN_KINETIC_MODES.

    ``False``/``None`` -> ``"none"`` (default, existing behaviour unchanged);
    ``True`` -> ``"symmetrise"``.  String values are lower-cased and de-aliased.
    """
    if spec is None or spec is False:
        return "none"
    if spec is True:
        return "symmetrise"
    key = str(spec).strip().lower()
    key = _EVEN_KINETIC_ALIASES.get(key, key)
    if key not in _EVEN_KINETIC_MODES:
        raise ValueError(
            f"even_kinetic must be one of {list(_EVEN_KINETIC_MODES)} "
            f"(or a bool), got {spec!r}"
        )
    return key


# ── Separable Hamiltonian (SRNN, Chen et al. ICLR 2020 https://arxiv.org/pdf/1909.13334) ──────────────────────


class SeparableHamiltonianNet(nn.Module):
    """Two independent MLPs enforcing H(q,p,u) = K(p) + V(q,u).

    K depends only on p; V depends on q and u.  Because no single network sees
    both q and p, dK/dp never touches q and dV/dq never touches p — the leapfrog
    integrator is then exactly symplectic, not merely approximately so.

    Args:
        even_kinetic: constrain K to be an EVEN function of p, which makes the
            leapfrog map exactly time-reversible (see the module-level note).
            Defaults to ``False`` = ``"none"`` so existing runs are unchanged.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        even_kinetic: bool | str = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.even_kinetic = normalize_even_kinetic(even_kinetic)
        if self.even_kinetic == "quadratic":
            # K(p) = 1/2 ||L p||^2.  Init L = I  =>  K = 1/2 ||p||^2 (unit mass),
            # the correct physical starting point rather than a random form.
            self.K_net = None
            self.L_kin = nn.Parameter(torch.eye(latent_dim))
        else:
            self.K_net = self._build_mlp(latent_dim, hidden_dim, n_layers)
        self.V_net = self._build_mlp(latent_dim + action_dim, hidden_dim, n_layers)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            is_out = i == len(dims) - 2
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not is_out))
            if not is_out:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def K(self, p: Tensor) -> Tensor:
        """Kinetic energy K(p), shape (B,).

        Even in p (K(-p) == K(p) bitwise, up to summation order) for every
        ``even_kinetic`` mode except ``"none"``.
        """
        mode = self.even_kinetic
        if mode == "none":
            return self.K_net(p).squeeze(-1)
        if mode == "symmetrise":
            # Both signs in ONE forward pass: cheaper, and it would also keep
            # any future normalisation layer from seeing different statistics
            # for the two branches.
            n = p.shape[0]
            both = self.K_net(torch.cat([p, -p], dim=0))
            return (0.5 * (both[:n] + both[n:])).squeeze(-1)
        if mode == "square":
            return self.K_net(p * p).squeeze(-1)
        # quadratic: 1/2 * p^T (L^T L) p, manifestly even and PSD.
        return 0.5 * torch.nn.functional.linear(p, self.L_kin).pow(2).sum(-1)

    def V(self, q: Tensor, u: Tensor) -> Tensor:
        """Potential energy V(q, u), shape (B,)."""
        return self.V_net(torch.cat([q, u], dim=-1)).squeeze(-1)

    def forward(self, q: Tensor, p: Tensor, u: Tensor) -> Tensor:
        """Total energy H = K(p) + V(q, u), shape (B,)."""
        return self.K(p) + self.V(q, u)


class SeparableHamiltonianFlow(nn.Module):
    """Leapfrog flow on a separable Hamiltonian H = K(p / s) + V(q, u).

    Implements equations 2 & 4 from Chen et al., ICLR 2020 (SRNN):

        p_{n+1/2} = p_n        - (dt/2) · dV/dq(q_n, u)
        q_{n+1}   = q_n        + dt     · dK/dp(p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (dt/2) · dV/dq(q_{n+1}, u)

    Exactly symplectic because K never sees q and V never sees p — contrast with
    stormer_verlet_step which uses a single H(q,p,u) net and is only approximately
    symplectic for non-separable Hamiltonians.

    Momentum scale.  K is evaluated on p / s, with s a per-dimension scale
    (``p_scale``; learned when ``learnable_scale``, as s = exp(log_p_scale)).  It
    rescales the momentum coordinates, i.e. acts as a learned diagonal mass, and
    absorbs the relative scale of q and p that self-supervised training leaves
    free.  Because s is part of H, dK/dp = (1/s) K'(p/s) carries it into the
    position update only, and the step above stays the exact leapfrog of H.

    With ``even_kinetic`` set (K even in p) the map is additionally exactly
    time-reversible: R o Psi_h o R = Psi_h^{-1} for R(q,p) = (q,-p).  Symplectic
    and reversible are independent properties; see the module-level note.

    Drop-in for HamiltonianFlow: same forward(q, p, u) and eval_H signatures.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dt: float = 1.0,
        p_scale: float = 1.0,
        learnable_scale: bool = False,
        even_kinetic: bool | str = False,
    ):
        super().__init__()
        self.H_net = SeparableHamiltonianNet(
            latent_dim, action_dim, hidden_dim, n_layers, even_kinetic=even_kinetic
        )
        self.even_kinetic = self.H_net.even_kinetic
        self.dt = dt
        self.latent_dim = latent_dim
        if learnable_scale:
            # Per-dimension log scale: exp(log_p_scale[i]) rescales p[:,i] before K_net.
            # Initialized from p_scale so a non-unit starting value is preserved.
            # Log parameterization keeps the scale strictly positive.
            self.log_p_scale = nn.Parameter(
                torch.full((latent_dim,), math.log(float(p_scale)))
            )
        else:
            self.register_buffer("p_scale", torch.tensor(float(p_scale)))
            self.log_p_scale = None  # sentinel: use buffer path

    def _get_p_scale(self) -> Tensor:
        """Return per-dimension scale as a (latent_dim,) tensor.

        Note for ``even_kinetic``: K is applied to p / scale, and p -> -p
        commutes with division by any (elementwise) scale, so the rescaling
        never breaks evenness — including the learnable log-parameterised
        variant, whose exp() is strictly positive at every step of training.
        An additive OFFSET on p would break it; there is none, and none
        should be added.
        """
        if self.log_p_scale is not None:
            return self.log_p_scale.exp()
        return self.p_scale  # scalar buffer, broadcasts fine

    def eval_H(self, q: Tensor, p: Tensor, u: Tensor) -> Tensor:
        """K(p/scale) + V(q, u) without autograd setup."""
        q32, p32, u32 = q.float(), p.float(), u.float()
        scale = self._get_p_scale().to(p32.dtype)
        return self.H_net.K(p32 / scale) + self.H_net.V(q32, u32)

    def _resolve_dt(self, dt: Tensor | float | None, q: Tensor):
        """Scalar `self.dt`, or a per-sample step broadcast to (B, 1)."""
        if dt is None:
            return self.dt
        if isinstance(dt, Tensor):
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            if dt.shape[0] != q.shape[0]:
                raise ValueError(
                    f"per-sample dt has batch {dt.shape[0]}, expected {q.shape[0]}")
            return dt.to(device=q.device, dtype=torch.float32)
        return dt

    def forward(self, q: Tensor, p: Tensor, u: Tensor,
                dt: Tensor | float | None = None) -> tuple[Tensor, Tensor]:
        """One leapfrog step (eq 4, SRNN paper).

        Args:
            q: (B, latent_dim) position-like latent coordinates.
            p: (B, latent_dim) momentum-like latent coordinates.
            u: (B, action_dim) encoded action embedding.
            dt: optional per-sample step size, (B,) or (B, 1). Defaults to the
                scalar `self.dt`.

        MIXED SAMPLING RATES. `dt` is the integrator step in GAUGE time, not
        seconds: h = dt_episode / dt_ref, so h = 1 on the reference rate
        (and whenever no per-episode step is given).  Only the
        RATIO across episodes is identifiable — rescaling t -> λt with H -> H/λ
        leaves every trajectory invariant, so the absolute unit is a gauge
        freedom absorbed into the learned H, and fixing dt_ref merely picks
        where to put it.  Passing per-episode h is what makes ONE V valid across
        several sampling rates: in frame units the dynamics carry a factor
        dt^2, and dividing it out through h removes it from the target.
        The momentum must be built on the same clock (p = dq/dh, i.e. the
        frame-unit stencil divided by h) or K(p) sees an inconsistent scale.

        Returns:
            q_next, p_next — each (B, latent_dim).
        """
        assert q.shape == p.shape, f"q/p shape mismatch: {q.shape} vs {p.shape}"
        assert q.shape[-1] == self.latent_dim, (
            f"expected latent_dim={self.latent_dim}, got {q.shape[-1]}"
        )
        dt = self._resolve_dt(dt, q)
        scale = self._get_p_scale()

        create_graph = torch.is_grad_enabled()
        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            with torch.enable_grad():
                q32 = _ensure_grad(q.float())
                p32 = _ensure_grad(p.float())
                u32 = u.float()

                # p_{n+1/2} = p_n - (dt/2) * dV/dq(q_n, u)
                V1 = self.H_net.V(q32, u32)
                (dVdq1,) = torch.autograd.grad(
                    V1.sum(), [q32], create_graph=create_graph, retain_graph=create_graph
                )
                p_half = p32 - 0.5 * dt * dVdq1

                # q_{n+1} = q_n + dt * dK/dp(p_{n+1/2})
                K1 = self.H_net.K(p_half / scale.to(p_half.dtype))
                (dKdp,) = torch.autograd.grad(K1.sum(), [p_half], create_graph=create_graph)
                q_next = q32 + dt * dKdp

                # p_{n+1} = p_{n+1/2} - (dt/2) * dV/dq(q_{n+1}, u)
                V2 = self.H_net.V(q_next, u32)
                (dVdq2,) = torch.autograd.grad(V2.sum(), [q_next], create_graph=create_graph)
                p_next = p_half - 0.5 * dt * dVdq2

        return q_next.to(q.dtype), p_next.to(q.dtype)

    def symplectic_grads(
        self, q: Tensor, p: Tensor, u: Tensor, create_graph: bool | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return the symplectic gradient (dH/dq, dH/dp) at (q, p, u).

        These are the two fields the HNN loss supervises directly (rather than
        the integrated leapfrog step): Hamilton's equations read
        dq/dt = dH/dp and dp/dt = -dH/dq, so a caller matches dH/dp against an
        observed dq/dt and dH/dq against -dp/dt.

        For the separable H = K(p/scale) + V(q, u):
          dH/dq = dV/dq(q, u)
          dH/dp = (1/scale) · K'(p/scale)
        which are exactly the dVdq / dKdp the leapfrog step computes internally,
        kept here as a standalone, integrator-free entry point.

        Args:
            q, p: (B, latent_dim).  u: (B, action_dim) action embedding.
            create_graph: keep the graph for second-order backprop. Defaults to
                ``self.training`` (True during training so the HNN loss is
                differentiable w.r.t. H_net params; False otherwise).

        Returns:
            (dHdq, dHdp), each (B, latent_dim), in the input dtype.
        """
        assert q.shape == p.shape, f"q/p shape mismatch: {q.shape} vs {p.shape}"
        assert q.shape[-1] == self.latent_dim, (
            f"expected latent_dim={self.latent_dim}, got {q.shape[-1]}"
        )
        if create_graph is None:
            create_graph = self.training
        scale = self._get_p_scale()

        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            with torch.enable_grad():
                q32 = _ensure_grad(q.float())
                p32 = _ensure_grad(p.float())
                u32 = u.float()
                V = self.H_net.V(q32, u32)
                (dHdq,) = torch.autograd.grad(V.sum(), [q32], create_graph=create_graph)
                K = self.H_net.K(p32 / scale.to(p32.dtype))
                (dHdp,) = torch.autograd.grad(K.sum(), [p32], create_graph=create_graph)
        return dHdq.to(q.dtype), dHdp.to(p.dtype)


class HamiltonianPredictor(nn.Module):
    """ARPredictor drop-in that applies a Hamiltonian flow in latent (q, p) space.

    Two modes, selected by ``phase_space``:

    **z-space mode** (``phase_space=False``, default)
        The encoder produces z ∈ R^{embed_dim}.  Linear projections
        z → (q, p) → z' bridge between JEPA space and phase space.
        Loss is computed in z space (embed_dim).

    **Direct phase-space mode** (``phase_space=True``)
        The encoder already outputs (q, p) ∈ R^{2·latent_dim}
        (``embed_dim`` must equal ``2 * latent_dim``).  No projections
        are created; the flow operates directly in the representation space
        and Liouville's theorem applies without interference from the
        round-trip z ↔ (q, p).  Loss is computed in phase space (2·latent_dim).

    Args:
        embed_dim:      Input/output dimension (z-space or phase-space, depending on mode).
        latent_dim:     d — phase space is 2·d; q and p each have dimension d.
        action_emb_dim: Dimension of action embeddings from Embedder.
        phase_space:    If True, operate directly in (q, p) space (no projections).
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        action_emb_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        integrator: str = "stormer_verlet",
        dt: float = 1.0,
        phase_space: bool = False,
        p_scale: float = 1.0,
        separable: bool = False,
        learnable_scale: bool = False,
        even_kinetic: bool | str = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.phase_space = phase_space
        self.even_kinetic = normalize_even_kinetic(even_kinetic)

        if phase_space:
            if embed_dim != 2 * latent_dim:
                raise ValueError(
                    f"phase_space=True requires embed_dim == 2 * latent_dim, "
                    f"got embed_dim={embed_dim}, latent_dim={latent_dim}"
                )
        else:
            self.z_to_qp = nn.Linear(embed_dim, 2 * latent_dim)
            self.qp_to_z = nn.Linear(2 * latent_dim, embed_dim)

        if separable:
            self.flow: HamiltonianFlow | SeparableHamiltonianFlow = SeparableHamiltonianFlow(
                latent_dim, action_emb_dim, hidden_dim, n_layers, dt, p_scale,
                learnable_scale=learnable_scale, even_kinetic=self.even_kinetic,
            )
        else:
            if self.even_kinetic != "none":
                raise ValueError(
                    "even_kinetic requires separable=True. There is no K(p) to make "
                    "even in the joint-H model, and — more importantly — the explicit "
                    "stormer_verlet_step on a non-separable H is not a self-adjoint "
                    "composition, so making H even in p would still not give exact "
                    "time-reversibility."
                )
            self.flow = HamiltonianFlow(
                latent_dim, action_emb_dim, hidden_dim, n_layers, integrator, dt, p_scale,
                learnable_scale=learnable_scale,
            )

    def forward(self, x: Tensor, c: Tensor,
                dt: Tensor | float | None = None) -> Tensor:
        """Apply one Hamiltonian step at each position in the context window.

        Args:
            x: (B, T, embed_dim) — embeddings in encoder/JEPA space.
            c: (B, T, action_emb_dim) — action embeddings from Embedder.
            dt: optional per-sample integrator step, (B,) or (B, T). Broadcast
                over the context window and flattened to (B*T, 1) alongside x.
                None keeps the scalar the flow was constructed with.

        Returns:
            (B, T, embed_dim) — predicted embeddings in the same space as x.
        """
        B, T, _ = x.shape
        d = self.latent_dim

        x_flat = x.reshape(B * T, self.embed_dim)
        c_flat = c.reshape(B * T, -1)

        if self.phase_space:
            q, p = x_flat[:, :d], x_flat[:, d:]
        else:
            qp = self.z_to_qp(x_flat)
            q, p = qp[:, :d], qp[:, d:]

        dt_flat = None
        if dt is not None:
            dt_t = dt if isinstance(dt, Tensor) else torch.as_tensor(dt, device=x.device)
            if dt_t.dim() == 0:
                dt_flat = dt_t
            else:
                if dt_t.dim() == 1:                 # (B,) -> one step per sample
                    dt_t = dt_t.unsqueeze(1).expand(B, T)
                dt_flat = dt_t.reshape(B * T, 1)
        q_next, p_next = self.flow(q, p, c_flat, dt_flat)
        qp_next = torch.cat([q_next, p_next], dim=-1)   # (B*T, 2*d)

        if self.phase_space:
            return qp_next.reshape(B, T, self.embed_dim)
        else:
            return self.qp_to_z(qp_next).reshape(B, T, self.embed_dim)


class HamiltonianFlow(nn.Module):
    """Learned Hamiltonian flow: (q_t, p_t, u_t) -> (q_{t+1}, p_{t+1}).

    Replaces the unconstrained ARPredictor in LeWorldModel.
    Do NOT wrap with torch.compile.

    Args:
        p_scale: scalar divisor applied to p before H_net (default 1.0 = no rescaling).
            Set to the empirical std of the FD momentum (e.g. 0.52 for the pendulum with
            temporal correlation ≈ 0.81) so that H_net sees p with unit variance.
            Stored as a non-learnable buffer; persisted in checkpoints.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        integrator: str = "stormer_verlet",
        dt: float = 1.0,
        p_scale: float = 1.0,
        learnable_scale: bool = False,  # accepted for config compat; only effective for separable=True
    ):
        super().__init__()
        assert integrator in _INTEGRATORS, f"integrator must be one of {list(_INTEGRATORS)}"
        self.H_net = HamiltonianNet(latent_dim, action_dim, hidden_dim, n_layers)
        self._step = _INTEGRATORS[integrator]
        self.dt = dt
        self.latent_dim = latent_dim
        # Non-learnable buffer: rescales p to unit variance before H_net.
        self.register_buffer("p_scale", torch.tensor(float(p_scale)))

    def _H_fn(self, q: Tensor, p: Tensor, u: Tensor) -> Tensor:
        """H_net evaluated with p rescaled to unit variance."""
        return self.H_net(q, p / self.p_scale.to(p.dtype), u)

    def eval_H(self, q: Tensor, p: Tensor, u: Tensor) -> Tensor:
        """Scalar Hamiltonian H(q,p,u) without autograd setup.

        Use this for monitoring and conservation loss — it is cheaper than
        calling the integrator step functions which set up create_graph.
        """
        return self._H_fn(q.float(), p.float(), u.float())

    def forward(self, q: Tensor, p: Tensor, u: Tensor,
                dt: Tensor | float | None = None) -> tuple[Tensor, Tensor]:
        """Predict (q_{t+1}, p_{t+1}) from current state and action.

        Args:
            q: (B, d) position-like latent coordinates.
            p: (B, d) momentum-like latent coordinates.
            u: (B, action_dim) encoded action embedding.

        Returns:
            q_next, p_next — each (B, d).
        """
        assert q.shape == p.shape, f"q/p shape mismatch: {q.shape} vs {p.shape}"
        assert q.shape[-1] == self.latent_dim, (
            f"expected latent_dim={self.latent_dim}, got {q.shape[-1]}"
        )
        step = self.dt
        if dt is not None:
            step = dt.unsqueeze(-1) if isinstance(dt, Tensor) and dt.dim() == 1 else dt
            if isinstance(step, Tensor):
                step = step.to(device=q.device, dtype=torch.float32)
        return self._step(self._H_fn, q, p, u, step)

    def symplectic_grads(
        self, q: Tensor, p: Tensor, u: Tensor, create_graph: bool | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return the symplectic gradient (dH/dq, dH/dp) at (q, p, u).

        Joint-H counterpart to SeparableHamiltonianFlow.symplectic_grads: here a
        single H(q, p/scale, u) net is differentiated w.r.t. both q and p. Used by
        the HNN loss, which matches dH/dp against dq/dt and dH/dq against -dp/dt.
        See that method for the full contract.
        """
        assert q.shape == p.shape, f"q/p shape mismatch: {q.shape} vs {p.shape}"
        assert q.shape[-1] == self.latent_dim, (
            f"expected latent_dim={self.latent_dim}, got {q.shape[-1]}"
        )
        if create_graph is None:
            create_graph = self.training

        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            with torch.enable_grad():
                q32 = _ensure_grad(q.float())
                p32 = _ensure_grad(p.float())
                u32 = u.float()
                H = self._H_fn(q32, p32, u32)
                dHdq, dHdp = torch.autograd.grad(
                    H.sum(), [q32, p32], create_graph=create_graph
                )
        return dHdq.to(q.dtype), dHdp.to(p.dtype)


# ── time-reversal diagnostics ────────────────────────────────────────────────


def momentum_parity(
    flow: HamiltonianFlow | SeparableHamiltonianFlow,
    q: Tensor,
    p: Tensor,
    u: Tensor,
) -> dict[str, float]:
    """Pointwise (integrator-free) test of whether H is even in p.

    Two numbers, both scale-free:

      ``rho_odd``  -- odd energy fraction of the field dH/dp, using the same
          odd/even split convention as ``eval_time_reversal.py`` uses for the
          momentum head::

              g_odd  = (dHdp(p) - dHdp(-p)) / 2
              g_even = (dHdp(p) + dHdp(-p)) / 2
              rho    = ||g_odd||^2 / (||g_odd||^2 + ||g_even||^2)

          ``rho_odd == 1`` exactly iff dH/dp is odd, i.e. iff K is even, which
          is the condition for exact time-reversibility of the leapfrog map.

      ``H_asym_rel`` -- |H(q,p,u) - H(q,-p,u)| averaged and normalised by the
          p-dependent part |H(q,p,u) - H(q,0,u)|.  Zero iff H is even in p.
          The denominator guards the degenerate pass in which K is even only
          because it is constant (a flow with no drift is trivially reversible
          and trivially useless).

      ``drift_norm`` -- mean ||dH/dp||, the size of the leapfrog drift term.
          Report it alongside ``rho_odd`` so a collapsed K cannot be mistaken
          for a success.
    """
    _, g_pos = flow.symplectic_grads(q, p, u, create_graph=False)
    _, g_neg = flow.symplectic_grads(q, -p, u, create_graph=False)
    g_odd = 0.5 * (g_pos - g_neg)
    g_even = 0.5 * (g_pos + g_neg)
    e_odd = g_odd.double().pow(2).sum()
    e_even = g_even.double().pow(2).sum()

    with torch.no_grad():
        H_pos = flow.eval_H(q, p, u).double()
        H_neg = flow.eval_H(q, -p, u).double()
        H_nil = flow.eval_H(q, torch.zeros_like(p), u).double()
    denom = (H_pos - H_nil).abs().mean()

    return {
        "rho_odd": float(e_odd / (e_odd + e_even + 1e-30)),
        "H_asym_rel": float((H_pos - H_neg).abs().mean() / (denom + 1e-30)),
        "drift_norm": float(g_pos.double().norm(dim=-1).mean()),
    }
