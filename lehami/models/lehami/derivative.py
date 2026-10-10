"""Polynomial-interpolation differentiation (Savitzky-Golay) for latent momentum.

The momentum p is defined as the time derivative of the encoder trajectory q(t),
estimated by least-squares quadratic interpolation rather than backward finite
differences (FD):

  centered : fit over frames [t-2 .. t+2], differentiate at t.
             Used wherever both-side GT frames exist: training targets, SIGReg
             pairs, eval p_gt references, probe features.
  causal   : fit over frames [t-3 .. t], differentiate at t.
             Used at rollout anchors, where future frames are unavailable
             (using them would leak the first prediction targets).

Both estimate the derivative AT frame t — no half-step lag, unlike backward FD.
LSQ fitting is linear in the data, so each estimator is a fixed stencil
(closed-form constant weights); there is nothing learned and nothing that can
collapse. `p_interp_series` falls back to the causal stencil (and, for the first
frames, backward FD) where the centered window does not fit, so it is defined
for every frame t >= 1 of a trajectory.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

CENTERED_HALFWIDTH = 2          # centered window: t-2 .. t+2
CAUSAL_WINDOW = 4               # causal window:   t-3 .. t
ANCHOR_OFFSET = CAUSAL_WINDOW - 1   # first frame with a full causal window (= 3)


def _lsq_deriv_weights(offsets: list[int], order: int = 2) -> np.ndarray:
    """Weights w s.t. sum_m w[m] * f(t + offsets[m]) = LSQ-poly derivative at t."""
    x = np.asarray(offsets, dtype=np.float64)
    A = np.stack([x**k for k in range(order + 1)], axis=1)   # (W, order+1)
    return np.linalg.pinv(A)[1]                              # slope-coefficient row


CENTERED_OFFSETS = list(range(-CENTERED_HALFWIDTH, CENTERED_HALFWIDTH + 1))
CAUSAL_OFFSETS = list(range(-(CAUSAL_WINDOW - 1), 1))
CENTERED_WEIGHTS = _lsq_deriv_weights(CENTERED_OFFSETS)      # (5,)
CAUSAL_WEIGHTS = _lsq_deriv_weights(CAUSAL_OFFSETS)          # (4,)


def _weights_like(w: np.ndarray, ref: Tensor) -> Tensor:
    return torch.tensor(w, dtype=ref.dtype, device=ref.device)


def p_causal_last(z_window: Tensor, dt: float = 1.0) -> Tensor:
    """Causal derivative at the LAST frame of a window.

    z_window: (..., W, d) with W == CAUSAL_WINDOW (frames t-3 .. t, oldest first).
    Returns (..., d): dq/dt at frame t. Differentiable.
    """
    assert z_window.shape[-2] == CAUSAL_WINDOW, (
        f"expected window of {CAUSAL_WINDOW} frames, got {z_window.shape[-2]}"
    )
    w = _weights_like(CAUSAL_WEIGHTS, z_window)
    return torch.einsum("...wd,w->...d", z_window, w) / dt


def p_centered_at(z: Tensor, t: int, dt: float = 1.0) -> Tensor:
    """Centered derivative at frame t of a trajectory z: (..., T, d) -> (..., d)."""
    hw = CENTERED_HALFWIDTH
    w = _weights_like(CENTERED_WEIGHTS, z)
    window = z[..., t - hw : t + hw + 1, :]
    return torch.einsum("...wd,w->...d", window, w) / dt


def p_interp_series(z: Tensor, dt: float = 1.0) -> Tensor:
    """Interpolation-derivative series for a whole trajectory.

    z: (..., T, d)  ->  p: (..., T, d), where
      t in [2, T-3]   : centered quadratic (the definition of p)
      t in {T-2, T-1} : causal quadratic  (tail fallback: no future frames)
      t == 1          : backward FD       (head fallback: one past frame only)
      t == 0          : copy of t == 1    (no derivative defined; never used as a
                        target — kept only so indexing matches frame indexing)
    Differentiable end to end.
    """
    T = z.shape[-2]
    hw = CENTERED_HALFWIDTH
    assert T >= CAUSAL_WINDOW + 1, f"trajectory too short for interpolation: T={T}"

    w_c = _weights_like(CENTERED_WEIGHTS, z)
    # centered interior via stacked shifted slices: frames hw .. T-1-hw
    interior = sum(
        w_c[m] * z[..., hw + off : T - hw + off, :]
        for m, off in enumerate(CENTERED_OFFSETS)
    ) / dt                                                     # (..., T-2*hw, d)

    tail = torch.stack(
        [p_causal_last(z[..., t - CAUSAL_WINDOW + 1 : t + 1, :], dt)
         for t in range(T - hw, T)],
        dim=-2,
    )                                                          # (..., hw, d)

    p1 = (z[..., 1, :] - z[..., 0, :]) / dt                    # backward FD at t=1
    head = torch.stack([p1, p1], dim=-2)                       # t=0 (copy), t=1

    return torch.cat([head, interior, tail], dim=-2)
