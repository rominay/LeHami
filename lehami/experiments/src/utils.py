"""Shared rollout and probe utilities for lehami/experiments/src.

Provides:
    RolloutStep       — per-step result (q_pred, p_pred, q_gt, p_gt as CPU numpy)
    rollout_episode() — autoregressive rollout for one episode (lehami + lewm)
    probe_features()  — build probe input (q or qp) from a RolloutStep
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

ROOT      = Path(__file__).resolve().parents[3]
_SRC_DIR  = Path(__file__).resolve().parent
_LEWM_PATH = ROOT / "lehami" / "models" / "third_party" / "leworldmodel"
_MODELS    = ROOT / "lehami" / "models"
for _p in [str(_LEWM_PATH), str(_MODELS), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_eval import (
    _encode_episode,
    _encode_z_lewm,
    _build_anchor,
    _zero_action_emb,
    _latent_dim,
    _mom_type,
    _anchor_frame,
)


class RolloutStep(NamedTuple):
    k: int
    q_pred: np.ndarray   # (d,) predicted position coordinate
    p_pred: np.ndarray   # (d,) predicted momentum (zeros for lewm)
    q_gt:   np.ndarray   # (d,) encoder output at ground-truth frame h-1+k
    p_gt:   np.ndarray   # (d,) ground-truth momentum at frame h-1+k (zeros for lewm)


def _p_gt_at(model, cls: torch.Tensor, q_enc: torch.Tensor,
             frame: int, mtype: str, fd_dt: float) -> np.ndarray:
    """Ground-truth momentum at absolute frame index `frame` (requires frame >= 1)."""
    if mtype == "freemom":
        # Reference p from the encoder's own head (self-consistency, not GT --
        # see run_eval._has_reference_p). W is read off the head's input width.
        W = model.mom_projector.net[0].in_features // cls.shape[-1]
        return model.mom_projector(
            cls[frame - W + 1 : frame + 1].reshape(1, -1)).squeeze(0).cpu().numpy()
    if mtype == "learned":
        c = torch.cat([cls[frame].unsqueeze(0), cls[frame - 1].unsqueeze(0)], dim=-1)
        return model.mom_projector(c).squeeze(0).cpu().numpy()
    if mtype == "sg":
        from lehami.derivative import p_interp_series
        return p_interp_series(q_enc, fd_dt)[frame].cpu().numpy()
    return ((q_enc[frame] - q_enc[frame - 1]) / fd_dt).cpu().numpy()


@torch.no_grad()
def rollout_episode(
    model,
    pixels: torch.Tensor,
    cfg,
    device: str,
    horizon: int,
    enc_batch: int = 50,
) -> list[RolloutStep]:
    """Autoregressive rollout for one episode.

    Args:
        pixels:   (L, C, H, W) ImageNet-normalised frames on `device`.
        horizon:  max steps beyond the anchor frame.

    Returns:
        List of RolloutStep for k = 0 .. min(horizon, L - history_size).
        All arrays are CPU numpy of shape (d,).
    """
    h      = cfg.history_size
    d      = _latent_dim(cfg)
    L      = pixels.shape[0]
    mtype  = _mom_type(cfg.model.name)
    a      = _anchor_frame(cfg, mtype)
    k_max  = min(horizon, L - 1 - a)

    if mtype == "lewm":
        return _rollout_lewm(model, pixels, cfg, device, h, d, k_max, enc_batch)

    # ── lehami path ─────────────────────────────────────────────────────────────
    cls, q_enc = _encode_episode(model, pixels, enc_batch)  # (L, vit_d), (L, d)
    q0, p0     = _build_anchor(model, cls, q_enc, mtype, cfg, device)
    u_emb      = _zero_action_emb(model, cfg, device)       # (1, phase_dim)
    fd_dt      = float(cfg.model.get("fd_dt", 1.0))

    steps = [RolloutStep(
        k=0,
        q_pred=q0.cpu().numpy(),
        p_pred=p0.cpu().numpy(),
        q_gt=q_enc[a].cpu().numpy(),
        p_gt=_p_gt_at(model, cls, q_enc, a, mtype, fd_dt),
    )]

    qp = torch.cat([q0, p0], dim=-1).unsqueeze(0)  # (1, phase_dim)
    for k in range(1, k_max + 1):
        qp = model.predictor(qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0]
        steps.append(RolloutStep(
            k=k,
            q_pred=qp[0, :d].cpu().numpy(),
            p_pred=qp[0, d:].cpu().numpy(),
            q_gt=q_enc[a + k].cpu().numpy(),
            p_gt=_p_gt_at(model, cls, q_enc, a + k, mtype, fd_dt),
        ))

    return steps


@torch.no_grad()
def _rollout_lewm(
    model, pixels: torch.Tensor, cfg, device: str,
    h: int, d: int, k_max: int, enc_batch: int,
) -> list[RolloutStep]:
    z_enc = _encode_z_lewm(model, pixels, enc_batch)  # (L, d)

    zeros = np.zeros(d, dtype=np.float32)
    steps = [RolloutStep(
        k=0,
        q_pred=z_enc[h - 1].cpu().numpy(),
        p_pred=zeros,
        q_gt=z_enc[h - 1].cpu().numpy(),
        p_gt=zeros,
    )]

    act_emb = model.action_encoder(
        torch.zeros(1, h, cfg.data.action_dim, device=device)
    )  # (1, h, d)
    buf = z_enc[:h].unsqueeze(0)  # (1, h, d)

    for k in range(1, k_max + 1):
        # model.predict (full context + pred_proj, last slot) — matches run_eval's
        # lewm rollout; model.predictor(buf, act_emb)[:, 0] gives wrong embeddings
        z_pred = model.predict(buf, act_emb)[:, -1]  # (1, d)
        buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1)
        steps.append(RolloutStep(
            k=k,
            q_pred=z_pred[0].cpu().numpy(),
            p_pred=zeros,
            q_gt=z_enc[h - 1 + k].cpu().numpy(),
            p_gt=zeros,
        ))

    return steps


def probe_features(step: RolloutStep, probe_input: str) -> np.ndarray:
    """Build the probe input vector from a RolloutStep.

    Args:
        probe_input: ``"qp"`` for concatenated position+momentum,
                     ``"q"`` for position only.
    """
    if probe_input == "qp":
        return np.concatenate([step.q_pred, step.p_pred])
    return step.q_pred
