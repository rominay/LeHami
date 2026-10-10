#!/usr/bin/env python
"""Latent phase portrait: the (q, p) embedding of ONE episode, plotted like a phase
diagram, next to the true (theta, omega) portrait of the same episode.

Standalone -- reads a checkpoint and the dataset, writes a figure. It touches no
notebook and no result CSV.

    python lehami/experiments/src/plot_latent_phase_portrait.py --episode 0

What it plots, and why there is more than one panel
---------------------------------------------------
q and p are d-dimensional (d = 32 for the exp027 conservative pendulum), so "p vs q"
is not one curve but d of them, in a space with no canonical 2-D projection. Three
views, none of which is privileged:

  (a) TRUE portrait -- (theta, omega) from the dataset's own `state`. The reference.
  (b) PCA view      -- first PC of q against first PC of p, both fitted on THIS
                       episode. Shows the dominant closed orbit if there is one.
  (c) per-dimension -- p_j vs q_j for the highest-variance coordinates j. This is the
                       honest view: a Hamiltonian flow is symplectic in the FULL
                       (q, p), and a clean loop in one coordinate pair is neither
                       necessary nor sufficient for that.

A closed loop in (b) or (c) is suggestive, not proof: PCA can manufacture a loop from
any smooth periodic signal, and the latent is only defined up to the symplectic gauge.
Read it as "does the representation trace a consistent orbit", not as a conservation
measurement -- for that use the energy-drift scripts.

Momentum convention
-------------------
p is NOT a separate encoder output for the stencil models. It is the finite-difference
/ Savitzky-Golay derivative of q along the episode, exactly as `run_eval.evaluate_rollout`
builds its reference p (`_p_series`), including the per-episode dt rescaling when the
dataset carries a variable clock. So the p plotted here is the same quantity every
rollout metric is scored against, not a re-derivation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import (  # noqa: E402
    _align_states,
    _encode_episode,
    _enc_window,
    _episode_dt_ratio,
    _h5_path,
    _mom_type,
    _p_series,
    _preprocess_batch,
    load_model,
)

# The conservative (undamped, unactuated) pendulum arm, EXP-027.
DEFAULT_CKPT = (
    "${STABLEWM_HOME}/checkpoints/"
    "lehami_sg_sep_sreg_joint_ms_pendulum_sg_sep_sreg_joint_ms_d32_exp027/best.ckpt"
)


def encode_episode_qp(model, cfg, h5_path: Path, ep: int, device: str, enc_batch: int = 50):
    """Return (q, p, states) for every latent frame of episode `ep`.

    q: (T_eff, d)   p: (T_eff, d)   states: (T_eff, state_dim)  -- all numpy, aligned.
    """
    import h5py

    mom_type = _mom_type(cfg.model.name)
    if mom_type in ("learned", "freemom", "encqp"):
        raise SystemExit(
            f"mom_type={mom_type!r} emits p from a head/projector, not a stencil over q. "
            "This script covers the stencil models (fd, sg); extend it deliberately "
            "rather than silently plotting a different quantity."
        )

    with h5py.File(h5_path, "r") as f:
        start = int(f["ep_offset"][ep])
        L = int(f["ep_len"][ep])
        pixels_raw = f["pixels"][start : start + L]
        states_raw = f["state"][start : start + L]
        h_arr = _episode_dt_ratio(h5_path, cfg)

    pixels = _preprocess_batch(pixels_raw, cfg.img_size, device)
    with torch.no_grad():
        _cls, q_enc = _encode_episode(model, pixels, enc_batch)

    fd_dt = float(cfg.model.get("fd_dt", 1.0))
    p_enc = _p_series(q_enc, mom_type, fd_dt)
    if h_arr is not None:                       # put p on this episode's clock
        p_enc = p_enc / float(h_arr[ep])

    states = _align_states(model, states_raw)   # pixel time -> latent time
    q = q_enc.detach().cpu().numpy()
    p = p_enc.detach().cpu().numpy()
    n = min(len(q), len(p), len(states))
    return q[:n], p[:n], np.asarray(states)[:n]


def _pc1(X: np.ndarray) -> np.ndarray:
    """First principal component score of X (T, d), centred. Sign-fixed so the
    largest-|loading| coordinate is positive, otherwise the plot flips run to run."""
    Xc = X - X.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    v = vt[0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    return Xc @ v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--episode", type=int, default=0, help="episode index in the HDF5 file")
    ap.add_argument("--dims", type=int, default=4,
                    help="how many per-dimension (p_j vs q_j) panels to draw")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enc-batch", type=int, default=50)
    ap.add_argument("--out", default=None, help="save here instead of plt.show()")
    args = ap.parse_args()

    model, cfg = load_model(Path(args.checkpoint), args.device)
    h5_path = _h5_path(cfg)
    print(f"model   : {cfg.model.name}  (mom_type={_mom_type(cfg.model.name)}, "
          f"W={_enc_window(model)}, history_size={cfg.history_size})")
    print(f"dataset : {h5_path}")

    q, p, states = encode_episode_qp(model, cfg, h5_path, args.episode,
                                     args.device, args.enc_batch)
    T, d = q.shape
    print(f"episode {args.episode}: T={T} latent frames, d={d}")
    print(f"  std(q) = {q.std(0).mean():.4f}   std(p) = {p.std(0).mean():.4f}"
          f"   ratio p/q = {p.std(0).mean() / max(q.std(0).mean(), 1e-12):.3f}")

    t = np.arange(T)
    # Rank coordinates by how much they move in this episode; a coordinate that is
    # constant here carries no orbit and a panel of it would be pure noise.
    order = np.argsort(-(q.std(0) + p.std(0)))
    dims = order[: args.dims]

    ncol = 2 + len(dims)
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.2))

    def _traj(ax, x, y, title, xlabel, ylabel):
        ax.plot(x, y, color="#999", lw=0.7, zorder=1)
        s = ax.scatter(x, y, c=t, cmap="viridis", s=11, zorder=2)
        ax.scatter(x[0], y[0], marker="o", s=55, facecolor="none",
                   edgecolor="#d1495b", lw=1.5, zorder=3, label="start")
        ax.scatter(x[-1], y[-1], marker="X", s=55, color="#d1495b", zorder=3, label="end")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return s

    # (a) the truth
    theta, omega = states[:, 0], states[:, 1]
    _traj(axes[0], theta, omega, "TRUE  (theta, omega)", r"$\theta$", r"$\omega$")
    axes[0].legend(fontsize=7, loc="upper right")

    # (b) PCA view
    s = _traj(axes[1], _pc1(q), _pc1(p), "latent, PC1 of each",
              "PC1 of $q$", "PC1 of $p$")

    # (c) per-dimension views
    for ax, j in zip(axes[2:], dims):
        _traj(ax, q[:, j], p[:, j], f"latent dim {j}", f"$q_{{{j}}}$", f"$p_{{{j}}}$")

    cb = fig.colorbar(s, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("frame index within the episode", fontsize=8)

    fig.suptitle(
        f"Latent phase portrait — {cfg.model.name}, episode {args.episode} "
        f"(d={d}, T={T}).  Colour = time; circle = first frame, X = last.",
        fontsize=9, y=1.04,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
