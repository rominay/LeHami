"""Generate a DIVERSIFIED pendulum pixel dataset: several rod lengths x several
sampling timesteps, over a fixed physical time window.

Motivation (paper/notes/hjepa_status_audit.tex, ideas 2a + 2b):

  2a  Mixed-dt training.  The Hamiltonian flow takes dt as an argument and LeWM
      structurally cannot.  Training across several dt and evaluating on an
      *unseen* dt is the cleanest "LeWM cannot enter this comparison" result
      available.  For that claim to be tested, some dt values must be HELD OUT --
      see --train-dt-mults / --test-dt-mults, recorded in the HDF5 attrs.

  2b  Varying rod length.  L changes the natural frequency omega_0 = sqrt(g/L),
      so this is a system-identification / conditioning task, not merely more
      data: the latent must encode a *parameter* as well as a state, which is
      where V(q; u) conditioning is more natural than an unconditioned map.

Design decisions, and why (all are overridable):

  Fixed time window, not fixed frame count.
      Every episode spans the same WINDOW seconds regardless of dt, so a
      horizon in seconds is matched across arms and the dt comparison is not
      confounded by episode duration.  Frame counts therefore differ per dt and
      are recorded per episode in `ep_len` (data_prep.py already handles this).

  Physics integrated at a fine substep, then SUBSAMPLED.
      All dt arms share the *same* underlying trajectory: we integrate with RK4
      at SUBSTEP seconds and record every k-th sample.  So dt changes the
      observation rate only, never the physics.  Without this, coarse-dt arms
      would carry extra integration error and the comparison would be confounded.

  Geometric lengths, capped by pixel resolution.
      omega_0 ~ L^{-1/2}, so lengths should be spaced geometrically.  But the
      renderer is 64x64 with the pivot at the centre, leaving ~27 px of radius;
      if L_min/L_max < 0.5 the shortest rod becomes shorter than the bob is wide
      and the short-L arm is resolution-limited rather than physics-limited.
      Default {0.7, 1.0, 1.4} keeps every rod in 13-27 px (ratio 0.5, omega_0
      spread 1.41x).  Push the ratio further only if IMG_SIZE grows too.

  Rod rendered TO SCALE.
      L is visible in a single frame.  The alternative (fixed on-screen rod)
      hides the parameter and makes the observation non-Markovian in a way that
      is an artifact of the renderer, not a property of the system.  Set
      --fixed-rod-px to render all lengths identically if you want the harder,
      dynamics-only system-ID task instead.

  Initial velocity sampled in DIMENSIONLESS units.
      omega_0 = omega_tilde * sqrt(g/L) with omega_tilde ~ U(-a, a).  Sampling
      omega directly in rad/s would give the long-L arm systematically less
      energy relative to its own scale, so phase-space coverage would not be
      matched across lengths.  Default a is set so that L=1 reproduces the
      U(-2, 2) rad/s of scripts/generate_pendulum.py exactly.

HDF5 layout (superset of pendulum.h5, so data_prep.py consumes it directly):
    ep_len    (N_ep,)                episode lengths (VARY with dt)
    ep_offset (N_ep,)                global start index per episode
    pixels    (N, 64, 64, 3)  uint8  RGB frames
    state     (N, 2)          float32 (theta, dtheta)
    length    (N_ep,)         float32 rod length L per episode
    dt        (N_ep,)         float32 sampling timestep per episode
    dt_mult   (N_ep,)         int64   dt / base_dt
    is_train_dt (N_ep,)       bool    dt in the training set of timesteps

Usage:
    python lehami/data/pendulum_multi.py                       # defaults
    python lehami/data/pendulum_multi.py --n-per-cell 100 --window 8.0
    python lehami/data/pendulum_multi.py --lengths 0.7 1.0 1.4 \
        --train-dt-mults 1 2 4 --test-dt-mults 3 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — activates compression codecs
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

G = 9.81
BASE_DT = 0.05        # the dt of the existing pendulum.h5; dt_mult multiplies this
SUBSTEP = 0.0125      # RK4 integration substep (BASE_DT / 4) — physics accuracy
IMG_SIZE = 64
BOB_R = 4
PIVOT = (IMG_SIZE // 2, IMG_SIZE // 2)
MAX_ROD_PX = IMG_SIZE // 2 - BOB_R - 1     # 27 px: keeps the bob inside the frame


# ── physics ───────────────────────────────────────────────────────────────────

def _rk4_step(theta: float, dtheta: float, L: float, h: float):
    """One RK4 step of theta'' = -(g/L) sin(theta)."""
    def deriv(th, dth):
        return dth, -(G / L) * np.sin(th)

    k1 = deriv(theta, dtheta)
    k2 = deriv(theta + h / 2 * k1[0], dtheta + h / 2 * k1[1])
    k3 = deriv(theta + h / 2 * k2[0], dtheta + h / 2 * k2[1])
    k4 = deriv(theta + h * k3[0], dtheta + h * k3[1])
    return (theta + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            dtheta + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def simulate(theta0: float, dtheta0: float, L: float, dt: float, n_frames: int) -> np.ndarray:
    """Integrate at SUBSTEP and record every `dt` seconds.  Returns (n_frames, 2)."""
    sub_per_frame = int(round(dt / SUBSTEP))
    assert abs(sub_per_frame * SUBSTEP - dt) < 1e-9, f"dt={dt} not a multiple of {SUBSTEP}"
    out = np.empty((n_frames, 2), dtype=np.float32)
    th, om = theta0, dtheta0
    for t in range(n_frames):
        out[t] = (th, om)
        for _ in range(sub_per_frame):
            th, om = _rk4_step(th, om, L, SUBSTEP)
    return out


# ── rendering ─────────────────────────────────────────────────────────────────

def render_frame(theta: float, rod_px: float) -> np.ndarray:
    """Render angle theta with a rod of `rod_px` pixels.  Matches the palette and
    conventions of scripts/generate_pendulum.py (theta=0 hangs down, +ve clockwise)."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    bob_x = PIVOT[0] + rod_px * np.sin(theta)
    bob_y = PIVOT[1] + rod_px * np.cos(theta)          # y increases downward
    draw.line([PIVOT, (bob_x, bob_y)], fill=(50, 50, 50), width=2)
    draw.ellipse([bob_x - BOB_R, bob_y - BOB_R, bob_x + BOB_R, bob_y + BOB_R],
                 fill=(200, 50, 50))
    draw.ellipse([PIVOT[0] - 2, PIVOT[1] - 2, PIVOT[0] + 2, PIVOT[1] + 2],
                 fill=(50, 50, 200))
    return np.array(img, dtype=np.uint8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.7, 1.0, 1.4],
                    help="Rod lengths in metres. Keep L_min/L_max >= 0.5 at 64px.")
    ap.add_argument("--train-dt-mults", type=int, nargs="+", default=[1, 2, 4],
                    help="dt multipliers of BASE_DT seen during training.")
    ap.add_argument("--test-dt-mults", type=int, nargs="+", default=[3],
                    help="HELD-OUT dt multipliers, for the unseen-dt generalisation claim.")
    ap.add_argument("--n-per-cell", type=int, default=100,
                    help="Episodes per (length, dt) cell.")
    ap.add_argument("--window", type=float, default=8.0,
                    help="Physical duration of every episode, seconds.")
    ap.add_argument("--omega-scale", type=float, default=2.0 / np.sqrt(G / 1.0),
                    help="Dimensionless |omega_tilde| bound; default reproduces "
                         "U(-2,2) rad/s at L=1 exactly.")
    ap.add_argument("--fixed-rod-px", type=float, default=None,
                    help="Render every length with this rod size (hides L from a "
                         "single frame -> dynamics-only system ID). Default: to scale.")
    ap.add_argument("--out", type=str, default="~/.stable_worldmodel/pendulum_multi.h5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L_max = max(args.lengths)
    all_mults = sorted(set(args.train_dt_mults) | set(args.test_dt_mults))
    cells = [(L, k) for L in args.lengths for k in all_mults]

    print(f"lengths      : {args.lengths}   (omega_0 = "
          f"{[round(float(np.sqrt(G / L)), 2) for L in args.lengths]} rad/s)")
    print(f"dt multiples : train {args.train_dt_mults}  held-out {args.test_dt_mults}")
    print(f"window       : {args.window} s   |   {args.n_per_cell} episodes x {len(cells)} cells "
          f"= {args.n_per_cell * len(cells)} episodes")
    if args.fixed_rod_px is None:
        print(f"rod          : to scale, {[round(MAX_ROD_PX * L / L_max, 1) for L in args.lengths]} px")
    else:
        print(f"rod          : FIXED {args.fixed_rod_px} px (L hidden from single frames)")

    ep_states, ep_pixels, ep_meta = [], [], []
    for L, k in cells:
        dt = BASE_DT * k
        n_frames = int(round(args.window / dt))
        rod_px = args.fixed_rod_px if args.fixed_rod_px is not None else MAX_ROD_PX * L / L_max
        om_bound = args.omega_scale * np.sqrt(G / L)     # dimensionless -> rad/s
        for _ in range(args.n_per_cell):
            th0 = rng.uniform(-np.pi, np.pi)
            om0 = rng.uniform(-om_bound, om_bound)
            st = simulate(th0, om0, L, dt, n_frames)
            ep_states.append(st)
            ep_pixels.append(np.stack([render_frame(float(a), rod_px) for a in st[:, 0]]))
            ep_meta.append((L, dt, k, k in args.train_dt_mults))
        print(f"  L={L:4.2f}  dt=x{k} ({dt:.3f}s)  {n_frames:3d} frames/ep  "
              f"|omega_0| <= {om_bound:.2f} rad/s   done")

    ep_len = np.array([len(s) for s in ep_states], dtype=np.int64)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(np.int64)
    states = np.concatenate(ep_states).astype(np.float32)
    pixels = np.concatenate(ep_pixels).astype(np.uint8)
    meta = np.array(ep_meta, dtype=object)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("pixels", data=pixels, chunks=(1, IMG_SIZE, IMG_SIZE, 3),
                         **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("state", data=states)
        f.create_dataset("length", data=np.array([m[0] for m in meta], dtype=np.float32))
        f.create_dataset("dt", data=np.array([m[1] for m in meta], dtype=np.float32))
        f.create_dataset("dt_mult", data=np.array([m[2] for m in meta], dtype=np.int64))
        f.create_dataset("is_train_dt", data=np.array([m[3] for m in meta], dtype=bool))
        # Per-FRAME copy of the episode's dt. The per-episode `dt` column above
        # is for analysis; this one exists so the training dataloader can carry
        # dt into the batch (swm's HDF5Dataset indexes frame-aligned columns
        # only), which is what lets the Hamiltonian flow take a per-episode
        # integrator step h = dt / dt_ref instead of a single global scalar.
        f.create_dataset("dt_frame", data=np.repeat(
            np.array([m[1] for m in meta], dtype=np.float32), ep_len)[:, None])
        f.attrs.update(base_dt=BASE_DT, substep=SUBSTEP, window=args.window,
                       lengths=np.array(args.lengths, dtype=np.float32),
                       train_dt_mults=np.array(args.train_dt_mults, dtype=np.int64),
                       test_dt_mults=np.array(args.test_dt_mults, dtype=np.int64),
                       omega_scale=args.omega_scale, seed=args.seed,
                       rod_to_scale=args.fixed_rod_px is None)

    print(f"\n{len(ep_len)} episodes, {len(states)} frames -> {out_path}")
    print(f"file size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
