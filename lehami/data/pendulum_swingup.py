"""Generate an ACTUATED, diversified pendulum pixel dataset: torque input, several
rod lengths x several bob masses, one sampling rate.

This is `pendulum_multi.py` with two additions and one removal, and every one of
them is deliberate:

  ADD  torque.  theta'' = -(g/L) sin(theta) + u / (m L^2).  The dataset carries a
       per-frame `action` column, so the models' V(q, u) finally has a `u`.  This
       is the prerequisite for a swing-up planning experiment: with u == 0 the
       system is autonomous and there is nothing to plan.

  ADD  mass.  m enters ONLY through the torque term, so it is meaningless in the
       unactuated dataset — which is why `pendulum_multi.py` has no mass axis and
       why the two axes must be added together.  m rescales the control
       authority u/(m L^2), i.e. it is the parameter that decides whether a given
       torque can pump the pendulum over the top.

  DROP the dt axis.  `pendulum_multi.py` sweeps dt to test unseen-rate
       generalisation.  The builders feed one scalar step size and (at W=1) have
       no channel to say which dt an episode used, so a mixed-dt manifest is
       ill-posed — see the header of
       `configs/experiment/phase2_lehami_sg_sep_conv_w1_multi_exp067_a0m.yaml`.
       One rate here: `--dt-mult 2`, i.e. dt = 0.1 s, matching EXP-067/068.

Both diversification axes are VISIBLE IN THE FRAME, and that is not cosmetic.
At W = 1 the encoder sees a single image, so any parameter it cannot see is
unidentifiable and V(q, u) cannot condition on it:
  * rod length  -> rod drawn to scale, as in `pendulum_multi.py`;
  * bob mass    -> bob radius proportional to m^(1/3), i.e. a constant-density
    sphere.  Hiding the mass would penalise the Hamiltonian arm specifically,
    since it is the arm that must scale a torque by 1/(m L^2).

Exploration policy.  Piecewise-constant torque, uniform on [-u_max, u_max], held
for `--hold` frames.  This is COVERAGE, not demonstrations: the models are
trained to predict, and a dataset of successful swing-ups would be a narrow slice
of phase space.  `u_max = 4` is not a free choice — it comes from the catch-basin
analysis in `pendulum_swingup_preview.ipynb` (at u_max = 2 the basin admits only
0.42x of the measured latent velocity error, so the planning outcome is decided
by the probe tier rather than by the model).

Columns
-------
    ep_len      (N_ep,)        int64    frames per episode
    ep_offset   (N_ep,)        int64    start index of each episode
    pixels      (N, 64, 64, 3) uint8
    state       (N, 2)         float32  (theta, dtheta) -- probing targets only
    action      (N, 1)         float32  torque applied AT that frame; action[-1]
                                        of each episode is a zero pad and drives
                                        no transition (the pusht_data convention)
    length      (N_ep,)        float32  rod length L, metres
    mass        (N_ep,)        float32  bob mass m, kg
    dt          (N_ep,)        float32
    dt_mult     (N_ep,)        int64
    dt_frame    (N, 1)         float32  per-frame copy of the episode dt

Usage
-----
The defaults ARE the paper's dataset: 3 rod lengths x 3 bob masses x 100 episodes of
8 s at dt = 0.1 s (900 episodes, 72,000 frames), torque |u| <= 4 held for 3 frames,
seed 42. lehami/data/prepare_actuated_data.sh runs exactly this. Write
the file into $STABLEWM_HOME, where training and evaluation look for it:

    python lehami/data/pendulum_swingup.py \
        --out "$STABLEWM_HOME/pendulum_swingup.h5"

    # every flag spelled out, same file:
    python lehami/data/pendulum_swingup.py \
        --lengths 0.7 1.0 1.4 --masses 0.5 1.0 2.0 --dt-mult 2 \
        --n-per-cell 100 --window 8.0 --u-max 4.0 --hold 3 --seed 42 \
        --out "$STABLEWM_HOME/pendulum_swingup.h5"

The output is deterministic: its sha256 equals the `dataset_sha256` recorded in the
shipped split, lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json
(train / val / test = 630 / 135 / 135 episodes, split 70/15/15 within each rod length).
That split is what the paper uses, so there is no need to rebuild it; to do so anyway
(this overwrites the shipped copy):

    python lehami/data/data_prep.py \
        --dataset "$STABLEWM_HOME/pendulum_swingup.h5" \
        --split-method episode_srnn --num-steps 16 --train-frames 80 \
        --train-ratio 0.7 --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — activates compression codecs
import numpy as np
from PIL import Image, ImageDraw

G = 9.81
BASE_DT = 0.05        # dt_mult multiplies this; dt_mult=2 -> 0.1 s, as in EXP-067
SUBSTEP = 0.0125      # RK4 substep (BASE_DT / 4)
IMG_SIZE = 64
PIVOT = (IMG_SIZE // 2, IMG_SIZE // 2)
BOB_R_REF = 4.0       # bob radius at m = 1 kg
MAX_BOB_R = 6.0       # cap so the largest bob still fits inside the frame
MAX_ROD_PX = IMG_SIZE // 2 - MAX_BOB_R - 1     # 25 px, sized for the LARGEST bob


def bob_radius(m: float) -> float:
    """Radius of a constant-density bob, so mass is legible from one frame."""
    return float(min(MAX_BOB_R, BOB_R_REF * m ** (1.0 / 3.0)))


# ── physics ───────────────────────────────────────────────────────────────────

def _rk4_step(theta: float, dtheta: float, L: float, m: float, u: float, h: float):
    """One RK4 step of theta'' = -(g/L) sin(theta) + u / (m L^2).

    The torque is held constant across the step, which is exact for the
    piecewise-constant policy used here (u changes only on frame boundaries, and
    a frame is a whole number of substeps).
    """
    def deriv(th, dth):
        return dth, -(G / L) * np.sin(th) + u / (m * L * L)

    k1 = deriv(theta, dtheta)
    k2 = deriv(theta + h / 2 * k1[0], dtheta + h / 2 * k1[1])
    k3 = deriv(theta + h / 2 * k2[0], dtheta + h / 2 * k2[1])
    k4 = deriv(theta + h * k3[0], dtheta + h * k3[1])
    return (theta + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            dtheta + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def simulate(theta0, dtheta0, L, m, dt, n_frames, actions):
    """Integrate at SUBSTEP, recording every `dt`.  Returns states (n_frames, 2).

    `actions[t]` is applied over the interval [t, t+1), so `state[t+1]` is the
    result of `state[t]` under `action[t]` — the same convention as
    `pusht_data.rollout_episode`.  `actions[-1]` is never applied.
    """
    sub_per_frame = int(round(dt / SUBSTEP))
    assert abs(sub_per_frame * SUBSTEP - dt) < 1e-9, f"dt={dt} not a multiple of {SUBSTEP}"
    out = np.empty((n_frames, 2), dtype=np.float32)
    th, om = theta0, dtheta0
    for t in range(n_frames):
        out[t] = (th, om)
        if t == n_frames - 1:
            break                      # the pad action drives nothing
        u = float(actions[t])
        for _ in range(sub_per_frame):
            th, om = _rk4_step(th, om, L, m, u, SUBSTEP)
    return out


def torque_sequence(rng, n_frames, u_max, hold):
    """Piecewise-constant torque, uniform on [-u_max, u_max], held `hold` frames.

    Held rather than i.i.d. because an i.i.d. torque at 10 Hz is close to white
    noise: it averages out over a swing period and the pendulum barely leaves the
    small-angle regime, so the dataset would never visit the high-energy states a
    swing-up planner has to traverse.
    """
    n_blocks = int(np.ceil(n_frames / hold))
    blocks = rng.uniform(-u_max, u_max, size=n_blocks)
    u = np.repeat(blocks, hold)[:n_frames]
    u[-1] = 0.0                        # explicit pad, never applied
    return u.astype(np.float32)


# ── rendering ─────────────────────────────────────────────────────────────────

def render_frame(theta: float, rod_px: float, bob_px: float) -> np.ndarray:
    """theta = 0 hangs down, +ve clockwise — the convention of every other
    pendulum dataset in this repo, so probes and figures transfer unchanged."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    bob_x = PIVOT[0] + rod_px * np.sin(theta)
    bob_y = PIVOT[1] + rod_px * np.cos(theta)          # y increases downward
    draw.line([PIVOT, (bob_x, bob_y)], fill=(50, 50, 50), width=2)
    draw.ellipse([bob_x - bob_px, bob_y - bob_px, bob_x + bob_px, bob_y + bob_px],
                 fill=(200, 50, 50))
    draw.ellipse([PIVOT[0] - 2, PIVOT[1] - 2, PIVOT[0] + 2, PIVOT[1] + 2],
                 fill=(50, 50, 200))
    return np.array(img, dtype=np.uint8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.7, 1.0, 1.4],
                    help="Rod lengths in metres; drawn to scale.")
    ap.add_argument("--masses", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="Bob masses in kg; drawn as radius ~ m^(1/3).")
    ap.add_argument("--dt-mult", type=int, default=2,
                    help="dt = BASE_DT * this. ONE value only: the builders carry a "
                         "single scalar step size and cannot condition on dt at W=1.")
    ap.add_argument("--n-per-cell", type=int, default=100,
                    help="Episodes per (length, mass) cell.")
    ap.add_argument("--window", type=float, default=8.0,
                    help="Physical duration of every episode, seconds.")
    ap.add_argument("--u-max", type=float, default=4.0,
                    help="Torque bound. 4.0 comes from the catch-basin analysis; "
                         "see pendulum_swingup_preview.ipynb sec 7.3.")
    ap.add_argument("--hold", type=int, default=3,
                    help="Frames each torque value is held for.")
    ap.add_argument("--omega-scale", type=float, default=2.0 / np.sqrt(G / 1.0),
                    help="Dimensionless |omega_tilde| bound on the initial state.")
    ap.add_argument("--omega-cap", type=float, default=2.5,
                    help="Reject an episode whose |omega| exceeds this multiple of "
                         "sqrt(g/L) anywhere. Enforces PIXEL IDENTIFIABILITY: at "
                         "2.5*sqrt(g/L) and dt=0.1 s the frame-to-frame angle step "
                         "is ~0.8 rad, well inside the +-pi aliasing limit, so theta "
                         "is recoverable from the image sequence. Without it a random "
                         "torque walk spins the rod up to >20 rad/s (2.2 rad/frame) "
                         "and the pixels alias.")
    ap.add_argument("--max-retries", type=int, default=200,
                    help="Resampling attempts per episode before giving up on the cap.")
    ap.add_argument("--out", type=str,
                    default="~/.stable_worldmodel/pendulum_swingup.h5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L_max = max(args.lengths)
    dt = BASE_DT * args.dt_mult
    n_frames = int(round(args.window / dt))
    cells = [(L, m) for L in args.lengths for m in args.masses]

    print(f"lengths   : {args.lengths}   (omega_0 = "
          f"{[round(float(np.sqrt(G / L)), 2) for L in args.lengths]} rad/s)")
    print(f"masses    : {args.masses}   (bob radius = "
          f"{[round(bob_radius(m), 2) for m in args.masses]} px)")
    print(f"rod       : to scale, {[round(MAX_ROD_PX * L / L_max, 1) for L in args.lengths]} px")
    print(f"dt        : {dt:.3f} s  (x{args.dt_mult})   {n_frames} frames/episode")
    print(f"torque    : U(-{args.u_max}, {args.u_max}), held {args.hold} frames")
    print(f"episodes  : {args.n_per_cell} x {len(cells)} cells = "
          f"{args.n_per_cell * len(cells)}")
    # control authority: the torque-only angular acceleration at the pivot, against
    # gravity's maximum.  <1 means the arm CANNOT lift the bob statically and must
    # pump -- which is what makes swing-up a planning problem rather than a reach.
    print("\n  authority u_max/(m L^2) vs g/L  (underactuated iff < 1):")
    for L, m in cells:
        print(f"    L={L:4.2f} m={m:4.2f}  ratio = "
              f"{(args.u_max / (m * L * L)) / (G / L):5.2f}")

    ep_states, ep_pixels, ep_actions, ep_meta = [], [], [], []
    n_rejected = n_uncapped = 0
    for L, m in cells:
        rod_px = MAX_ROD_PX * L / L_max
        bob_px = bob_radius(m)
        om_bound = args.omega_scale * np.sqrt(G / L)
        om_cap = args.omega_cap * np.sqrt(G / L)
        for _ in range(args.n_per_cell):
            # Rejection sampling on the whole episode: keep the policy simple and
            # u_max at its analysed value, and throw away the trajectories that
            # leave the frame-rate's identifiable range.
            for attempt in range(args.max_retries):
                th0 = rng.uniform(-np.pi, np.pi)
                om0 = rng.uniform(-om_bound, om_bound)
                u = torque_sequence(rng, n_frames, args.u_max, args.hold)
                st = simulate(th0, om0, L, m, dt, n_frames, u)
                if np.abs(st[:, 1]).max() <= om_cap:
                    break
                n_rejected += 1
            else:
                n_uncapped += 1
            ep_states.append(st)
            ep_actions.append(u[:, None])
            ep_pixels.append(np.stack([render_frame(float(a), rod_px, bob_px)
                                       for a in st[:, 0]]))
            ep_meta.append((L, m))
        print(f"  L={L:4.2f}  m={m:4.2f}  |omega_0| <= {om_bound:.2f} rad/s  "
              f"|omega| cap {om_cap:.2f}   done")

    ep_len = np.array([len(s) for s in ep_states], dtype=np.int64)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(np.int64)
    states = np.concatenate(ep_states).astype(np.float32)
    actions = np.concatenate(ep_actions).astype(np.float32)
    pixels = np.concatenate(ep_pixels).astype(np.uint8)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("pixels", data=pixels, chunks=(1, IMG_SIZE, IMG_SIZE, 3),
                         **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("state", data=states)
        f.create_dataset("action", data=actions)
        f.create_dataset("length", data=np.array([mm[0] for mm in ep_meta], dtype=np.float32))
        f.create_dataset("mass", data=np.array([mm[1] for mm in ep_meta], dtype=np.float32))
        f.create_dataset("dt", data=np.full(len(ep_len), dt, dtype=np.float32))
        f.create_dataset("dt_mult", data=np.full(len(ep_len), args.dt_mult, dtype=np.int64))
        f.create_dataset("dt_frame", data=np.full((len(states), 1), dt, dtype=np.float32))
        f.attrs.update(base_dt=BASE_DT, substep=SUBSTEP, window=args.window,
                       lengths=np.array(args.lengths, dtype=np.float32),
                       masses=np.array(args.masses, dtype=np.float32),
                       dt_mult=args.dt_mult, u_max=args.u_max, hold=args.hold,
                       omega_scale=args.omega_scale, seed=args.seed,
                       actuated=True, rod_to_scale=True, bob_scales_with_mass=True)

    print(f"\n{len(ep_len)} episodes, {len(states)} frames -> {out_path}")
    print(f"file size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"rejected {n_rejected} draws for the omega cap; "
          f"{n_uncapped} episodes kept over cap after {args.max_retries} retries")
    print(f"max |omega|*dt = {np.abs(states[:, 1]).max() * dt:.2f} rad/frame "
          f"(aliasing at pi)")
    print(f"|u| mean {np.abs(actions).mean():.2f}   "
          f"theta range [{states[:, 0].min():.2f}, {states[:, 0].max():.2f}]   "
          f"|omega| max {np.abs(states[:, 1]).max():.2f}")


if __name__ == "__main__":
    main()
