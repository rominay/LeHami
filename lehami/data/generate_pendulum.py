"""Generate a pendulum pixel dataset and save as HDF5.

Simulates a conservative pendulum (no friction, no actions) and renders
each frame as a 64x64 RGB image. Saves in the format expected by
stable-worldmodel's HDF5Dataset.

The state of a pendulum is given by the angle theta and angular velocity dtheta at that time step. It has some initial conditions (theta from  [-π, π] and dtheta from  [-5, 5]) and evolves.
The physics is integrated using the RK4 method for accuracy:

θ'' = -(g/L) sin(θ)

with g = 9.81, L = 1.0 m, and a timestep DT = 0.05 s (20 Hz).

Note that the pendulum's natural frequency is ~0.5 Hz (one full swing every 2 seconds). Sampling at 20 Hz puts you 20× below the Nyquist limit — the signal is massively oversampled from a signal-processing standpoint, which is intentional for two reasons:

- Smoothness for the model. Consecutive frames change by at most ~18° in the fastest case (near-escape trajectory at θ=0). That's a small, learnable visual difference. If you doubled DT to 0.1s, fast trajectories would jump 35° per frame — harder for an encoder to track.
- RK4 accuracy. RK4 error scales as O(DT⁵). At DT=0.05 the integrator is very accurate; energy is conserved to numerical precision over 100 steps.


The initial conditions are randomly sampled for each episode.

Each episode is one independent pendulum trajectory starting from a randomly sampled (θ₀, dθ₀/dt). After 100 steps, it stops and the next episode starts from a completely different initial condition.
The reason to use this structure rather than one long continuous simulation is that it gives the model diverse coverage of the state space — many different starting angles and velocities — 
instead of one long trajectory that might only explore a narrow region. It also matches the format expected by world model training code (like LeWorldModel), which is built around episodic data.

Note that one period is T = 2π / √(g/L) = 2π / √9.81 ≈ 2.0 seconds. Given that in each episode we have 100 frames at 20 Hz, each episode is 5 seconds long — about 2.5 full swings for a typical trajectory. This is long enough to see interesting dynamics but short enough to keep the dataset size manageable.

Episode structure

- 500 episodes (default), each 100 frames long → 50,000 total frames
- Each episode gets an independent initial condition: θ₀ ~ Uniform(-π, π), dθ₀ ~ Uniform(-2, 2)
- Episodes are statistically independent — no continuity across episode boundaries
- This gives broad phase-space coverage rather than a single long chaotic trajectory

Usage:
    python scripts/generate_pendulum.py

Output:
    ~/.stable-wm/pendulum.h5

HDF5 layout:
    ep_len    (N_episodes,)              episode lengths
    ep_offset (N_episodes,)              global start index per episode
    pixels    (total_frames, 64, 64, 3)  uint8 RGB frames  [H, W, C]
    state     (total_frames, 2)          float32 (theta, dtheta) for probing
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — needed to activate compression codecs
import numpy as np
from PIL import Image, ImageDraw

# ── physics ──────────────────────────────────────────────────────────────────

G = 9.81
L = 1.0       # rod length (metres)
DT = 0.05     # integration step (seconds)


def _rk4_step(theta: float, dtheta: float) -> tuple[float, float]:
    """One RK4 step of θ'' = -(g/L) sin(θ)."""
    def deriv(th, dth):
        return dth, -(G / L) * np.sin(th)

    k1 = deriv(theta, dtheta)
    k2 = deriv(theta + DT / 2 * k1[0], dtheta + DT / 2 * k1[1])
    k3 = deriv(theta + DT / 2 * k2[0], dtheta + DT / 2 * k2[1])
    k4 = deriv(theta + DT * k3[0], dtheta + DT * k3[1])

    theta_next = theta + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    dtheta_next = dtheta + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta_next, dtheta_next


def simulate_episode(ep_len: int, theta0: float, dtheta0: float) -> np.ndarray:
    """Return state array (ep_len, 2) with columns [theta, dtheta]."""
    states = np.empty((ep_len, 2), dtype=np.float32)
    theta, dtheta = theta0, dtheta0
    for t in range(ep_len):
        states[t] = [theta, dtheta]
        theta, dtheta = _rk4_step(theta, dtheta)
    return states


# ── rendering ────────────────────────────────────────────────────────────────

IMG_SIZE = 64
ROD_PX = 24       # rod length in pixels
BOB_R = 4         # bob radius in pixels
PIVOT = (IMG_SIZE // 2, IMG_SIZE // 2)


def render_frame(theta: float) -> np.ndarray:
    """Render pendulum angle theta to (64, 64, 3) uint8 array."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # theta=0 → hanging down; positive = clockwise
    bob_x = PIVOT[0] + ROD_PX * np.sin(theta)
    bob_y = PIVOT[1] + ROD_PX * np.cos(theta)   # y increases downward

    draw.line([PIVOT, (bob_x, bob_y)], fill=(50, 50, 50), width=2)
    draw.ellipse(
        [bob_x - BOB_R, bob_y - BOB_R, bob_x + BOB_R, bob_y + BOB_R],
        fill=(200, 50, 50),
    )
    draw.ellipse(
        [PIVOT[0] - 2, PIVOT[1] - 2, PIVOT[0] + 2, PIVOT[1] + 2],
        fill=(50, 50, 200),
    )
    return np.array(img, dtype=np.uint8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--ep-len", type=int, default=100,
                        help="Frames per episode")
    parser.add_argument("--out-dir", type=str, default="~/.stable-wm")
    parser.add_argument("--name", type=str, default="pendulum")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out_dir).expanduser() / f"{args.name}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ep = args.n_episodes
    ep_len = args.ep_len
    total = n_ep * ep_len

    print(f"Generating {n_ep} episodes × {ep_len} frames → {total} total frames")
    print(f"Output: {out_path}")

    ep_lens = np.full(n_ep, ep_len, dtype=np.int64)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lens[:-1])]).astype(np.int64)

    all_pixels = np.empty((total, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    all_states = np.empty((total, 2), dtype=np.float32)

    for ep in range(n_ep):
        # Sample initial conditions: angle in [-pi, pi], small angular velocity
        theta0 = rng.uniform(-np.pi, np.pi)
        dtheta0 = rng.uniform(-2.0, 2.0)

        states = simulate_episode(ep_len, theta0, dtheta0)
        start = ep_offsets[ep]
        all_states[start : start + ep_len] = states

        for t in range(ep_len):
            all_pixels[start + t] = render_frame(states[t, 0])

        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{n_ep} episodes done")

    print("Writing HDF5 file...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ep_len",    data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("pixels",    data=all_pixels,
                         chunks=(1, IMG_SIZE, IMG_SIZE, 3),
                         **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("state",     data=all_states)

    print(f"Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
