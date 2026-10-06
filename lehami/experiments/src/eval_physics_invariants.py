#!/usr/bin/env python
"""Physics-invariant evaluations: the things a Hamiltonian latent flow can do
that an unconstrained predictor structurally cannot.

Three tests, all decoded to PHYSICAL units so they are model-agnostic and
directly comparable between the Hamiltonian runs and the LeWM baseline.

  1. energy    -- Is the TRUE physical energy conserved along the rollout?
                  E = 1/2 w^2 + (g/L)(1 - cos th), computed from each model's
                  probe-decoded (th_hat, w_hat) at every step. This is NOT the
                  learned H_phi: H_phi lives in arbitrary latent units and LeWM
                  has no such quantity at all, so it cannot ground a comparison.
                  The physical energy can, and it is exactly conserved in the
                  data (the dataset is an undamped free pendulum).

  2. dt        -- Roll out with a DIFFERENT step size. The separable leapfrog
                  takes dt as an explicit parameter, so the same learned
                  Hamiltonian can be integrated at dt/M with M times as many
                  steps and should trace the SAME continuous trajectory. This
                  tests whether the model learned a continuous vector field or
                  merely a discrete one-frame map. LeWM has no dt parameter --
                  its predictor is a fixed discrete map -- so this test is
                  inapplicable to it by construction, which is itself the
                  finding.

  3. reversal  -- Integrate forward K steps, then integrate BACK K steps, and
                  ask whether the initial state is recovered. Two variants:

                    (a) dt -> -dt   : exact for any H (leapfrog is a symmetric
                                      composition), so this is a sanity check on
                                      the integrator. Expect ~machine precision.
                    (b) p -> -p     : returns to the start only if the learned
                                      kinetic energy K(p) is EVEN in p, i.e.
                                      only if the model discovered the physical
                                      time-reversal symmetry of the pendulum.
                                      This one is a real scientific test.

                  LeWM's ARPredictor has no inverse, so (a) and (b) have no
                  counterpart there.

Ground truth for the dataset (recovered from data/pendulum.h5 by regression,
R^2 = 0.999979): th'' = -(g/L) sin th with g/L = 9.7715, dt = 0.05 s per frame.
One model step (dt=1.0 in model units) therefore equals 0.05 s of physical time.

Usage
-----
  # all three tests, both models
  python eval_physics_invariants.py --exp exp027_sg_ms_d32 --test all
  python eval_physics_invariants.py --exp exp030_lewm_ms_srnn_k100 --test energy

  # smoke test
  python eval_physics_invariants.py --exp exp027_sg_ms_d32 --test all \
      --episodes 5 --horizon 20

Outputs land in lehami/experiments/results/<exp>/:
  physics_energy.csv        per-k physical-energy drift
  physics_dt.csv            per-(dt, k) trajectory error vs the true ODE
  physics_reversal.csv      per-K round-trip recovery error
  physics_invariants.json   scalar summary of all three
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_eval as RE  # noqa: E402

# Class must be resolvable in __main__ for the saved probe pickles.
sys.modules["__main__"].Probe = RE.Probe

# ── dataset physics (regressed from data/pendulum.h5) ────────────────────────
G_OVER_L = 9.771490
DT_PHYS = 0.05  # seconds of physical time per frame == per unit of model dt


G_GRAV = 9.81   # pendulum_multi.py:83 -- the g used to generate the multi-rod set


def episode_constants(h5_path) -> dict | None:
    """Per-episode (g/L, dt) for datasets whose episodes do NOT share one rod.

    `pendulum_multi_v2` mixes three rod lengths and several sampling timesteps,
    so the module-level G_OVER_L / DT_PHYS are wrong for it by up to 1.4x in
    frequency.  Returns None for the single-rod sets, where the globals hold.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "length" not in f:
            return None
        L = np.asarray(f["length"][:], dtype=float)
        dt = (np.asarray(f["dt"][:], dtype=float) if "dt" in f
              else np.full(L.shape, DT_PHYS))
    return {"g_over_L": G_GRAV / L, "dt": dt}


def phys_energy(theta: np.ndarray, omega: np.ndarray,
                g_over_L: float | np.ndarray = G_OVER_L) -> np.ndarray:
    """Specific mechanical energy of a unit-mass pendulum of length L.

    `g_over_L` may be a scalar or an (n_ep, 1) column, so a multi-rod set gets
    each episode its own potential term.
    """
    return 0.5 * omega**2 + g_over_L * (1.0 - np.cos(theta))


def true_ode(theta0: float, omega0: float, t: np.ndarray,
             g_over_L: float = G_OVER_L) -> np.ndarray:
    """Integrate th'' = -(g/L) sin th at times `t` (seconds). Returns (len(t), 2).

    RK4 on a fine uniform grid, then sampled at `t` -- accurate enough that the
    integration error is far below any model error we are measuring.
    """
    t = np.asarray(t, dtype=float)
    t_max = float(t.max())
    n_fine = max(int(t_max / (DT_PHYS / 200.0)) + 1, 2)
    tf = np.linspace(0.0, t_max, n_fine)
    hs = tf[1] - tf[0]

    def f(y):
        return np.array([y[1], -g_over_L * np.sin(y[0])])

    ys = np.empty((n_fine, 2))
    y = np.array([theta0, omega0], dtype=float)
    ys[0] = y
    for i in range(1, n_fine):
        k1 = f(y)
        k2 = f(y + 0.5 * hs * k1)
        k3 = f(y + 0.5 * hs * k2)
        k4 = f(y + hs * k3)
        y = y + (hs / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i] = y
    return np.stack([np.interp(t, tf, ys[:, 0]), np.interp(t, tf, ys[:, 1])], axis=1)


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


# ── model / probe loading ────────────────────────────────────────────────────
def load_probe(results_dir: Path, key: str = "knn_k1") -> RE.Probe:
    """Load a saved probe. knn_k1 is the FIDELITY probe: fit on the model's own
    k=1 rollout output, so it is matched to the distribution we decode here.

    Do not reuse this choice for canonicality claims -- knn_k1's fit
    distribution has velocity smeared across the whole representation (one
    predictor step mixes p into q), so it cannot show WHERE velocity lives.
    For decoding a trajectory to physical units, which is all we do here, that
    is irrelevant and knn_k1 is the most accurate option available.
    """
    fname = RE.PROBE_FILENAMES[key]
    with open(results_dir / fname, "rb") as fh:
        return pickle.load(fh)


def decode(probe: RE.Probe, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, D) latent -> (theta, omega) in physical units."""
    pred = probe.ridge.predict(X)
    return np.arctan2(pred[:, 0], pred[:, 1]), pred[:, 2]


# ── trajectory collection ────────────────────────────────────────────────────
@torch.no_grad()
def _episode_pixels(f, ep: int, cfg, device: str):
    start = int(f["ep_offset"][ep])
    L = int(f["ep_len"][ep])
    pixels = RE._preprocess_batch(f["pixels"][start : start + L], cfg.img_size, device)
    return pixels, f["state"][start : start + L], L


def rollout_latents(
    model, cfg, h5_path: Path, episodes: list[int], horizon: int, device: str,
    dt_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Autoregressive rollout, returning the raw latent at every step.

    dt_scale: multiplies the flow's dt and divides the number of *model* steps
    per emitted frame, so the emitted trajectory always lands on the same
    physical times. dt_scale=1 -> 1 step/frame (the trained setting);
    dt_scale=0.5 -> 2 half-steps per frame; dt_scale=2 -> one double step per
    two frames (only frames at even k are reachable). Hamiltonian models only.

    Returns (probe_input, states_gt, ks):
      probe_input (n_ep, K+1, D) -- latent to feed the probe
      states_gt   (n_ep, K+1, 2) -- true (theta, omega) at the matching frames
      ks          (K+1,)         -- rollout step index (in frames) of each entry
      eps_used    (n_ep,)        -- episode indices kept, in row order.  Episodes
                                    too short for the horizon are skipped, so this
                                    is NOT `episodes`; it is what indexes any
                                    per-episode constant onto the rows.
    """
    import h5py

    model_name = cfg.model.name
    mom_type = RE._mom_type(model_name)
    is_lewm = mom_type == "lewm"
    latent_dim = RE._latent_dim(cfg)

    if is_lewm:
        if dt_scale != 1.0:
            raise ValueError(
                "LeWM has no dt parameter: its predictor is a fixed discrete "
                "one-frame map, so a different step size is undefined. This is "
                "a structural property of the baseline, not a missing feature."
            )
        h = cfg.history_size
        act_emb = model.action_encoder(torch.zeros(1, h, cfg.data.action_dim, device=device))
    else:
        u_emb = RE._zero_action_emb(model, cfg, device)
        flow = model.predictor.flow
        dt_orig = flow.dt
        # sub-steps per emitted frame; must be an integer >= 1
        sub = 1.0 / dt_scale
        if abs(sub - round(sub)) > 1e-9 and abs(dt_scale - round(dt_scale)) > 1e-9:
            raise ValueError(f"dt_scale={dt_scale} must be 1/integer or integer")

    X_all, S_all, eps_used = [], [], []
    with h5py.File(h5_path, "r") as f:
        a = 0 if is_lewm else RE._anchor_frame(cfg, mom_type)
        if is_lewm:
            a = cfg.history_size - 1

        for ep in episodes:
            pixels, states, L = _episode_pixels(f, ep, cfg, device)
            k_range = min(horizon, L - a - 1)
            if k_range < 1:
                continue

            if is_lewm:
                z_enc = RE._encode_z_lewm(model, pixels)
                buf = z_enc[: cfg.history_size].unsqueeze(0)
                xs = [z_enc[a].cpu().numpy()]
                for _ in range(1, k_range + 1):
                    z_pred = model.predict(buf, act_emb)[:, -1].detach()
                    xs.append(z_pred[0].cpu().numpy())
                    buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1)
            else:
                cls, q_enc = RE._encode_episode(model, pixels)
                q0, p0 = _anchor_qp(model, cfg, cls, q_enc, mom_type, device)
                qp = torch.cat([q0, p0], dim=-1).unsqueeze(0)
                xs = [qp[0].cpu().numpy()]
                try:
                    flow.dt = dt_orig * dt_scale
                    n_sub = int(round(1.0 / dt_scale)) if dt_scale <= 1.0 else 1
                    stride = int(round(dt_scale)) if dt_scale > 1.0 else 1
                    for _ in range(1, k_range + 1, stride):
                        for _ in range(n_sub):
                            # detach every step: the flow re-enables grad
                            # internally for the autodiff through H, so the
                            # graph would otherwise grow over the whole rollout
                            qp = model.predictor(
                                qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0].detach()
                        xs.append(qp[0].cpu().numpy())
                finally:
                    flow.dt = dt_orig

            n = len(xs)
            stride = int(round(dt_scale)) if (not is_lewm and dt_scale > 1.0) else 1
            idx = a + np.arange(n) * stride
            idx = np.clip(idx, 0, L - 1)
            X_all.append(np.stack(xs))
            S_all.append(states[idx])
            eps_used.append(int(ep))

    n_min = min(x.shape[0] for x in X_all)
    X = np.stack([x[:n_min] for x in X_all])
    S = np.stack([s[:n_min] for s in S_all])
    stride = int(round(dt_scale)) if (not is_lewm and dt_scale > 1.0) else 1
    ks = np.arange(n_min) * stride
    return X, S, ks, np.asarray(eps_used)


def _anchor_qp(model, cfg, cls, q_enc, mom_type, device):
    """Anchor (q,p), mirroring run_eval.evaluate_rollout's per-mom_type logic."""
    q0, p0 = RE._build_anchor(model, cls, q_enc, mom_type, cfg, device)
    return q0, p0


# ── test 1: physical energy conservation ─────────────────────────────────────
def test_energy(model, cfg, h5_path, episodes, horizon, device, probe, out_rows):
    X, S, ks, eps_used = rollout_latents(model, cfg, h5_path, episodes, horizon, device)
    n_ep, K1, _ = X.shape

    th_hat, om_hat = decode(probe, X.reshape(n_ep * K1, -1))
    th_hat = th_hat.reshape(n_ep, K1)
    om_hat = om_hat.reshape(n_ep, K1)

    const = episode_constants(h5_path)
    gL = (const["g_over_L"][eps_used][:, None] if const else G_OVER_L)
    dt_ep = (const["dt"][eps_used] if const else np.full(n_ep, DT_PHYS))

    E_hat = phys_energy(th_hat, om_hat, gL)
    E_true = phys_energy(S[:, :, 0], S[:, :, 1], gL)
    E0_true = E_true[:, :1]                      # exactly conserved in the data
    E0_hat = E_hat[:, :1]                        # the probe's own k=0 estimate

    for j, k in enumerate(ks):
        out_rows.append({
            "k": int(k),
            "t_sec": float(k * dt_ep.mean()),
            "E_true_mean": float(E_true[:, j].mean()),
            "E_hat_mean": float(E_hat[:, j].mean()),
            # drift relative to each episode's own decoded anchor energy:
            # isolates DRIFT from the probe's constant decoding bias
            "rel_drift_vs_hat0": float(np.mean(np.abs(E_hat[:, j] - E0_hat[:, 0]) / np.abs(E0_hat[:, 0]))),
            # absolute error against the true conserved energy
            "rel_err_vs_true": float(np.mean(np.abs(E_hat[:, j] - E0_true[:, 0]) / np.abs(E0_true[:, 0]))),
            "E_true_rel_drift": float(np.mean(np.abs(E_true[:, j] - E0_true[:, 0]) / np.abs(E0_true[:, 0]))),
            "n_ep": int(n_ep),
        })
    return {
        "energy_rel_drift_final": out_rows[-1]["rel_drift_vs_hat0"],
        "energy_probe_floor": out_rows[0]["rel_err_vs_true"],
        "energy_data_drift_final": out_rows[-1]["E_true_rel_drift"],
    }


# ── test 2: different dt ─────────────────────────────────────────────────────
def test_dt(model, cfg, h5_path, episodes, horizon, device, probe, out_rows,
            dt_scales=(1.0, 0.5, 0.25, 2.0)):
    if RE._mom_type(cfg.model.name) == "lewm":
        raise ValueError("dt test is inapplicable to LeWM (no dt parameter).")

    summary = {}
    for ds in dt_scales:
        X, S, ks, eps_used = rollout_latents(model, cfg, h5_path, episodes, horizon,
                                             device, dt_scale=ds)
        n_ep, K1, _ = X.shape
        th_hat, om_hat = decode(probe, X.reshape(n_ep * K1, -1))
        th_hat = th_hat.reshape(n_ep, K1)
        om_hat = om_hat.reshape(n_ep, K1)

        # On a multi-rod set every episode has its own natural frequency and may
        # have its own sampling dt, so the ODE reference is built per episode.
        # Using one global g/L here would charge the flow for a mis-specified
        # reference of up to sqrt(1.4/0.7) = 1.41x in frequency.
        const = episode_constants(h5_path)
        gL_ep = (const["g_over_L"][eps_used] if const else np.full(n_ep, G_OVER_L))
        dt_ep = (const["dt"][eps_used] if const else np.full(n_ep, DT_PHYS))
        t = ks * float(np.median(dt_ep))
        # true ODE from each episode's own decoded anchor state, so a constant
        # anchor decoding bias is not charged to the flow
        th_ode = np.empty_like(th_hat)
        om_ode = np.empty_like(om_hat)
        for i in range(n_ep):
            y = true_ode(th_hat[i, 0], om_hat[i, 0], ks * dt_ep[i], gL_ep[i])
            th_ode[i], om_ode[i] = y[:, 0], y[:, 1]

        for j, k in enumerate(ks):
            out_rows.append({
                "dt_scale": ds,
                "dt_model": float(ds),
                "k_frames": int(k),
                "t_sec": float(t[j]),
                "g_over_L_mean": float(np.mean(gL_ep)),
                "n_model_steps": int(round(k / ds)) if ds > 0 else 0,
                "mae_theta_vs_ode": float(np.mean(np.abs(wrap(th_hat[:, j] - th_ode[:, j])))),
                "mae_omega_vs_ode": float(np.mean(np.abs(om_hat[:, j] - om_ode[:, j]))),
                "mae_theta_vs_data": float(np.mean(np.abs(wrap(th_hat[:, j] - S[:, j, 0])))),
                "n_ep": int(n_ep),
            })
        last = out_rows[-1]
        summary[f"dt{ds}_mae_theta_vs_ode_final"] = last["mae_theta_vs_ode"]
    return summary


# ── test 3: time reversal ────────────────────────────────────────────────────
@torch.no_grad()
def test_reversal(model, cfg, h5_path, episodes, horizon, device, out_rows,
                  Ks=(1, 2, 5, 10, 20, 30, 50)):
    import h5py

    mom_type = RE._mom_type(cfg.model.name)
    if mom_type == "lewm":
        raise ValueError("reversal test is inapplicable to LeWM (predictor has no inverse).")

    u_emb = RE._zero_action_emb(model, cfg, device)
    flow = model.predictor.flow
    dt_orig = flow.dt
    d = RE._latent_dim(cfg)

    def step(qp, n):
        for _ in range(n):
            qp = model.predictor(qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0].detach()
        return qp

    acc = {K: {"dt": [], "pflip": [], "scale": []} for K in Ks}
    with h5py.File(h5_path, "r") as f:
        a = RE._anchor_frame(cfg, mom_type)
        for ep in episodes:
            pixels, states, L = _episode_pixels(f, ep, cfg, device)
            if L - a - 1 < max(Ks):
                continue
            cls, q_enc = RE._encode_episode(model, pixels)
            q0, p0 = _anchor_qp(model, cfg, cls, q_enc, mom_type, device)
            qp0 = torch.cat([q0, p0], dim=-1).unsqueeze(0)
            nrm = float(qp0.norm())

            for K in Ks:
                # (a) dt -> -dt : exact for a symmetric composition, any H
                fwd = step(qp0, K)
                try:
                    flow.dt = -dt_orig
                    back = step(fwd, K)
                finally:
                    flow.dt = dt_orig
                acc[K]["dt"].append(float((back - qp0).norm()) / nrm)

                # (b) p -> -p : returns only if K(p) is EVEN in p
                fl = torch.cat([fwd[:, :d], -fwd[:, d:]], dim=-1)
                back2 = step(fl, K)
                back2 = torch.cat([back2[:, :d], -back2[:, d:]], dim=-1)
                acc[K]["pflip"].append(float((back2 - qp0).norm()) / nrm)
                acc[K]["scale"].append(nrm)

    for K in Ks:
        if not acc[K]["dt"]:
            continue
        out_rows.append({
            "K": K,
            "roundtrip_rel_err_dtflip": float(np.mean(acc[K]["dt"])),
            "roundtrip_rel_err_pflip": float(np.mean(acc[K]["pflip"])),
            "n_ep": len(acc[K]["dt"]),
        })
    return {
        "reversal_dtflip_maxK": out_rows[-1]["roundtrip_rel_err_dtflip"] if out_rows else None,
        "reversal_pflip_maxK": out_rows[-1]["roundtrip_rel_err_pflip"] if out_rows else None,
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="results dir name, e.g. exp027_sg_ms_d32")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--test", default="all", choices=["all", "energy", "dt", "reversal"])
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=50, help="max test episodes")
    ap.add_argument("--probe", default="knn_k1", choices=list(RE.PROBE_FILENAMES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    root = _HERE.parents[2]
    results = root / "lehami" / "experiments" / "results" / args.exp
    results.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else _guess_ckpt(args.exp)
    print(f"[load] {ckpt}")
    model, cfg = RE.load_model(ckpt, args.device, model_name=args.model_name)
    model.eval()

    manifest = args.manifest or str(
        root / "lehami" / "data" / "split_pendulum_episode_srnn_K100_seed42.json")
    _, test_eps = RE._load_manifest(manifest)
    test_eps = test_eps[: args.episodes]
    h5_path = RE._h5_path(cfg)
    is_lewm = RE._mom_type(cfg.model.name) == "lewm"
    print(f"[setup] model={cfg.model.name} lewm={is_lewm} episodes={len(test_eps)} "
          f"horizon={args.horizon} probe={args.probe}")

    probe = load_probe(results, args.probe)
    summary: dict = {"exp": args.exp, "model_name": cfg.model.name,
                     "is_lewm": is_lewm, "horizon": args.horizon,
                     "n_episodes": len(test_eps), "probe": args.probe,
                     "g_over_L": G_OVER_L, "dt_phys": DT_PHYS,
                     "per_episode_constants": episode_constants(h5_path) is not None}

    want = {"all": {"energy", "dt", "reversal"}}.get(args.test, {args.test})

    if "energy" in want:
        print("\n=== test 1: physical energy conservation ===")
        rows: list[dict] = []
        summary.update(test_energy(model, cfg, h5_path, test_eps, args.horizon,
                                   args.device, probe, rows))
        RE._write_rows_csv(results / "physics_energy.csv", rows)
        print(f"  -> physics_energy.csv ({len(rows)} rows)")
        for r in rows[:: max(1, len(rows) // 6)]:
            print(f"    k={r['k']:3d}  E_hat={r['E_hat_mean']:.4f}  "
                  f"drift={100*r['rel_drift_vs_hat0']:.2f}%  "
                  f"(data drift {100*r['E_true_rel_drift']:.4f}%)")

    if "dt" in want:
        print("\n=== test 2: rollout at a different dt ===")
        if is_lewm:
            print("  SKIPPED -- LeWM has no dt parameter. Its predictor is a fixed\n"
                  "  discrete one-frame map, so 'the same dynamics at half the step\n"
                  "  size' is undefined. This is the structural point of the test.")
            summary["dt_applicable"] = False
        else:
            rows = []
            summary.update(test_dt(model, cfg, h5_path, test_eps, args.horizon,
                                   args.device, probe, rows))
            summary["dt_applicable"] = True
            RE._write_rows_csv(results / "physics_dt.csv", rows)
            print(f"  -> physics_dt.csv ({len(rows)} rows)")

    if "reversal" in want:
        print("\n=== test 3: time reversal round trip ===")
        if is_lewm:
            print("  SKIPPED -- LeWM's ARPredictor has no inverse. There is no\n"
                  "  backward map to run, by construction.")
            summary["reversal_applicable"] = False
        else:
            rows = []
            Ks = tuple(K for K in (1, 2, 5, 10, 20, 30, 50) if K <= args.horizon)
            summary.update(test_reversal(model, cfg, h5_path, test_eps, args.horizon,
                                         args.device, rows, Ks=Ks))
            summary["reversal_applicable"] = True
            RE._write_rows_csv(results / "physics_reversal.csv", rows)
            print(f"  -> physics_reversal.csv ({len(rows)} rows)")
            for r in rows:
                print(f"    K={r['K']:3d}  dt-flip {r['roundtrip_rel_err_dtflip']:.3e}   "
                      f"p-flip {r['roundtrip_rel_err_pflip']:.3e}")

    (results / "physics_invariants.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[done] {results / 'physics_invariants.json'}")


def _guess_ckpt(exp: str) -> Path:
    """Map a results-dir name to its local checkpoint directory."""
    import os
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    ckpt_root = home / "checkpoints"
    tag = exp.split("_")[0]                       # 'exp027'
    cands = [d for d in ckpt_root.iterdir() if d.is_dir() and tag in d.name]
    if not cands:
        raise FileNotFoundError(
            f"No checkpoint dir under {ckpt_root} matching '{tag}'. "
            f"Pass --checkpoint explicitly.")
    if len(cands) > 1:
        raise ValueError(f"Ambiguous checkpoint dirs for '{tag}': "
                         f"{[c.name for c in cands]}. Pass --checkpoint.")
    return cands[0] / "best.ckpt"


if __name__ == "__main__":
    main()
