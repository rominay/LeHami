#!/usr/bin/env python
"""Does the one-step latent map preserve phase-space volume?

The DIRECT test of the paper's failure-mode claim. For a map z -> Phi(z) the
Jacobian's determinant IS the local volume expansion factor: an infinitesimal
ball of radius eps maps to an ellipsoid with semi-axes eps*sigma_i, so
vol(image)/vol(source) = prod_i sigma_i = |det J|.

A symplectic map satisfies M^T J_s M = J_s, hence (det M)^2 = 1 and |det M| = 1
EXACTLY, everywhere, every step (Liouville). `SeparableHamiltonianFlow` is an
exactly symplectic leapfrog: each substep updates one variable using only the
other, so its Jacobian is unit-triangular and the learned K and V sit strictly
off the diagonal, where they cannot touch the determinant. LeHami reading ~0 here
is therefore a CORRECTNESS CONTROL on the implementation, not evidence of a
benefit -- say so when quoting it. The claim only becomes a mechanism result if
the baseline's value is non-zero AND tracks something it measurably loses.

    # panel (a): one Jacobian per episode, at the anchor
    uv run python .../eval_jacobian_volume.py
    # panel (b): a Jacobian at EVERY step of the rollout
    uv run python .../eval_jacobian_volume.py --horizon 70
    # the actuated replication (true per-step torque, NOT the zero action)
    uv run python .../eval_jacobian_volume.py --arms a0q,b0s --horizon 76
    # independent estimator, no autograd
    uv run python .../eval_jacobian_volume.py --eps-ball

THE REFERENCE LINE IS EXACT AND FREE. The true pendulum flow is symplectic, so
ground truth is log|det J| = 0 by theorem -- there is no estimator to trust.
That is the advantage over an ensemble-covariance proxy, whose ground-truth
control does NOT come out flat: a cloud of trajectories at different energies
shears and filaments, and covariance is a second moment, not a volume, so it
grows even under an exactly volume-preserving flow (measured 1.00 -> 1.15 and
1.00 -> 1.55 on the TRUE dynamics).

═══ THE h=3 COMPLICATION, AND WHY TWO NUMBERS ARE REPORTED ══════════════════

LeHami (history_size 1) has a genuinely square map on its phase space:
(q, p) -> (q', p'), 2d x 2d. Its determinant is the phase-space volume factor.

LeWM (history_size 3) does not. Its predictor consumes three latents and emits
one, and the square object is the CONTEXT map
    (z_{t-2}, z_{t-1}, z_t) -> (z_{t-1}, z_t, z_{t+1})
whose Jacobian is block-companion,
    [ 0  I  0 ]
    [ 0  0  I ]      det = +/- det(A),  A = d z_{t+1} / d z_{t-2}
    [ A  B  C ]
so ALL of its volume behaviour is carried by the dependence on the OLDEST frame.
If the predictor largely ignores that frame, det ~ 0 and the arm reports
near-infinite contraction -- true, but a statement about finite context rather
than about learned dynamics, and a reviewer will say so.

Both are therefore reported:
  full    the literal phase-space volume (2d x 2d for LeHami, hE x hE for LeWM)
  newest  d z_{t+1} / d z_t alone, d x d for both arms -- dimension-matched and
          interpretable as "how the newest state is rescaled", but NOT a volume.
Quote `full` for the theorem and `newest` for the comparison; never one alone.

Values are per-dimension, (1/n) log|det J|, so maps of different size are
comparable. 0 = volume preserved, <0 = contraction, >0 = expansion.

═══ ACTUATION ═══════════════════════════════════════════════════════════════

On the swing-up the map is Phi(z, u), not Phi(z). What is measured is
d Phi / d z at the episode's TRUE per-step action, held fixed. Torque enters
through V(q, u), which is still a shear in the leapfrog, so the Jacobian stays
unit-triangular and the det = 1 reference line survives unchanged. The true
action is used rather than the zero token deliberately: the zero-action rollout
path is a known defect that once produced a spurious -29.8% LeWM gain, and here
it would measure a counterfactual dynamics the model was never trained on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import (  # noqa: E402
    _act_emb_at,
    _act_emb_window,
    _anchor_frame,
    _build_anchor,
    _enc_window,
    _encode_episode,
    _encode_z_lewm,
    _episode_actions,
    _episode_dt_ratio,
    _h5_path,
    _latent_dim,
    _latent_to_pixel_offset,
    _load_manifest,
    _mom_type,
    _preprocess_batch,
    _zero_action_emb,
    load_model,
)

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "lehami" / "experiments" / "results"
DATA = ROOT / "lehami" / "data"

MAN_MULTI = DATA / "split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json"
MAN_SWING = DATA / "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"

# key -> (label, checkpoint subdir, manifest, default horizon, results dir).
# The results dir is where the notebook finds probe_metrics.csv for panel (c).
ARMS = {
    "a0m": ("LeHami (EXP-067)",
            "lehami_sg_sep_sreg_joint_ms_conv_pendulum_multi_v2_sg_sep_conv_w1_multi_exp067_a0m",
            MAN_MULTI, 70, "exp067_a0m_lehami_w1_multi"),
    "b0m": ("LeWM  (EXP-068)",
            "lewm_ms_conv_pendulum_multi_v2_ms_conv_w1_multi_exp068_b0m",
            MAN_MULTI, 70, "exp068_b0m_lewm_w1_multi"),
    "a0q": ("LeHami (EXP-079, actuated)",
            "lehami_sg_sep_sreg_q_ms_conv_pendulum_swingup_lehami_sg_sep_conv_w1_swingup"
            "_exp079_a0q_s0",
            MAN_SWING, 76, "exp079_a0q_lehami_s0"),
    "b0s": ("LeWM  (EXP-072, actuated)",
            "lewm_ms_conv_pendulum_swingup_lewm_ms_conv_w1_swingup_exp072_b0s_s0",
            MAN_SWING, 76, "exp072_b0s_lewm_s0"),
}

# Extra seeds from conservative_multi_rod.sh. Seed 42 is the UNSUFFIXED dir
# above -- its run_suffix carries no seed component -- so the n=3 set per arm is
# {a0m, a0m_s1, a0m_s2}. Do not rename the unsuffixed dirs: published numbers
# point at those paths.
for _k, _tag in (("a0m", "_s1"), ("a0m", "_s2"), ("b0m", "_s1"), ("b0m", "_s2")):
    _lab, _sub, _man, _hor, _res = ARMS[_k]
    ARMS[f"{_k}{_tag}"] = (f"{_lab.strip()} {_tag[1:]}", _sub + _tag, _man, _hor,
                           _res + _tag)


def _ckpt_root() -> Path:
    """Checkpoint root: $STABLEWM_HOME/checkpoints, default ~/.stable_worldmodel."""
    import os
    return Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")) / "checkpoints"


def _spectrum(J: torch.Tensor, rel_tol: float = 1e-6) -> dict:
    """Singular-value summary of one Jacobian.

    `n_crushed` counts directions with sigma_i < rel_tol * sigma_max: those are
    numerically annihilated by the step, which is MERGING -- two states differing
    only along such a direction map to the same image, and no probe downstream can
    tell them apart. A symplectic map cannot do this (|det| = 1 bounds every
    sigma_i away from 0), so the count is 0 for LeHami by construction.
    """
    # On CPU: consumer GPUs run float64 at 1/32 rate and a per-step 192x192
    # cuSOLVER SVD dominates the whole run (11 min vs 2 min measured). The matrix
    # is tiny, so the transfer is free and LAPACK is far faster here.
    sv = torch.linalg.svdvals(J.detach().double().cpu())
    smax = float(sv[0])
    return {"sigma_max": smax, "sigma_min": float(sv[-1]),
            "log10_cond": float(torch.log10(sv[0] / sv[-1]).item())
            if float(sv[-1]) > 0 else float("inf"),
            "n_crushed": int((sv < rel_tol * smax).sum().item())}


def _slogdet_per_dim(J: torch.Tensor) -> float:
    """(1/n) log|det J|, via slogdet.

    slogdet rather than log(det(...)): a contracting map underflows det to 0 in
    float32 long before the log is meaningful, and returns -inf instead of a
    number. Computed in float64 because the whole point is to resolve a value
    that should be 0 to within float noise.
    """
    n = J.shape[0]
    sign, logabsdet = torch.linalg.slogdet(J.double())
    if sign.item() == 0:
        return float("-inf")
    return float(logabsdet.item() / n)


# ── the one-step maps ────────────────────────────────────────────────────────

def _step_lehami(model, qp, u, h_scalar):
    """(1, phase_dim) -> (1, phase_dim): one leapfrog step under action u."""
    return model.predictor(
        qp.unsqueeze(1), u.unsqueeze(1),
        None if h_scalar is None
        else torch.full((1,), float(h_scalar), device=qp.device),
    )[:, 0]


@torch.enable_grad()
def jac_lehami(model, qp, u, h_scalar):
    """Jacobian of the LeHami step AT state `qp`. Square, phase_dim x phase_dim."""
    def step(v):
        return _step_lehami(model, v.unsqueeze(0), u, h_scalar).squeeze(0)

    v0 = qp.reshape(-1).detach().requires_grad_(True)
    return torch.autograd.functional.jacobian(step, v0, vectorize=True)


@torch.enable_grad()
def jac_lewm(model, buf, act):
    """Jacobians of the LeWM context map AT context `buf`.

    Returns (J_full, A_newest, G):
      J_full  d(z_{t-1}, z_t, z_{t+1}) / d(z_{t-2}, z_{t-1}, z_t)   (hE x hE)
      A_new   d z_{t+1} / d z_t                                     (E x E)
      G       d z_{t+1} / d(every slot)                             (E x hE)
    The companion rows are a constant shift, so J_full is assembled analytically
    from the learned block rather than differentiated through -- identical result,
    and it makes the block structure explicit instead of implicit.
    """
    h, E = buf.shape[1], buf.shape[2]

    def newest(flat):
        return model.predict(flat.view(1, h, E), act)[:, -1].squeeze(0)

    flat0 = buf.reshape(-1).detach().requires_grad_(True)
    G = torch.autograd.functional.jacobian(newest, flat0, vectorize=True)

    J_full = torch.zeros(h * E, h * E, dtype=G.dtype, device=G.device)
    if h > 1:
        J_full[: (h - 1) * E, E:] = torch.eye((h - 1) * E, dtype=G.dtype, device=G.device)
    J_full[(h - 1) * E:, :] = G
    return J_full, G[:, (h - 1) * E:], G


# ── epsilon-ball: an independent estimator that never calls autograd ─────────

def eps_ball_logdet(phi, z0, eps: float, n_samples: int, seed: int = 0) -> float:
    """(1/n) log|det| of the local linear map, from a perturbation cloud.

    Recovers the Jacobian by LEAST SQUARES instead of by autograd: perturb z0 with
    n_samples isotropic displacements of scale eps, push each through phi, and
    regress output displacement on input displacement. With eps inside the linear
    regime and n_samples > n the recovered A ~ J, so comparing slogdet(A) against
    slogdet(J) cross-checks the whole autograd path -- which is exactly the
    "your Jacobian is wrong" objection, foreclosed.

    n_samples MUST exceed n or the regression is underdetermined and its
    determinant is meaningless; lstsq would return a rank-deficient fit silently.
    """
    n = z0.numel()
    if n_samples <= n:
        raise ValueError(f"n_samples={n_samples} must exceed state dim n={n}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    dz = (torch.randn(n_samples, n, generator=g) * eps).to(z0.device, z0.dtype)

    base = phi(z0).detach().reshape(-1)
    out = torch.stack([phi(z0 + d).detach().reshape(-1) - base for d in dz])

    # out ~ dz @ A^T, solved in float64 for the same reason slogdet is.
    A = torch.linalg.lstsq(dz.double(), out.double()).solution.T
    sign, logabsdet = torch.linalg.slogdet(A)
    return float("-inf") if sign.item() == 0 else float(logabsdet.item() / n)


# ── per-episode walkers ──────────────────────────────────────────────────────

def _rollout_rows(model, cfg, f, ep, offs, lens, device, horizon, u_zero,
                  h_arr, mom, lat2pix, k_stride):
    """Yield one row per rollout step for a single episode.

    The stepping mirrors run_eval.evaluate_rollout / evaluate_rollout_lewm
    exactly -- same anchor, same k_range, same action indexing. The Jacobian at
    step k is evaluated at the state BEING STEPPED OUT OF, which is the state the
    chain rule composes at.
    """
    start, L = int(offs[ep]), int(lens[ep])
    n_lat = L - _enc_window(model) + 1
    pix = _preprocess_batch(f["pixels"][start:start + L], cfg.img_size, device)
    acts = _episode_actions(f, start, L, cfg, device)
    hs = None if h_arr is None else float(h_arr[ep])
    hist = cfg.history_size

    if mom == "lewm":
        k_range = min(horizon, n_lat - hist)
        if k_range < 1:
            return
        z = _encode_z_lewm(model, pix)
        buf = z[:hist].unsqueeze(0).detach()
        E = buf.shape[2]
        u_base = model.action_encoder(
            torch.zeros(1, hist, cfg.data.action_dim, device=device))
        for k in range(1, k_range + 1):
            act_k = _act_emb_window(model, acts, u_base, k - 1, hist, lat2pix)
            if (k - 1) % k_stride == 0:
                J_full, A_new, G = jac_lewm(model, buf, act_k)
                row = {
                    "k": k - 1, "n_full": int(J_full.shape[0]),
                    "n_new": int(A_new.shape[0]),
                    "logdet_per_dim_full": _slogdet_per_dim(J_full),
                    "logdet_per_dim_newest": _slogdet_per_dim(A_new),
                    **_spectrum(J_full),
                }
                for i in range(hist):
                    row[f"slot{i}"] = _slogdet_per_dim(G[:, i * E:(i + 1) * E])
                yield row
            with torch.no_grad():
                z_pred = model.predict(buf, act_k)[:, -1]
            buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1).detach()
    else:
        a = _anchor_frame(cfg, mom)
        k_range = min(horizon, n_lat - a - 1)
        if k_range < 1:
            return
        cls, q_enc = _encode_episode(model, pix)
        q0, p0 = _build_anchor(model, cls, q_enc, mom, cfg, device)
        if h_arr is not None:
            p0 = p0 / hs
        qp = torch.cat([q0, p0], dim=-1).unsqueeze(0).detach()
        for k in range(1, k_range + 1):
            u_k = _act_emb_at(model, acts, u_zero, a + k - 1, lat2pix)
            if (k - 1) % k_stride == 0:
                J_full = jac_lehami(model, qp, u_k, hs)
                # history_size 1: the state IS the newest state, so the
                # newest-state map is the whole map. NOT the q-block -- the
                # q-block of a symplectic map is not itself volume-preserving,
                # so slicing it tests a claim the theory never makes.
                ld = _slogdet_per_dim(J_full)
                yield {
                    "k": k - 1, "n_full": int(J_full.shape[0]),
                    "n_new": int(J_full.shape[0]),
                    "logdet_per_dim_full": ld, "logdet_per_dim_newest": ld,
                    **_spectrum(J_full),
                }
            with torch.no_grad():
                qp = _step_lehami(model, qp, u_k, hs).detach()


def _eps_ball_rows(model, cfg, f, ep, offs, lens, device, u_zero, h_arr, mom,
                   lat2pix, eps_list, n_samples):
    """Epsilon-ball estimate at the anchor, for each eps in `eps_list`."""
    start, L = int(offs[ep]), int(lens[ep])
    pix = _preprocess_batch(f["pixels"][start:start + L], cfg.img_size, device)
    acts = _episode_actions(f, start, L, cfg, device)
    hs = None if h_arr is None else float(h_arr[ep])
    hist = cfg.history_size

    if mom == "lewm":
        z = _encode_z_lewm(model, pix)
        buf = z[:hist].unsqueeze(0).detach()
        E = buf.shape[2]
        u_base = model.action_encoder(
            torch.zeros(1, hist, cfg.data.action_dim, device=device))
        act_k = _act_emb_window(model, acts, u_base, 0, hist, lat2pix)

        def phi(flat):
            """The FULL context map on the flattened state: shift, then append."""
            b = flat.view(1, hist, E)
            with torch.no_grad():
                zp = model.predict(b, act_k)[:, -1]
            return torch.cat([b[:, 1:].reshape(-1), zp.reshape(-1)])

        z0 = buf.reshape(-1).detach()
        J_full, _, _ = jac_lewm(model, buf, act_k)
    else:
        a = _anchor_frame(cfg, mom)
        cls, q_enc = _encode_episode(model, pix)
        q0, p0 = _build_anchor(model, cls, q_enc, mom, cfg, device)
        if h_arr is not None:
            p0 = p0 / hs
        u_k = _act_emb_at(model, acts, u_zero, a, lat2pix)

        def phi(v):
            with torch.no_grad():
                return _step_lehami(model, v.unsqueeze(0), u_k, hs).squeeze(0)

        z0 = torch.cat([q0, p0], dim=-1).detach()
        J_full = jac_lehami(model, z0.unsqueeze(0), u_k, hs)

    jac_ref = _slogdet_per_dim(J_full)
    return [{"eps": e, "n_dim": int(z0.numel()), "logdet_per_dim_jac": jac_ref,
             "logdet_per_dim_ball": eps_ball_logdet(phi, z0, e, n_samples)}
            for e in eps_list]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="a0m,b0m")
    ap.add_argument("--max-episodes", type=int, default=45)
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = anchor only (panel a); >0 walks the rollout (panel b)")
    ap.add_argument("--k-stride", type=int, default=1,
                    help="compute a Jacobian every k-stride steps")
    ap.add_argument("--eps-ball", action="store_true",
                    help="run the independent epsilon-ball estimator instead")
    ap.add_argument("--eps", default="1e-3,5e-4",
                    help="comma list of scales; halving must not move the estimate")
    ap.add_argument("--ball-samples", type=int, default=0, help="0 = 2*state_dim")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import h5py
    import pandas as pd

    rows = []
    for key in [k.strip() for k in args.arms.split(",")]:
        if key not in ARMS:
            raise SystemExit(f"unknown arm {key!r}; known: {sorted(ARMS)}")
        label, ckpt_sub, manifest, default_h, _ = ARMS[key]
        ckpt = _ckpt_root() / ckpt_sub / "best.ckpt"
        if not ckpt.exists():
            raise SystemExit(f"{key}: checkpoint missing at {ckpt}")

        model, cfg = load_model(ckpt, args.device)
        model.eval()
        mom = _mom_type(cfg.model.name)
        h5 = _h5_path(cfg)
        _, test_eps = _load_manifest(str(manifest))
        ep_ids = test_eps[: args.max_episodes]
        h_arr = _episode_dt_ratio(h5, cfg)
        u_zero = _zero_action_emb(model, cfg, args.device)
        lat2pix = _latent_to_pixel_offset(model)
        horizon = args.horizon if args.horizon > 0 else default_h

        with h5py.File(h5, "r") as f:
            has_act = "action" in f
            print(f"\n[{key}] {label}  mom={mom}  history_size={cfg.history_size}  "
                  f"latent_dim={_latent_dim(cfg)}  episodes={len(ep_ids)}  "
                  f"actions={'TRUE per-step' if has_act else 'none (unactuated)'}")
            offs, lens = f["ep_offset"][:], f["ep_len"][:]

            for ep in ep_ids:
                if args.eps_ball:
                    n_state = (cfg.history_size * _latent_dim(cfg) if mom == "lewm"
                               else 2 * _latent_dim(cfg))
                    ns = args.ball_samples or 2 * n_state
                    for r in _eps_ball_rows(model, cfg, f, ep, offs, lens, args.device,
                                            u_zero, h_arr, mom, lat2pix,
                                            [float(x) for x in args.eps.split(",")], ns):
                        rows.append({"arm": key, "label": label, "episode": int(ep),
                                     "n_samples": ns, **r})
                else:
                    hz = horizon if args.horizon > 0 else 1
                    for r in _rollout_rows(model, cfg, f, ep, offs, lens, args.device,
                                           hz, u_zero, h_arr, mom, lat2pix,
                                           args.k_stride):
                        rows.append({"arm": key, "label": label, "episode": int(ep), **r})

    df = pd.DataFrame(rows)
    if args.out is None:
        name = ("epsball.csv" if args.eps_ball
                else "jacobian_rollout.csv" if args.horizon > 0 else "jacobian_k0.csv")
        args.out = RESULTS / "latent_volume" / name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)\n")

    if args.eps_ball:
        print("Epsilon-ball vs autograd Jacobian, (1/n) log|det|.\n"
              "Independent estimators of the same quantity: they must agree, and\n"
              "halving eps must not move the ball estimate.\n")
        g = df.groupby(["label", "eps"]).agg(
            jac=("logdet_per_dim_jac", "mean"), ball=("logdet_per_dim_ball", "mean"),
            ball_std=("logdet_per_dim_ball", "std"), n=("episode", "count"))
        g["abs_diff"] = (g.jac - g.ball).abs()
        print(g.round(5).to_string(), "\n")
        return

    print("(1/n) log|det J|.  0 = volume preserved, <0 contraction, >0 expansion.\n"
          "Ground truth for the true flow is EXACTLY 0.\n")
    for col, what in (("logdet_per_dim_full", "FULL map (the phase-space volume)"),
                      ("logdet_per_dim_newest", "NEWEST block (dimension-matched)")):
        print(f"--- {what} ---")
        summ = df.groupby("label")[col].agg(["mean", "std", "min", "max"]).round(5)
        summ["n_dim"] = df.groupby("label")["n_full" if "full" in col else "n_new"].first()
        print(summ.to_string(), "\n")

    if "n_crushed" in df:
        print("--- MERGING: directions annihilated per step (sigma_i < 1e-6 sigma_max) ---")
        sp = df.groupby("label").agg(
            n_crushed=("n_crushed", "mean"), sigma_min=("sigma_min", "mean"),
            log10_cond=("log10_cond", "mean"), n_dim=("n_full", "first")).round(4)
        print(sp.to_string(), "\n")

    if args.horizon > 0:
        print("--- CONSTANT along k, or relaxing? (newest block, mean over episodes) ---")
        show = [k for k in (0, 1, 2, 5, 10, 20, 40, 60, 69, 75) if k in set(df.k)]
        piv = (df[df.k.isin(show)]
               .pivot_table(index="k", columns="label",
                            values="logdet_per_dim_newest", aggfunc="mean").round(4))
        print(piv.to_string(), "\n")


if __name__ == "__main__":
    main()
