"""Train physics probes: embeddings → [sinθ, cosθ, ω].

Fits Ridge/KNN regression probes on latents from training episodes. Two
training distributions, both dense (all valid frames of every train episode,
not just one anchor per episode):

  GT probes (ridge_gt, knn_gt)   — input is the frozen encoder's own (q, p)
                                    (or z for LeWM) at every frame. ridge_gt is
                                    the original probe (see run_eval.py's
                                    _train_probe_episodes); knn_gt is the same
                                    data, fit with a KNeighborsRegressor.
  k=1 probes (ridge_k1, knn_k1)  — input is the MODEL'S OWN one-step rollout
                                    output: encode frame i, build (q_i, p_i),
                                    run model.predictor once, and fit against
                                    the GT target at frame i+1. This trains the
                                    probe on the same kind of (slightly biased)
                                    embeddings it will see when applied frozen
                                    along a real rollout, rather than on clean
                                    GT embeddings only.

run_eval.py imports train_probe_knn_gt / collect_k1_probe_data / fit_ridge_probe
/ fit_knn_probe from this module and calls them during its "3.a" probe-training
step, saving each to its own file (probe.pkl, probe_knn.pkl, probe_k1_ridge.pkl,
probe_k1_knn.pkl) in lehami/experiments/results/<exp_name>/.

This file also keeps its original standalone CLI (anchor-frame-only Ridge probe
→ probe_physics.pkl, used by make_rollout_error.py) — see main() below.

Usage:
    python lehami/experiments/src/train_probe_physics.py \\
        --checkpoint ~/.stable_worldmodel/checkpoints/lehami_fd_sep_lmom_pendulum_fd_sep_lmom_d192/best.ckpt \\
        --manifest   lehami/data/split_pendulum_episode_seed42.json \\
        --exp-name   exp013_lehami_fd_sep_lmom \\
        --device     cuda

    # smoke test
    python lehami/experiments/src/train_probe_physics.py \\
        --checkpoint <ckpt> --manifest <manifest> --exp-name smoke \\
        --device cpu --max-episodes 10
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_LEWM_PATH = ROOT / "lehami" / "models" / "third_party" / "leworldmodel"
if str(_LEWM_PATH) not in sys.path:
    sys.path.insert(0, str(_LEWM_PATH))
if str(ROOT / "lehami" / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "lehami" / "models"))

import hdf5plugin  # noqa: F401
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_targets import (  # leaf module: the target rule, shared with run_eval
    PENDULUM_TARGET_NAMES,      # noqa: F401 — re-exported for existing importers
    THREE_BODY_TARGET_NAMES,    # noqa: F401
    n_targets,                  # noqa: F401
    probe_targets,
    target_names,               # noqa: F401
)
from run_eval import (
    load_model, _load_manifest, _h5_path, _preprocess_batch,
    _encode_episode, _encode_z_lewm, _build_anchor, _latent_dim, _mom_type,
    _anchor_offset, _causal_p_batch, _zero_action_emb,
    _act_emb_at, _act_emb_window, _episode_actions, _latent_to_pixel_offset,
    _collect_probe_episode_data, _align_states, _enc_window,
    _episode_lengths, _probe_targets_block,
    Probe,
)


# ── feature extraction ────────────────────────────────────────────────────────

def _anchor_features(
    model, cls: torch.Tensor, q_enc: torch.Tensor,
    mom_type: str, cfg, device: str, probe_input: str,
) -> np.ndarray:
    """Return probe input vector for one anchor frame."""
    q0, p0 = _build_anchor(model, cls, q_enc, mom_type, cfg, device)
    q = q0.cpu().numpy()
    if probe_input == "qp":
        return np.concatenate([q, p0.cpu().numpy()])
    return q


# Probe targets are ENVIRONMENT-SPECIFIC and dispatched on the width of the stored
# state, because there is no generic map from a state vector to "the physics you
# want decoded". The pendulum path below is byte-identical to what it always was.
# The target rule, the names, and the widths now live in `probe_targets.py`, a
# leaf module imported by both this file and run_eval.  Re-exported here so the
# existing callers of `_targets` / `target_names` keep working unchanged.
_targets = probe_targets


# ── collection ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_anchor_data(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, probe_input: str, enc_batch: int = 50,
    max_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode anchor frame of each episode → (features, targets).

    Returns:
        X: (N, feature_dim) — probe inputs
        Y: (N, 3)           — [sinθ, cosθ, ω]
    """
    import h5py
    if max_episodes:
        episodes = episodes[:max_episodes]

    h         = cfg.history_size
    img_size  = cfg.img_size
    mom_type  = _mom_type(cfg.model.name)
    is_lewm   = (mom_type == "lewm")
    W         = _enc_window(model)   # 1 for every per-frame encoder

    X_list, Y_list = [], []
    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for ep_num, ep in enumerate(episodes):
            if ep_num % 100 == 0:
                print(f"  encoding episode {ep_num}/{len(episodes)}...", flush=True)

            start, L = int(ep_offsets[ep]), int(ep_lengths[ep])
            if L - W + 1 < h:
                continue

            pixels_raw = f["pixels"][start:start + L]
            states_raw = f["state"][start:start + L]
            pixels = _preprocess_batch(pixels_raw, img_size, device)
            # Features below live in LATENT time (index j ↔ pixel frame j+W-1-lag
            # for a temporal encoder); the GT array is in pixel time. Re-base it
            # or the anchor is scored against θ W-1 frames late. Identity at W=1.
            states_raw = _align_states(model, states_raw)

            if is_lewm:
                z = _encode_z_lewm(model, pixels, enc_batch)
                feat = z[h - 1].cpu().numpy()
            else:
                cls, q_enc = _encode_episode(model, pixels, enc_batch)
                feat = _anchor_features(model, cls, q_enc, mom_type, cfg, device, probe_input)

            X_list.append(feat)
            Y_list.append(_targets(states_raw[h - 1]))

    return np.stack(X_list), np.stack(Y_list)


# ── probe fitting (shared by GT and k=1 probes) ───────────────────────────────

def fit_ridge_probe(X: np.ndarray, Y: np.ndarray) -> Ridge:
    """Ridge(alpha=1.0), matching the convention used everywhere else in this repo."""
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X, Y)
    return ridge


def fit_knn_probe(X: np.ndarray, Y: np.ndarray, n_neighbors: int = 25) -> KNeighborsRegressor:
    """Distance-weighted KNN regressor — the non-linear counterpart to Ridge.

    No feature standardization: (q, p) already sit close to N(0, I) per-dim
    (SIGReg regularizes towards it during training), so raw features are
    already roughly on a common scale, consistent with how the Ridge probe
    is fit (also unstandardized).
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X, Y)
    return knn


def fit_mlp_probe(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> Pipeline:
    """Two-layer MLP probe, the third tier alongside ridge and kNN.

    WHY A THIRD TIER, and why an MLP specifically. Ridge is linear and kNN is
    local, so a quantity that is globally nonlinear but smooth can be invisible to
    both while remaining perfectly decodable: ridge cannot represent it, and kNN
    needs neighbours dense enough to interpolate it, which they are not in 128
    dimensions. On the actuated pendulum the ridge/kNN gap was the whole story of
    the omega result, and the conclusion there turned on which tier was used. A
    capacity-matched nonlinear readout closes that argument in one direction: if a
    quantity is absent under all three, it is absent from the representation
    rather than from the reader.

    Standardised inputs, because the latent blocks (q and p) can differ in scale by
    an order of magnitude and an unscaled MLP would spend its first layer undoing
    that. `early_stopping` holds out 10% internally, so the probe cannot be fit to
    convergence on noise; that matters because a probe with capacity is the one
    tier that CAN overfit, and an overfitting probe would manufacture a positive
    result out of nothing.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), mlp).fit(X, Y)


def _probe_train_r2(regressor, X: np.ndarray, Y: np.ndarray) -> float:
    return float(r2_score(Y, regressor.predict(X)))


# ── GT-embedding probes (frozen encoder output, dense over all train frames) ──

def train_probe_knn_gt(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
    n_neighbors: int = 25,
) -> Probe:
    """KNN counterpart of run_eval.py's GT ridge probe (_train_probe_episodes):
    same dense (q, p) features from the frozen encoder, same [sinθ,cosθ,ω]
    targets, fit with a KNeighborsRegressor instead of Ridge.
    """
    X, Y = _collect_probe_episode_data(
        model, cfg, h5_path, episodes, device, enc_batch, train_frames,
    )
    knn = fit_knn_probe(X, Y, n_neighbors)
    print(
        f"[probe:knn_gt] fit on {len(X)} samples from {len(episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  k={n_neighbors}  train R²={_probe_train_r2(knn, X, Y):.4f}"
    )
    return Probe(
        ridge=knn, latent_dim=_latent_dim(cfg),
        uses_p=(_mom_type(cfg.model.name) != "lewm"),
    )


def train_probe_mlp_gt(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
    seed: int = 0,
) -> Probe:
    """MLP counterpart of the GT ridge/kNN probes: same frozen-encoder features,
    same targets, third readout tier.

    Signature and return type deliberately mirror `train_probe_knn_gt`. The earlier
    version took a `probe_input` positional and returned a bare dict, which matched
    neither `_collect_probe_episode_data` (7 parameters, no `probe_input`) nor the
    `Probe` that every downstream consumer unpacks -- so it raised TypeError on its
    first call and had never been executed.
    """
    X, Y = _collect_probe_episode_data(
        model, cfg, h5_path, episodes, device, enc_batch, train_frames,
    )
    mlp = fit_mlp_probe(X, Y, seed)
    print(
        f"[probe:mlp_gt] fit on {len(X)} samples from {len(episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  train R²={_probe_train_r2(mlp, X, Y):.4f}"
    )
    return Probe(
        ridge=mlp, latent_dim=_latent_dim(cfg),
        uses_p=(_mom_type(cfg.model.name) != "lewm"),
    )


# ── k=1 probes (trained on the model's OWN one-step rollout output) ───────────

@torch.no_grad()
def collect_k1_probe_data(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONE predictor step from every valid GT anchor frame i in each
    episode, collecting the model's own k=1 output as probe input, paired
    with the GT physics target at frame i+1.

    Unlike collect_anchor_data / _collect_probe_episode_data (which see only
    the frozen encoder's clean output), this is the training distribution the
    "k=1" probes actually need: the same one-step-prediction biases they will
    be evaluated against later, applied frozen along a real rollout.

    All valid anchor positions within each episode share the same "history
    needed" constraint, so each episode is processed as ONE batched predictor
    call (batch = number of valid anchors), not one call per anchor.

    Returns:
        X: (N, 2·latent_dim) for lehami models [cat(q_pred, p_pred)],
           (N, embed_dim) for lewm [z_pred]
        Y: (N, 3) — [sinθ, cosθ, ω] at frame i+1, or (N, 4) with the rod length L
           appended when the dataset varies it (matching _collect_probe_episode_data,
           so the k1 tiers score `mae_length`/`r2_length` like the gt tiers do)
    """
    import h5py

    ep_lengths_arr = _episode_lengths(h5_path)   # None unless the dataset varies L
    mom_type = _mom_type(cfg.model.name)
    is_lewm  = (mom_type == "lewm")
    img_size = cfg.img_size
    fd_dt    = float(cfg.model.get("fd_dt", 1.0))
    a_off    = (cfg.history_size - 1) if is_lewm else _anchor_offset(mom_type, cfg)
    W        = _enc_window(model)   # 1 for every per-frame encoder

    X_list, Y_list = [], []
    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for ep_num, ep in enumerate(episodes):
            if ep_num % 100 == 0:
                print(f"  [k1-probe] episode {ep_num}/{len(episodes)}...", flush=True)

            L        = int(ep_lengths[ep])
            n_frames = min(L, train_frames) if train_frames else L

            start      = int(ep_offsets[ep])
            pixels_raw = f["pixels"][start : start + n_frames]
            states_raw = f["state"][start : start + n_frames]
            pixels     = _preprocess_batch(pixels_raw, img_size, device)
            # The k=1 feature is the model's state after ONE predictor step, and that
            # step is driven by an action. Feeding u = 0 here (the pre-2026-08-18
            # behaviour) makes the probe read a counterfactual "what if no torque"
            # state on an actuated dataset, which is not the distribution the
            # predictor produces.
            ep_actions = _episode_actions(f, start, n_frames, cfg, device)
            lat2pix = _latent_to_pixel_offset(model)
            # Every feature array below (z from _encode_z_lewm, cls/q_enc from
            # _encode_episode) is in LATENT time and is W-1 rows SHORTER than the
            # pixel sequence. Re-base the GT onto the same clock before deriving
            # anchor indices from it — otherwise `idx` overruns z by W-1 (a CUDA
            # gather assert) and every in-bounds row is off by W-1 frames.
            # Identity at W=1, so per-frame-encoder numbers are unchanged.
            states_raw = _align_states(model, states_raw)
            n_lat      = states_raw.shape[0]
            if n_lat - 1 <= a_off:
                continue   # not enough frames for even one anchor + its k=1 target

            idx = torch.arange(a_off, n_lat - 1, device=device)   # anchor positions i
            if idx.numel() == 0:
                continue

            if is_lewm:
                h = cfg.history_size
                z = _encode_z_lewm(model, pixels, enc_batch)          # (n_frames, embed_dim)
                offs = torch.arange(-(h - 1), 1, device=device)
                window_idx = idx.unsqueeze(1) + offs.unsqueeze(0)     # (N, h)
                buf = z[window_idx]                                    # (N, h, embed_dim)
                zero_ctx = model.action_encoder(
                    torch.zeros(1, h, cfg.data.action_dim, device=device))
                # One action per context frame, latent [i-h+1 .. i], matching
                # builders._build_lewm_ms_conv. Stacked per anchor.
                act = torch.cat([
                    _act_emb_window(model, ep_actions, zero_ctx,
                                    int(i) - h + 1, h, lat2pix)
                    for i in idx
                ], dim=0) if ep_actions is not None else zero_ctx.expand(idx.shape[0], -1, -1)
                z_pred = model.predict(buf, act)[:, -1]                # (N, embed_dim)
                feat = z_pred.cpu().numpy()
            else:
                cls, q_enc = _encode_episode(model, pixels, enc_batch)
                u_zero1 = _zero_action_emb(model, cfg, device)         # (1, phase_dim)
                N = idx.shape[0]
                q_i = q_enc[idx]
                p_i = _causal_p_batch(q_enc, idx, mom_type, fd_dt)
                # The step out of anchor i is driven by the action AT latent frame i.
                u_emb = (torch.cat([_act_emb_at(model, ep_actions, u_zero1,
                                                int(i), lat2pix) for i in idx], dim=0)
                         if ep_actions is not None else u_zero1.expand(N, -1))
                qp = torch.cat([q_i, p_i], dim=-1)
                qp_pred = model.predictor(qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0]
                feat = qp_pred.cpu().numpy()   # (N, 2d) = cat(q_pred, p_pred)

            idx_np = idx.cpu().numpy()
            X_list.append(feat)
            Y_list.append(_probe_targets_block(
                states_raw[idx_np + 1],
                None if ep_lengths_arr is None else ep_lengths_arr[ep],
            ))

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    return X, Y


def train_probe_ridge_k1(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
) -> Probe:
    """Ridge probe fit on the model's own k=1 rollout output (see collect_k1_probe_data)."""
    X, Y = collect_k1_probe_data(model, cfg, h5_path, episodes, device, enc_batch, train_frames)
    ridge = fit_ridge_probe(X, Y)
    print(
        f"[probe:ridge_k1] fit on {len(X)} samples from {len(episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  train R²={_probe_train_r2(ridge, X, Y):.4f}"
    )
    return Probe(
        ridge=ridge, latent_dim=_latent_dim(cfg),
        uses_p=(_mom_type(cfg.model.name) != "lewm"),
    )


def train_probe_knn_k1(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
    n_neighbors: int = 25,
) -> Probe:
    """KNN probe fit on the model's own k=1 rollout output (see collect_k1_probe_data)."""
    X, Y = collect_k1_probe_data(model, cfg, h5_path, episodes, device, enc_batch, train_frames)
    knn = fit_knn_probe(X, Y, n_neighbors)
    print(
        f"[probe:knn_k1] fit on {len(X)} samples from {len(episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  k={n_neighbors}  train R²={_probe_train_r2(knn, X, Y):.4f}"
    )
    return Probe(
        ridge=knn, latent_dim=_latent_dim(cfg),
        uses_p=(_mom_type(cfg.model.name) != "lewm"),
    )


def train_probe_mlp_k1(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
    seed: int = 0,
) -> Probe:
    """MLP probe fit on the model's OWN one-step rollout output, k=1.

    Mirrors `train_probe_knn_k1`; see `train_probe_mlp_gt` on why the previous
    signature could not be called.
    """
    X, Y = collect_k1_probe_data(
        model, cfg, h5_path, episodes, device, enc_batch, train_frames,
    )
    mlp = fit_mlp_probe(X, Y, seed)
    print(
        f"[probe:mlp_k1] fit on {len(X)} samples from {len(episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  train R²={_probe_train_r2(mlp, X, Y):.4f}"
    )
    return Probe(
        ridge=mlp, latent_dim=_latent_dim(cfg),
        uses_p=(_mom_type(cfg.model.name) != "lewm"),
    )


# ── metrics ───────────────────────────────────────────────────────────────────

def _wrapped_mae_deg(sincos_pred: np.ndarray, theta_gt: np.ndarray) -> float:
    theta_hat = np.arctan2(sincos_pred[:, 0], sincos_pred[:, 1])
    diff = np.arctan2(np.sin(theta_hat - theta_gt), np.cos(theta_hat - theta_gt))
    return float(np.degrees(np.abs(diff)).mean())


def _eval_probe_three_body(regressor, X: np.ndarray, Y: np.ndarray) -> dict:
    """Position and velocity are reported as SEPARATE blocks, never pooled.

    The two are the two clauses of the planning criterion, and the swing-up result
    turned entirely on their separation: the arms were indistinguishable on pose
    and differed only on velocity. A single pooled R^2 would have hidden that.
    Per-body numbers are kept as well, because the actuated body is the one the
    planning task scores and the one with the least pixel evidence.
    """
    P = regressor.predict(X)
    out = {
        "r2_pos": float(r2_score(Y[:, :6], P[:, :6], multioutput="uniform_average")),
        "r2_vel": float(r2_score(Y[:, 6:], P[:, 6:], multioutput="uniform_average")),
        "mae_pos": float(np.abs(P[:, :6] - Y[:, :6]).mean()),
        "mae_vel": float(np.abs(P[:, 6:] - Y[:, 6:]).mean()),
    }
    for b in range(3):
        pq, pv = slice(2 * b, 2 * b + 2), slice(6 + 2 * b, 6 + 2 * b + 2)
        out[f"r2_pos_body{b + 1}"] = float(r2_score(Y[:, pq], P[:, pq],
                                                    multioutput="uniform_average"))
        out[f"r2_vel_body{b + 1}"] = float(r2_score(Y[:, pv], P[:, pv],
                                                    multioutput="uniform_average"))
    return out


def _eval_probe(ridge: Ridge, X: np.ndarray, Y: np.ndarray, theta_gt: np.ndarray) -> dict:
    if Y.shape[1] == 12:
        return _eval_probe_three_body(ridge, X, Y)
    preds = ridge.predict(X)
    return {
        "r2_sincos": float(r2_score(Y[:, :2], preds[:, :2], multioutput="uniform_average")),
        "r2_omega":  float(r2_score(Y[:, 2], preds[:, 2])),
        "angle_mae_deg": _wrapped_mae_deg(preds[:, :2], theta_gt),
        "omega_mae": float(np.abs(preds[:, 2] - Y[:, 2]).mean()),
    }


def _format_probe_metrics(m: dict) -> str:
    """One line per split, keyed off the metric dict rather than the environment.

    `_eval_probe` already returns two different key sets, and the reporting used to
    assume the pendulum's. On three-body that raised KeyError('r2_sincos') AFTER the
    probe had been fitted and scored -- the numbers were computed, then thrown away
    by the print statement, and the `|| die` in run_variant_swingup.sh killed the
    remaining arms. Branching on the keys keeps the two in step by construction.
    """
    if "r2_pos" in m:
        return (f"R²(pos)={m['r2_pos']:.4f}  R²(vel)={m['r2_vel']:.4f}  "
                f"MAE(pos)={m['mae_pos']:.4f}  MAE(vel)={m['mae_vel']:.4f}")
    return (f"R²(sincos)={m['r2_sincos']:.4f}  R²ω={m['r2_omega']:.4f}  "
            f"MAE°={m['angle_mae_deg']:.2f}  ωMAE={m['omega_mae']:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint",   required=True, type=Path)
    parser.add_argument("--manifest",     required=True)
    parser.add_argument("--exp-name",     required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enc-batch",    type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap episodes per split (for smoke tests)")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    train_eps, test_eps = _load_manifest(args.manifest)
    h5_path   = _h5_path(cfg)
    d         = _latent_dim(cfg)
    mom_type  = _mom_type(cfg.model.name)
    probe_input = "q" if mom_type == "lewm" else "qp"

    print(f"Model      : {cfg.model.name}  |  latent_dim={d}  |  probe_input={probe_input}")
    print(f"H5         : {h5_path}")
    print(f"Train eps  : {len(train_eps)}  |  Test eps: {len(test_eps)}")

    # ── collect train features ────────────────────────────────────────────────
    print(f"\nCollecting train anchor features ({len(train_eps)} episodes)...")
    X_tr, Y_tr = collect_anchor_data(
        model, cfg, h5_path, train_eps, args.device, probe_input,
        args.enc_batch, args.max_episodes,
    )
    print(f"Train set: {len(X_tr)} anchors  |  feature_dim={X_tr.shape[1]}")

    # ── fit probe ─────────────────────────────────────────────────────────────
    print("\nFitting Ridge probe...")
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_tr, Y_tr)

    tr_metrics = _eval_probe(ridge, X_tr, Y_tr, Y_tr[:, 0])  # quick sanity on train
    # Note: theta_gt from Y_tr is not right — Y_tr has sinθ,cosθ not θ itself.
    # Recompute theta_gt properly:
    theta_tr = np.arctan2(Y_tr[:, 0], Y_tr[:, 1])
    tr_metrics = _eval_probe(ridge, X_tr, Y_tr, theta_tr)
    print(f"  Train  — {_format_probe_metrics(tr_metrics)}")

    # ── evaluate on test split ────────────────────────────────────────────────
    print(f"\nEvaluating on test split ({len(test_eps)} episodes)...")
    X_te, Y_te = collect_anchor_data(
        model, cfg, h5_path, test_eps, args.device, probe_input,
        args.enc_batch, args.max_episodes,
    )
    theta_te = np.arctan2(Y_te[:, 0], Y_te[:, 1])
    te_metrics = _eval_probe(ridge, X_te, Y_te, theta_te)
    print(f"  Test   — {_format_probe_metrics(te_metrics)}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = ROOT / "lehami" / "experiments" / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_physics.pkl"

    payload = {
        "ridge":        ridge,
        "probe_input":  probe_input,    # "q" or "qp"
        "latent_dim":   d,
        "model_name":   cfg.model.name,
        "train_metrics": tr_metrics,
        "test_metrics":  te_metrics,
    }
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"\nProbe saved → {out_path}")


if __name__ == "__main__":
    main()
