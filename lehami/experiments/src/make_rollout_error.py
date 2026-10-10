"""Rollout evaluation: apply a trained physics probe along the predicted trajectory.

Loads:
    probe_physics.pkl  — Ridge probe trained by train_probe_physics.py
    model checkpoint   — frozen encoder + predictor

For each episode in the chosen split, rolls out the predictor for `horizon`
steps and at each step k computes:

    Embedding metrics (q-space):
        rmse_norm_q   RMSE / √d vs ground-truth encoded q
        r2_q          R² vs ground-truth encoded q
        cosine_q      cosine similarity vs ground-truth encoded q

    Physics metrics (via probe):
        angle_mae_deg  wrapped MAE of predicted θ vs ground truth (degrees)
        r2_sincos      R²(sinθ, cosθ) multioutput uniform average
        omega_mae      MAE of predicted ω vs ground truth (rad/s)
        omega_r2       R² of predicted ω

    Energy metric (lehami only):
        energy_mean, energy_std  mean ± std of H_φ(q_k, p_k) across episodes

Output:
    lehami/experiments/results/<exp_name>/rollout_errors.csv

Usage:
    python lehami/experiments/src/make_rollout_error.py \\
        --checkpoint ~/.stable_worldmodel/checkpoints/lehami_fd_sep_lmom_pendulum_fd_sep_lmom_d192/best.ckpt \\
        --manifest   lehami/data/split_pendulum_episode_seed42.json \\
        --exp-name   exp013_lehami_fd_sep_lmom \\
        --horizon    97 --split test --device cuda

    # smoke test
    python lehami/experiments/src/make_rollout_error.py \\
        --checkpoint <ckpt> --manifest <manifest> --exp-name smoke \\
        --horizon 5 --device cpu --max-episodes 5
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from collections import defaultdict
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
from sklearn.metrics import r2_score

from run_eval import (
    load_model, _load_manifest, _h5_path, _preprocess_batch,
    _zero_action_emb, _latent_dim, _mom_type, _has_energy,
)
from utils import rollout_episode, probe_features


# ── per-step metric helpers ────────────────────────────────────────────────────

def _r2(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean(axis=0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _wrapped_mae_deg(sincos_pred: np.ndarray, theta_gt: np.ndarray) -> float:
    theta_hat = np.arctan2(sincos_pred[:, 0], sincos_pred[:, 1])
    diff = np.arctan2(np.sin(theta_hat - theta_gt), np.cos(theta_hat - theta_gt))
    return float(np.degrees(np.abs(diff)).mean())


def _embedding_metrics(pred: np.ndarray, gt: np.ndarray, d: int, prefix: str = "q") -> dict:
    mse = float(np.mean((pred - gt) ** 2))
    norm_p = np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
    norm_g = np.linalg.norm(gt,   axis=1, keepdims=True) + 1e-8
    cosine = float(((pred / norm_p) * (gt / norm_g)).sum(axis=1).mean())
    return {
        f"rmse_norm_{prefix}": math.sqrt(mse) / math.sqrt(d),
        f"r2_{prefix}":        _r2(pred, gt),
        f"cosine_{prefix}":    cosine,
    }


def _physics_metrics(sincos_omega_pred: np.ndarray, states: np.ndarray) -> dict:
    theta_gt = states[:, 0]
    omega_gt = states[:, 1]
    sincos_gt = np.stack([np.sin(theta_gt), np.cos(theta_gt)], axis=1)
    return {
        "angle_mae_deg": _wrapped_mae_deg(sincos_omega_pred[:, :2], theta_gt),
        "r2_sincos":     float(r2_score(sincos_gt, sincos_omega_pred[:, :2],
                                        multioutput="uniform_average")),
        "omega_mae":     float(np.abs(sincos_omega_pred[:, 2] - omega_gt).mean()),
        "omega_r2":      float(r2_score(omega_gt, sincos_omega_pred[:, 2])),
    }


@torch.no_grad()
def _energy_metrics(model, q_pred: np.ndarray, p_pred: np.ndarray,
                    device: str, cfg) -> dict:
    q_t   = torch.from_numpy(q_pred).float().to(device)
    p_t   = torch.from_numpy(p_pred).float().to(device)
    N     = len(q_t)
    u_emb = _zero_action_emb(model, cfg, device).expand(N, -1)  # (N, phase_dim)
    H     = model.predictor.flow.eval_H(q_t, p_t, u_emb)
    return {
        "energy_mean": float(H.mean().cpu()),
        "energy_std":  float(H.std().cpu()),
    }


# ── rollout evaluation ────────────────────────────────────────────────────────

def evaluate(
    model, cfg, h5_path: Path, episodes: list[int],
    probe: dict, horizon: int, device: str,
    split_name: str, enc_batch: int = 50,
    max_episodes: int | None = None,
) -> list[dict]:
    import h5py

    if max_episodes:
        episodes = episodes[:max_episodes]

    h          = cfg.history_size
    d          = _latent_dim(cfg)
    img_size   = cfg.img_size
    probe_in   = probe["probe_input"]
    ridge      = probe["ridge"]
    has_energy = _has_energy(cfg.model.name)
    is_lewm    = (_mom_type(cfg.model.name) == "lewm")

    # per-k accumulators
    acc: dict[int, dict[str, list]] = defaultdict(lambda: {
        "q_pred": [], "p_pred": [], "q_gt": [], "p_gt": [], "states": [],
    })

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for ep_num, ep in enumerate(episodes):
            if ep_num % 50 == 0:
                print(f"  [{split_name}] episode {ep_num}/{len(episodes)}...", flush=True)

            start, L = int(ep_offsets[ep]), int(ep_lengths[ep])
            if L < h + 1:
                continue

            pixels_raw = f["pixels"][start:start + L]
            states_raw = f["state"][start:start + L]  # (L, 2) [θ, ω]

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            steps  = rollout_episode(model, pixels, cfg, device, horizon, enc_batch)

            for step in steps:
                frame_idx = h - 1 + step.k
                if frame_idx >= L:
                    break
                a = acc[step.k]
                a["q_pred"].append(step.q_pred)
                a["p_pred"].append(step.p_pred)
                a["q_gt"].append(step.q_gt)
                a["p_gt"].append(step.p_gt)
                a["states"].append(states_raw[frame_idx])

    # ── aggregate per-k metrics ────────────────────────────────────────────────
    rows = []
    for k in sorted(acc.keys()):
        a       = acc[k]
        n       = len(a["q_pred"])
        q_pred  = np.stack(a["q_pred"])   # (N, d)
        p_pred  = np.stack(a["p_pred"])   # (N, d)
        q_gt    = np.stack(a["q_gt"])     # (N, d)
        p_gt    = np.stack(a["p_gt"])     # (N, d)
        states  = np.array(a["states"])   # (N, 2) dtype float

        row = {"split": split_name, "k": k, "n": n}
        row.update(_embedding_metrics(q_pred, q_gt, d, prefix="q"))
        if not is_lewm:
            row.update(_embedding_metrics(p_pred, p_gt, d, prefix="p"))
        else:
            row["rmse_norm_p"] = float("nan")
            row["r2_p"]        = float("nan")
            row["cosine_p"]    = float("nan")

        # build probe input for each step and predict physics
        if probe_in == "qp":
            feat = np.concatenate([q_pred, p_pred], axis=1)
        else:
            feat = q_pred
        sincos_omega_pred = ridge.predict(feat)  # (N, 3)
        row.update(_physics_metrics(sincos_omega_pred, states))

        if has_energy and not is_lewm:
            row.update(_energy_metrics(model, q_pred, p_pred, device, cfg))
        else:
            row["energy_mean"] = float("nan")
            row["energy_std"]  = float("nan")

        rows.append(row)
        if k % 10 == 0 or k == 0:
            p_str = (
                f"  RMSE/√d_p={row['rmse_norm_p']:.4f}  R²p={row['r2_p']:.4f}"
                if not math.isnan(row["rmse_norm_p"]) else ""
            )
            print(
                f"  k={k:3d}: RMSE/√d={row['rmse_norm_q']:.4f}"
                f"  R²q={row['r2_q']:.4f}"
                + p_str
                + f"  MAE°={row['angle_mae_deg']:.2f}"
                f"  R²θ={row['r2_sincos']:.4f}"
                f"  ωMAE={row['omega_mae']:.4f}"
                f"  R²ω={row['omega_r2']:.4f}"
                + (f"  H={row['energy_mean']:.3f}±{row['energy_std']:.3f}"
                   if not math.isnan(row["energy_mean"]) else "")
                + f"  n={n}"
            )

    return rows


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint",   required=True, type=Path)
    parser.add_argument("--manifest",     required=True)
    parser.add_argument("--exp-name",     required=True)
    parser.add_argument("--horizon",      type=int, default=97)
    parser.add_argument("--split",        choices=["train", "test", "both"],
                        default="test")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enc-batch",    type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap episodes per split (for smoke tests)")
    args = parser.parse_args()

    results_dir  = ROOT / "lehami" / "experiments" / "results" / args.exp_name
    probe_path   = results_dir / "probe_physics.pkl"
    if not probe_path.exists():
        print(f"Probe not found at {probe_path}")
        print("Run train_probe_physics.py first:")
        print(f"  python lehami/experiments/src/train_probe_physics.py "
              f"--checkpoint {args.checkpoint} --manifest {args.manifest} "
              f"--exp-name {args.exp_name}")
        raise SystemExit(1)

    with open(probe_path, "rb") as fh:
        probe = pickle.load(fh)

    model, cfg = load_model(args.checkpoint, args.device)
    train_eps, test_eps = _load_manifest(args.manifest)
    h5_path = _h5_path(cfg)

    print(f"Model      : {cfg.model.name}  |  probe_input={probe['probe_input']}")
    print(f"Horizon    : {args.horizon}  |  Split: {args.split}")
    print(f"Probe      : train R²θ={probe['train_metrics']['r2_sincos']:.4f}"
          f"  test R²θ={probe['test_metrics']['r2_sincos']:.4f}")

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", train_eps))
    if args.split in ("test", "both"):
        splits.append(("test", test_eps))

    all_rows = []
    for split_name, episodes in splits:
        print(f"\n[{split_name}] Rolling out {len(episodes)} episodes...")
        rows = evaluate(
            model, cfg, h5_path, episodes,
            probe, args.horizon, args.device,
            split_name, args.enc_batch, args.max_episodes,
        )
        all_rows.extend(rows)

    # ── write CSV ──────────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "rollout_errors.csv"

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nResults saved → {out_path}  ({len(all_rows)} rows)")
    else:
        print("No results to save (all episodes too short?)")


if __name__ == "__main__":
    main()
