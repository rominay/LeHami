#!/usr/bin/env python
"""Dump (ground truth, probe prediction) pairs on the probe's OWN TRAINING DATA.

PROTOCOL item 3' asks to "show the results of the training of the probes".  This writes
the raw scatter data behind that: for each probe tier, the prediction it makes on the
exact samples it was fit on, so a predicted-vs-true plot shows FIT QUALITY -- how much of
theta / omega the representation carries at all -- separately from the generalisation and
rollout-drift questions that the test-split CSVs answer.

    uv run python lehami/experiments/src/dump_probe_fit_scatter.py \
        --checkpoint <ckpt>/best.ckpt --manifest <manifest.json> --exp-name exp071_a0s_hjepa_s0

Output: `results/<exp_name>/probe_fit_scatter.csv`, tidy, one row per (tier, sample).
The columns follow the DATASET, dispatched on target width exactly as
`run_eval._probe_targets_block` does:

    PENDULUM (3 or 4 targets)
        tier, k, theta_true, theta_pred, omega_true, omega_pred, sin_true, sin_pred,
        cos_true, cos_pred

    THREE-BODY (12 targets)
        tier, k, {x,y}_{0,1,2}_{true,pred}, {vx,vy}_{0,1,2}_{true,pred}

    Body indices are 0-based, as in the generator's state layout; `probe_metrics.csv`
    numbers the same bodies from 1. Body 2 (`x_2, y_2`) is the ACTUATED one.

Tiers are dumped AT THE k THEY ARE FIT AT -- there is no other k at which "fit quality"
is defined:

    ridge_gt, knn_gt   k=0   frozen-encoder embeddings, dense over train frames
    ridge_k1, knn_k1   k=1   the model's own one-step rollout output

── WHY THE PROBES ARE LOADED, NOT REFIT ─────────────────────────────────────────────
The pickles in the results dir are the objects that produced every number in the eval
CSVs.  Refitting here would introduce a second, silently different probe (KNN in
particular is sensitive to the exact sample set), and the scatter would then not explain
the metrics it sits next to.  A tier whose pickle is missing is skipped with a warning.

── WHY TRAIN AND NOT TEST ───────────────────────────────────────────────────────────
On TRAIN, a KNN with k=25 distance-weighted neighbours is near-interpolating, so a tight
diagonal is expected and only a LOOSE one is informative: it means the features do not
determine the target even locally.  Ridge has no such freedom, so its train scatter is a
direct read of how linearly decodable the quantity is.  Read the two tiers accordingly --
knn_gt lying on the diagonal is weak evidence, ridge_gt lying on it is strong.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import (  # noqa: E402
    PROBE_FILENAMES,
    Probe,
    _collect_probe_episode_data,
    _h5_path,
    _latent_dim,
    _load_manifest,
    _mom_type,
    evaluate_rollout,
    evaluate_rollout_lewm,
    load_model,
)
from probe_targets import (  # noqa: E402
    THREE_BODY_STATE_DIM, TWO_BODY_STATE_DIM, target_names,
)
from train_probe_physics import collect_k1_probe_data  # noqa: E402

# The probe pickles were written by run_eval.py while it was `__main__`, so pickle
# recorded their class as `__main__.Probe` and resolves it against whatever __main__ is
# at LOAD time -- this script.  Without this alias, `pickle.load` dies with
# "Can't get attribute 'Probe' on <module '__main__' ...>".  Binding the real class into
# this module's __main__ entry is the standard fix and is safe: it is the same class
# object run_eval defines, not a stand-in.
__main__ = sys.modules["__main__"]
if not hasattr(__main__, "Probe"):
    __main__.Probe = Probe

ROOT = Path(__file__).resolve().parents[3]

# tier -> the k at which that tier is fit.  Not a free parameter.
TIER_K = {"ridge_gt": 0, "knn_gt": 0, "ridge_k1": 1, "knn_k1": 1}


def _theta(sin_c: np.ndarray, cos_c: np.ndarray) -> np.ndarray:
    return np.arctan2(sin_c, cos_c)


def _scatter_frame(tier: str, k: int, Y: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """(true, pred) pairs as a tidy frame, with columns named for the actual state.

    Names come from `probe_targets.target_names`, the single definition shared
    with run_eval, so this file holds no naming table of its own.  The pendulum
    path is unchanged.  The three-body path previously fell through to it, which
    wrote `Y[:, 0:3]` under the names `sin`, `cos`, `omega` and took an `atan2` of
    two Cartesian coordinates: the numbers were real (they are the first three
    state coordinates) but the headers named quantities that do not exist here,
    and nine of the twelve channels -- including both coordinates of the ACTUATED
    body, the one planning scores -- were dropped on the floor.
    """
    if Y.shape[1] in (THREE_BODY_STATE_DIM, TWO_BODY_STATE_DIM):
        cols: dict = {"tier": tier, "k": k}
        for j, name in enumerate(target_names(Y.shape[1])):
            cols[f"{name}_true"] = Y[:, j]
            cols[f"{name}_pred"] = P[:, j]
        return pd.DataFrame(cols)

    return pd.DataFrame({
        "tier": tier, "k": k,
        "sin_true": Y[:, 0], "sin_pred": P[:, 0],
        "cos_true": Y[:, 1], "cos_pred": P[:, 1],
        "omega_true": Y[:, 2], "omega_pred": P[:, 2],
        "theta_true": _theta(Y[:, 0], Y[:, 1]),
        "theta_pred": _theta(P[:, 0], P[:, 1]),
    })


def _scatter_summary(df: pd.DataFrame) -> str:
    """One line of fit quality, in the units the dataset actually has."""
    # Try the multi-body namings in turn; both start with "x_0", so the first hit
    # is decided by which width's full column set is present.
    names = next(
        (n for n in (target_names(THREE_BODY_STATE_DIM), target_names(TWO_BODY_STATE_DIM))
         if all(f"{c}_true" in df.columns for c in n)),
        target_names(THREE_BODY_STATE_DIM),
    )
    if f"{names[0]}_true" in df.columns:
        pos = [c for c in names if not c.startswith("v")]
        vel = [c for c in names if c.startswith("v")]
        mae = lambda names: float(np.mean([  # noqa: E731
            np.abs(df[f"{n}_pred"] - df[f"{n}_true"]).mean() for n in names]))
        return f"MAE_pos={mae(pos):.4f}  MAE_vel={mae(vel):.4f}  (world units)"
    # Wrapped angular error, so the +/-pi seam is not counted as a huge miss.
    dth = np.abs(np.angle(np.exp(1j * (df.theta_pred - df.theta_true))))
    return (f"MAE_theta={dth.mean():.4f} rad  "
            f"MAE_omega={np.abs(df.omega_pred - df.omega_true).mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enc-batch", type=int, default=50)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap episodes (smoke test)")
    ap.add_argument("--subsample", type=int, default=20000,
                    help="max rows kept per tier (uniform random, seeded); 0 = keep all")
    ap.add_argument("--train-frames", type=int, default=80,
                    help="must match the training protocol; 80 for the swing-up data")
    ap.add_argument("--no-dump-features", action="store_true",
                    help="skip probe_features.npz (the scatter CSV alone is still written)")
    ap.add_argument("--dump-rollout", action="store_true",
                    help="also write rollout_latents.npz: the predictor's (q, p) and the "
                         "[theta, omega] targets at EVERY k on the test split, so a frozen "
                         "probe of any kind can be applied along the rollout locally")
    ap.add_argument("--dump-rollout-train", action="store_true",
                    help="also write rollout_latents_train.npz — the same dump on the TRAIN "
                         "episodes. Needed to REFIT a probe at each k (fit on train "
                         "features at k, score on test features at the same k), which is "
                         "what separates 'the quantity moved' from 'the quantity is gone'. "
                         "Implies --dump-rollout.")
    ap.add_argument("--horizon", type=int, default=76,
                    help="rollout horizon for --dump-rollout; ks with no data are skipped, "
                         "so 76 is safe for the W=3 arms that only reach 74")
    args = ap.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    train_eps, test_eps = _load_manifest(args.manifest)
    if args.max_episodes:
        train_eps = train_eps[: args.max_episodes]
        test_eps = test_eps[: args.max_episodes]
    h5_path = _h5_path(cfg)
    mom_type = _mom_type(cfg.model.name)

    print(f"Model     : {cfg.model.name}  latent_dim={_latent_dim(cfg)}  mom={mom_type}")
    print(f"Train eps : {len(train_eps)}  (probe training set)")
    print(f"Test  eps : {len(test_eps)}   (held out; encoded only for the feature dump)")

    out_dir = ROOT / "lehami" / "experiments" / "results" / args.exp_name

    # Encode once per k and share across the two tiers fit at that k -- encoding is the
    # expensive part and the tiers differ only in the regressor.
    feats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    print("\nCollecting k=0 features (frozen encoder, dense over train frames)...")
    feats[0] = _collect_probe_episode_data(
        model, cfg, h5_path, train_eps, args.device, args.enc_batch, args.train_frames,
    )
    print(f"  k=0: {feats[0][0].shape[0]} samples, dim={feats[0][0].shape[1]}")

    print("Collecting k=1 features (one step through the predictor)...")
    feats[1] = collect_k1_probe_data(
        model, cfg, h5_path, train_eps, args.device, args.enc_batch, args.train_frames,
    )
    print(f"  k=1: {feats[1][0].shape[0]} samples, dim={feats[1][0].shape[1]}")

    # ── raw feature dump ─────────────────────────────────────────────────────
    # Encoding is the ONLY step that needs the GPU, the checkpoint and the pixel H5.
    # Writing (X, Y) out means every later probe question -- a higher-capacity readout,
    # a different KNN k, per-episode diagnostics -- is a local CPU job on a ~15 MB file
    # instead of another cluster round-trip.  TEST features are included precisely so a
    # capacity test can be scored on held-out episodes; without them the dump would only
    # support in-sample statements.
    if not args.no_dump_features:
        print("\nCollecting TEST-split features for the held-out score...")
        te0 = _collect_probe_episode_data(
            model, cfg, h5_path, test_eps, args.device, args.enc_batch, args.train_frames,
        )
        te1 = collect_k1_probe_data(
            model, cfg, h5_path, test_eps, args.device, args.enc_batch, args.train_frames,
        )
        # Record the target names rather than leaving the consumer to guess from the
        # width. Derived from the width via the shared table, so the three-body dump
        # no longer carries three pendulum names beside a twelve-column array.
        n_y = feats[0][1].shape[1]
        # Widths that name themselves: the multi-body sets' targets ARE the state,
        # so n_y is the state width. Anything else is the pendulum, whose 4th column
        # is the rod length.
        y_cols = target_names(
            n_y if n_y in (THREE_BODY_STATE_DIM, TWO_BODY_STATE_DIM) else 2,
            with_length=(n_y == 4),
        )

        npz = out_dir / "probe_features.npz"
        np.savez_compressed(
            npz,
            X_train_k0=feats[0][0].astype(np.float32), Y_train_k0=feats[0][1].astype(np.float32),
            X_train_k1=feats[1][0].astype(np.float32), Y_train_k1=feats[1][1].astype(np.float32),
            X_test_k0=te0[0].astype(np.float32), Y_test_k0=te0[1].astype(np.float32),
            X_test_k1=te1[0].astype(np.float32), Y_test_k1=te1[1].astype(np.float32),
            y_cols=np.array(y_cols),
            latent_dim=np.array(_latent_dim(cfg)),
            uses_p=np.array(mom_type != "lewm"),
            model_name=np.array(str(cfg.model.name)),
            exp_name=np.array(args.exp_name),
        )
        mb = npz.stat().st_size / 1e6
        print(f"  wrote {npz.name}  ({mb:.1f} MB)  "
              f"train k0={feats[0][0].shape} k1={feats[1][0].shape}  "
              f"test k0={te0[0].shape} k1={te1[0].shape}  y_cols={y_cols}")

    # ── rollout latents ──────────────────────────────────────────────────────
    # evaluate_rollout already accumulates the predicted (q, p) and the [theta, omega]
    # targets at every k and then throws them away once the per-k metrics are computed.
    # dump_path makes it persist them instead; the metrics it returns are unchanged, and
    # they are discarded here because run_eval.py is the authority on those numbers.
    if args.dump_rollout or args.dump_rollout_train:
        # LeWM rolls out through a different function; both accept dump_path and write
        # the same key names, so the consumer never branches on the arm.
        _roll = evaluate_rollout_lewm if mom_type == "lewm" else evaluate_rollout
        dumps = [("test", test_eps, "rollout_latents.npz")]
        if args.dump_rollout_train:
            dumps.append(("train", train_eps, "rollout_latents_train.npz"))
        for split_name, eps, fname in dumps:
            print(f"\nRolling out the {split_name.upper()} split to k={args.horizon} "
                  f"to dump per-k latents...")
            _roll(
                model, cfg, h5_path, eps, {}, args.horizon, args.device,
                split_name, args.enc_batch, dump_path=out_dir / fname,
            )

    rng = np.random.default_rng(0)
    rows = []
    for tier, k in TIER_K.items():
        p_path = out_dir / PROBE_FILENAMES[tier]
        if not p_path.exists():
            print(f"  [skip] {tier}: {p_path.name} not found")
            continue
        with open(p_path, "rb") as fh:
            probe = pickle.load(fh)

        X, Y = feats[k]
        expected = 2 * probe.latent_dim if probe.uses_p else probe.latent_dim
        if X.shape[1] != expected:
            print(f"  [skip] {tier}: feature dim {X.shape[1]} != probe expects {expected}")
            continue

        P = probe.ridge.predict(X)          # .ridge holds Ridge or KNeighborsRegressor
        idx = np.arange(len(X))
        if args.subsample and len(idx) > args.subsample:
            idx = rng.choice(idx, args.subsample, replace=False)

        df = _scatter_frame(tier, k, Y[idx], P[idx])
        rows.append(df)

        print(f"  {tier:9s} k={k}  n={len(df):6d}  {_scatter_summary(df)}")

    if not rows:
        raise SystemExit("no probe tiers dumped — are the probe pickles present?")

    out = out_dir / "probe_fit_scatter.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
