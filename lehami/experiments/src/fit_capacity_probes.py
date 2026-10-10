#!/usr/bin/env python
"""Probe CAPACITY test: how much of (sin θ, cos θ, ω) is in the latent at all?

Runs entirely on the `probe_features.npz` files written by `dump_probe_fit_scatter.py`
-- no GPU, no checkpoint, no pixel H5.  Local CPU job.

    uv run python lehami/experiments/src/fit_capacity_probes.py \
        --exps exp071_a0s_lehami_s0,exp072_b0s_lewm_s0,exp075_a3s_lehami_s0

── THE QUESTION ─────────────────────────────────────────────────────────────────────
The shipped probes disagree sharply on LeHami: Ridge gets ~52° on theta while a 25-NN
gets ~10.5°, and LeWM gets ~0.8° with either.  Two readings fit that:

  (A) REPARAMETRISATION.  The angle is in LeHami's (q, p) but on a curved manifold, so a
      linear readout cannot see it.  Predicts: a higher-capacity probe closes most of the
      gap to LeWM.
  (B) INFORMATION LOSS.  The angle is largely gone -- LeHami's q is 32-dim against LeWM's
      64, and its other 32 inputs are a Savitzky-Golay stencil OF q, not an independent
      view.  Predicts: capacity does not help; the MLP stalls near the KNN number.

This script fits the same three regressors on every arm and reports both, so the answer
is a comparison rather than an anecdote.  All probes are MULTI-OUTPUT over
[sin θ, cos θ, ω] (the same targets the shipped probes use); theta is recovered with
atan2 and scored with a WRAPPED error, so the ±π seam is not counted as a large miss.

── WHAT MAKES THE COMPARISON FAIR ───────────────────────────────────────────────────
* Same X, same Y, same train/test episode split for all three regressors.
* Ridge and KNN reproduce the shipped numbers (they are the same estimators with the
  same hyperparameters), so they act as controls: if they do not land near the values in
  the notebook, the features were dumped wrong and the MLP number means nothing.
* The MLP standardises its inputs; Ridge and KNN deliberately do NOT, because the shipped
  probes do not, and changing that would make them stop being controls.  Standardisation
  is part of the MLP arm, not a correction applied to everyone.
* TEST rows are held-out EPISODES, so a high test score cannot come from frame-level
  leakage between neighbouring frames of one episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "lehami" / "experiments" / "results"


def _theta(sc: np.ndarray) -> np.ndarray:
    """[.., sin, cos, ..] -> angle. Column order matches y_cols from the dump."""
    return np.arctan2(sc[:, 0], sc[:, 1])


def _wrapped_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.angle(np.exp(1j * (a - b)))).mean())


def _score(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    th_t, th_p = _theta(Y_true), _theta(Y_pred)
    return {
        "r2_sincos": r2_score(Y_true[:, :2], Y_pred[:, :2]),
        "mae_theta_rad": _wrapped_mae(th_p, th_t),
        "mae_theta_deg": np.degrees(_wrapped_mae(th_p, th_t)),
        "r2_omega": r2_score(Y_true[:, 2], Y_pred[:, 2]),
        "mae_omega": float(np.abs(Y_pred[:, 2] - Y_true[:, 2]).mean()),
    }


def _regressors(seed: int, hidden: tuple[int, ...], max_iter: int) -> dict:
    return {
        # Controls — identical to the shipped probes, unstandardised on purpose.
        "ridge": Ridge(alpha=1.0, fit_intercept=True),
        "knn25": KNeighborsRegressor(n_neighbors=25, weights="distance"),
        # The capacity arm. early_stopping holds out 10% of TRAIN internally, so the
        # reported test score stays untouched by model selection.
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         max_iter=max_iter, early_stopping=True, n_iter_no_change=15,
                         random_state=seed),
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", required=True,
                    help="comma-separated results dir names (each needs probe_features.npz)")
    ap.add_argument("--k", default="0,1", help="which feature sets to score (0 and/or 1)")
    ap.add_argument("--hidden", default="256,256",
                    help="MLP hidden layer sizes, comma-separated")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0,
                    help="subsample TRAIN rows (0 = all); KNN predict is the slow step")
    ap.add_argument("--out", default="capacity_probes.csv",
                    help="written under results/ ; '' to skip")
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    ks = [int(x) for x in args.k.split(",")]
    rng = np.random.default_rng(args.seed)
    rows = []

    for exp in args.exps.split(","):
        npz_path = RESULTS / exp / "probe_features.npz"
        if not npz_path.exists():
            print(f"[skip] {exp}: no probe_features.npz — run dump_probe_fit_scatter.py first")
            continue
        d = np.load(npz_path, allow_pickle=False)
        y_cols = [str(c) for c in d["y_cols"]]
        assert y_cols[:3] == ["sin_theta", "cos_theta", "omega"], y_cols

        for k in ks:
            Xtr, Ytr = d[f"X_train_k{k}"], d[f"Y_train_k{k}"]
            Xte, Yte = d[f"X_test_k{k}"], d[f"Y_test_k{k}"]
            if args.max_train and len(Xtr) > args.max_train:
                idx = rng.choice(len(Xtr), args.max_train, replace=False)
                Xtr, Ytr = Xtr[idx], Ytr[idx]

            print(f"\n=== {exp}  k={k}   train {Xtr.shape} -> test {Xte.shape}"
                  f"   (input dim {Xtr.shape[1]}, latent_dim {int(d['latent_dim'])},"
                  f" uses_p={bool(d['uses_p'])})")
            for name, reg in _regressors(args.seed, hidden, args.max_iter).items():
                reg.fit(Xtr, Ytr[:, :3])
                for split, X, Y in (("train", Xtr, Ytr), ("test", Xte, Yte)):
                    m = _score(Y[:, :3], reg.predict(X))
                    rows.append({"exp": exp, "k": k, "probe": name, "split": split, **m})
                    if split == "test":
                        tr = rows[-2]
                        print(f"  {name:6s}  theta MAE {tr['mae_theta_deg']:6.2f}° (train) "
                              f"-> {m['mae_theta_deg']:6.2f}° (test)   "
                              f"r2_sincos {m['r2_sincos']:+.4f}   "
                              f"omega r2 {m['r2_omega']:+.4f}  MAE {m['mae_omega']:.4f}")

    if not rows:
        raise SystemExit("nothing scored — no probe_features.npz found")

    df = pd.DataFrame(rows)
    if args.out:
        out = RESULTS / args.out
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")
    print("\nTEST-split summary (theta MAE in degrees):")
    piv = df[df.split == "test"].pivot_table(index=["exp", "k"], columns="probe",
                                             values="mae_theta_deg")
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
