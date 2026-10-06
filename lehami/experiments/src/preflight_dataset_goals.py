"""Can the instrument resolve a difference? Ask before spending GPU-hours.

A planning benchmark is only informative between two bounds:

    FLOOR    what a null policy already achieves. Anything at or below it is not
             evidence about a world model.
    CEILING  what a planner with EXACT dynamics achieves. Nothing learned can beat
             it, so if the ceiling is 1.00 the comparison has no room in it.

EXP-085 failed at the floor: an oracle with perfect dynamics scored 0/80, so both
arms scoring ~0 said nothing about either. The `--goal-mode dataset` protocol fails
at the OTHER end under LeWorldModel's "success at any step" criterion -- 300 random
torque sequences (one CEM iteration against a 300x30 search) solve essentially every
goal, at every tolerance tried. This script measures both ends so the design point
is fixed on the plant, before any checkpoint is loaded.

It is the pendulum analogue of the PushT SNR pre-flight, and the rule is the same:
if floor and ceiling are not well separated, do not run the experiment.

Usage
-----
    python lehami/experiments/src/preflight_dataset_goals.py
    python .../preflight_dataset_goals.py --n-eval 135 --cem-iters 30 --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lehami" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_pendulum_swingup as P  # noqa: E402
from pendulum_swingup import SUBSTEP, G  # noqa: E402

wrap = P.wrap


def rollout_batch(th0, om0, L, m, acts, dt):
    """Vectorised true plant: `acts` (N, T) -> (theta, omega), each (N, T).

    Same RK4 and substep as `plant_step`, batched over candidates so an oracle CEM
    is affordable. `tests/test_preflight_dataset_goals.py::
    test_rollout_batch_matches_plant_step` pins the equivalence. (Until 2026-09-01
    this line named a `test_preflight_matches_plant_step` that had never been
    written -- the equivalence was asserted in a comment and nowhere else.)
    """
    n_sub = int(round(dt / SUBSTEP))
    th = np.full(acts.shape[0], float(th0))
    om = np.full(acts.shape[0], float(om0))
    TH, OM = [], []
    for t in range(acts.shape[1]):
        u = acts[:, t]
        for _ in range(n_sub):
            def d(a, b):
                return b, -(G / L) * np.sin(a) + u / (m * L * L)
            k1 = d(th, om)
            k2 = d(th + SUBSTEP / 2 * k1[0], om + SUBSTEP / 2 * k1[1])
            k3 = d(th + SUBSTEP / 2 * k2[0], om + SUBSTEP / 2 * k2[1])
            k4 = d(th + SUBSTEP * k3[0], om + SUBSTEP * k3[1])
            th = th + SUBSTEP / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            om = om + SUBSTEP / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        TH.append(th.copy())
        OM.append(om.copy())
    return np.array(TH).T, np.array(OM).T


def _hit(TH, OM, g, tol, at):
    """(N,) bool: did each candidate satisfy the criterion?

    `at` is None for 'any step' or the 1-indexed step for 'target'.
    """
    tt, ot = tol
    ok = (np.abs(wrap(TH - g["theta"])) < tt) & (np.abs(OM - g["omega"]) < ot)
    return ok.any(1) if at is None else ok[:, at - 1]


def _cost(TH, OM, g, tol, at):
    """What the oracle MINIMISES. Normalised by the tolerances so the two terms are
    commensurable -- the same normalisation `demo_swingup_planning` uses."""
    tt, ot = tol
    d = (wrap(TH - g["theta"]) / tt) ** 2 + ((OM - g["omega"]) / ot) ** 2
    return d.min(1) if at is None else d[:, at - 1]


def oracle_cem(e, args, tol, at, rng):
    """CEM on the TRUE plant. The ceiling: no learned model can do better.

    CEILING division on the block count, then truncate to `budget` -- exactly what
    `plan_pendulum_swingup.cem_plan` does. FLOOR division was wrong in two ways at
    once whenever `action_block` did not divide `budget`, and only one of them was
    noisy: `target` raised `IndexError: index 24 is out of bounds for axis 1 with
    size 24` (budget 25, block 3), while `any` -- which reduces with `.any(1)` and
    never indexes -- silently rolled out only `floor(B/blk)*blk` steps and so
    UNDERSTATED the ceiling this script exists to measure. A ceiling that is too low
    is the one error that cannot be caught by reading the output, because it makes
    the gap look wider and the experiment look more informative than it is.
    """
    dt = P.BASE_DT * args.dt_mult
    B, blk = args.budget, args.action_block
    n_blk = -(-B // blk)
    mean = np.zeros(n_blk)
    std = np.full(n_blk, args.u_max / 2)
    for _ in range(args.cem_iters):
        cand = rng.normal(mean, std, size=(args.cem_samples, n_blk))
        cand[0] = mean
        cand = np.clip(cand, -args.u_max, args.u_max)
        steps = np.repeat(cand, blk, axis=1)[:, :B]
        TH, OM = rollout_batch(e["theta0"], e["omega0"], e["L"], e["m"], steps, dt)
        el = cand[np.argsort(_cost(TH, OM, e["goal"], tol, at))[:args.cem_elites]]
        mean, std = el.mean(0), el.std(0) + 1e-6
    steps = np.repeat(mean, blk)[None, :B]
    TH, OM = rollout_batch(e["theta0"], e["omega0"], e["L"], e["m"], steps, dt)
    return int(_hit(TH, OM, e["goal"], tol, at)[0])


def measure(eps, args, tol, at, seed=0):
    """Floors and ceiling for one criterion. Returns a dict of rates."""
    dt = P.BASE_DT * args.dt_mult
    rng = np.random.default_rng(seed)
    out = {}

    def openloop(fn, n_try):
        hits = []
        for e in eps:
            ok = 0
            for _ in range(n_try):
                a = np.asarray(fn(), float)[None, :]
                TH, OM = rollout_batch(e["theta0"], e["omega0"], e["L"], e["m"], a, dt)
                ok = max(ok, int(_hit(TH, OM, e["goal"], tol, at)[0]))
                if ok:
                    break
            hits.append(ok)
        return float(np.mean(hits))

    B = args.budget
    out["zero"] = openloop(lambda: np.zeros(B), 1)
    out["rand1"] = openloop(lambda: rng.uniform(-args.u_max, args.u_max, B), 1)
    out["rand300"] = openloop(lambda: rng.uniform(-args.u_max, args.u_max, B), 300)
    out["oracle"] = float(np.mean([oracle_cem(e, args, tol, at, rng) for e in eps]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--split", default="test")
    ap.add_argument("--goal-offset", type=int, default=25)
    # Must MIRROR whatever the arms were run with, or the floor and the ceiling are
    # measured on a different goal set than the arms score -- the same mismatch that
    # made a 135-goal pre-flight share exactly ONE episode with a 50-goal arm run.
    ap.add_argument("--draw-offset", type=int, default=None,
                    help="offset used to DRAW the goals, decoupled from "
                         "--goal-offset (the reach distance). Default None = same "
                         "as --goal-offset. Pass the SAME pair the arms used.")
    ap.add_argument("--u-max", type=float, default=4.0)
    ap.add_argument("--dt-mult", type=int, default=2)
    ap.add_argument("--ic-seed", type=int, default=1234)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--cem-samples", type=int, default=300)
    ap.add_argument("--cem-iters", type=int, default=30)
    ap.add_argument("--cem-elites", type=int, default=30)
    ap.add_argument("--sweep", action="store_true",
                    help="also report the criteria that were rejected")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--manifest",
                    default=str(ROOT / "lehami" / "data" /
                                "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"))
    args = ap.parse_args()
    if args.dataset is None:
        import os
        args.dataset = str(Path(os.environ.get(
            "STABLEWM_HOME", Path.home() / ".stable_worldmodel")) / "pendulum_swingup.h5")

    args.budget = args.goal_offset          # scored at the deadline; later steps are moot
    args.success_at, args.theta_tol, args.omega_tol = "target", 0.1, 0.5
    eps = P.dataset_goals(args)
    _dr = getattr(args, "draw_offset", None) or args.goal_offset
    if _dr != args.goal_offset:
        print(f"[draw] goals drawn at offset {_dr}, start {args.goal_offset} steps "
              f"before the goal -- paired with the offset-{_dr} arms.")
    print(f"{len(eps)} goals from {args.split}, offset {args.goal_offset}, "
          f"u_max {args.u_max}, budget {args.budget}")
    print(f"oracle CEM: {args.cem_samples} x {args.cem_iters}, top-{args.cem_elites}\n")

    crit = [("target", (0.1, 0.5))]
    if args.sweep:
        crit = [("any", (0.2, 1.0)), ("any", (0.05, 0.2)),
                ("target", (0.2, 1.0)), ("target", (0.1, 0.5)), ("target", (0.05, 0.2))]

    print(f"{'when':>7} {'tol':>13} {'u=0':>7} {'rand1':>7} {'rand300':>8} "
          f"{'ORACLE':>7} {'gap':>6}   verdict")
    print("-" * 80)
    for when, tol in crit:
        at = None if when == "any" else args.goal_offset
        # 'any' must be given the full LeWM budget or it is not the same criterion.
        args.budget = args.goal_offset if when == "target" else 2 * args.goal_offset
        r = measure(eps, args, tol, at)
        # The ranking quantity is ORACLE - RAND300, not oracle - floor. rand300 is
        # what a search with perfect dynamics gets from 300 blind samples; if the
        # oracle cannot beat it, the SEARCH is doing the work and no world model can
        # be shown to matter. `any (0.05, 0.2)` has a healthy-looking floor of 0.067
        # and a gap of exactly 0.000 -- it is the case this column exists to catch.
        gap = r["oracle"] - r["rand300"]
        verdict = ("SEARCH-BOUND" if gap < 0.05 else
                   "weak" if gap < 0.2 else "USABLE")
        print(f"{when:>7} {str(tol):>13} {r['zero']:>7.3f} {r['rand1']:>7.3f} "
              f"{r['rand300']:>8.3f} {r['oracle']:>7.3f} {gap:>6.3f}   {verdict}")

    print("\ngap = ORACLE - rand300: the room a world model has to earn. A high")
    print("floor is a nuisance (report lift over it); a small gap is fatal, because")
    print("it means blind search already does what perfect dynamics can do.")


if __name__ == "__main__":
    main()
