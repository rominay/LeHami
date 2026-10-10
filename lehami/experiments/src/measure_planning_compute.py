"""Planning compute of the K=30 swing-up replanning ladder: FLOPs and isolated wall time.

WHY THIS EXISTS
    Fig 8'''' puts a0q (EXP-079 LeHami), b0s (EXP-072 LeWM W=1) and b3s (EXP-076 LeWM
    W=3) on one CEM budget, 300 x 30 -- but equal budget is not equal compute: each
    candidate is scored by a different model. The runs DID log time (`wall_s`, `plan_s`,
    `cell_wall_s`), but those numbers track how the cluster packed the jobs, not the
    models. The same b0s single-shot cell took 308 s in `plan_replan_swingup` and 925 s
    in `plan_swingup_h30` (same budget, same L40S), and b3s/b0s is 2.5x in one sweep and
    1.03x in another. So this script measures compute directly, two ways:

      gflops   FlopCounterMode around the planner's OWN `run_episode` -- every encoder
               call, every CEM iteration, every predictor step, including the
               autograd.grad calls inside LeHami's leapfrog. Hardware- and
               co-tenancy-independent, and deterministic for a given rung.
      time     the same `run_episode`, one process, one GPU, nothing else running,
               arms interleaved per episode so drift hits all three alike. Median
               over episodes after a warm-up.

    FlopCounterMode counts matmul/conv/attention FLOPs only (2*M*N*K per matmul);
    elementwise ops are not counted, which is the standard convention.

CHECKPOINTS
    Found under $STABLEWM_HOME/checkpoints (default ~/.stable_worldmodel) by the same
    `*_<cfg>_s0` glob actuated_planning.sh uses. The b3s checkpoint is LOADED when
    present, and otherwise BUILT at random init from b0s's resolved config with the one
    architectural change b3s makes, `enc_window: 3`. FLOPs and time do not depend on the weights
    (no data-dependent control flow in either predictor, and under `--replan` the number
    of solves is fixed by the ladder). Two guards make the fallback checkable:
      1. the two experiment YAMLs are diffed and may differ ONLY in
         {run_suffix, num_preds, model.enc_window} (num_preds is a training clip length);
      2. b0s built the same way at random init must reproduce the b0s checkpoint's
         state_dict key-for-key and shape-for-shape.
    The `weights` column of every CSV says which one a row came from.

CUDA ONLY
    On CPU, SDPA dispatches to `_scaled_dot_product_flash_attention_for_cpu`, which
    FlopCounterMode has no formula for, so LeWM's attention would be counted as 0 (-1.0%).
    `count_flops` refuses any LeWM count with no attention FLOPs in it.

USAGE
    python lehami/experiments/src/measure_planning_compute.py            # 300x30
    python lehami/experiments/src/measure_planning_compute.py --smoke    # 1 iter
    python .../measure_planning_compute.py --arms lehami --cem-iters 48        # one budget
    python .../measure_planning_compute.py --arms lehami --cells cell3 \
        --receding 6 --cell-label cell6_goal_H30_R6                           # a new rung
    python .../measure_planning_compute.py --build-grid                       # merge
    bash lehami/experiments/actuated_planning.sh                        # all, with the ladder

Writes to outputs/planning_compute/ (or --out-dir): one CSV (+ meta.json) per
(arms, budget), and `--build-grid` merges the four of the compute-matched grid into
swingup_K30_compute_grid.csv, the table Fig 8^(6) reads.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

import plan_pendulum_swingup as P
from run_eval import _enc_shift, _enc_window, load_model
from lehami.builders import REGISTRY

ROOT = P.ROOT
SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
CKPT = SWM / "checkpoints"
RUN = ROOT / "outputs" / "plan_replan_swingup" / "test"
OUT = ROOT / "outputs" / "planning_compute"
EXPCFG = ROOT / "lehami" / "training" / "configs" / "experiment"

CELLS = ["cell1_single_shot_H30_R30", "cell2_goal_H30_R15", "cell3_goal_H30_R9"]

#: key -> (tag, experiment, run-name suffix). Keys match the notebook's DA.key; the
#: suffix is the one actuated_planning.sh uses, so both find the same `*_<cfg>_s0` dir.
ARMS = {
    "lehami":   ("a0q", "EXP-079", "lehami_sg_sep_conv_w1_swingup_exp079_a0q"),
    "lewm":    ("b0s", "EXP-072", "lewm_ms_conv_w1_swingup_exp072_b0s"),
    "lewm_w3": ("b3s", "EXP-076", "lewm_ms_conv_w3_swingup_exp076_b3s"),
}

#: The compute-matched grid, one CSV per (arms, cem_iters) -- None is the cells' own
#: 300x30. LEVEL pairs the iso-FLOP points: `low` ~ a0q@300x30, `high` ~ b0s@300x30.
GRID = [("lehami,lewm,lewm_w3", None), ("lehami", 48), ("lewm", 19), ("lewm_w3", 18)]
LEVEL = {("lehami", 30): "low", ("lehami", 48): "high",
         ("lewm", 19): "low", ("lewm", 30): "high",
         ("lewm_w3", 18): "low", ("lewm_w3", 30): "high"}


def csv_name(arms, cem_iters, cem_samples=300, receding=None):
    """The file one (arms, budget[, rung]) measurement writes; the 300x30 table keeps its name."""
    name = "swingup_K30_ladder_compute"
    if cem_iters is not None or cem_samples != 300:
        # the population override reaches here with cem_iters unset: name it by the
        # budget that will actually run, so a 600x30 file never lands on a 300x30 one
        name += f"_{'-'.join(arms)}_cem{cem_samples}x{cem_iters if cem_iters is not None else 30}"
    if receding is not None:
        name += (f"_{'-'.join(arms)}" if cem_iters is None else "") + f"_R{receding}"
    return name


def build_grid(out):
    """Merge the four GRID measurements into the 18-row table Fig 8^(6) reads."""
    g = pd.concat([pd.read_csv(out / f"{csv_name(a.split(','), it)}.csv")
                   .assign(source=f"{csv_name(a.split(','), it)}.csv") for a, it in GRID],
                  ignore_index=True)
    g["level"] = [LEVEL[(k, it)] for k, it in zip(g.key, g.cem_iters)]
    assert len(g) == 18 and g.groupby("key").size().eq(6).all(), "grid is not 3 x 3 x 2"
    assert not g.duplicated(["key", "cell", "cem_iters"]).any()
    assert g.groupby(["cell", "level"]).size().eq(3).all(), "not one point per arm per level"
    g.to_csv(out / "swingup_K30_compute_grid.csv", index=False)
    (out / "swingup_K30_compute_grid.meta.json").write_text(json.dumps(dict(
        built_from=sorted(set(g.source)), rows=len(g), devices=sorted(set(g.device)),
        note="x-values for Fig 8^(6): 3 arms x 3 rungs x 2 budgets; 'level' pairs the "
             "iso-FLOP points (low ~ a0q@300x30, high ~ b0s@300x30)"), indent=1))
    spread = g.groupby(["cell", "level"]).gflops_total.agg(lambda s: s.max() / s.min() - 1)
    print("iso-FLOP spread within each (rung, level), %:")
    print((100 * spread).round(2).to_string())
    print(g.pivot_table(index="cell", columns=["key", "cem_iters"],
                        values="gflops_total").div(1e3).round(2).to_string())
    print(f"wrote {out / 'swingup_K30_compute_grid.csv'}")


def _ckpt(cfg_name):
    """The s0 checkpoint for a run name, or None if it is not on this machine."""
    dirs = sorted(CKPT.glob(f"*_{cfg_name}_s0"))
    assert len(dirs) <= 1, f"ambiguous checkpoint dirs for {cfg_name}: {dirs}"
    return dirs[0] / "best.ckpt" if dirs and (dirs[0] / "best.ckpt").exists() else None


def _flat(cfg):
    return {k: v for k, v in _walk(OmegaConf.to_container(cfg, resolve=True))}


def _walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk(v, f"{pre}{k}.")
        else:
            yield f"{pre}{k}", v


def build_b3s(cfg_b0s, sd_b0s, device):
    """b3s at random init, from b0s's resolved config -- see the module docstring."""
    y1 = _flat(OmegaConf.load(EXPCFG / "phase0_lewm_ms_conv_w1_swingup_exp072_b0s.yaml"))
    y3 = _flat(OmegaConf.load(EXPCFG / "phase0_lewm_ms_conv_w3_swingup_exp076_b3s.yaml"))
    diff = {k for k in set(y1) | set(y3) if y1.get(k) != y3.get(k)}
    assert diff == {"run_suffix", "num_preds", "model.enc_window"}, \
        f"EXP-072 and EXP-076 differ in more than the encoder window: {sorted(diff)}"

    # guard 2: the builder path reproduces a checkpoint we DO have
    torch.manual_seed(0)
    probe, _, _ = REGISTRY[cfg_b0s.model.name](cfg_b0s)
    got = {k: tuple(v.shape) for k, v in probe.state_dict().items()}
    want = {k: tuple(v.shape) for k, v in sd_b0s.items()}
    assert got == want, "config -> model does not reproduce the b0s checkpoint"

    cfg = copy.deepcopy(cfg_b0s)
    cfg.model.enc_window = int(y3["model.enc_window"])
    cfg.num_preds = int(y3["num_preds"])
    cfg.run_suffix = "_lewm_ms_conv_w3_swingup_exp076_b3s_RANDOM_INIT"
    torch.manual_seed(0)
    model, _, _ = REGISTRY[cfg.model.name](cfg)
    return model.eval().to(device), cfg


def load_arms(device):
    out = {}
    for key, (tag, exp, cfg_name) in ARMS.items():
        path = _ckpt(cfg_name)
        if path is None:
            assert key == "lewm_w3", f"{key}: no *_{cfg_name}_s0/best.ckpt under {CKPT}"
            continue
        model, cfg = load_model(path, str(device))
        out[key] = (model.eval(), cfg, "checkpoint s0")
    if "lewm_w3" not in out:
        sd_b0s = out["lewm"][0].state_dict()
        model, cfg = build_b3s(out["lewm"][1], sd_b0s, device)
        out["lewm_w3"] = (model, cfg, "random init from EXP-076 config")
    m3 = out["lewm_w3"][0]
    assert _enc_window(m3) == 3 and _enc_window(m3) - 1 - _enc_shift(m3) == 1, \
        "b3s should be a symmetric W=3 encoder with lag 1"
    for key, (_, _, w) in out.items():
        print(f"[arms] {key:8s} {w}")
    return out


def cell_args(cell, device, smoke, cem_iters=None, receding=None, cem_samples=None):
    """The run's own args, read from its meta.json, with local paths swapped in.

    `cem_iters` overrides the budget for a compute-matched re-run whose outputs are
    not back yet: FLOPs depend on the protocol and the budget, not on the results, so
    the 300x30 cell's args with the iteration count swapped ARE that run's args.
    `receding` does the same for a new rung of the replanning ladder (e.g. cell 6):
    it is taken from a REPLANNING cell, so only the stride changes.
    """
    metas = {k: json.loads((RUN / cell / f"{t}_s0" / "meta.json").read_text())["args"]
             for k, (t, _, _) in ARMS.items()}
    keep = ["horizon", "action_block", "receding", "replan", "budget", "u_max",
            "dt_mult", "cost", "goal_mode", "goal_offset", "n_eval", "split",
            "success_at", "cem_samples", "cem_iters", "cem_elites", "ic_seed"]
    for k in keep:        # the three arms ran one protocol, or the plot is meaningless
        assert len({json.dumps(m[k]) for m in metas.values()}) == 1, f"{cell}: {k} differs"
    a = dict(metas["lehami"])
    a.update(dataset=str(SWM / "pendulum_swingup.h5"),
             manifest=str(ROOT / "lehami" / "data" / Path(a["manifest"]).name),
             device=str(device), p_weight=None)
    if cem_iters is not None:
        a["cem_iters"] = cem_iters
    if cem_samples is not None:
        # the POPULATION axis (2026-09-12). FLOPs are linear in it -- every iteration
        # scores every candidate -- but the encode term is not, so count rather than
        # scale. Elites follow the 10% fraction the planning run used.
        a["cem_samples"] = cem_samples
        a["cem_elites"] = max(1, round(cem_samples * a["cem_elites"] / 300))
    if receding is not None:
        assert a["replan"], f"{cell} is single-shot; take a new rung from a --replan cell"
        # A shrinking horizon that is not a whole number of blocks is fine since the
        # 2026-09-12 truncation in `cem_plan` (it scores steps[:, :horizon]); without
        # that, the terminal cost would land 1-2 steps past the goal. Checked here so
        # a stale checkout cannot quietly mis-count.
        hs = list(range(a["goal_offset"], 0, -receding))
        if not all(h % a["action_block"] == 0 for h in hs):
            src = (ROOT / "lehami" / "experiments" / "src"
                   / "plan_pendulum_swingup.py").read_text()
            assert "[:, :horizon]" in src, \
                (f"receding {receding} gives horizons {hs}, not all multiples of "
                 f"{a['action_block']}, and this checkout's cem_plan does not truncate")
        a["receding"] = receding
    if smoke:
        a["cem_iters"] = 1
    return SimpleNamespace(**a)


def count_flops(model, cfg, ep, args, device):
    """Total FLOPs of one episode, split by CALL SITE rather than by module.

    Module attribution is unreliable here twice over: LeHami's leapfrog runs
    autograd.grad, whose backward matmuls the engine executes outside any module's
    forward (a0q: 91 GF filed under HamiltonianPredictor vs ~265 GF actually spent),
    and b3s's TemporalConvEncoder is driven through its children, so it appears as a
    bare `Sequential`. Instead the planner's two entry points are wrapped and the
    running total is read before and after each call:
      encode  every `encode_frames` call -- context + goal, incl. the projector/anchor
      cem     every `cem_plan` call -- rollouts, action embedding, cost
    """
    calls, site = [0], {"encode": 0, "cem": 0}
    h = model.predictor.register_forward_hook(
        lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    orig = {"encode": P.encode_frames, "cem": P.cem_plan}

    def _wrap(name):
        def f(*a, **k):
            t0 = fc.get_total_flops()
            out = orig[name](*a, **k)
            site[name] += fc.get_total_flops() - t0
            return out
        return f

    P.encode_frames, P.cem_plan = _wrap("encode"), _wrap("cem")
    try:
        with FlopCounterMode(display=False) as fc:
            r = P.run_episode(model, cfg, ep, 0, args, device)
    finally:
        h.remove()
        P.encode_frames, P.cem_plan = orig["encode"], orig["cem"]
    tot = fc.get_total_flops()
    part = dict(site, other=tot - sum(site.values()))
    # LeWM's predictor is a transformer, so a count with no attention in it means the
    # attention kernel had no FlopCounterMode formula (CPU SDPA) -- refuse, don't undercount.
    attn = sum(v for op, v in fc.get_flop_counts()["Global"].items() if "attention" in str(op))
    if P._mom_type(cfg.model.name) == "lewm":
        assert attn > 0, (f"{cfg.model.name}: no attention FLOPs counted on {device} -- the "
                          f"SDPA kernel in use has no FlopCounterMode formula")
    part["attention"] = attn        # a SUBSET of cem, not an extra bucket
    return tot, part, calls[0], r


def time_episode(model, cfg, ep, args, device):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    r = P.run_episode(model, cfg, ep, 0, args, device)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0, r["plan_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-time", type=int, default=5, help="timed episodes per arm x rung")
    ap.add_argument("--n-flop", type=int, default=2,
                    help="episodes FLOP-counted per arm x rung (must agree exactly)")
    ap.add_argument("--smoke", action="store_true", help="1 CEM iteration, 1 timed ep")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"comma list of {list(ARMS)}")
    ap.add_argument("--cells", default=",".join(c.split("_")[0] for c in CELLS),
                    help="comma list of cell1,cell2,cell3")
    ap.add_argument("--cem-iters", type=int, default=None,
                    help="override the cells' CEM iterations (compute-matched runs)")
    ap.add_argument("--cem-samples", type=int, default=None,
                    help="override the cells' CEM population (the search-axis runs)")
    ap.add_argument("--receding", type=int, default=None,
                    help="a NEW rung: one replanning --cells entry with its stride overridden")
    ap.add_argument("--cell-label", default=None,
                    help="the new rung's cell name (e.g. cell6_goal_H30_R6)")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="where the CSVs go (the torch job writes to a subfolder)")
    ap.add_argument("--build-grid", action="store_true",
                    help="only merge the four GRID CSVs in --out-dir, measure nothing")
    args_cli = ap.parse_args()
    out = Path(args_cli.out_dir)
    if args_cli.build_grid:
        return build_grid(out)
    if args_cli.smoke:
        args_cli.n_time, args_cli.n_flop = 1, 1
    want = args_cli.arms.split(",")
    assert set(want) <= set(ARMS), f"unknown arm in {want}"
    cells = [c for c in CELLS if c.split("_")[0] in args_cli.cells.split(",")]
    if args_cli.receding is not None:
        assert len(cells) == 1 and args_cli.cell_label, \
            "--receding takes exactly one base cell (--cells) and a --cell-label"

    # CUDA on purpose, not a convenience. On CPU, SDPA dispatches to
    # `_scaled_dot_product_flash_attention_for_cpu`, which FlopCounterMode has no
    # formula for, so LeWM's attention is silently counted as 0 (-1.0%; measured
    # 193.459 vs 195.450 GF at R=30, CEM 300x1). LeHami has no attention and matches
    # to the FLOP. The CUDA kernels (efficient / flash / cuDNN) are all counted, and
    # `count_flops` refuses a LeWM count with no attention in it.
    device = torch.device("cuda")
    arms = {k: v for k, v in load_arms(device).items() if k in want}
    rows = []
    for base in cells:
        args = cell_args(base, device, args_cli.smoke, args_cli.cem_iters,
                         args_cli.receding, args_cli.cem_samples)
        cell = args_cli.cell_label if args_cli.receding is not None else base
        eps = P.dataset_goals(args)
        for key, (model, cfg, weights) in arms.items():
            P._check_plannable(cfg, args, key)
        print(f"\n=== {cell}: receding {args.receding}, CEM {args.cem_samples}x"
              f"{args.cem_iters}, {len(eps)} goals")

        flops = {}
        for key, (model, cfg, _) in arms.items():
            res = [count_flops(model, cfg, eps[i], args, device)
                   for i in range(args_cli.n_flop)]
            assert len({r[0] for r in res}) == 1, \
                f"{key}: FLOPs differ across episodes {[r[0] for r in res]}"
            flops[key] = res[0]
            tot, part, calls, r = res[0]
            print(f"  {key:8s} {tot / 1e12:7.3f} TFLOP  predictor calls {calls}  "
                  f"solves {r['n_solves']}  "
                  + "  ".join(f"{k} {v / 1e9:.1f}G" for k, v in part.items()))

        # timing: warm-up once per arm, then interleave arms per episode
        for key, (model, cfg, _) in arms.items():
            time_episode(model, cfg, eps[-1], args, device)
        times = {k: [] for k in arms}
        for i in range(args_cli.n_time):
            for key, (model, cfg, _) in arms.items():
                times[key].append(time_episode(model, cfg, eps[i], args, device))

        for key, (model, cfg, weights) in arms.items():
            tag, exp, _ = ARMS[key]
            tot, part, calls, r = flops[key]
            wall = np.array([t[0] for t in times[key]])
            plan = np.array([t[1] for t in times[key]])
            rows.append(dict(
                key=key, tag=tag, exp=exp, weights=weights, cell=cell,
                receding=args.receding, n_solves=r["n_solves"],
                cem_samples=args.cem_samples, cem_iters=args.cem_iters,
                predictor_calls=calls,
                rollout_steps_per_iter=calls // args.cem_iters,
                gflops_total=tot / 1e9,
                **{f"gflops_{k}": v / 1e9 for k, v in part.items()},
                # the cost of scoring ONE candidate for ONE step: predictor + action
                # embedding + (LeWM) pred_proj + the cost's share
                mflops_per_step_per_candidate=part["cem"] / 1e6
                / (calls * args.cem_samples),
                episode_s_median=float(np.median(wall)),
                plan_s_median=float(np.median(plan)),
                plan_s_min=float(plan.min()), plan_s_max=float(plan.max()),
                n_timed=len(wall), device=torch.cuda.get_device_name(device)))
            print(f"  {key:8s} episode {np.median(wall):6.2f} s  (plan {np.median(plan):6.2f} s)")

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    # a non-default budget gets its own file: never overwrite the 300x30 table
    name = csv_name(want, args_cli.cem_iters, args.cem_samples, args_cli.receding)
    name += "_SMOKE" if args_cli.smoke else ""
    df.to_csv(out / f"{name}.csv", index=False)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (out / f"{name}.meta.json").write_text(json.dumps(dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha=sha,
        torch=torch.__version__, device=torch.cuda.get_device_name(device),
        stablewm_home=str(SWM),
        weights={k: w for k, (_, _, w) in arms.items()},
        n_time=args_cli.n_time, n_flop=args_cli.n_flop, smoke=args_cli.smoke,
        source_runs=str(RUN.relative_to(ROOT))), indent=1))
    print(f"\nwrote {out / name}.csv")
    print(df[["key", "cell", "n_solves", "predictor_calls", "gflops_total",
              "episode_s_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
