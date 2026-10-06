"""Training cost per step of the pendulum_multi_v2 arms (EXP-096): FLOPs and isolated time.

WHY THIS EXISTS
    EXP-096 asks how much cheaper HJEPA is to train than LeWM. The W&B runtimes of the
    100-epoch runs (87-102 min on an L40S) cannot answer it: the four te_w3_multi arms
    trained CONCURRENTLY on one GPU, so their wall time measures the packing -- the same
    failure as the planners' logged `wall_s`. This script measures one TRAINING STEP of
    each arm directly, two ways:

      gflops   FlopCounterMode around forward + backward of one real batch, under the
               training precision (bf16-mixed autocast). Includes HJEPA's leapfrog
               `autograd.grad(create_graph=True)` and the backward through it -- the
               second-order cost that makes "the Hamiltonian is cheaper" non-obvious.
      time     forward + backward + AdamW step, one process, arms interleaved per step,
               batches pre-staged on the GPU so data loading is excluded. Median after
               warm-up.

    Steps per epoch come from the REAL training loader (the manifest split, batch 32,
    drop_last), and they differ by arm: clips per episode = train_frames - num_steps + 1,
    and num_steps is history_size + num_preds (14 for HJEPA W=1, 11 for LeWM W=1, 13 for
    LeWM W=3). So an "epoch" is not the same amount of work across arms; per-step and
    per-epoch numbers are both reported.

    Each arm is built from its checkpoint's own `config.yaml` -- the exact resolved config
    it trained with -- at fresh init: FLOPs and time do not depend on the weights.
    CUDA only, for the reason in measure_planning_compute.py (CPU SDPA is not counted).

USAGE
    python lehami/experiments/src/measure_training_compute.py            # all arms
    python .../measure_training_compute.py --arms a0m,b0m --n-time 20          # quick

Writes outputs/exp096/training_compute_<gpu>.csv (+ .meta.json).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

import run_eval  # noqa: F401  -- puts lehami/models (hjepa, leworldmodel) on sys.path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lehami" / "training"))
import train as T  # noqa: E402  -- the trainer's own dataset + manifest split
import stable_pretraining as spt  # noqa: E402
from hjepa.builders import REGISTRY  # noqa: E402

SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
CKPT = SWM / "checkpoints"
OUT = ROOT / "outputs" / "exp096"

#: arm -> (experiment, label, checkpoint dir). Only config.yaml is read from the dir.
ARMS = {
    "a0m": ("EXP-067", "HJEPA W=1",
            "hjepa_sg_sep_sreg_joint_ms_conv_pendulum_multi_v2_sg_sep_conv_w1_multi_exp067_a0m"),
    "b0m": ("EXP-068", "LeWM W=1",
            "lewm_ms_conv_pendulum_multi_v2_ms_conv_w1_multi_exp068_b0m"),
    "b1m": ("EXP-066", "LeWM W=3",
            "lewm_ms_conv_pendulum_multi_v2_ms_conv_te_w3_multi_exp066_b1m"),
    "a1m": ("EXP-065", "HJEPA W=3",
            "hjepa_sg_sep_sreg_joint_ms_conv_pendulum_multi_v2_sg_sep_conv_te_w3_multi_exp065_a1m"),
}


def _to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def build(arm, device, n_batches):
    """(module, optimizer, cfg, steps_per_epoch, staged batches) for one arm."""
    cfg = OmegaConf.load(CKPT / ARMS[arm][2] / "config.yaml")
    # the cluster path in the saved config -> the same manifest in this checkout
    cfg.manifest = str(ROOT / "lehami" / "data" / Path(cfg.manifest).name)
    ds = T._build_dataset(cfg)
    train_set, _ = T._split_from_manifest(ds, cfg.manifest, cfg)
    dl = torch.utils.data.DataLoader(
        train_set, batch_size=int(cfg.loader.batch_size), shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(0), num_workers=2)
    it = iter(dl)
    batches = [_to(next(it), device) for _ in range(n_batches)]

    torch.manual_seed(0)
    model, sigreg, fwd = REGISTRY[cfg.model.name](cfg)
    module = spt.Module(model=model, sigreg=sigreg, forward=fwd, optim=None).to(device)
    module.train()
    opt = torch.optim.AdamW(module.model.parameters(), lr=float(cfg.optim.optimizer.lr),
                            weight_decay=float(cfg.optim.optimizer.weight_decay))
    return module, opt, cfg, len(dl), batches


def make_step(module, opt, cfg):
    bf16 = "bf16" in str(OmegaConf.select(cfg, "trainer.precision", default=""))
    clip = OmegaConf.select(cfg, "trainer.gradient_clip_val", default=None)

    def fwd_bwd(batch):
        batch = dict(batch, batch_idx=0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            state = module(batch, stage="fit")
        state["loss"].backward()

    def step(batch):
        fwd_bwd(batch)
        if clip:
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), float(clip))
        opt.step()
        opt.zero_grad(set_to_none=True)

    return fwd_bwd, step, bf16


def count_flops(fwd_bwd, batch, opt, is_lewm):
    with FlopCounterMode(display=False) as fc:
        fwd_bwd(batch)
    opt.zero_grad(set_to_none=True)
    attn = sum(v for op, v in fc.get_flop_counts()["Global"].items() if "attention" in str(op))
    if is_lewm:
        assert attn > 0, "LeWM step with no attention FLOPs counted -- kernel not covered"
    return fc.get_total_flops(), attn


def main():
    ap = argparse.ArgumentParser()
    # a1m (HJEPA W=3) is left out of the default: it does not train (val loss bottoms at
    # epoch 2 and diverges), so its cost is not a comparison anyone should read.
    ap.add_argument("--arms", default="a0m,b0m,b1m", help=f"comma list of {list(ARMS)}")
    ap.add_argument("--n-warm", type=int, default=10)
    ap.add_argument("--n-time", type=int, default=50, help="timed steps per arm")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="where the CSV goes (the torch job writes to a subfolder)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    arms = args.arms.split(",")
    assert set(arms) <= set(ARMS), arms
    device = torch.device("cuda")

    built = {}
    for a in arms:
        module, opt, cfg, spe, batches = build(a, device, n_batches=4)
        fwd_bwd, step, bf16 = make_step(module, opt, cfg)
        is_lewm = cfg.model.name.startswith("lewm")
        f = [count_flops(fwd_bwd, batches[i], opt, is_lewm) for i in range(2)]
        assert f[0][0] == f[1][0], f"{a}: FLOPs differ across batches {f}"
        built[a] = dict(module=module, cfg=cfg, spe=spe, batches=batches, step=step,
                        bf16=bf16, gflops=f[0][0] / 1e9, gflops_attn=f[0][1] / 1e9)
        print(f"[{a}] {cfg.model.name}  steps/epoch {spe}  {f[0][0] / 1e9:.1f} GFLOP/step "
              f"(fwd+bwd, bf16={bf16})")

    # timing: warm up each arm, then interleave arms step by step so drift hits all alike
    for a in arms:
        for i in range(args.n_warm):
            built[a]["step"](built[a]["batches"][i % 4])
    torch.cuda.synchronize()
    times = {a: [] for a in arms}
    for i in range(args.n_time):
        for a in arms:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            built[a]["step"](built[a]["batches"][i % 4])
            torch.cuda.synchronize()
            times[a].append(time.perf_counter() - t0)

    rows = []
    for a in arms:
        b, cfg = built[a], built[a]["cfg"]
        m = b["module"].model
        parts = {n: sum(p.numel() for p in c.parameters()) for n, c in m.named_children()}
        t = np.array(times[a])
        s_step = float(np.median(t))
        rows.append(dict(
            arm=a, exp=ARMS[a][0], label=ARMS[a][1], model_name=cfg.model.name,
            enc_window=int(OmegaConf.select(cfg, "model.enc_window", default=1) or 1),
            history_size=int(cfg.history_size), num_preds=int(cfg.num_preds),
            clip_frames=int(cfg.history_size) + int(cfg.num_preds),
            batch_size=int(cfg.loader.batch_size), steps_per_epoch=b["spe"],
            params_total=sum(parts.values()), **{f"params_{k}": v for k, v in parts.items()},
            gflops_per_step=b["gflops"], gflops_attention_per_step=b["gflops_attn"],
            s_per_step_median=s_step, s_per_step_p10=float(np.percentile(t, 10)),
            s_per_step_p90=float(np.percentile(t, 90)),
            tflops_per_epoch=b["gflops"] * b["spe"] / 1e3,
            s_per_epoch=s_step * b["spe"],
            tflops_100_epochs=b["gflops"] * b["spe"] * 100 / 1e3,
            gpu_h_100_epochs=s_step * b["spe"] * 100 / 3600,
            bf16=b["bf16"], n_timed=len(t), device=torch.cuda.get_device_name(device)))
    df = pd.DataFrame(rows)

    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(device).replace("NVIDIA ", "").split()[0].lower()
    name = f"training_compute_{gpu}"
    df.to_csv(out / f"{name}.csv", index=False)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (out / f"{name}.meta.json").write_text(json.dumps(dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha=sha, torch=torch.__version__,
        device=torch.cuda.get_device_name(device),
        n_warm=args.n_warm, n_time=args.n_time, arms=arms), indent=1))
    print(df[["arm", "label", "clip_frames", "steps_per_epoch", "params_total",
              "gflops_per_step", "s_per_step_median", "tflops_100_epochs",
              "gpu_h_100_epochs"]].to_string(index=False))
    print(f"wrote {out / name}.csv")


if __name__ == "__main__":
    main()
