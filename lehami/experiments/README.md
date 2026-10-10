# Experiments

Each script here trains and/or evaluates for the experiments we conducted for the paper publication. Run them from the
repository root. They read datasets from, and write checkpoints to, `$STABLEWM_HOME`
(default `~/.stable_worldmodel`); evaluation results go to
`lehami/experiments/results/<name>/` and logs to
`lehami/experiments/logs/<script>/`.

The paper's figures do **not** need any of this: `paper_figures/paper_figures.ipynb`
reads the frozen numbers in `paper_figures/data/`. These scripts regenerate the
numbers behind them.

## Before you start

1. **Datasets.** Generate them once with the two scripts in `lehami/data/`
   (see the *Datasets* section of the top-level README):
   `bash lehami/data/prepare_conservative_data.sh` and
   `bash lehami/data/prepare_actuated_data.sh`. Every script below checks for
   the file it needs and says which command makes it.
2. **DINOv2 weights**, for `conservative_dino.sh` only: run
   `bash lehami/experiments/download_dinov2.sh` once, on a machine with network
   access (~89 MB into the torch.hub cache).

## Running them

**Stage 1: train and evaluate.** Independent of each other; run any subset, in any
order.

| Script | What it does | Paper |
|---|---|---|
| `conservative_single_rod.sh` | LeHami vs LeWM on the single rod | single-rod latent figure |
| `conservative_multi_rod.sh` | LeHami vs LeWM on three rod lengths, 3 seeds; step-size and time-reversal tests of the flow; Jacobian log-volume of the one-step map | conservative rollout figure and tables, dt-and-reversal figure, Jacobian-volume figure and table |
| `conservative_dino.sh` | both predictors on a shared frozen DINOv2 encoder | DINOv2 figure |
| `actuated_pendulum.sh` | 9 models x 3 seeds on the actuated pendulum: LeHami (SIGReg placement x encoder width) and LeWM; Jacobian log-volume; probe capacity (ridge / 25-NN / MLP) | actuated prediction tables, canonicality figure, actuated Jacobian-volume figure, probe-capacity tables |

**Stage 2: experiments that reuse stage-1 checkpoints** (no training).

| Script | Needs | What it does | Paper |
|---|---|---|---|
| `conservative_training_compute.sh` | `conservative_multi_rod.sh` | accuracy per training epoch + FLOPs/time per step | iso-compute table |
| `actuated_planning.sh` | `actuated_pendulum.sh` | CEM goal-reaching  planning ladder + planning cost | goal-reaching  planning tables |

Each script documents its models, outputs and options in the comment at the top of the
file, and each accepts a subset of models/seeds (the training ones also `--eval-only`),
so it can be resumed or run piecemeal. Jobs run one after another on one GPU.

## Models trained on datasets

The short codes used by the scripts and in the result folder names:

| Code | System | Model | Encoder |
|---|---|---|---|
| `a0m` | conservative, 3 rods | LeHami | per-frame (W=1) |
| `b0m` | conservative, 3 rods | LeWM | per-frame |
| `b1m` | conservative, 3 rods | LeWM | 3-frame window (W=3) |
| `a0d` / `b0d` | conservative, single rod | LeHami / DINO-WM predictor | frozen DINOv2 |
| `a0q` | actuated | LeHami, SIGReg on q (main) | ConvNet w/ W=1 |
| `a0p` / `a0s` | actuated | LeHami, SIGReg on q and p separately / on [q,p] | ConvNet w/ W=1 |
| `a3q` / `a3p` / `a3s` | actuated | as above | ConvNet w/ W=3 |
| `b0s` / `b3s` | actuated | LeWM | ConvNet w/ W=1 / W=3 |
| `b3f` | actuated | LeWM, untied temporal window | ConvNet w/ W=3 |

Each code maps to one config in `lehami/training/configs/experiment/`, named
`phase<N>_<model>_<details>_exp<NNN>_<code>.yaml`. The `expNNN` number is the
internal run id; it is kept so result folders match the configs.

## Smoke test

`TRAIN_OVERRIDES` appends Hydra overrides to every `train.py` call a script makes. For
a quick end-to-end check of the whole chain (data, train, eval, probes, dumps), train
one seed of the main model for one epoch, in a separate `STABLEWM_HOME` so the smoke
checkpoint does not sit where the real run will write:

```bash
export STABLEWM_HOME=/tmp/lehami_smoke
bash lehami/data/prepare_actuated_data.sh
TRAIN_OVERRIDES="trainer.max_epochs=1" \
    bash lehami/experiments/actuated_pendulum.sh --models a0q --seeds 0
```

