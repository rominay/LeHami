# LeHami: Learning Symplectic World Models End-to-End

Anonymous code release accompanying the submission. LeHami's predictor is a learned symplectic flow whose phase-space-like latent preserves volume by construction, avoiding some collapse modes. By design it represents the environment's complete physics, suiting velocity-dependent planning.


## Contents

This repository contains the code to the model's architecture, their training and evaluation,
as well as the notebook that draws every figure in the paper.

```
paper_figures/
  paper_figures.ipynb   every figure and table in the paper, end to end
  data/                 the frozen numbers it reads (CSV / JSON / NPZ)
lehami/                 all the code; the method is called LeHami in the paper (its
                        Python package keeps its original name, `hjepa`)
  models/hjepa/         the method: the Hamiltonian flow predictor (ham_flow.py), the
                        momentum stencils (derivative.py), the model builders
                        (builders.py) and the adapter over DINO-WM (dino_wm_adapter.py)
  training/             train.py + Hydra configs (configs/model, configs/experiment, ...)
  data/                 dataset generators (conservative and actuated pendulum), the
                        prepare_*.sh scripts that run them with the paper's flags, and
                        the train/val/test split manifests used in the paper
  experiments/*.sh      runners that train and evaluate every arm reported in the
                        paper (see experiments/README.md)
  experiments/src/      the evaluation, probing and planning scripts they call
  tests/                unit tests (shape assertions, integrator properties, planning)
```

`paper_figures.ipynb` reads **only** `paper_figures/data/`, so every figure reproduces
on a laptop with no GPU, no datasets and no retraining. 


The runner scripts read and write under `${STABLEWM_HOME}`
(default `~/.stable_worldmodel`) and run each job in sequence on one GPU.

## Setup

```bash
uv sync
source .venv/bin/activate
```

We use third-party libraries. Some of them are cloned and we specify the commits we used,
into `lehami/models/third_party/`; the code adds them to `sys.path` at import
time and never modifies them:

```bash
T=lehami/models/third_party
git clone https://github.com/lucas-maes/le-wm      $T/leworldmodel
git -C $T/leworldmodel checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
git clone https://github.com/gaoyuezhou/dino_wm    $T/dino_wm
git -C $T/dino_wm checkout 0a9492fa12044b852ae9e001cc74604b79c8bb0c
```

Only the frozen-DINOv2 experiment needs `dino_wm`; its tests skip when the clone is
absent. That experiment also needs the DINOv2 weights (~89 MB), downloaded once with
`bash lehami/experiments/download_dinov2.sh` on a machine with network access.

Some others are installed differently, see below.

### What comes from where

The comparison is built so that the **predictor is the only difference** between our
model and the baselines; everything else is shared upstream code.

| Source | What we use | Used by |
|---|---|---|
| [LeWorldModel](https://github.com/lucas-maes/le-wm) (`third_party/leworldmodel`) | `JEPA` (the encoder → projector → predictor wrapper and its rollout), `MLP` (projectors), `Embedder` (action encoder), `SIGReg` (anti-collapse regulariser), `ARPredictor` (the autoregressive transformer predictor) | all models; `ARPredictor` is the LeWM baseline's predictor, which our Hamiltonian flow (`models/hjepa/ham_flow.py`) replaces |
| [DINO-WM](https://github.com/gaoyuezhou/dino_wm) (`third_party/dino_wm`) | `DinoV2Encoder` (frozen DINOv2, CLS token) and `ViTPredictor` (its space-time transformer predictor) | the frozen-DINOv2 experiment only, both arms share the encoder; `dino_wm_adapter.py` wraps them with one fix (the causal mask becomes a buffer so it follows `.to(device)`) |
| [DINOv2](https://github.com/facebookresearch/dinov2) via `torch.hub` | `dinov2_vits14` weights | the frozen-DINOv2 experiment |
| [stable-worldmodel](https://pypi.org/project/stable-worldmodel/) (pip) | `HDF5Dataset` (episode/clip loading from the `.h5` files) and the `$STABLEWM_HOME` data/checkpoint directory | `train.py`, `run_eval.py` |
| [stable-pretraining](https://pypi.org/project/stable-pretraining/) (pip, installed with stable-worldmodel) | `Module` / `Manager` / `DataModule` (the Lightning training loop), `random_split`, image transforms | `train.py` |

Our own code is the Hamiltonian flow and its integrators, the momentum stencils, the
pixel encoders used on the pendulum (`ConvEncoder` / `TemporalConvEncoder` in
`builders.py`), the dataset generators, evaluation, probing and planning.

## Datasets

Each dataset takes two steps: a generator writes an HDF5 file, and a **split manifest**
(a `split_*.json`) says which episodes are train / val / test and how to cut them into
clips. Training and evaluation both take the manifest via `manifest=` / `--manifest`.
They load the frames from `${STABLEWM_HOME}/<data.name>.h5` (default
`~/.stable_worldmodel`), so the file names below must match exactly.

**1. Generate the HDF5 files** (all generators are seeded, `--seed 42` by default):

```bash
export STABLEWM_HOME=${STABLEWM_HOME:-$HOME/.stable_worldmodel}

# conservative pendulum: pendulum.h5 (single rod, 500 episodes x 100 frames)
#                    and pendulum_multi_v2.h5 (3 rod lengths x 4 dt)
bash lehami/data/prepare_conservative_data.sh

# actuated pendulum (swing-up), 3 lengths x 3 masses  -> pendulum_swingup.h5
bash lehami/data/prepare_actuated_data.sh
```

Both run on CPU in about a minute and only call the generators in `lehami/data/`
(`generate_pendulum.py`, `pendulum_multi.py`, `pendulum_swingup.py`) with the flags
used for the paper.

**2. Use the shipped manifests.** The three splits used in the paper are committed in
`lehami/data/` and can be passed straight to training / evaluation:

| Manifest | Dataset | Split |
|---|---|---|
| `split_pendulum_episode_srnn_K100_seed42.json` | `pendulum.h5` | 450 / – / 50 episodes, full 100-frame episodes |
| `split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json` | `pendulum_multi_v2.h5` | 210 / 45 / 45 episodes, dt×2 slice only, first 80 frames trained |
| `split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json` | `pendulum_swingup.h5` | 630 / 135 / 135 episodes, first 80 frames trained |

Each manifest records the `dataset_sha256` of the file it was cut from. You can check your
regenerated file against it with `sha256sum "$STABLEWM_HOME"/<name>.h5`.

**Optional: rebuild a manifest** with `lehami/data/data_prep.py` (or pass
`--manifests` to either prepare script). It writes `split_<dataset>_….json` next to
itself, overwriting the shipped copy of the same name. The exact recipes:

```bash
uv run python lehami/data/data_prep.py --dataset "$STABLEWM_HOME/pendulum.h5" \
    --split-method episode_srnn --train-ratio 0.9 --train-frames 100 --num-steps 10 --seed 42

uv run python lehami/data/data_prep.py --dataset "$STABLEWM_HOME/pendulum_multi_v2.h5" \
    --split-method episode_srnn --train-ratio 0.70 --val-ratio 0.15 \
    --restrict-dt-mults 2 --train-frames 80 --num-steps 16 --seed 42

uv run python lehami/data/data_prep.py --dataset "$STABLEWM_HOME/pendulum_swingup.h5" \
    --split-method episode_srnn --train-ratio 0.7 --val-ratio 0.15 \
    --train-frames 80 --num-steps 16 --seed 42
```

The split is drawn with the seed *within* each design cell recorded in the HDF5 file
(rod length, and frame rate where it varies), so a file generated with the commands
above yields the same episode lists as the shipped manifest.

## Running the experiments

With the datasets in place, every block of the paper has one script in
`lehami/experiments/` that trains and evaluates all of its arms and seeds:

| Stage | Script | Paper |
|---|---|---|
| 1 | `conservative_single_rod.sh` | single-rod latent figure |
| 1 | `conservative_multi_rod.sh` | conservative rollout, dt-and-reversal and Jacobian-volume results |
| 1 | `conservative_dino.sh` (first run `download_dinov2.sh`) | DINOv2 figure |
| 1 | `actuated_pendulum.sh` | actuated prediction, canonicality, Jacobian-volume and probe-capacity results |
| 2 | `conservative_training_compute.sh` (after `conservative_multi_rod.sh`) | iso-compute table |
| 2 | `actuated_planning.sh` (after `actuated_pendulum.sh`) | swing-up planning tables |

```bash
bash lehami/experiments/actuated_pendulum.sh            # e.g. every actuated arm, 3 seeds
```

[`lehami/experiments/README.md`](lehami/experiments/README.md) describes each script,
the arm codes (`a0q`, `b0s`, ...) and a one-epoch smoke test of the whole chain.

### A single model by hand

The scripts are thin wrappers around two commands. To train the main actuated model
(a0q, one seed) and evaluate it:

```bash
uv run python lehami/training/train.py \
    +experiment=phase2_hjepa_sg_sep_conv_w1_swingup_exp079_a0q \
    manifest=lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json

uv run python lehami/experiments/src/run_eval.py \
    --checkpoint "$STABLEWM_HOME/checkpoints/hjepa_sg_sep_sreg_q_ms_conv_pendulum_swingup_sg_sep_conv_w1_swingup_exp079_a0q/best.ckpt" \
    --manifest   lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \
    --exp-name   a0q --horizon 76
```

An experiment config names a model config, which names a builder in
`lehami/models/hjepa/builders.py`. Checkpoints and the resolved config are
written to `${STABLEWM_HOME}/checkpoints/<model>_<dataset><run_suffix>/`. Logging to
Weights & Biases is off by default; add `wandb.enabled=true` (and
`wandb.entity=<your entity>` if it is not your default one) to turn it on.

`run_eval.py` writes `rollout_metrics.csv`, `probe_metrics.csv`, `summary.json` and the
energy-stratified variants into `lehami/experiments/results/<name>/`. It rebuilds the
model from the `config.yaml` saved next to the checkpoint, so evaluation never depends
on flags being re-typed correctly. Planning is driven by
`lehami/experiments/src/plan_pendulum_swingup.py` (CEM in latent space);
`actuated_planning.sh` runs the paper's full planning ladder.

## Figures

Every figure in the paper is produced by one notebook, which needs nothing but
this repository:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace paper_figures/paper_figures.ipynb
```

It ships already executed, so the figures are also readable without running
anything. Its data lives beside it in `paper_figures/data/`, organised by system
(`conservative_pendulum/`, `actuated_pendulum/`, `datasets/`); running it writes the
PNGs to `paper_figures/figures/<system>/`.

## Tests

```bash
uv run pytest lehami/tests/
```
