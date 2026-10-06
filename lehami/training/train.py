"""Model-agnostic training entry point.

Model builders live in lehami/models/hjepa/builders.py. This script owns
data loading, split logic and the Lightning training loop (and appends a run entry to
a LABBOOK.md at the repository root if one exists; the release ships none).

Split modes (set via manifest=<path> Hydra override):
  episode manifest  → torch.utils.data.Subset filtered by episode index
  clip manifest     → spt.data.random_split with the manifest's seed + ratio
  no manifest       → random clip-level split using cfg.train_split + cfg.seed

Usage (the runner scripts in lehami/experiments/ make these calls for every
arm and seed of the paper):
    python lehami/training/train.py \\
        +experiment=phase2_hjepa_sg_sep_conv_w1_swingup_exp079_a0q \\
        manifest=lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json

    # any Hydra override works, e.g. a one-epoch check with a different seed
    python lehami/training/train.py \\
        +experiment=phase0_lewm_ms_conv_w1_swingup_exp072_b0s \\
        manifest=lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \\
        trainer.max_epochs=1 seed=1 run_suffix=_check
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root

# ── paths: model package + third-party (leworldmodel) ────────────────────────
_LEWM = ROOT / "lehami" / "models" / "third_party" / "leworldmodel"
if str(_LEWM) not in sys.path:
    sys.path.insert(0, str(_LEWM))
if str(ROOT / "lehami" / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "lehami" / "models"))

import hdf5plugin  # noqa: F401 — activates Zstd codec before any h5py import
import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_pretraining import data as dt
from omegaconf import DictConfig, OmegaConf

from hjepa.builders import REGISTRY


# ── data helpers ──────────────────────────────────────────────────────────────

def _build_dataset(cfg: DictConfig):
    imagenet  = dt.dataset_stats.ImageNet
    transform = dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet, source="pixels", target="pixels"),
        dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"),
    )
    return swm.data.HDF5Dataset(
        cfg.data.name,
        frameskip=cfg.data.frameskip,
        num_steps=cfg.history_size + cfg.num_preds,
        keys_to_load=list(cfg.data.keys_to_load),
        transform=transform,
    )


def _val_episodes(manifest: dict) -> tuple[set[int], str]:
    """Episodes the val dataloader is built from, and a label for the printout.

    A manifest written with `--val-ratio` carries a genuine third split, and the
    val loader uses it: the checkpoint is then selected on episodes that are
    neither trained on nor reported.  A two-way manifest has no such split, and
    the historical behaviour was to feed the *test* episodes to a dataloader
    named "validate" — so `best.ckpt`, monitored on `validate/*`, was selected
    on the test set.  That path is kept only so
    old manifests stay loadable, and it is loud about what it is: with no val
    episodes, `validate/*` metrics are TEST metrics and must not drive
    checkpoint selection.  `checkpoint_monitor` defaults to `fit/pred_loss_epoch`
    for exactly this reason.
    """
    if manifest.get("val_episode_indices"):
        return set(manifest["val_episode_indices"]), "val"
    print(
        "[train] WARNING: manifest has no val_episode_indices — the 'validate' "
        "dataloader is built from the TEST episodes. Every validate/* metric is "
        "a test metric. Do not point checkpoint_monitor at it; regenerate the "
        "manifest with data_prep.py --val-ratio to get a real val split."
    )
    return set(manifest["test_episode_indices"]), "val(=TEST, leaky)"


def _split_from_manifest(dataset, manifest_path: str, cfg: DictConfig | None = None):
    manifest = json.loads(Path(manifest_path).read_text())
    method   = manifest["split_method"]
    if method == "episode":
        train_eps = set(manifest["train_episode_indices"])
        val_eps, vlabel = _val_episodes(manifest)
        train_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in train_eps]
        val_idx   = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in val_eps]
        print(f"[train] episode split: {len(train_idx)} train / {len(val_idx)} {vlabel} clips")
        return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)
    if method == "episode_srnn":
        train_eps       = set(manifest["train_episode_indices"])
        val_eps, vlabel = _val_episodes(manifest)
        train_frames    = int(manifest["train_frames"])
        num_steps    = cfg.history_size + cfg.num_preds
        # Only clips whose last frame falls within the first train_frames of the episode.
        # clip start + num_steps <= train_frames  →  start <= train_frames - num_steps
        max_start = train_frames - num_steps
        if max_start < 0:
            raise ValueError(
                f"cfg.history_size + cfg.num_preds = {num_steps} exceeds the manifest's "
                f"train_frames = {train_frames}: no training clip fits in the frame budget."
            )
        m_num_steps = manifest.get("num_steps")
        if m_num_steps is not None and int(m_num_steps) != num_steps:
            if cfg.get("allow_num_steps_mismatch", False):
                print(
                    f"[train] num_steps mismatch allowed: manifest={m_num_steps}, "
                    f"using cfg history_size+num_preds={num_steps}. Episode split and "
                    f"train_frames cap still come from the manifest."
                )
            else:
                raise ValueError(
                    f"Manifest specifies num_steps={m_num_steps} but cfg gives "
                    f"history_size+num_preds={num_steps}. The manifest is authoritative for "
                    f"clip generation: set history_size/num_preds to sum to {m_num_steps}, "
                    f"regenerate the manifest with data_prep.py --num-steps {num_steps}, "
                    f"or pass allow_num_steps_mismatch=true to opt out deliberately "
                    f"(e.g. baselines with a different clip length on the same episode split)."
                )
        clip_stride = int(manifest.get("clip_stride", 1))
        train_idx = [
            i for i, (ep, start) in enumerate(dataset.clip_indices)
            if ep in train_eps and start <= max_start and start % clip_stride == 0
        ]
        if manifest.get("val_episode_indices"):
            # A real val split is capped to the same frame budget as train: an
            # uncapped val set would score the model on late-episode frames it
            # is never trained to reach, so the selection signal would track
            # long-horizon drift rather than the objective being optimised.
            val_idx = [
                i for i, (ep, start) in enumerate(dataset.clip_indices)
                if ep in val_eps and start <= max_start and start % clip_stride == 0
            ]
        else:
            # Legacy two-way manifest: uncapped, exactly as before, so the
            # (leaky, non-selecting) validate/* numbers of past runs still
            # reproduce bit-for-bit.
            val_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in val_eps]
        n_eps_used = len({dataset.clip_indices[i][0] for i in train_idx})
        print(
            f"[train] episode_srnn split: {len(train_idx)} train clips "
            f"from {n_eps_used}/{len(train_eps)} train episodes "
            f"(first {train_frames} frames, num_steps={num_steps}, "
            f"clip_stride={clip_stride}, max_start={max_start}) / "
            f"{len(val_idx)} {vlabel} clips"
        )
        return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)
    if method == "clip":
        ratio = manifest["train_ratio"]
        rng   = torch.Generator().manual_seed(manifest["seed"])
        print(f"[train] clip split: ratio={ratio}, seed={manifest['seed']}")
        return spt.data.random_split(dataset, [ratio, 1.0 - ratio], generator=rng)
    raise ValueError(f"Unknown split_method in manifest: {method!r}")


def _split_random(dataset, cfg: DictConfig):
    rng = torch.Generator().manual_seed(cfg.seed)
    print(f"[train] random clip split: ratio={cfg.train_split}, seed={cfg.seed}")
    return spt.data.random_split(dataset, [cfg.train_split, 1.0 - cfg.train_split], generator=rng)


# ── labbook logging ───────────────────────────────────────────────────────────

_LABBOOK = ROOT / "LABBOOK.md"


def _log_run_start(cfg: DictConfig, model_name: str, run_name: str) -> None:
    if not _LABBOOK.exists():
        return
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    text    = _LABBOOK.read_text()
    exp_num = text.count("### EXP-") + 1
    wandb_ref = (
        f"[{run_name}](https://wandb.ai/{cfg.wandb.entity}/{cfg.wandb.project})"
        if cfg.wandb.enabled else run_name
    )
    entry = (
        f"\n### EXP-{exp_num:03d} — {run_name}\n"
        f"**Date:** {date.today()}  \n"
        f"**Git SHA:** `{sha}`  \n"
        f"**W&B run:** {wandb_ref}  \n\n"
        f"**Config:**\n"
        f"- `model`: {model_name}\n"
        f"- `data`: {cfg.data.name}\n"
        f"- `img_size`: {cfg.img_size}\n"
        f"- `embed_dim`: {cfg.embed_dim}\n"
        f"- `history_size`: {cfg.history_size}\n"
        f"- `max_epochs`: {cfg.trainer.max_epochs}\n"
        f"- `batch_size`: {cfg.loader.batch_size}\n"
        f"- `lr`: {cfg.optim.optimizer.lr}\n"
        f"- `sigreg.weight`: {cfg.loss.sigreg.weight}\n\n"
        f"**Goal:** *(fill in)*\n\n"
        f"**Results:**\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| rollout_mse_k1  | |\n| rollout_mse_k5  | |\n"
        f"| rollout_mse_k10 | |\n| probe_theta_r2  | |\n"
        f"| probe_dtheta_r2 | |\n\n"
        f"**Observations:** *(fill in)*\n\n---\n"
    )
    marker = "## Phase 0"
    if marker in text:
        idx = text.index(marker)
        _LABBOOK.write_text(text[:idx] + entry + "\n" + text[idx:])
    else:
        _LABBOOK.write_text(text + entry)


# ── entry point ───────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dataset = _build_dataset(cfg)
    manifest_path = cfg.get("manifest", None)
    train_set, val_set = (
        _split_from_manifest(dataset, manifest_path, cfg) if manifest_path
        else _split_random(dataset, cfg)
    )

    rng = torch.Generator().manual_seed(cfg.seed)
    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    train_dl = torch.utils.data.DataLoader(
        train_set, shuffle=True, drop_last=True, generator=rng, **loader_kwargs)
    val_dl = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kwargs)

    model_name = cfg.model.name
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Known: {list(REGISTRY)}")
    model, sigreg, forward_fn = REGISTRY[model_name](cfg)

    pl_module = spt.Module(
        model=model, sigreg=sigreg, forward=forward_fn,
        optim={"model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW",
                          "lr": cfg.optim.optimizer.lr,
                          "weight_decay": cfg.optim.optimizer.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }},
    )

    logger = None
    if cfg.wandb.enabled:
        # The seed and run_suffix MUST reach W&B. Seeds of one arm share model_name
        # and data.name, so without them every seed logs under an identical display
        # name, and there is no way to tell which curve belongs to which checkpoint
        # directory (the checkpoint dir is keyed by run_suffix).
        run_suffix = cfg.get("run_suffix", "")
        logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity,
                             name=f"{model_name}_{cfg.data.name}{run_suffix}",
                             tags=[model_name, cfg.data.name, f"seed{cfg.seed}"])
        # config= on the constructor is merged into, not substituted for, the config
        # Lightning/swm already populate, so this adds keys rather than replacing them.
        logger.experiment.config.update(
            {"seed": cfg.seed, "run_suffix": run_suffix,
             "model_name": model_name, "data_name": cfg.data.name},
            allow_val_change=True,
        )

    ckpt_dir = (Path(swm.data.utils.get_cache_dir()) / "checkpoints"
                / f"{model_name}_{cfg.data.name}{cfg.get('run_suffix', '')}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Checkpoint selection metric, chosen from what the manifest actually offers.
    #
    #   three-way manifest (data_prep.py --val-ratio) → `validate/pred_loss`, a
    #       genuine held-out signal: the val episodes are neither trained on nor
    #       reported, so selection is honest AND sensitive to overfitting.
    #   two-way manifest → `fit/pred_loss_epoch`.  The `validate` dataloader is
    #       then built from the *test* episodes, so monitoring `validate/*`
    #       would select the checkpoint on the test set (the 2026-08-09 leak).
    #       Training loss is blunt — it falls monotonically, so this effectively
    #       picks a late epoch — but it is leak-free, which matters more.
    #
    # An explicit `checkpoint_monitor=...` always wins, including to reproduce
    # pre-2026-08-09 runs.
    has_val = bool(manifest_path) and bool(
        json.loads(Path(manifest_path).read_text()).get("val_episode_indices"))
    default_monitor = "validate/pred_loss" if has_val else "fit/pred_loss_epoch"
    ckpt_monitor = cfg.get("checkpoint_monitor", default_monitor)
    ckpt_mode    = cfg.get("checkpoint_mode", "min")
    print(f"[train] checkpoint selection: monitor={ckpt_monitor!r} mode={ckpt_mode!r} "
          f"(manifest has {'a real val split' if has_val else 'no val split'})")
    if not has_val and str(ckpt_monitor).startswith("validate/"):
        print("[train] WARNING: monitoring validate/* on a manifest with no val split "
              "selects the checkpoint on the TEST episodes. Every number derived from "
              "this checkpoint is optimistically biased.")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="best",
                            monitor=ckpt_monitor, mode=ckpt_mode,
                            save_top_k=1, save_last=True),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="epoch{epoch:03d}",
                            every_n_epochs=10, save_top_k=-1, save_last=False),
        ],
        num_sanity_val_steps=1,
        enable_checkpointing=True,
        default_root_dir=str(ckpt_dir),
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )

    _log_run_start(cfg, model_name, f"{model_name}_{cfg.data.name}")
    # `resume_from` is EXACT, not weights-only: the checkpoint carries optimizer,
    # scheduler, epoch, global_step and callback state, and `max_epochs` is
    # absolute, so resuming epoch 19 with max_epochs 35 runs the missing 16.
    # Without it an interrupted run restarts from scratch. The runner scripts in
    # lehami/experiments/ refuse to train into a checkpoint dir that already
    # holds checkpoints, so resume by calling train.py directly with resume_from.
    manager = spt.Manager(trainer=trainer, module=pl_module,
                          data=spt.data.DataModule(train=train_dl, val=val_dl),
                          ckpt_path=cfg.get("resume_from", None))
    manager()


if __name__ == "__main__":
    main()
