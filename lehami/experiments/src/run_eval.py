"""Standardized evaluation for any lehami checkpoint.

Two modes, selected with --mode:

  standard (default)
      Steps executed:
        3.a  Probe training  — for episode manifests, trains 4 probes on train-split
             frames (skip with --skip-extra-probes to train only the first):
               ridge_gt  Ridge(GT encoder (q,p) → [sinθ, cosθ, ω])       — original probe
               knn_gt    KNN   on the same GT (q,p)                       (train_probe_physics.py)
               ridge_k1  Ridge on the model's OWN k=1 rollout output       (train_probe_physics.py)
               knn_k1    KNN   on the model's OWN k=1 rollout output       (train_probe_physics.py)
             Clip manifests only get ridge_gt.
        3.b  Rollout metrics — RMSE/√d, R², and energy H_φ at each k (both splits).
        3.c  Probe rollout   — apply each frozen probe to predicted q_k → MAE & R² for θ and ω.
      One anchor per episode (or per manifest clip for clip manifests).
      Outputs in lehami/experiments/results/<exp_name>/:
        rollout_metrics.csv  — columns: split, k, rmse_norm_q, r2_q, energy_mean, energy_std
                               (+ rmse_norm_p, r2_p for models with a momentum coordinate)
        probe_metrics.csv    — columns: split, k, mae_theta, r2_theta, mae_omega, r2_omega
                               (+ mae_length, r2_length when the dataset varies rod
                               length, e.g. pendulum_multi_v2.h5. L is a PARAMETER, not
                               a state: read r2_length vs k, not at k=0. At k=0 it is
                               near-trivial — the rod is rendered to scale, so L is
                               readable from one frame — but a frozen probe on the
                               predicted q_k must hold it FLAT, and decay means the flow
                               is leaking a conserved parameter into the state.)
                               ridge_gt is unsuffixed; the same columns appear suffixed
                               __knn_gt / __ridge_k1 / __knn_k1 for the extra probes
        probe.pkl / probe_knn.pkl / probe_k1_ridge.pkl / probe_k1_knn.pkl
        summary.json         — key-horizon subset of the above + metadata
        rollout_metrics_by_energy.csv / probe_metrics_by_energy.csv
                             — same metrics per episode-energy bin (terciles of max |ω|
                               by default; --energy-bins N, 0 disables). Pooled CSVs are
                               unchanged; bin edges recorded in summary.json.

  multiclip
      Dense sliding-window clips from each test episode: the anchor slides over
      every valid starting position (anchor c+1, for c = 0 .. L - horizon - 2),
      giving ~70× more evaluation clips per episode and closely matching the
      training distribution. Loads whichever of probe.pkl / probe_knn.pkl /
      probe_k1_ridge.pkl / probe_k1_knn.pkl exist in the results dir (probe
      metrics embedded inline, columns suffixed as in probe_metrics.csv above);
      does NOT train a probe. --probe-path only overrides the ridge_gt location.
      LeWM checkpoints use a rolling h-frame context buffer (no p/energy metrics).
      Outputs in lehami/experiments/results/<exp_name>/:
        multiclip_rollout_metrics.csv
        multiclip_rollout_metrics_by_energy.csv  — per episode-energy bin (no extra
                                                   rollout compute; --energy-bins N)

The model architecture is reconstructed from a config.yaml saved alongside the
checkpoint by lehami/training/train.py.

Usage:
    python lehami/experiments/src/run_eval.py \\
        --checkpoint "$STABLEWM_HOME/checkpoints/lehami_sg_sep_sreg_q_ms_conv_pendulum_swingup_sg_sep_conv_w1_swingup_exp079_a0q/best.ckpt" \\
        --manifest   lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \\
        --exp-name   a0q --horizon 76

    # multiclip (default horizon 30, test split only)
    python lehami/experiments/src/run_eval.py --mode multiclip \\
        --checkpoint <ckpt> --manifest lehami/data/split_pendulum_episode_srnn_K100_seed42.json \\
        --exp-name multiclip_check --device cpu

    # smoke tests (short horizon, CPU)
    python lehami/experiments/src/run_eval.py \\
        --checkpoint <path> --manifest <path> --exp-name smoke \\
        --horizon 5 --device cpu
    python lehami/experiments/src/run_eval.py --mode multiclip \\
        --checkpoint <path> --manifest <path> --exp-name smoke \\
        --horizon 5 --device cpu --max-episodes 5
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR  = Path(__file__).resolve().parent
_LEWM     = ROOT / "lehami" / "models" / "third_party" / "leworldmodel"
_MODELS   = ROOT / "lehami" / "models"
# Insert _SRC_DIR first so that experiments/src/utils.py wins over leworldmodel/utils.py
for _p in [str(_LEWM), str(_MODELS), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# train_probe_physics.py does `from run_eval import ...` at module level; when
# this file is executed directly (__name__ == "__main__") and later does a
# deferred `from train_probe_physics import ...` (see main()'s 3.a section),
# that top-level import would otherwise re-exec this whole file as a SECOND,
# distinct "run_eval" module (duplicate Probe class, double-loaded imports).
# Registering this already-running module under its on-disk name up front
# makes both names resolve to the same module object.
sys.modules.setdefault("run_eval", sys.modules[__name__])

import hdf5plugin  # noqa: F401 — activates Zstd codec
import torch
import torch.nn.functional as F
import stable_worldmodel as swm
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from probe_targets import (  # leaf module, no cycle — see its docstring
    probe_targets,
    state_names,
    target_names,  # noqa: F401 — re-exported for importers of this module
)

from lehami.builders import REGISTRY, ConvEncoder, TemporalConvEncoder
from lehami.dino_wm_adapter import DinoEncoder

# Encoders that return a bare (B, embed_dim) tensor, as opposed to a HuggingFace
# ViT output object that needs `.last_hidden_state[:, 0]` and an
# `interpolate_pos_encoding=True` kwarg. Dispatching on this rather than on
# "is it the ConvEncoder" is what lets the frozen DINOv2 arms through:
# DinoEncoder pools inside its own forward and has no such kwarg, so the ViT
# branch raises TypeError on it.
_PLAIN_TENSOR_ENCODERS = (ConvEncoder, TemporalConvEncoder, DinoEncoder)

# ── model-family classification ────────────────────────────────────────────────

_LEWM = {
    # Unconstrained predictors with no q/p split: no momentum is synthesised for them.
    "lewm_ms",        # LeWM, ViT-tiny encoder, multi-step rollout training
    "lewm_ms_conv",   # the same with the ConvNet encoder
    "dino_wm_cls",    # DINO-WM's ViTPredictor on frozen DINOv2 CLS features
}
_SG = {
    # The Hamiltonian flow with Savitzky-Golay momentum. The variants share one builder
    # and differ only in cfg.model.sigreg_mode / cfg.model.encoder, so all of them need
    # the same SG stencil at eval.
    "lehami_sg_sep_sreg_joint_ms",       # ViT-tiny encoder
    "lehami_sg_sep_sreg_joint_ms_conv",  # ConvNet encoder, SIGReg on [q, p]
    "lehami_sg_sep_sreg_q_ms_conv",      # SIGReg on q only
    "lehami_sg_sep_sreg_split_ms_conv",  # SIGReg on q and p independently
    "lehami_sg_sep_sreg_q_ms_dino",      # frozen DINOv2 encoder
}


def _mom_type(name: str) -> str:
    """Return 'lewm' (unconstrained predictor, no p) or 'sg' (Hamiltonian flow whose
    momentum is the Savitzky-Golay derivative of q; see lehami/derivative.py —
    causal stencil at anchors, centered elsewhere)."""
    if name in _LEWM:
        return "lewm"
    if name in _SG:
        return "sg"
    raise ValueError(f"unknown model '{name}'; expected one of {sorted(_LEWM | _SG)}")


def _anchor_offset(mom_type: str, cfg=None) -> int:
    """Frames of history a rollout anchor needs before it: ANCHOR_OFFSET (3, the
    causal SG window) for sg, 1 otherwise."""
    from lehami.derivative import ANCHOR_OFFSET
    return ANCHOR_OFFSET if mom_type == "sg" else 1


def _p_series(q_enc: torch.Tensor, mom_type: str, fd_dt: float) -> torch.Tensor:
    """Reference momentum at every frame of an encoded trajectory q_enc (L, d): the SG
    interpolation series (centered interior, causal tail, backward FD at t=1)."""
    assert mom_type == "sg", f"no reference momentum for mom_type={mom_type!r}"
    from lehami.derivative import p_interp_series
    return p_interp_series(q_enc, fd_dt)


def _causal_p_batch(q_enc: torch.Tensor, idx: torch.Tensor, mom_type: str, fd_dt: float) -> torch.Tensor:
    """GT momentum at a batch of frame indices, via the CAUSAL stencil the model
    itself uses to build inputs (matches _build_anchor's per-frame formula,
    generalized to idx: (N,) LongTensor of frame indices). Returns (N, d).

    Not defined for "lewm", which has no momentum.
    """
    assert mom_type == "sg", f"no causal momentum for mom_type={mom_type!r}"
    from lehami.derivative import CAUSAL_WINDOW, p_causal_last
    offs = torch.arange(-(CAUSAL_WINDOW - 1), 1, device=idx.device)
    window = q_enc[idx.unsqueeze(-1) + offs]        # (N, W, d)
    return p_causal_last(window, fd_dt)


def _has_energy(name: str) -> bool:
    return name not in _LEWM


def _latent_dim(cfg) -> int:
    """Embedding dimension: latent_dim for lehami models, embed_dim for lewm."""
    return cfg.model.get("latent_dim", cfg.embed_dim)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: str, model_name: str | None = None):
    """Load model from checkpoint, rebuilding it from the config.yaml that train.py
    saves next to every checkpoint. `model_name` is unused; it is kept so existing
    callers need not change.
    """
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found alongside checkpoint (expected {cfg_path}); "
            f"train.py writes it next to every checkpoint."
        )
    cfg = OmegaConf.load(cfg_path)
    mn = cfg.model.name
    if mn not in REGISTRY:
        raise ValueError(f"Unknown model '{mn}'. Known: {list(REGISTRY)}")

    model, _sigreg, _fwd = REGISTRY[mn](cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"]
    if any(k.startswith("model.") for k in raw):
        sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    else:
        sd = dict(raw)
    missing, unexpected = model.load_state_dict(sd, strict=False).missing_keys, \
                          model.load_state_dict(sd, strict=False).unexpected_keys
    allowed_missing = {"predictor.flow.p_scale"}
    assert set(missing) <= allowed_missing, f"Unexpected missing keys: {missing}"
    assert not unexpected, f"Unexpected keys in checkpoint: {unexpected}"

    model.eval().to(device)
    return model, cfg


def apply_flow_substeps(model, n: int) -> None:
    """Make each predictor step run `n` leapfrog sub-steps instead of one.

    This is the cross-dt transfer arm: a model trained at dt is deployed on data
    sampled at n*dt by integrating its OWN learned Hamiltonian n times per frame
    gap, rather than by retraining. Composing n symplectic steps is still exactly
    symplectic, so the deployed map remains a valid flow of the same H — nothing
    about the model is refit, only how far it is integrated.

    Implemented by wrapping `predictor.flow.forward` rather than by scaling
    `flow.dt`, and the difference matters: scaling dt would take ONE coarser
    step, which changes the integrator's truncation error and conflates
    "transfers across rates" with "tolerates a coarser step". Sub-stepping holds
    the per-step error fixed.

    Unconstrained predictors (LeWM) have no `flow`, and no equivalent operation —
    that asymmetry is the point of the arm, so this raises rather than no-ops.
    """
    if n < 1:
        raise ValueError(f"flow-substeps must be >= 1, got {n}")
    flow = getattr(getattr(model, "predictor", None), "flow", None)
    if flow is None:
        raise ValueError(
            "--flow-substeps requires a model with predictor.flow (a Hamiltonian "
            f"flow); '{type(model).__name__}' has none. Unconstrained predictors "
            "cannot be re-stepped at a different rate — retrain instead."
        )
    if n == 1:
        return
    inner = flow.forward

    def _substepped(q, p, u, _inner=inner, _n=n):
        for _ in range(_n):
            q, p = _inner(q, p, u)
        return q, p

    flow.forward = _substepped
    print(f"[flow] each predictor step now runs {n} leapfrog sub-steps")


def apply_flow_dt_scale(model, s: float) -> None:
    """Integrate the SAME learned H with a step size of `s * dt`, holding the
    physical gap per emitted frame fixed.

    This is the step-size ablation, and it is the complement of
    `apply_flow_substeps`, not a duplicate:

        --flow-substeps N   holds h fixed and covers N*h of physics per frame
                            (the cross-dt transfer arm: more physics, same h)
        --flow-dt-scale  s  holds the physics per frame fixed and changes h
                            (this: same physics, different h)

    One emitted frame still advances exactly one dataset frame, so `1/s` inner
    steps of size `s*dt` are composed per predictor call — the model is asked to
    cross the *same* interval at a different resolution. Nothing else moves: same
    episodes, same anchors, same GT frames, same probe. The only variable is h,
    which is what makes the resulting error-vs-h curve interpretable (leapfrog is
    second order, so h < dt should be flat and h > dt should grow like h^2).

    Because the frame grid is unchanged, the FD/SG stencil still sees dt-spaced
    frames and `fd_dt` must NOT be rescaled — unlike the transfer arm, where the
    frame spacing itself changed and omitting the rescale silently handed K(p) a
    momentum on the wrong scale.

    Only s <= 1 (refinement) is supported. s > 1 would make one predictor step
    cover several dataset frames, which this eval's frame-indexed rollout cannot
    express; do that with a strided dataset plus --flow-substeps instead.

    LeWM has no `flow` and no step size to scale — it can only re-apply its fixed
    map an integer number of times. That asymmetry is the point of the ablation,
    so this raises rather than no-ops.
    """
    if not (0 < s <= 1):
        raise ValueError(
            f"--flow-dt-scale must be in (0, 1]; got {s}. Values > 1 need one "
            "predictor step to span several dataset frames — use a strided "
            "dataset with --flow-substeps for the coarsening direction."
        )
    n_inner = round(1.0 / s)
    if abs(n_inner - 1.0 / s) > 1e-9:
        raise ValueError(
            f"--flow-dt-scale must have integral 1/s so that {n_inner} steps of "
            f"size s*dt exactly cover one frame; got s={s} (1/s={1.0 / s})."
        )
    flow = getattr(getattr(model, "predictor", None), "flow", None)
    if flow is None:
        raise ValueError(
            "--flow-dt-scale requires a model with predictor.flow (a Hamiltonian "
            f"flow); '{type(model).__name__}' has none. An unconstrained predictor "
            "has no step size to change — it can only be re-applied."
        )
    if s == 1.0:
        return
    old_dt = float(flow.dt)
    flow.dt = old_dt * s
    inner = flow.forward

    def _refined(q, p, u, dt=None, _inner=inner, _n=n_inner, _s=s):
        # Composes with a per-episode step: the caller's h is the full-frame
        # step, so each inner step takes h*s and _n of them cover one frame.
        for _ in range(_n):
            q, p = _inner(q, p, u) if dt is None else _inner(q, p, u, dt * _s)
        return q, p

    flow.forward = _refined
    print(f"[flow] dt {old_dt:g} → {flow.dt:g} (x{s:g}); "
          f"{n_inner} steps per frame, same physical gap per frame")


# ── data helpers ──────────────────────────────────────────────────────────────

def _h5_path(cfg) -> Path:
    return Path(swm.data.utils.get_cache_dir()) / f"{cfg.data.name}.h5"


def _load_manifest(manifest_path: str) -> tuple[list[int], list[int]]:
    """Return (train_episode_indices, test_episode_indices).

    For episode manifests the stored indices are used directly.
    For clip manifests there are no episode-level indices, so we derive a
    random episode split using the same seed and train_ratio stored in the
    manifest.  Note: clips from the same episode may appear on both sides of
    the clip split, so the derived episode split is approximate — whole
    episodes are held out from the probe but not from model training.
    """
    import random as _random
    m = json.loads(Path(manifest_path).read_text())
    if m["split_method"] in ("episode", "episode_srnn"):
        return m["train_episode_indices"], m["test_episode_indices"]
    if m["split_method"] == "clip":
        n_eps = m["dataset_stats"]["n_episodes"]
        ratio = float(m.get("train_ratio", 0.9))
        seed  = int(m.get("seed", 42))
        all_eps = list(range(n_eps))
        rng = _random.Random(seed)
        rng.shuffle(all_eps)
        n_train = int(n_eps * ratio)
        print(
            f"[manifest] clip split → deriving episode split: "
            f"{n_train} train / {n_eps - n_train} test episodes "
            f"(seed={seed}, ratio={ratio})"
        )
        return all_eps[:n_train], all_eps[n_train:]
    raise ValueError(
        f"Unsupported split_method {m['split_method']!r}. "
        "Expected 'episode' or 'clip'."
    )


def _preprocess_batch(pixels_hwc: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    """(L, H, W, C) uint8 → (L, C, H, W) float32 ImageNet-normalized on device."""
    x = torch.from_numpy(pixels_hwc).float() / 255.0         # (L, H, W, C)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    x = (x - mean) / std                                      # (L, H, W, C)
    x = x.permute(0, 3, 1, 2)                                 # (L, C, H, W)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.to(device)


def _cfg_get(cfg, dotted: str, default=None):
    """Dotted lookup that works for DictConfig, SimpleNamespace and plain dicts.

    OmegaConf raises KeyError (not AttributeError) for a missing key, so a bare
    getattr(..., default) does not shield it; SimpleNamespace (used throughout
    the tests) is not an OmegaConf node at all, so OmegaConf.select does not
    work on it either. This handles all three.
    """
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(part, None)
            continue
        try:
            node = getattr(node, part)
        except Exception:
            try:
                node = node[part]
            except Exception:
                return default
    return default if node is None else node


def _episode_dt_ratio(h5_path, cfg) -> "np.ndarray | None":
    """Per-episode integrator step h = dt_episode / dt_ref, or None.

    None (the scalar path, h = 1 everywhere) unless the dataset carries a
    per-episode `dt` column AND the config sets `model.dt_ref` — so every
    single-rate run and every checkpoint trained before this existed is scored
    exactly as before.

    This must MATCH TRAINING. `builders._episode_h` divides the momentum
    stencil by h and hands h to the flow; if eval integrates the same weights at
    h = 1, a dt = 0.2 s episode is advanced half as far per step as it was
    trained to be, and the arm looks broken for a reason that has nothing to do
    with the model.
    """
    dt_ref = _cfg_get(cfg, "model.dt_ref")
    if dt_ref is None:
        return None
    if float(dt_ref) <= 0:
        raise ValueError(f"model.dt_ref must be > 0, got {dt_ref}")
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "dt" not in f:
            return None
        return f["dt"][:].astype(np.float64) / float(dt_ref)


def _scale_p(p, h):
    """Put a frame-unit momentum on the episode's clock: p = dq/dh."""
    return p if h is None else p / h


def _episode_lengths(h5_path) -> "np.ndarray | None":
    """Per-episode rod length L, or None when the dataset does not vary it.

    L is a *parameter* of the system, not a state: the flow advances (q, p) and
    must leave L untouched.  That is what makes it worth probing — see
    `_apply_probes_standard`.  Returns None for single-length datasets
    (pendulum.h5), where R^2 against a constant target is undefined, so every
    length metric is simply omitted rather than reported as nan.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "length" not in f:
            return None
        lengths = f["length"][:].astype(np.float64)
    return lengths if len(np.unique(lengths)) > 1 else None


def _with_length(states: np.ndarray, ep_lengths_arr, ep: int) -> np.ndarray:
    """Append rod length as a 3rd state column, so it rides along per-k with theta/omega."""
    if ep_lengths_arr is None:
        return states
    return np.column_stack([states, np.full(len(states), ep_lengths_arr[ep], dtype=np.float64)])


def _probe_n_targets(estimator) -> int:
    """Number of output channels a fitted probe estimator predicts.

    `Probe.ridge` holds whichever regressor the tier uses — Ridge for the
    `ridge_*` tiers, KNeighborsRegressor for the `knn_*` ones — so the count has
    to be read per estimator kind rather than off `coef_` alone:

        Ridge                  coef_  (n_targets, n_features)
        KNeighborsRegressor    _y     (n_samples, n_targets)   [no coef_]

    Reading only `coef_` with a 3-row default silently reported 3 for every KNN
    probe, which is what dropped `mae_length__knn_gt` / `r2_length__knn_gt` from
    every probe_metrics.csv written before 2026-08-24 even though the KNN probes
    were fit with L as a 4th target.
    """
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        return int(np.atleast_2d(coef).shape[0])
    y = getattr(estimator, "_y", None)          # KNeighborsRegressor's stored targets
    if y is not None:
        y = np.asarray(y)
        return int(y.shape[1]) if y.ndim > 1 else 1
    return int(getattr(estimator, "n_outputs_", 3))


def _probe_has_length(probe: "Probe") -> bool:
    """4 output channels => [sin, cos, omega, L]; 3 => the historical [sin, cos, omega].

    Inferred from the fitted estimator rather than stored on `Probe`, so probe
    pickles written before this existed still load and still score correctly.
    """
    return _probe_n_targets(probe.ridge) >= 4


def _enc_window(model) -> int:
    """Pixel frames consumed per latent embedding (1 for every per-frame encoder)."""
    return int(getattr(model.encoder, "window", 1))


def _enc_shift(model) -> int:
    """Pixel frames between latent index 0 and the pixel frame that latent represents.

    A `TemporalConvEncoder` emits latents in LATENT TIME: index j is built from
    pixel frames j … j+W-1 and, because the kernel is symmetric about its own
    centroid, represents the physical time of pixel frame j + (W-1) - lag.
    Every index used by training lives in this same latent time (the anchor
    offset, n_unroll, the SG stencil), so eval indices need no change — but the
    *ground-truth state array* is in pixel time and must be re-based, or q_j is
    scored against θ at the wrong instant.  The induced error ~ ω·lag·dt grows
    with speed and would look like an energy-dependent position error.

    W=1 → 0; W=3 symmetric → 1; W=2 'diff' (lag 0) → 1.  Always an integer for
    the shipped variants, so alignment is an exact slice, never an interpolation.
    """
    W   = _enc_window(model)
    lag = float(getattr(model.encoder, "lag", 0.0))
    sh  = (W - 1) - lag
    if abs(sh - round(sh)) > 1e-9:
        raise ValueError(
            f"non-integer encoder lag ({lag}) leaves latent time off-grid by {sh} "
            "frames; GT alignment would need interpolation.")
    return int(round(sh))


def _align_states(model, states_raw: np.ndarray) -> np.ndarray:
    """Re-base a pixel-time (L, ·) GT array onto latent time (L-W+1, ·).

    Identity (same object) for every per-frame encoder, so all pre-existing
    numbers are reproduced bit-for-bit.
    """
    W = _enc_window(model)
    if W == 1:
        return states_raw
    sh = _enc_shift(model)
    return states_raw[sh : sh + states_raw.shape[0] - W + 1]


@torch.no_grad()
def _encode_episode(model, pixels_chw: torch.Tensor, enc_batch: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all L frames of an episode.

    Returns (with W = encoder window, T_eff = L - W + 1):
        cls: (T_eff, enc_dim)    — raw CLS tokens / conv embeddings
        q:   (T_eff, latent_dim) — projected position coordinates

    For a per-frame encoder W = 1 and T_eff = L, i.e. unchanged.  For a
    temporal encoder the outputs are in latent time; pair them with
    `_align_states(model, states_raw)`.
    """
    L = pixels_chw.shape[0]
    W = _enc_window(model)
    is_plain = isinstance(model.encoder, _PLAIN_TENSOR_ENCODERS)
    cls_list, q_list = [], []
    if W > 1:
        assert isinstance(model.encoder, TemporalConvEncoder)
        if L < W:
            raise ValueError(f"episode too short for temporal encoder: L={L} < window={W}")
        # Chunks overlap by W-1 frames so the temporal window is never cut at a
        # chunk boundary; the per-frame stem is recomputed on the overlap only.
        for s in range(0, L - W + 1, enc_batch):
            e = min(s + enc_batch, L - W + 1)
            chunk = pixels_chw[s : e + W - 1].float().unsqueeze(0)
            c = model.encoder.forward_sequence(chunk)[0]   # (e-s, embed_dim)
            cls_list.append(c)
            q_list.append(model.projector(c))
        return torch.cat(cls_list), torch.cat(q_list)
    for s in range(0, L, enc_batch):
        chunk = pixels_chw[s:s + enc_batch]
        if is_plain:
            c = model.encoder(chunk.float())   # (B, embed_dim) directly, no CLS token
        else:
            c = model.encoder(chunk.float(), interpolate_pos_encoding=True).last_hidden_state[:, 0]
        cls_list.append(c)
        q_list.append(model.projector(c))
    return torch.cat(cls_list), torch.cat(q_list)


def _anchor_frame(cfg, mom_type: str) -> int:
    """Return the episode frame index used as the rollout anchor.

    sg: the anchor sits at clip frame 3, the first frame with a full causal
    interpolation window (frames 0..3) -- the same frame training anchors at.
    Otherwise the last context frame, h-1 (frame 1 when history_size == 1, so
    that q[a-1] never wraps to the end of the episode).
    """
    if mom_type == "sg":
        return _anchor_offset(mom_type, cfg)
    h = cfg.history_size
    return 1 if h == 1 else h - 1


def _build_anchor(model, cls: torch.Tensor, q: torch.Tensor,
                  mom_type: str, cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build anchor (q_a, p_a) at the episode frame returned by _anchor_frame().

    cls: (L, vit_dim), q: (L, latent_dim)
    Returns q0 (latent_dim,) and p0 (latent_dim,).
    """
    fd_dt = float(cfg.model.get("fd_dt", 1.0))
    a = _anchor_frame(cfg, mom_type)   # frame index of the anchor

    q0 = q[a]
    assert mom_type == "sg", f"no anchor momentum for mom_type={mom_type!r}"
    from lehami.derivative import CAUSAL_WINDOW, p_causal_last
    p0 = p_causal_last(q[a - CAUSAL_WINDOW + 1 : a + 1], fd_dt)
    return q0.detach(), p0.detach()


@torch.no_grad()
@torch.no_grad()
def _zero_action_emb(model, cfg, device: str) -> torch.Tensor:
    """(1, phase_dim) zero-action embedding."""
    act = torch.zeros(1, 1, cfg.data.action_dim, device=device)
    return model.action_encoder(act)[:, 0]  # (1, phase_dim)


def _latent_to_pixel_offset(model) -> int:
    """Frames to add to a LATENT index to get the pixel frame it stands for.

    Same rule as `builders._action_emb`, and it must stay the same rule: a
    window-W encoder turns T pixel frames into T-W+1 latents, and which pixel
    frame latent i represents depends on the stencil's centroid lag, not on W.
        full / depthwise (symmetric, odd W): lag = (W-1)/2 -> offset (W-1)/2
        diff             (W = 2, causal)   : lag = 0       -> offset 1
        per-frame        (W = 1)           : offset 0
    """
    w = int(getattr(model.encoder, "window", 1))
    lag = float(getattr(model.encoder, "lag", 0.0))
    off = (w - 1) - lag
    assert float(off).is_integer(), (
        f"non-integral latent->pixel offset {off} for window={w}, lag={lag}")
    return int(off)


def _episode_actions(f, ep_offset: int, n_pixel: int, cfg, device: str):
    """(n_pixel, action_dim) float tensor of the episode's actions, or None.

    None whenever the dataset carries no `action` column -- every pendulum
    dataset before the swing-up one -- so those runs keep the zero-token path
    and their numbers are unchanged.
    """
    if "action" not in f:
        return None
    a = f["action"][ep_offset : ep_offset + n_pixel]
    a = np.asarray(a, dtype=np.float32).reshape(n_pixel, -1)
    assert a.shape[1] == int(cfg.data.action_dim), (
        f"action_dim mismatch: h5 has {a.shape[1]}, cfg says {cfg.data.action_dim}")
    return torch.as_tensor(a, device=device)


def _act_emb_window(model, actions, zero_ctx, start_latent: int, n: int, off: int):
    """(1, n, embed_dim) action embeddings for latent frames start_latent .. +n-1.

    LeWM's ARPredictor consumes one action per context frame, and
    `builders._build_lewm_ms_conv` feeds it latent frames [step, step+h-1] when
    predicting latent frame h+step. Eval must use the same window or the baseline
    is handicapped relative to the Hamiltonian arm, which would make the whole
    comparison meaningless.
    """
    if actions is None:
        return zero_ctx
    idx = [min(max(i + off, 0), actions.shape[0] - 1)
           for i in range(start_latent, start_latent + n)]
    return model.action_encoder(actions[idx].unsqueeze(0))   # (1, n, embed_dim)


def _act_emb_at(model, actions, zero_emb, latent_idx: int, off: int) -> torch.Tensor:
    """(1, phase_dim) embedding of the action that DRIVES the transition out of
    `latent_idx`.  Falls back to the zero token when the dataset is unactuated.

    `builders._forward` drives the transition into latent frame A+1+step with the
    action at latent frame A+step, so eval must use the action at the latent frame
    it is stepping OUT of -- an off-by-one here is a half-step phase error that no
    metric would flag.
    """
    if actions is None:
        return zero_emb
    j = latent_idx + off
    if j >= actions.shape[0]:          # past the episode's last real action
        return zero_emb
    return model.action_encoder(actions[j].view(1, 1, -1))[:, 0]


# ── clip manifest loader ──────────────────────────────────────────────────────

def _load_clips_from_manifest(
    manifest_path: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Return (train_clips, test_clips, num_steps).

    Each clip is a (ep_idx, start_frame_in_episode) tuple.
    Reproduces the torch.randperm split used by spt.data.random_split during training,
    so train/test assignment is identical to what the model saw.

    Only works for clip manifests (split_method == 'clip').
    """
    m = json.loads(Path(manifest_path).read_text())
    if m["split_method"] != "clip":
        raise ValueError(
            f"_load_clips_from_manifest requires a clip manifest; "
            f"got split_method='{m['split_method']}'"
        )
    ep_lengths = m["dataset_stats"]["ep_lengths"]
    num_steps  = int(m["num_steps"])
    step       = int(m.get("frameskip", 1))
    seed       = int(m["seed"])
    n_train    = int(m["n_train_clips"])

    # All clips in episode-major order — matches HDF5Dataset iteration order.
    all_clips: list[tuple[int, int]] = [
        (ep, start)
        for ep, L in enumerate(ep_lengths)
        for start in range(max(0, L - (num_steps - 1) * step))
    ]
    assert len(all_clips) == m["n_total_clips"], (
        f"Clip count mismatch: reconstructed {len(all_clips)} "
        f"vs manifest {m['n_total_clips']}"
    )
    perm = torch.randperm(
        len(all_clips), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_clips = [all_clips[i] for i in perm[:n_train]]
    test_clips  = [all_clips[i] for i in perm[n_train:]]
    print(
        f"[manifest] clip split: {len(train_clips)} train / {len(test_clips)} test  "
        f"(seed={seed}, num_steps={num_steps})"
    )
    return train_clips, test_clips, num_steps


# ── 3.a — probe training ─────────────────────────────────────────────────────

class Probe(NamedTuple):
    ridge: Ridge
    latent_dim: int   # dim of q (= dim of p for LeHami models)
    uses_p: bool = True  # True → input is cat(q,p) ∈ ℝ^{2d}; False → q/z only ∈ ℝ^d


def train_probe(
    model,
    cfg,
    h5_path: Path,
    train_clips: list[tuple[int, int]],
    num_steps: int,
    device: str,
    enc_batch: int = 50,
    max_clips: int | None = None,
) -> Probe:
    """Fit Ridge(cat(q,p) → [sinθ, cosθ, ω]) on all frame pairs from train clips.

    For each clip (ep, start) of length ``num_steps``, extracts
    ``num_steps - 1`` samples at local frame indices i = 1, …, num_steps - 1:

        input  : cat(q_i, p_i) ∈ ℝ^{2·latent_dim}   (LeHami)
                 or  q_i/z_i ∈ ℝ^{latent_dim}         (LeWM / FD-only)
        target : [sin θ_i, cos θ_i, ω_i]
                  θ in radians (sinθ/cosθ are unitless), ω in rad/s

    Episodes are loaded once each regardless of how many of their clips
    appear in ``train_clips``, so H5 I/O is O(n_episodes), not O(n_clips).

    Parameters
    ----------
    train_clips : list of (ep_idx, start_frame_in_episode) pairs
    num_steps   : clip length (= history_size + num_preds, e.g. 4)
    """
    import h5py

    latent_dim = _latent_dim(cfg)
    img_size   = cfg.img_size
    mom_type   = _mom_type(cfg.model.name)
    fd_dt      = float(cfg.model.get("fd_dt", 1.0))
    has_p      = mom_type != "lewm"

    if max_clips is not None and max_clips < len(train_clips):
        n_orig = len(train_clips)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_orig, size=max_clips, replace=False)
        train_clips = [train_clips[i] for i in idx]
        print(f"[probe] subsampled to {max_clips}/{n_orig} train clips")

    vec_list:    list[np.ndarray]  = []
    target_list: list[np.ndarray]  = []
    ep_lengths_arr = _episode_lengths(h5_path)   # None unless the dataset varies L
    h_arr = _episode_dt_ratio(h5_path, cfg)      # None -> scalar (h=1) path

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]

        for clip_num, (ep, start) in enumerate(train_clips): ## ep refers to the episode and start to the starting point of the clip in that episode 
            if clip_num % 5000 == 0:
                print(f"  [probe] clip {clip_num}/{len(train_clips)}...", flush=True)

            # Load exactly the num_steps frames of this clip.
            abs_start  = int(ep_offsets[ep]) + start ## this is the absolute start index in the HDF5 dataset
            pixels_raw = f["pixels"][abs_start : abs_start + num_steps]
            states_raw = f["state"][abs_start : abs_start + num_steps]  # (num_steps, 2)

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            cls, q_enc = _encode_episode(model, pixels, enc_batch)
            # local indices 0..n_lat-1 are LATENT time (== clip frames for a
            # per-frame encoder, W-1 shorter for a temporal one); states are
            # re-based to match.
            states_raw = _align_states(model, states_raw)
            n_lat = q_enc.shape[0]
            # Vectorized over frames i = 1..n_lat-1 — per-frame
            # cat/.cpu()/.numpy() calls fragment the heap (~10 MB/episode leak,
            # OOM-killed the eval on 450×100-frame runs).
            if has_p:
                p_seq = _p_series(q_enc, mom_type, fd_dt)[1:n_lat]
                if h_arr is not None:
                    p_seq = p_seq / float(h_arr[ep])
                vecs = torch.cat([q_enc[1:n_lat], p_seq], dim=-1).cpu().numpy()
            else:
                vecs = q_enc[1:n_lat].cpu().numpy()

            theta = states_raw[1:n_lat, 0].astype(np.float64)  # rad
            omega = states_raw[1:n_lat, 1].astype(np.float64)  # rad/s
            vec_list.append(vecs)
            tgt = [np.sin(theta), np.cos(theta), omega]
            if ep_lengths_arr is not None:            # 4th channel: rod length L (m)
                tgt.append(np.full_like(theta, ep_lengths_arr[ep]))
            target_list.append(np.stack(tgt, axis=1))

    X = np.concatenate(vec_list)      # (N, 2·d) or (N, d)
    Y = np.concatenate(target_list)   # (N, 3) — [sinθ, cosθ, ω (rad/s)]
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X, Y)
    r2_train = r2_score(Y, ridge.predict(X))
    print(
        f"[probe] fit on {len(X)} samples  "
        f"({len(train_clips)} clips × {num_steps - 1} frames/clip)  "
        f"input ∈ ℝ^{X.shape[1]}  target: [sinθ, cosθ, ω (rad/s)]  "
        f"train R²={r2_train:.4f}"
    )
    return Probe(ridge=ridge, latent_dim=latent_dim, uses_p=has_p)


def _probe_targets_block(states_block: np.ndarray,
                         ep_length: float | None = None) -> np.ndarray:
    """(T, state_dim) raw states -> (T, n_targets) probe targets, keyed on width.

    Thin wrapper kept for the callers that already import this name; the rule now
    lives in `probe_targets.py`, a leaf module both this file and
    `train_probe_physics` import, so the per-frame and per-block forms can no
    longer drift apart. See that module's docstring for the rule itself.
    """
    return probe_targets(states_block, ep_length)


def _collect_probe_episode_data(
    model, cfg, h5_path: Path, episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (cat(q,p), [sinθ, cosθ, ω]) pairs from GT frames of episodes.

    For each episode, uses frames i=1..min(train_frames, ep_len)-1.
    Consistent with the episode_srnn training protocol: same frame range and
    same (q_i, p_i) computation as the training forward pass. This is the raw
    data collector shared by the GT-anchor probes (Ridge here, KNN in
    train_probe_physics.py) — encoding is the expensive part, so both probe
    kinds are fit from a single pass over the episodes.

    Returns:
        X: (N, 2·latent_dim) or (N, latent_dim) — probe inputs
        Y: (N, 3)                                — [sinθ, cosθ, ω]
    """
    import h5py

    img_size   = cfg.img_size
    mom_type   = _mom_type(cfg.model.name)
    fd_dt      = float(cfg.model.get("fd_dt", 1.0))
    has_p      = mom_type != "lewm"

    vec_list:    list[np.ndarray]  = []
    target_list: list[np.ndarray]  = []
    ep_lengths_arr = _episode_lengths(h5_path)   # None unless the dataset varies L
    h_arr = _episode_dt_ratio(h5_path, cfg)      # None -> scalar (h=1) path

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for ep_num, ep in enumerate(episodes):
            if ep_num % 100 == 0:
                print(f"  [probe] episode {ep_num}/{len(episodes)}...", flush=True)

            start    = int(ep_offsets[ep])
            L        = int(ep_lengths[ep])
            n_frames = min(L, train_frames) if train_frames else L

            pixels_raw = f["pixels"][start : start + n_frames]
            states_raw = f["state"][start : start + n_frames]

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            cls, q_enc = _encode_episode(model, pixels, enc_batch)
            states_raw = _align_states(model, states_raw)
            n_lat = q_enc.shape[0]      # == n_frames for a per-frame encoder
            # Vectorized over frames i = 1..n_lat-1 — per-frame
            # cat/.cpu()/.numpy() calls fragment the heap (~10 MB/episode leak,
            # OOM-killed the eval on 450×100-frame runs).
            if has_p:
                p_seq = _p_series(q_enc, mom_type, fd_dt)[1:n_lat]
                if h_arr is not None:
                    p_seq = p_seq / float(h_arr[ep])
                vecs = torch.cat([q_enc[1:n_lat], p_seq], dim=-1).cpu().numpy()
            else:
                vecs = q_enc[1:n_lat].cpu().numpy()

            vec_list.append(vecs)
            target_list.append(_probe_targets_block(
                states_raw[1:n_lat],
                None if ep_lengths_arr is None else ep_lengths_arr[ep],
            ))

    X = np.concatenate(vec_list)
    Y = np.concatenate(target_list)
    return X, Y


def _train_probe_episodes(
    model, cfg, h5_path: Path, train_episodes: list[int],
    device: str, enc_batch: int = 50, train_frames: int | None = None,
) -> Probe:
    """Fit Ridge(cat(q,p) → [sinθ, cosθ, ω]) on frames from train episodes."""
    latent_dim = _latent_dim(cfg)
    has_p      = _mom_type(cfg.model.name) != "lewm"

    X, Y = _collect_probe_episode_data(
        model, cfg, h5_path, train_episodes, device, enc_batch, train_frames,
    )
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X, Y)
    r2_train = r2_score(Y, ridge.predict(X))
    print(
        f"[probe] fit on {len(X)} samples from {len(train_episodes)} episodes  "
        f"input ∈ ℝ^{X.shape[1]}  target: [sinθ, cosθ, ω]  train R²={r2_train:.4f}"
    )
    return Probe(ridge=ridge, latent_dim=latent_dim, uses_p=has_p)


# ── energy stratification ─────────────────────────────────────────────────────

def _episode_energy_bins(
    h5_path: Path, episodes: list[int], n_bins: int = 3,
) -> tuple[dict[str, list[int]], list[float], list[str]]:
    """Partition episodes into energy bins by max |ω| over the episode (energy proxy).

    Bin edges are quantiles computed over `episodes`, so for a fixed manifest split
    every model gets identical bins (comparable across models). Returns
    (label → episode list, bin edges, ordered labels).
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
        proxy = {}
        for ep in episodes:
            s = int(ep_off[ep])
            proxy[ep] = float(np.abs(f["state"][s : s + int(ep_len[ep]), 1]).max())
    vals   = np.array([proxy[ep] for ep in episodes])
    edges  = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    labels = ["low", "mid", "high"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    idx    = np.clip(np.digitize(vals, edges[1:-1]), 0, n_bins - 1)
    bins   = {lab: [ep for ep, i in zip(episodes, idx) if i == b]
              for b, lab in enumerate(labels)}
    return bins, [float(e) for e in edges], labels


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))   # union, first-seen order
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {path}  ({len(rows)} rows)")


# ── 3.b + 3.c — rollout ──────────────────────────────────────────────────────

def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


PRIMARY_PROBE = "ridge_gt"   # keeps the original unsuffixed column names

# Cache filename per probe kind, shared by the 3.a training section and the
# multiclip probe-loading section below.
PROBE_FILENAMES = {
    "ridge_gt": "probe.pkl",         # Ridge on GT (frozen-encoder) embeddings — original probe
    "knn_gt":   "probe_knn.pkl",     # KNN on GT embeddings
    "ridge_k1": "probe_k1_ridge.pkl",  # Ridge on the model's own k=1 rollout output
    "knn_k1":   "probe_k1_knn.pkl",    # KNN on the model's own k=1 rollout output
    "mlp_gt":   "probe_mlp.pkl",       # MLP on GT embeddings — nonlinear readout tier
    "mlp_k1":   "probe_k1_mlp.pkl",    # MLP on the model's own k=1 rollout output
}


def _apply_probes_standard(
    probes: dict[str, Probe], probe_input: np.ndarray,
    theta_gt: np.ndarray, omega_gt: np.ndarray,
    length_gt: np.ndarray | None = None,
) -> dict:
    """Apply every probe in `probes` to the same probe_input, merging results
    into one row dict. `PRIMARY_PROBE`'s metrics get the original unsuffixed
    column names (mae_theta, r2_theta, mae_omega, r2_omega); every other key
    gets `<name>__<key>` columns. Matches evaluate_rollout / evaluate_rollout_lewm's
    wrapped-angle-error convention exactly (see _multiclip_compute_metrics for
    the distinct multiclip convention, kept separate on purpose).
    """
    row: dict = {}
    for key, probe in probes.items():
        pred = probe.ridge.predict(probe_input)
        theta_pred = np.arctan2(pred[:, 0], pred[:, 1])
        omega_pred = pred[:, 2]
        delta_theta = np.arctan2(np.sin(theta_pred - theta_gt), np.cos(theta_pred - theta_gt))
        suffix = "" if key == PRIMARY_PROBE else f"__{key}"
        row[f"mae_theta{suffix}"] = float(np.mean(np.abs(delta_theta)))
        row[f"r2_theta{suffix}"]  = float(r2_score(theta_gt, theta_pred))
        row[f"mae_omega{suffix}"] = float(np.mean(np.abs(omega_pred - omega_gt)))
        row[f"r2_omega{suffix}"]  = float(r2_score(omega_gt, omega_pred))
        # Rod length L. At k=0 this is near-trivial: the rod is rendered to
        # scale, so L is readable from a single frame and any non-collapsed
        # encoder scores ~1. The informative reading is r2_length vs k — L is a
        # PARAMETER, so a frozen probe on the predicted q_k must hold it flat.
        # Decay means the flow is leaking a conserved parameter into the state.
        if length_gt is not None and _probe_has_length(probe):
            len_pred = pred[:, 3]
            row[f"mae_length{suffix}"] = float(np.mean(np.abs(len_pred - length_gt)))
            row[f"r2_length{suffix}"]  = (float(r2_score(length_gt, len_pred))
                                          if np.ptp(length_gt) > 0 else float("nan"))
        if key == PRIMARY_PROBE:
            # The two angle channels scored separately, before atan2 collapses them.
            # `mae_theta` is the wrapped error of the *combined* prediction and can hide
            # which channel is failing; these two do not wrap and are directly comparable
            # across models. Emitted to probe_metrics_sincos.csv by _save_results.
            row["mae_sin"] = float(np.mean(np.abs(pred[:, 0] - np.sin(theta_gt))))
            row["mae_cos"] = float(np.mean(np.abs(pred[:, 1] - np.cos(theta_gt))))
    return row


def _apply_probes_three_body(
    probes: dict[str, Probe], probe_input: np.ndarray, states: np.ndarray,
    n_bodies: int = 3,
) -> dict:
    """Multi-body counterpart of `_apply_probes_standard`, same suffix convention.

    `n_bodies` defaults to 3 so the three-body call site is unchanged: with 3 the
    split below is states[:, :6] / states[:, 6:] and the loop is range(3), exactly
    what this function did before the two-body dataset was added. Two-body passes 2.

    Position and velocity are reported as SEPARATE blocks and never pooled, matching
    `train_probe_physics._eval_probe_three_body`: they are the two clauses of the
    planning criterion, and the swing-up result turned entirely on their separation.
    Per-body columns are kept because the actuated body is the one planning scores.

    There is no angle here, so nothing wraps and no atan2 is taken. Applying the
    pendulum path to a 12-D state does not fail -- it reads q_0x, q_0y, q_1x as
    theta, omega and rod length, and writes plausible-looking numbers for all three.
    """
    row: dict = {}
    half = 2 * n_bodies                       # 6 for three-body, 4 for two-body
    q_gt, v_gt = states[:, :half], states[:, half:]
    for key, probe in probes.items():
        pred = probe.ridge.predict(probe_input)
        suffix = "" if key == PRIMARY_PROBE else f"__{key}"
        row[f"r2_pos{suffix}"]  = float(r2_score(q_gt, pred[:, :half],
                                                 multioutput="uniform_average"))
        row[f"r2_vel{suffix}"]  = float(r2_score(v_gt, pred[:, half:],
                                                 multioutput="uniform_average"))
        row[f"mae_pos{suffix}"] = float(np.abs(pred[:, :half] - q_gt).mean())
        row[f"mae_vel{suffix}"] = float(np.abs(pred[:, half:] - v_gt).mean())
        for b in range(n_bodies):
            pq_, pv_ = slice(2 * b, 2 * b + 2), slice(half + 2 * b, half + 2 * b + 2)
            row[f"r2_pos_body{b + 1}{suffix}"] = float(
                r2_score(states[:, pq_], pred[:, pq_], multioutput="uniform_average"))
            row[f"r2_vel_body{b + 1}{suffix}"] = float(
                r2_score(states[:, pv_], pred[:, pv_], multioutput="uniform_average"))
    return row


def _apply_probes(
    probes: dict[str, Probe], probe_input: np.ndarray, states: np.ndarray,
) -> dict:
    """Dispatch on the ground-truth state width, exactly as `_eval_probe` does.

    The two call sites used to unpack `states` themselves as (theta, omega, L), which
    is silently wrong for any environment whose state is not the pendulum's. Deriving
    the targets inside the branch keeps that decision in one place.
    """
    if states.shape[1] == 12:
        return _apply_probes_three_body(probes, probe_input, states, n_bodies=3)
    if states.shape[1] == 8:
        # Two-body (lehami/data/two_body.py). Same layout, one fewer body, so
        # the per-body columns come out as body1 (the centre) and body2 (the actuated
        # orbiter, ACTUATED_BODY = 1 zero-based). NOTE the numbering: three-body's
        # actuated body is body3 here, two-body's is body2.
        return _apply_probes_three_body(probes, probe_input, states, n_bodies=2)
    theta_gt = states[:, 0]
    omega_gt = states[:, 1]
    length_gt = states[:, 2] if states.shape[1] > 2 else None
    return _apply_probes_standard(probes, probe_input, theta_gt, omega_gt, length_gt)


def evaluate_rollout(
    model, cfg, h5_path: Path, episodes: list[int],
    probes: dict[str, Probe], horizon: int, device: str,
    split_name: str, enc_batch: int = 50, dump_path: Path | None = None,
    zero_action: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Roll out the predictor for each episode and compute all metrics.

    Args:
        zero_action: force the pre-2026-08-18 behaviour of driving the rollout with
            u = 0 at every step. The default (False) feeds the episode's TRUE
            per-frame action, which is what training did. On an unactuated dataset
            the two are identical, so every pre-swing-up number is unchanged; on the
            actuated swing-up data the old path scored a u = 0 rollout against a
            torque-driven ground truth, i.e. it answered "predict the future
            assuming no torque".
        dump_path: if given, ALSO write the per-k predicted latents and their [θ, ω]
            targets to this .npz. Off by default and inert when None — the metrics
            returned are byte-identical either way. It exists so a frozen probe of any
            kind (MLP, a different KNN, …) can be applied along the rollout LOCALLY,
            without a second GPU pass: these arrays live only inside this loop and are
            otherwise discarded once the per-k metrics are computed.

    Returns:
        rollout_rows: list of dicts with rollout metrics per k
        probe_rows:   list of dicts with probe metrics per k
    """
    import h5py
    h = cfg.history_size
    latent_dim = _latent_dim(cfg)
    img_size = cfg.img_size
    model_name = cfg.model.name
    mom_type = _mom_type(model_name)
    has_energy = _has_energy(model_name)
    has_p = (mom_type != "lewm")

    u_zero = _zero_action_emb(model, cfg, device)   # (1, phase_dim)
    lat2pix = _latent_to_pixel_offset(model)

    # accumulators: per k → list of (pred_q, gt_q, pred_p, gt_p, states)
    per_k_pq: list[list] = [[] for _ in range(horizon + 1)]
    per_k_pp: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gq: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gp: list[list] = [[] for _ in range(horizon + 1)]
    per_k_en: list[list] = [[] for _ in range(horizon + 1)]  # energy
    per_k_st: list[list] = [[] for _ in range(horizon + 1)]  # states [θ, ω]

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        a = _anchor_frame(cfg, mom_type)   # episode frame index of the anchor
        ep_lengths_arr = _episode_lengths(h5_path)
        h_arr = _episode_dt_ratio(h5_path, cfg)   # None -> scalar (h=1) path

        for ep_num, ep in enumerate(episodes):
            if ep_num % 50 == 0:
                print(f"  [{split_name}] episode {ep_num}/{len(episodes)}...", flush=True)

            start, L = int(ep_offsets[ep]), int(ep_lengths[ep])
            # `a` and every frame index below are in LATENT time; a temporal
            # encoder turns L pixel frames into L-W+1 of them.
            n_lat = L - _enc_window(model) + 1
            max_k = n_lat - a - 1   # frames available after the anchor
            k_range = min(horizon, max_k)
            if k_range < 1:
                continue

            pixels_raw = f["pixels"][start : start + L]
            states_raw = f["state"][start : start + L]
            ep_actions = None if zero_action else _episode_actions(f, start, L, cfg, device)

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            cls, q_enc = _encode_episode(model, pixels, enc_batch)
            states_raw = _with_length(_align_states(model, states_raw), ep_lengths_arr, ep)

            fd_dt = float(cfg.model.get("fd_dt", 1.0))

            # Anchor (q_a, p_a) and the reference p at every frame, from the SG stencils.
            q0, p0 = _build_anchor(model, cls, q_enc, mom_type, cfg, device)
            p_gt_all = _p_series(q_enc, mom_type, fd_dt)
            if h_arr is not None:                 # put p on the episode clock
                p0 = p0 / float(h_arr[ep])
                p_gt_all = p_gt_all / float(h_arr[ep])

            def _gt_p(fi: int) -> np.ndarray:
                return p_gt_all[fi].cpu().numpy()

            # k=0: anchor — GT is the anchor frame itself
            per_k_pq[0].append(q0.cpu().numpy())
            per_k_gq[0].append(q_enc[a].cpu().numpy())
            per_k_st[0].append(states_raw[a])
            if has_p:
                per_k_pp[0].append(p0.cpu().numpy())
                per_k_gp[0].append(p0.cpu().numpy())  # p0 IS the GT at k=0
                if has_energy:
                    H0 = model.predictor.flow.eval_H(
                        q0.unsqueeze(0), p0.unsqueeze(0),
                        _act_emb_at(model, ep_actions, u_zero, a, lat2pix)).item()
                    per_k_en[0].append(H0)

            # k=1..k_range: autoregressive rollout
            qp = torch.cat([q0, p0], dim=-1).unsqueeze(0)  # (1, phase_dim) for lehami
            for k in range(1, k_range + 1):
                frame_idx = a + k   # GT frame at rollout step k
                if mom_type == "lewm":
                    # lewm uses ARPredictor — not (q,p) space; handle separately below
                    break

                # The transition INTO latent frame a+k is driven by the action at
                # latent frame a+k-1 -- the frame being stepped OUT of. Same
                # convention as builders._forward; an off-by-one here is a half-step
                # phase error that no metric would flag.
                u_k = _act_emb_at(model, ep_actions, u_zero, frame_idx - 1, lat2pix)
                qp = model.predictor(
                    qp.unsqueeze(1), u_k.unsqueeze(1),
                    None if h_arr is None
                    else torch.full((1,), float(h_arr[ep]), device=device),
                )[:, 0]  # (1, phase_dim)
                q_pred_k = qp[0, :latent_dim]
                p_pred_k = qp[0, latent_dim:]

                per_k_pq[k].append(q_pred_k.detach().cpu().numpy())
                per_k_gq[k].append(q_enc[frame_idx].cpu().numpy())
                per_k_st[k].append(states_raw[frame_idx])
                if has_p:
                    per_k_pp[k].append(p_pred_k.detach().cpu().numpy())
                    per_k_gp[k].append(_gt_p(frame_idx))
                    if has_energy:
                        H_k = model.predictor.flow.eval_H(
                            q_pred_k.unsqueeze(0), p_pred_k.unsqueeze(0), u_k
                        ).item()
                        per_k_en[k].append(H_k)

    # ── optional: persist the per-k latents before they are reduced to metrics ──
    if dump_path is not None:
        blob: dict[str, np.ndarray] = {}
        ks_dumped = []
        for k in range(horizon + 1):
            if not per_k_pq[k]:
                continue
            ks_dumped.append(k)
            blob[f"pred_q_k{k}"] = np.stack(per_k_pq[k]).astype(np.float32)
            blob[f"gt_q_k{k}"] = np.stack(per_k_gq[k]).astype(np.float32)
            if has_p and per_k_pp[k]:
                blob[f"pred_p_k{k}"] = np.stack(per_k_pp[k]).astype(np.float32)
                blob[f"gt_p_k{k}"] = np.stack(per_k_gp[k]).astype(np.float32)
            if per_k_st[k]:
                # (N, 2) = [theta rad, omega rad/s] — the probe targets at this k —
                # or (N, 3) with the rod length L appended on a dataset that varies it
                # (_with_length). `state_cols` below names them so a consumer never has
                # to infer the layout from the width.
                blob[f"states_k{k}"] = np.stack(per_k_st[k]).astype(np.float32)
        blob["ks"] = np.array(ks_dumped)
        blob["split"] = np.array(split_name)
        blob["latent_dim"] = np.array(latent_dim)
        blob["uses_p"] = np.array(has_p)
        blob["model_name"] = np.array(str(model_name))
        n_state = blob[f"states_k{ks_dumped[0]}"].shape[1] if ks_dumped else 2
        blob["state_cols"] = np.array(state_names(n_state))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **blob)
        print(f"[dump] per-k latents -> {dump_path} "
              f"({dump_path.stat().st_size / 1e6:.1f} MB, k={ks_dumped[0]}..{ks_dumped[-1]}, "
              f"N={blob[f'pred_q_k{ks_dumped[0]}'].shape[0]} clips)")

    # ── assemble per-k metrics ─────────────────────────────────────────────────
    rollout_rows, probe_rows = [], []
    key_horizons = sorted(set([0, 1, 5, 10, 20, 50, horizon]))

    for k in range(horizon + 1):
        if not per_k_pq[k]:
            continue
        pq = np.stack(per_k_pq[k])   # (N, latent_dim)
        gq = np.stack(per_k_gq[k])

        # rollout metrics: q
        mse_q = float(np.mean((pq - gq) ** 2))
        rmse_norm_q = math.sqrt(mse_q)
        r2_q = _r2(pq, gq)

        row_r = {"split": split_name, "k": k,
                 "rmse_norm_q": rmse_norm_q, "r2_q": r2_q,
                 "energy_mean": float("nan"), "energy_std": float("nan")}

        if has_p and per_k_pp[k]:
            pp = np.stack(per_k_pp[k])
            gp = np.stack(per_k_gp[k])
            mse_p = float(np.mean((pp - gp) ** 2))
            row_r["rmse_norm_p"] = math.sqrt(mse_p)
            row_r["r2_p"] = _r2(pp, gp)

        if has_energy and per_k_en[k]:
            en = np.array(per_k_en[k])
            row_r["energy_mean"] = float(np.mean(en))
            row_r["energy_std"]  = float(np.std(en))

        rollout_rows.append(row_r)

        # probe metrics: apply frozen probe(s) to predicted (q, p)
        states = np.stack(per_k_st[k])      # (N, 2) — [θ, ω] (+L when varied); (N, 12) three-body

        # probe input: cat(q,p) when has_p, else q only (consistent with train_probe)
        if has_p and per_k_pp[k]:
            pp_k = np.stack(per_k_pp[k])
            probe_input = np.concatenate([pq, pp_k], axis=-1)  # (N, 2·d)
        else:
            probe_input = pq                                     # (N, d)

        probe_rows.append({
            "split": split_name, "k": k,
            **_apply_probes(probes, probe_input, states),
        })

    return rollout_rows, probe_rows


# ── lewm rollout (ARPredictor path) ──────────────────────────────────────────

@torch.no_grad()
def _encode_z_lewm(model, pixels_chw: torch.Tensor, enc_batch: int = 50) -> torch.Tensor:
    """Encode frames through LeWM's full pipeline: encoder → projector.

    Returns (L-W+1, embed_dim) in latent time — (L, embed_dim) for the
    per-frame encoders, i.e. W = 1.  See `_encode_episode` / `_align_states`.
    """
    L = pixels_chw.shape[0]
    if _enc_window(model) > 1:
        return _encode_episode(model, pixels_chw, enc_batch)[1]
    is_plain = isinstance(model.encoder, _PLAIN_TENSOR_ENCODERS)
    z_list = []
    for s in range(0, L, enc_batch):
        chunk = pixels_chw[s:s + enc_batch]
        if is_plain:
            cls = model.encoder(chunk.float())   # (B, embed_dim) directly, no CLS token
        else:
            out = model.encoder(chunk.float(), interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
        z_list.append(model.projector(cls))
    return torch.cat(z_list)


def evaluate_rollout_lewm(
    model, cfg, h5_path: Path, episodes: list[int],
    probes: dict[str, Probe], horizon: int, device: str,
    split_name: str, enc_batch: int = 50, zero_action: bool = False,
    dump_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Rollout evaluation for LeWM (ARPredictor — no (q,p) split, no energy).

    `zero_action` forces the pre-2026-08-18 u = 0 rollout; see `evaluate_rollout`.
    `dump_path` persists the per-k predicted latents exactly as `evaluate_rollout`
    does, under the same key names (`pred_q_k*`, `gt_q_k*`, `states_k*`, with
    `uses_p=False`), so one consumer reads either arm's dump without branching.
    """
    import h5py
    h = cfg.history_size
    embed_dim = cfg.embed_dim
    img_size = cfg.img_size
    latent_dim = _latent_dim(cfg)

    per_k_pz: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gz: list[list] = [[] for _ in range(horizon + 1)]
    per_k_st: list[list] = [[] for _ in range(horizon + 1)]

    u_emb_base = torch.zeros(1, h, cfg.data.action_dim, device=device)
    act_emb_base = model.action_encoder(u_emb_base)  # (1, h, embed_dim) — zero fallback
    lat2pix = _latent_to_pixel_offset(model)

    ep_lengths_arr = _episode_lengths(h5_path)

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for ep_num, ep in enumerate(episodes):
            if ep_num % 50 == 0:
                print(f"  [{split_name}/lewm] episode {ep_num}/{len(episodes)}...", flush=True)

            start, L = int(ep_offsets[ep]), int(ep_lengths[ep])
            # h and every frame index below are in LATENT time (see _enc_shift).
            n_lat = L - _enc_window(model) + 1
            k_range = min(horizon, n_lat - h)
            if k_range < 1:
                continue

            pixels_raw = f["pixels"][start : start + L]
            states_raw = f["state"][start : start + L]
            ep_actions = None if zero_action else _episode_actions(f, start, L, cfg, device)
            pixels = _preprocess_batch(pixels_raw, img_size, device)
            z_enc = _encode_z_lewm(model, pixels, enc_batch)  # (n_lat, embed_dim)
            states_raw = _with_length(_align_states(model, states_raw), ep_lengths_arr, ep)

            # k=0: anchor is last context embedding
            z_anchor = z_enc[h - 1]
            per_k_pz[0].append(z_anchor.cpu().numpy())
            per_k_gz[0].append(z_enc[h - 1].cpu().numpy())
            per_k_st[0].append(states_raw[h - 1])

            # rolling buffer of the last h embeddings
            buf = z_enc[:h].unsqueeze(0)  # (1, h, embed_dim)
            for k in range(1, k_range + 1):
                frame_idx = h - 1 + k
                # Context is latent frames [k-1, k+h-2]; training pairs exactly that
                # window with the actions at the same latent frames (builders, step=k-1).
                act_k = _act_emb_window(model, ep_actions, act_emb_base, k - 1, h, lat2pix)
                z_pred = model.predict(buf, act_k)[:, -1]  # (1, embed_dim) — full context, with pred_proj
                buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1)

                per_k_pz[k].append(z_pred[0].detach().cpu().numpy())
                per_k_gz[k].append(z_enc[frame_idx].cpu().numpy())
                per_k_st[k].append(states_raw[frame_idx])

    # ── optional: persist the per-k latents before they are reduced to metrics ──
    if dump_path is not None:
        blob: dict[str, np.ndarray] = {}
        ks_dumped = []
        for k in range(horizon + 1):
            if not per_k_pz[k]:
                continue
            ks_dumped.append(k)
            # `pred_q` / `gt_q`, not `pred_z`: LeWM's z plays the role q does for the
            # Hamiltonian arms, and matching key names keeps consumers arm-agnostic.
            blob[f"pred_q_k{k}"] = np.stack(per_k_pz[k]).astype(np.float32)
            blob[f"gt_q_k{k}"] = np.stack(per_k_gz[k]).astype(np.float32)
            if per_k_st[k]:
                blob[f"states_k{k}"] = np.stack(per_k_st[k]).astype(np.float32)
        blob["ks"] = np.array(ks_dumped)
        blob["split"] = np.array(split_name)
        blob["latent_dim"] = np.array(latent_dim)
        blob["uses_p"] = np.array(False)
        blob["model_name"] = np.array(str(cfg.model.name))
        n_state = blob[f"states_k{ks_dumped[0]}"].shape[1] if ks_dumped else 2
        blob["state_cols"] = np.array(state_names(n_state))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **blob)
        print(f"[dump] per-k latents -> {dump_path} "
              f"({dump_path.stat().st_size / 1e6:.1f} MB, k={ks_dumped[0]}..{ks_dumped[-1]}, "
              f"N={blob[f'pred_q_k{ks_dumped[0]}'].shape[0]} episodes)")

    # ── assemble metrics ───────────────────────────────────────────────────────
    rollout_rows, probe_rows = [], []
    for k in range(horizon + 1):
        if not per_k_pz[k]:
            continue
        pz = np.stack(per_k_pz[k])
        gz = np.stack(per_k_gz[k])
        mse_z = float(np.mean((pz - gz) ** 2))
        rmse_norm_q = math.sqrt(mse_z)
        r2_q = _r2(pz, gz)
        rollout_rows.append({
            "split": split_name, "k": k,
            "rmse_norm_q": rmse_norm_q, "r2_q": r2_q,
            "energy_mean": float("nan"), "energy_std": float("nan"),
        })

        states = np.stack(per_k_st[k])
        probe_rows.append({
            "split": split_name, "k": k,
            **_apply_probes(probes, pz, states),
        })
    return rollout_rows, probe_rows


# ── clip-based rollout (README point 2) ───────────────────────────────────────

def _assert_scalar_dt_ok(cfg, mode: str) -> None:
    """Refuse to score a per-episode-dt model in a mode that ignores dt.

    Silently integrating a dt_ref model at h = 1 advances every off-reference
    episode by the wrong amount, which looks like a modelling failure and is
    not one. Loud beats subtly wrong.
    """
    if _cfg_get(cfg, "model.dt_ref") is not None:
        raise NotImplementedError(
            f"model.dt_ref is set (per-episode integrator step), but the '{mode}' "
            "eval path still uses the scalar step. Use --mode standard with an "
            "episode manifest, which is dt-aware.")


@torch.no_grad()
def evaluate_rollout_clips(
    model, cfg, h5_path: Path, test_clips: list[tuple[int, int]],
    probe: Probe, horizon: int, device: str,
    n_eval: int = 10_000, rng_seed: int = 0, enc_batch: int = 50,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Clip-based rollout on a random subset of test clips (README §Evaluation point 2).

    Anchor = (q_1, p_1) computed from the first two frames of each selected clip
    (local frames 0 and 1).  Ground truth at rollout step k is episode frame
    ``start + 1 + k``, where ``start`` is the clip's starting frame in the episode.

    Returns:
        rollout_rows : per-k metrics (q/p RMSE/√d, R², energy) — split='test'
        probe_rows   : per-k probe metrics (θ MAE rad, ω MAE rad/s) — split='test'
        eval_clips   : the sampled (ep, start) pairs in selection order (save to
                       reproduce the exact subset)
    """
    import h5py

    _assert_scalar_dt_ok(cfg, "clip-rollout")
    latent_dim = _latent_dim(cfg)
    img_size   = cfg.img_size
    model_name = cfg.model.name
    mom_type   = _mom_type(model_name)
    has_energy = _has_energy(model_name)
    has_p      = mom_type != "lewm"
    fd_dt      = float(cfg.model.get("fd_dt", 1.0))

    u_emb = _zero_action_emb(model, cfg, device)   # (1, phase_dim)

    # ── sample test clips ────────────────────────────────────────────────────
    rng      = np.random.default_rng(rng_seed)
    n_sample = min(n_eval, len(test_clips))
    sel_idx  = rng.choice(len(test_clips), size=n_sample, replace=False)
    eval_clips = [test_clips[i] for i in sel_idx]
    print(f"[rollout-clips] sampled {n_sample}/{len(test_clips)} test clips "
          f"(rng_seed={rng_seed})")

    # ── per-k accumulators ───────────────────────────────────────────────────
    per_k_pq: list[list] = [[] for _ in range(horizon + 1)]
    per_k_pp: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gq: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gp: list[list] = [[] for _ in range(horizon + 1)]
    per_k_en: list[list] = [[] for _ in range(horizon + 1)]
    per_k_st: list[list] = [[] for _ in range(horizon + 1)]   # (θ rad, ω rad/s)

    ep_lengths_arr = _episode_lengths(h5_path)

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for clip_num, (ep, start) in enumerate(eval_clips):
            if clip_num % 1000 == 0:
                print(f"  [rollout-clips] {clip_num}/{n_sample}...", flush=True)

            L         = int(ep_lengths[ep])
            ep_start  = int(ep_offsets[ep])
            # anchor is at LATENT index 1 of the clip (= clip frame 1 when W=1)
            anchor_ep = start + 1
            # a temporal encoder needs W-1 extra pixel frames to emit the same
            # number of latents; they are taken from the tail (never from before
            # the clip, which may be at the start of the episode).
            lead      = _enc_window(model) - 1
            k_max     = min(horizon, L - anchor_ep - 1 - lead)
            if k_max < 1:
                continue

            # Load frames: clip frame 0 through anchor + k_max (inclusive), + lead.
            # Local LATENT index i maps to episode frame start+i (+ the encoder
            # shift, which _align_states applies to the GT states).
            abs_s   = ep_start + start
            abs_e   = ep_start + anchor_ep + k_max + 1 + lead   # exclusive
            pixels_raw = f["pixels"][abs_s : abs_e]      # (2+k_max+lead, H, W, C)
            states_raw = f["state"] [abs_s : abs_e]      # (2+k_max+lead, 2) — [θ, ω]

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            cls, q_enc = _encode_episode(model, pixels, enc_batch)
            states_raw = _with_length(_align_states(model, states_raw), ep_lengths_arr, ep)
            # local 0 = clip frame 0, local 1 = clip frame 1 (anchor), local 1+k = GT at step k

            # ── anchor (q_1, p_1) from local frames 0 and 1 ──────────────────
            q0 = q_enc[1].detach()
            if has_p:
                p0 = ((q_enc[1] - q_enc[0]) / fd_dt).detach()
            else:
                p0 = torch.zeros_like(q0)

            # ── k=0: anchor ───────────────────────────────────────────────────
            per_k_pq[0].append(q0.cpu().numpy())
            per_k_gq[0].append(q_enc[1].cpu().numpy())
            per_k_st[0].append(states_raw[1])
            if has_p:
                per_k_pp[0].append(p0.cpu().numpy())
                per_k_gp[0].append(p0.cpu().numpy())   # GT p at k=0 is the anchor p
                if has_energy:
                    H0 = model.predictor.flow.eval_H(
                        q0.unsqueeze(0), p0.unsqueeze(0), u_emb).item()
                    per_k_en[0].append(H0)

            # ── k=1..k_max: autoregressive rollout ───────────────────────────
            qp = torch.cat([q0, p0]).unsqueeze(0)   # (1, phase_dim)
            for k in range(1, k_max + 1):
                local_gt = 1 + k     # local index of GT frame (= episode frame start+1+k)
                qp = model.predictor(qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0]
                q_pred_k = qp[0, :latent_dim]
                p_pred_k = qp[0, latent_dim:]

                per_k_pq[k].append(q_pred_k.detach().cpu().numpy())
                per_k_gq[k].append(q_enc[local_gt].cpu().numpy())
                per_k_st[k].append(states_raw[local_gt])
                if has_p:
                    per_k_pp[k].append(p_pred_k.detach().cpu().numpy())
                    p_gt_k = ((q_enc[local_gt] - q_enc[local_gt - 1]) / fd_dt).detach()
                    per_k_gp[k].append(p_gt_k.cpu().numpy())
                    if has_energy:
                        H_k = model.predictor.flow.eval_H(
                            q_pred_k.unsqueeze(0), p_pred_k.unsqueeze(0), u_emb).item()
                        per_k_en[k].append(H_k)

    # ── assemble per-k metrics ────────────────────────────────────────────────
    rollout_rows: list[dict] = []
    probe_rows:   list[dict] = []

    for k in range(horizon + 1):
        if not per_k_pq[k]:
            continue
        pq = np.stack(per_k_pq[k])   # (N, latent_dim)
        gq = np.stack(per_k_gq[k])

        mse_q       = float(np.mean((pq - gq) ** 2))
        rmse_norm_q = math.sqrt(mse_q)
        r2_q        = _r2(pq, gq)

        row_r: dict = {
            "split": "test", "k": k, "n_clips": len(per_k_pq[k]),
            "rmse_norm_q": rmse_norm_q, "r2_q": r2_q,
            "energy_mean": float("nan"), "energy_std": float("nan"),
        }
        if has_p and per_k_pp[k]:
            pp = np.stack(per_k_pp[k])
            gp = np.stack(per_k_gp[k])
            mse_p = float(np.mean((pp - gp) ** 2))
            row_r["rmse_norm_p"] = math.sqrt(mse_p)
            row_r["r2_p"]        = _r2(pp, gp)
        if has_energy and per_k_en[k]:
            en = np.array(per_k_en[k])
            row_r["energy_mean"] = float(np.mean(en))
            row_r["energy_std"]  = float(np.std(en))
        rollout_rows.append(row_r)

        # probe metrics: predicted (q,p) → predicted (θ, ω) vs GT states
        states   = np.stack(per_k_st[k])    # (N, 2) — [θ rad, ω rad/s]
        theta_gt = states[:, 0]
        omega_gt = states[:, 1]

        if probe.uses_p and per_k_pp[k]:
            pp_k        = np.stack(per_k_pp[k])
            probe_input = np.concatenate([pq, pp_k], axis=-1)
        else:
            probe_input = pq

        sincos_omega_pred = probe.ridge.predict(probe_input)   # (N, 3): [sinθ, cosθ, ω]
        theta_pred = np.arctan2(sincos_omega_pred[:, 0], sincos_omega_pred[:, 1])
        omega_pred = sincos_omega_pred[:, 2]

        delta_theta = np.arctan2(
            np.sin(theta_pred - theta_gt), np.cos(theta_pred - theta_gt)
        )
        probe_rows.append({
            "split": "test", "k": k, "n_clips": len(per_k_pq[k]),
            "mae_theta": float(np.mean(np.abs(delta_theta))),    # rad
            "r2_theta":  float(r2_score(theta_gt, theta_pred)),
            "mae_omega": float(np.mean(np.abs(omega_pred - omega_gt))),   # rad/s
            "r2_omega":  float(r2_score(omega_gt, omega_pred)),
        })

    return rollout_rows, probe_rows, eval_clips


# ── clip-based rollout for LeWM (rolling-buffer ARPredictor) ─────────────────

@torch.no_grad()
def evaluate_rollout_clips_lewm(
    model, cfg, h5_path: Path, test_clips: list[tuple[int, int]],
    probe: "Probe", horizon: int, device: str,
    n_eval: int = 10_000, rng_seed: int = 0, enc_batch: int = 50,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Clip-based rollout for LeWM (ARPredictor — rolling buffer, no (q,p) split).

    Anchor = z_{h-1} from the first h frames of each clip.
    GT at rollout step k is episode frame start + h - 1 + k.

    Returns:
        rollout_rows : per-k metrics (z RMSE/√d, R²) — split='test'
        probe_rows   : per-k probe metrics (θ MAE rad, ω MAE rad/s) — split='test'
        eval_clips   : the sampled (ep, start) pairs in selection order
    """
    import h5py

    h = cfg.history_size
    embed_dim = cfg.embed_dim
    img_size = cfg.img_size

    rng = np.random.default_rng(rng_seed)
    n_sample = min(n_eval, len(test_clips))
    sel_idx = rng.choice(len(test_clips), size=n_sample, replace=False)
    eval_clips = [test_clips[i] for i in sel_idx]
    print(f"[rollout-clips/lewm] sampled {n_sample}/{len(test_clips)} test clips "
          f"(rng_seed={rng_seed})")

    # zero-action context embedding: (1, h, embed_dim)
    u_base = torch.zeros(1, h, cfg.data.action_dim, device=device)
    act_emb_base = model.action_encoder(u_base)  # (1, h, embed_dim)

    per_k_pz: list[list] = [[] for _ in range(horizon + 1)]
    per_k_gz: list[list] = [[] for _ in range(horizon + 1)]
    per_k_st: list[list] = [[] for _ in range(horizon + 1)]

    ep_lengths_arr = _episode_lengths(h5_path)

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lengths = f["ep_len"][:]

        for clip_num, (ep, start) in enumerate(eval_clips):
            if clip_num % 1000 == 0:
                print(f"  [rollout-clips/lewm] {clip_num}/{n_sample}...", flush=True)

            L = int(ep_lengths[ep])
            ep_start = int(ep_offsets[ep])
            # local indices are LATENT time; a temporal encoder needs W-1 extra
            # pixel frames to emit h+k_max latents (taken from the tail).
            lead  = _enc_window(model) - 1
            k_max = min(horizon, L - start - h - lead)
            if k_max < 1:
                continue

            abs_s = ep_start + start
            abs_e = abs_s + h + k_max + lead  # h context latents + k_max GT latents
            pixels_raw = f["pixels"][abs_s:abs_e]
            states_raw = f["state"][abs_s:abs_e]

            pixels = _preprocess_batch(pixels_raw, img_size, device)
            z_enc = _encode_z_lewm(model, pixels, enc_batch)  # (h+k_max, embed_dim)
            states_raw = _with_length(_align_states(model, states_raw), ep_lengths_arr, ep)

            # k=0: anchor is the last context frame (local index h-1)
            per_k_pz[0].append(z_enc[h - 1].cpu().numpy())
            per_k_gz[0].append(z_enc[h - 1].cpu().numpy())
            per_k_st[0].append(states_raw[h - 1])

            buf = z_enc[:h].unsqueeze(0)    # (1, h, embed_dim) — projector space
            for k in range(1, k_max + 1):
                local_gt = h - 1 + k
                # model.predict applies pred_proj on top of the raw transformer,
                # matching the training pipeline. [:, -1] takes the prediction that
                # used the full h-frame context (causal transformer: position i
                # predicts i+1, so position h-1 is the last/best prediction).
                z_pred = model.predict(buf, act_emb_base)[:, -1]  # (1, embed_dim)
                buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1)

                per_k_pz[k].append(z_pred[0].detach().cpu().numpy())
                per_k_gz[k].append(z_enc[local_gt].cpu().numpy())
                per_k_st[k].append(states_raw[local_gt])

    rollout_rows: list[dict] = []
    probe_rows:   list[dict] = []

    for k in range(horizon + 1):
        if not per_k_pz[k]:
            continue
        pz = np.stack(per_k_pz[k])   # (N, embed_dim)
        gz = np.stack(per_k_gz[k])

        mse_z = float(np.mean((pz - gz) ** 2))
        rmse_norm_q = math.sqrt(mse_z)
        r2_q = _r2(pz, gz)
        rollout_rows.append({
            "split": "test", "k": k, "n_clips": len(per_k_pz[k]),
            "rmse_norm_q": rmse_norm_q, "r2_q": r2_q,
            "energy_mean": float("nan"), "energy_std": float("nan"),
        })

        states   = np.stack(per_k_st[k])        # (N, 2) — [θ rad, ω rad/s]
        theta_gt, omega_gt = states[:, 0], states[:, 1]
        sincos_omega_pred = probe.ridge.predict(pz)    # (N, 3): [sinθ, cosθ, ω rad/s]
        theta_pred = np.arctan2(sincos_omega_pred[:, 0], sincos_omega_pred[:, 1])
        omega_pred = sincos_omega_pred[:, 2]
        delta_theta = np.arctan2(np.sin(theta_pred - theta_gt), np.cos(theta_pred - theta_gt))
        row = {
            "split": "test", "k": k, "n_clips": len(per_k_pz[k]),
            "mae_theta": float(np.mean(np.abs(delta_theta))),     # rad
            "r2_theta":  float(r2_score(theta_gt, theta_pred)),
            "mae_omega": float(np.mean(np.abs(omega_pred - omega_gt))),    # rad/s
            "r2_omega":  float(r2_score(omega_gt, omega_pred)),
        }
        if states.shape[1] > 2 and _probe_has_length(probe):
            length_gt = states[:, 2]
            len_pred  = sincos_omega_pred[:, 3]
            row["mae_length"] = float(np.mean(np.abs(len_pred - length_gt)))
            row["r2_length"]  = (float(r2_score(length_gt, len_pred))
                                 if np.ptp(length_gt) > 0 else float("nan"))
        probe_rows.append(row)

    return rollout_rows, probe_rows, eval_clips


# ── multiclip mode: dense sliding-window rollout ─────────────────────────────

@torch.no_grad()
def _multiclip_rollout_clip(
    model,
    cls: torch.Tensor,
    q_enc: torch.Tensor,
    c: int,
    horizon: int,
    cfg,
    device: str,
    mtype: str,
    fd_dt: float,
    d: int,
    u_emb: torch.Tensor,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Autoregressive rollout for one clip starting at anchor index c+1.

    Returns list of (k, q_pred, p_pred, q_gt, p_gt) tuples (all CPU numpy, shape (d,)).
    k ranges from 0 to min(horizon, available_steps).
    """
    anchor = c + _anchor_offset(mtype, cfg)   # absolute episode frame used as anchor
    L = cls.shape[0]
    k_max = min(horizon, L - 1 - anchor)
    if k_max < 0:
        return []

    # Build anchor (q0, p0)
    q0 = q_enc[anchor]
    assert mtype == "sg", f"no anchor momentum for mom_type={mtype!r}"
    from lehami.derivative import CAUSAL_WINDOW, p_causal_last
    p0 = p_causal_last(q_enc[anchor - CAUSAL_WINDOW + 1 : anchor + 1], fd_dt)
    p_gt_all = _p_series(q_enc, mtype, fd_dt)

    def _pgt(frame: int) -> np.ndarray:
        return p_gt_all[frame].cpu().numpy()

    steps = [(0, q0.cpu().numpy(), p0.cpu().numpy(),
              q_enc[anchor].cpu().numpy(), _pgt(anchor))]

    qp = torch.cat([q0, p0], dim=-1).unsqueeze(0)   # (1, phase_dim)
    for k in range(1, k_max + 1):
        qp = model.predictor(qp.unsqueeze(1), u_emb.unsqueeze(1))[:, 0]
        steps.append((
            k,
            qp[0, :d].cpu().numpy(),
            qp[0, d:].cpu().numpy(),
            q_enc[anchor + k].cpu().numpy(),
            _pgt(anchor + k),
        ))
    return steps


def _multiclip_collect_episode(
    ep_idx: int,
    pixels_hwc: np.ndarray,
    state_ep: np.ndarray,
    cfg,
    model,
    device: str,
    horizon: int,
    mtype: str,
    fd_dt: float,
    d: int,
    img_size: int,
    enc_batch: int,
    eval_stride: int = 1,
) -> dict[int, list]:
    """Encode episode once, slide anchor window, return per-k result lists.

    eval_stride: step between consecutive anchor start positions (1 = every
    position). Purely an eval sampling-density choice — independent of the
    manifest's training clip_stride.

    Return dict: k → list of (q_pred, p_pred, q_gt, p_gt, H_phi, theta, omega)
    """
    # c, a_off and frame_abs are LATENT-time indices; a temporal encoder turns
    # L pixel frames into n_lat of them and re-bases the GT states.
    n_lat = pixels_hwc.shape[0] - _enc_window(model) + 1
    a_off = _anchor_offset(mtype, cfg)
    clip_len = horizon + 1 + a_off   # anchor history + anchor + horizon steps
    if n_lat < clip_len:
        return {}

    pixels = _preprocess_batch(pixels_hwc, img_size, device)
    cls, q_enc = _encode_episode(model, pixels, enc_batch)
    state_ep = _align_states(model, state_ep)
    u_emb = _zero_action_emb(model, cfg, device)
    has_H = _has_energy(cfg.model.name)

    per_k: dict[int, list] = defaultdict(list)

    n_clips = n_lat - clip_len + 1   # c = 0 .. n_lat - clip_len
    for c in range(0, n_clips, eval_stride):
        steps = _multiclip_rollout_clip(
            model, cls, q_enc, c, horizon, cfg, device, mtype, fd_dt, d, u_emb
        )
        for k, q_p, p_p, q_g, p_g in steps:
            # Hamiltonian energy of predicted state
            H_phi = float("nan")
            if has_H:
                with torch.no_grad():
                    q_t = torch.tensor(q_p, dtype=torch.float32, device=device).unsqueeze(0)
                    p_t = torch.tensor(p_p, dtype=torch.float32, device=device).unsqueeze(0)
                    H_phi = float(model.predictor.flow.eval_H(q_t, p_t, u_emb).item())

            # Ground-truth physics state at anchor + k = c + a_off + k
            frame_abs = c + a_off + k   # absolute index within the episode
            theta = float(state_ep[frame_abs, 0])
            omega = float(state_ep[frame_abs, 1])

            per_k[k].append((q_p, p_p, q_g, p_g, H_phi, theta, omega))

    return per_k


@torch.no_grad()
def _multiclip_collect_episode_lewm(
    ep_idx: int,
    pixels_hwc: np.ndarray,
    state_ep: np.ndarray,
    cfg,
    model,
    device: str,
    horizon: int,
    img_size: int,
    enc_batch: int,
    eval_stride: int = 1,
) -> dict[int, list]:
    """LeWM variant: encode episode once, slide an h-frame context window,
    roll out `horizon` steps with the ARPredictor's rolling buffer.

    Anchor for window starting at c is frame c+h-1; GT at step k is frame c+h-1+k.
    Items carry p=None and H=nan so _multiclip_compute_metrics skips those metrics.

    Return dict: k → list of (z_pred, None, z_gt, None, nan, theta, omega)
    """
    h = cfg.history_size
    # c, anchor and frame are LATENT-time indices (see _enc_shift).
    n_lat = pixels_hwc.shape[0] - _enc_window(model) + 1
    clip_len = h + horizon   # h context frames + horizon GT frames
    if n_lat < clip_len:
        return {}

    pixels = _preprocess_batch(pixels_hwc, img_size, device)
    z_enc = _encode_z_lewm(model, pixels, enc_batch)   # (n_lat, embed_dim)
    state_ep = _align_states(model, state_ep)

    u_base = torch.zeros(1, h, cfg.data.action_dim, device=device)
    act_emb_base = model.action_encoder(u_base)        # (1, h, embed_dim)

    per_k: dict[int, list] = defaultdict(list)

    n_clips = n_lat - clip_len + 1
    for c in range(0, n_clips, eval_stride):
        anchor = c + h - 1
        # k=0: anchor frame itself
        z_a = z_enc[anchor].cpu().numpy()
        per_k[0].append((z_a, None, z_a, None, float("nan"),
                         float(state_ep[anchor, 0]), float(state_ep[anchor, 1])))

        buf = z_enc[c : c + h].unsqueeze(0)            # (1, h, embed_dim)
        for k in range(1, horizon + 1):
            frame = anchor + k
            z_pred = model.predict(buf, act_emb_base)[:, -1]   # (1, embed_dim)
            buf = torch.cat([buf[:, 1:], z_pred.unsqueeze(1)], dim=1)
            per_k[k].append((
                z_pred[0].cpu().numpy(), None,
                z_enc[frame].cpu().numpy(), None, float("nan"),
                float(state_ep[frame, 0]), float(state_ep[frame, 1]),
            ))

    return per_k


def _multiclip_compute_metrics(
    per_k: dict[int, list],
    split: str,
    probes: dict[str, Probe] | None,
    mom_type: str = "sg",
) -> list[dict]:
    rows = []
    for k in sorted(per_k.keys()):
        items = per_k[k]
        has_p   = items[0][1] is not None   # LeWM items carry p=None
        q_preds = np.stack([x[0] for x in items])   # (N, d)
        q_gts   = np.stack([x[2] for x in items])
        H_vals  = np.array([x[4] for x in items])
        thetas  = np.array([x[5] for x in items])
        omegas  = np.array([x[6] for x in items])
        N, d = q_preds.shape

        rmse_q = float(np.sqrt(np.mean((q_preds - q_gts) ** 2)))
        r2_q   = float(r2_score(q_gts.reshape(-1), q_preds.reshape(-1)))

        H_finite = H_vals[np.isfinite(H_vals)]
        H_mean = float(H_finite.mean()) if len(H_finite) else float("nan")
        H_std  = float(H_finite.std())  if len(H_finite) else float("nan")

        row = {
            "split": split, "k": k,
            "n_clips": N,
            "rmse_norm_q": rmse_q / math.sqrt(d),
            "r2_q":        r2_q,
            "energy_mean": H_mean,
            "energy_std":  H_std,
        }
        if has_p:
            p_preds = np.stack([x[1] for x in items])
            p_gts   = np.stack([x[3] for x in items])
            rmse_p  = float(np.sqrt(np.mean((p_preds - p_gts) ** 2)))
            row["rmse_norm_p"] = rmse_p / math.sqrt(d)
            row["r2_p"] = float(r2_score(p_gts.reshape(-1), p_preds.reshape(-1)))
        rows.append(row)

        # probe metrics (in-line so we reuse thetas/omegas already collected)
        if probes:
            feat_p = feat_q = None   # lazily built, shared across probes at this k
            theta_gt = np.arctan2(np.sin(thetas), np.cos(thetas))
            for name, probe in probes.items():
                if probe.uses_p:
                    if feat_p is None:
                        feat_p = np.concatenate([q_preds, p_preds], axis=1)
                    feat = feat_p
                else:
                    if feat_q is None:
                        feat_q = q_preds
                    feat = feat_q
                pred_s = probe.ridge.predict(feat)        # (N, 3): [sinθ, cosθ, ω]
                if pred_s.shape[1] == 12:
                    # Unlike _apply_probes, this path has no three-body branch. Reading
                    # a 12-D target as [sinθ, cosθ, ω] does not fail, it silently scores
                    # q_0x, q_0y, q_1x as an angle and a velocity — so refuse instead.
                    raise NotImplementedError(
                        "multiclip probe metrics have no three-body branch; targets are "
                        "12-D (q in R^6, v in R^6), not [sinθ, cosθ, ω]. Use the standard "
                        "rollout path (_apply_probes), or add the branch here first.")
                theta_pred = np.arctan2(pred_s[:, 0], pred_s[:, 1])
                omega_pred = pred_s[:, 2]

                suffix = "" if name == PRIMARY_PROBE else f"__{name}"
                row[f"mae_theta{suffix}"] = float(np.mean(np.abs(theta_gt - theta_pred)))
                row[f"r2_theta{suffix}"]  = float(r2_score(theta_gt, theta_pred))
                row[f"mae_omega{suffix}"] = float(np.mean(np.abs(omegas - omega_pred)))
                row[f"r2_omega{suffix}"]  = float(r2_score(omegas, omega_pred))

    return rows


def run_multiclip(args, model, cfg) -> None:
    _assert_scalar_dt_ok(cfg, "multiclip")
    """Multiclip evaluation driver (--mode multiclip)."""
    import csv
    import h5py

    mtype = _mom_type(cfg.model.name)
    train_eps, test_eps = _load_manifest(args.manifest)
    h5      = _h5_path(cfg)
    d       = _latent_dim(cfg)
    img_sz  = cfg.img_size
    fd_dt   = float(cfg.model.get("fd_dt", 1.0))
    splits_to_run = [s.strip() for s in args.splits.split(",")]

    print(f"Model: {cfg.model.name}  d={d}  mtype={mtype}  device={args.device}")
    print(f"Horizon: {args.horizon}  h5: {h5}")

    out_dir = ROOT / "lehami" / "experiments" / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load probes if available ──────────────────────────────────────────────
    # --probe-path overrides only the primary (ridge_gt) probe's location; the
    # 3 extra probes (knn_gt, ridge_k1, knn_k1) are always looked up next to it
    # in this exp's own results dir (they're checkpoint-specific, not portable
    # across experiments the way --probe-path is used for ridge_gt).
    probes: dict[str, Probe] = {}
    for key, fname in PROBE_FILENAMES.items():
        p_path = args.probe_path if (key == "ridge_gt" and args.probe_path) else out_dir / fname
        if not p_path.exists():
            if key == "ridge_gt" and args.probe_path:
                raise SystemExit(f"--probe-path {p_path} does not exist.")
            continue
        with open(p_path, "rb") as f:
            loaded = pickle.load(f)
        expected_dim = 2 * d if loaded.uses_p else d
        if loaded.ridge.n_features_in_ != expected_dim:
            print(
                f"[{key}] probe at {p_path} expects input dim "
                f"{loaded.ridge.n_features_in_} but this model produces "
                f"{expected_dim} (latent_dim={d}, uses_p={loaded.uses_p}) — skipping."
            )
            continue
        probes[key] = loaded
        print(f"[{key}] probe loaded from {p_path}  uses_p={loaded.uses_p}")
    if not probes:
        print("No probe files found — skipping physics probe metrics.")

    # ── run per-split ─────────────────────────────────────────────────────────
    all_rows:   list[dict] = []
    strat_rows: list[dict] = []
    energy_bin_edges: dict[str, list[float]] = {}

    split_map = {"train": train_eps, "test": test_eps}
    for split_name in splits_to_run:
        eps = split_map[split_name]
        if args.max_episodes:
            eps = eps[:args.max_episodes]
        print(f"\n[{split_name}] {len(eps)} episodes")

        per_k_all: dict[int, list] = defaultdict(list)

        ep_bin: dict[int, str] = {}
        bin_meta = None
        if getattr(args, "energy_bins", 0) > 0:
            bins_map, edges, labels = _episode_energy_bins(h5, eps, args.energy_bins)
            ep_bin = {ep: lab for lab, ep_list in bins_map.items() for ep in ep_list}
            bin_meta = (bins_map, labels)
            energy_bin_edges[split_name] = edges
            print("  [energy-strat] bin edges (max|ω|): "
                  + ", ".join(f"{e:.2f}" for e in edges))
        per_k_bins: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

        with h5py.File(str(h5), "r") as hf:
            ep_offsets = hf["ep_offset"][:]
            ep_lengths = hf["ep_len"][:]

            for ep_num, ep_idx in enumerate(eps):
                if ep_num % 10 == 0:
                    print(f"  episode {ep_num}/{len(eps)}...", flush=True)
                off = int(ep_offsets[ep_idx])
                L   = int(ep_lengths[ep_idx])
                pixels_hwc = hf["pixels"][off : off + L]
                state_ep   = hf["state"][off : off + L]   # (L, 2)

                if mtype == "lewm":
                    ep_per_k = _multiclip_collect_episode_lewm(
                        ep_idx, pixels_hwc, state_ep,
                        cfg, model, args.device, args.horizon,
                        img_sz, args.enc_batch,
                        eval_stride=args.eval_stride,
                    )
                else:
                    ep_per_k = _multiclip_collect_episode(
                        ep_idx, pixels_hwc, state_ep,
                        cfg, model, args.device, args.horizon,
                        mtype, fd_dt, d, img_sz, args.enc_batch,
                        eval_stride=args.eval_stride,
                    )
                for k, items in ep_per_k.items():
                    per_k_all[k].extend(items)
                    if ep_bin:
                        per_k_bins[ep_bin[ep_idx]][k].extend(items)

        total_clips = len(per_k_all.get(0, []))
        print(f"  → {total_clips} clips collected at k=0")

        rows = _multiclip_compute_metrics(per_k_all, split_name, probes, mtype)
        all_rows.extend(rows)

        if bin_meta is not None:
            bins_map, labels = bin_meta
            for lab in labels:
                if not per_k_bins[lab]:
                    continue
                rows_b = _multiclip_compute_metrics(per_k_bins[lab], split_name, probes, mtype)
                for row in rows_b:
                    row["energy_bin"] = lab
                    row["n_episodes"] = len(bins_map[lab])
                strat_rows.extend(rows_b)

    # ── write CSV ─────────────────────────────────────────────────────────────
    if not all_rows:
        print("No results to write.")
        return

    fieldnames = list(all_rows[0].keys())
    out_csv = out_dir / "multiclip_rollout_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved → {out_csv}  ({len(all_rows)} rows)")

    if strat_rows:
        _write_rows_csv(out_dir / "multiclip_rollout_metrics_by_energy.csv", strat_rows)
        edges_path = out_dir / "multiclip_energy_bin_edges.json"
        edges_path.write_text(json.dumps(energy_bin_edges))
        print(f"[energy-strat] bin edges saved → {edges_path.name}")

    # ── quick summary ─────────────────────────────────────────────────────────
    print("\n--- Key horizons (test split) ---")
    test_rows = {r["k"]: r for r in all_rows if r["split"] == "test"}
    for k in [0, 1, 5, 10, 20, 30]:
        if k in test_rows:
            r = test_rows[k]
            line = f"k={k:2d}  n={r['n_clips']:5d}  r2_q={r['r2_q']:6.3f}"
            if "r2_p" in r:
                line += f"  r2_p={r['r2_p']:6.3f}"
            line += f"  rmse_norm_q={r['rmse_norm_q']:.4f}"
            if "mae_theta" in r:
                line += f"  mae_θ={r['mae_theta']:.3f}  r2_ω={r['r2_omega']:.3f}"
            print(line)


# ── save results ──────────────────────────────────────────────────────────────

def _save_results(
    exp_name: str, ckpt_path: Path, manifest_path: str, cfg,
    rollout_rows: list[dict], probe_rows: list[dict], horizon: int,
    out_root: Path, probes: dict[str, "Probe"],
    eval_clips: list[tuple[int, int]] | None = None,
) -> None:
    import csv
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(rows: list[dict], fname: str) -> None:
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
        with open(out_dir / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows)

    _write_csv(rollout_rows, "rollout_metrics.csv")
    _write_csv(probe_rows, "probe_metrics.csv")

    # The sin/cos decomposition of the primary probe, as its own narrow file.
    # Written only when the columns are
    # present (standard mode; the multiclip path builds its rows separately).
    SINCOS_COLS = ["split", "k", "mae_sin", "mae_cos", "mae_theta", "r2_theta"]
    if probe_rows and {"mae_sin", "mae_cos"} <= set(probe_rows[0]):
        _write_csv([{c: r[c] for c in SINCOS_COLS if c in r} for r in probe_rows],
                   "probe_metrics_sincos.csv")

    # Already saved individually as each was trained/loaded in 3.a; re-saved
    # here too so _save_results stays a complete checkpoint of its own.
    for key, probe in probes.items():
        probe_path = out_dir / PROBE_FILENAMES[key]
        with open(probe_path, "wb") as fh:
            pickle.dump(probe, fh)
        print(f"{key} probe saved to {probe_path}")

    if eval_clips is not None:
        clips_path = out_dir / "eval_clips.json"
        clips_path.write_text(json.dumps(eval_clips))
        print(f"Eval clips saved to {clips_path}  ({len(eval_clips)} clips)")

    # summary: key horizons only
    key_ks = sorted(set([0, 1, 5, 10, 20, 50, horizon]))
    summary: dict = {
        "exp_name": exp_name,
        "checkpoint": str(ckpt_path),
        "manifest": manifest_path,
        "model_name": cfg.model.name,
        "horizon": horizon,
        "metrics": {},
    }
    rollout_by_k = {(r["split"], r["k"]): r for r in rollout_rows}
    probe_by_k   = {(r["split"], r["k"]): r for r in probe_rows}
    for split in ("train", "test"):
        summary["metrics"][split] = {}
        for k in key_ks:
            rr = rollout_by_k.get((split, k))
            pr = probe_by_k.get((split, k))
            if rr:
                summary["metrics"][split][f"k{k}_rmse_norm_q"] = rr["rmse_norm_q"]
                summary["metrics"][split][f"k{k}_r2_q"]        = rr["r2_q"]
                if not math.isnan(rr["energy_mean"]):
                    summary["metrics"][split][f"k{k}_energy_mean"] = rr["energy_mean"]
            if pr:
                # Column names differ by environment (see _apply_probes): the pendulum
                # emits theta/omega, three-body emits pos/vel. Copy whichever the row
                # actually carries rather than assuming, so the summary neither
                # KeyErrors nor silently omits the metrics that exist.
                for col in ("mae_theta", "r2_theta", "mae_omega", "r2_omega",
                            "r2_pos", "r2_vel", "mae_pos", "mae_vel"):
                    if col in pr:
                        summary["metrics"][split][f"k{k}_{col}"] = pr[col]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to the Lightning .ckpt file")
    parser.add_argument("--manifest", required=True,
                        help="Path to episode-split manifest JSON from data_prep.py")
    parser.add_argument("--exp-name", required=True,
                        help="Experiment name — determines the output subdirectory")
    parser.add_argument("--mode", choices=["standard", "multiclip"], default="standard",
                        help="'standard': one anchor per episode/clip, trains probe, writes "
                             "rollout_metrics.csv + probe_metrics.csv + summary.json (default). "
                             "'multiclip': dense sliding-window anchors, loads probe.pkl if "
                             "present, writes multiclip_rollout_metrics.csv.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Max rollout steps (default: 97 for standard "
                             "= 100 frames − 3 context; 30 for multiclip)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enc-batch", type=int, default=50,
                        help="Frames per encoder forward pass (reduce if OOM)")
    parser.add_argument("--n-eval-clips", type=int, default=10_000,
                        help="Number of test clips to sample for rollout eval (default: 10000)")
    parser.add_argument("--rollout-seed", type=int, default=0,
                        help="RNG seed for sampling eval clips (default: 0)")
    parser.add_argument("--n-probe-train", type=int, default=None,
                        help="Max train clips used to fit the probe (default: all). "
                             "Subsample when training the full set causes OOM, e.g. --n-probe-train 5000.")
    parser.add_argument("--knn-neighbors", type=int, default=25,
                        help="n_neighbors for the KNN probes (knn_gt, knn_k1). Default: 25.")
    parser.add_argument(
        "--zero-action", action="store_true",
        help="drive the rollout with u = 0 instead of the dataset's true per-frame "
             "action. This was the ONLY behaviour before 2026-08-18 and is a no-op on "
             "any unactuated dataset (no `action` column -> zero token either way); on "
             "the actuated swing-up data it reproduces the old numbers, which scored a "
             "u = 0 rollout against a torque-driven ground truth.")
    parser.add_argument("--skip-extra-probes", action="store_true",
                        help="[episode manifests only] Train/apply only the original ridge_gt "
                             "probe — skip knn_gt, ridge_k1, knn_k1.")
    # multiclip-only options
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="[multiclip] Cap number of episodes per split (for smoke tests)")
    parser.add_argument("--splits", default="test",
                        help="[multiclip] Comma-separated: train,test (default: test only)")
    parser.add_argument("--probe-path", type=Path, default=None,
                        help="[multiclip] Path to an existing probe.pkl (e.g. from a standard-"
                             "mode run under a different --exp-name). Default: "
                             "results/<exp-name>/probe.pkl.")
    parser.add_argument("--eval-stride", type=int, default=1,
                        help="[multiclip] Step between consecutive anchor positions "
                             "(default: 1 = every position). Eval sampling density only — "
                             "independent of the manifest's training clip_stride.")
    parser.add_argument("--data-name", default=None,
                        help="Override cfg.data.name (the HDF5 stem resolved under "
                             "$STABLEWM_HOME). Use to evaluate a checkpoint on a "
                             "dataset it was not trained on (the cross-dt transfer arm).")
    parser.add_argument("--flow-substeps", type=int, default=1,
                        help="Run N leapfrog sub-steps per predictor step. N=2 "
                             "deploys a dt-trained Hamiltonian on 2*dt data "
                             "without retraining. Requires a model with "
                             "predictor.flow; errors for LeWM.")
    parser.add_argument("--flow-dt-scale", type=float, default=1.0,
                        help="Integrate the same learned H with step size s*dt, "
                             "composing 1/s steps per frame so the physical gap "
                             "per frame is unchanged (the step-size ablation). "
                             "Requires 0 < s <= 1 with integral 1/s. Complement of "
                             "--flow-substeps, which instead holds h fixed and "
                             "changes the physics covered. Errors for LeWM.")
    parser.add_argument("--fd-dt", type=float, default=None,
                        help="Override cfg.model.fd_dt, the denominator of the FD/SG "
                             "momentum stencil (p = Δq / fd_dt). Required alongside "
                             "--flow-substeps N on data sampled at N*dt: pass --fd-dt N "
                             "so the stencil's per-frame p is rescaled back into the "
                             "units K(p) was trained on. Without it the transfer arm "
                             "feeds K a p that is ~N x too large.")
    parser.add_argument("--energy-bins", type=int, default=3,
                        help="Also write *_by_energy.csv with metrics stratified into "
                             "this many episode-energy bins (quantiles of max |ω|). "
                             "0 disables. Pooled CSVs are always written unchanged. "
                             "Episode manifests only. Default: 3 (terciles).")
    args = parser.parse_args()

    if args.horizon is None:
        args.horizon = 30 if args.mode == "multiclip" else 97

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Manifest   : {args.manifest}")
    print(f"Exp name   : {args.exp_name}")
    print(f"Mode       : {args.mode}")
    print(f"Horizon    : {args.horizon}")
    print(f"Device     : {args.device}")

    model, cfg = load_model(args.checkpoint, args.device)

    # Cross-dt transfer arm: evaluate a checkpoint against a dataset
    # other than the one named in its own config. Deliberately explicit — the
    # default remains "the data it was trained on", so this can only happen when
    # asked for, and the override is echoed into the run log.
    if args.data_name is not None and args.data_name != cfg.data.name:
        print(f"[data] OVERRIDE: cfg.data.name {cfg.data.name!r} → {args.data_name!r}")
        cfg.data.name = args.data_name
    # Momentum-scale correction for the cross-dt transfer arm. The FD/SG stencil
    # computes p in PER-FRAME units (p = Δq / fd_dt), so the same physical
    # velocity yields a stencil p that scales with the frame spacing: at 2*dt the
    # displacement Δq roughly doubles, and K(p) — trained at the dt1 scale —
    # receives a p that is ~2x too large before a single step is taken. Setting
    # fd_dt to the rate ratio (2.0 for pendulum_dt2) puts p back in training
    # units. Applied to the anchor p0 AND to the GT p series, so r2_p stays
    # self-consistent.
    if args.fd_dt is not None:
        old_fd_dt = float(cfg.model.get("fd_dt", 1.0))
        if args.fd_dt != old_fd_dt:
            print(f"[momentum] OVERRIDE: cfg.model.fd_dt {old_fd_dt} → {args.fd_dt}")
            cfg.model.fd_dt = args.fd_dt
    if args.flow_substeps != 1:
        apply_flow_substeps(model, args.flow_substeps)
    if args.flow_dt_scale != 1.0:
        apply_flow_dt_scale(model, args.flow_dt_scale)

    if args.mode == "multiclip":
        run_multiclip(args, model, cfg)
        return

    h5_path    = _h5_path(cfg)
    model_name = cfg.model.name
    mom_type   = _mom_type(model_name)

    print(f"\nModel : {model_name}  |  momentum: {mom_type}")
    print(f"H5    : {h5_path}")

    # Detect manifest type and load train/test split accordingly
    m_meta       = json.loads(Path(args.manifest).read_text())
    split_method = m_meta.get("split_method", "episode")

    if split_method == "clip":
        train_clips, test_clips, num_steps = _load_clips_from_manifest(args.manifest)
    else:
        train_eps, test_eps = _load_manifest(args.manifest)
        print(f"Train episodes: {len(train_eps)}  |  Test episodes: {len(test_eps)}")

    # ── 3.a: probe training (reuse saved probes if input dim matches) ────────
    out_root           = ROOT / "lehami" / "experiments" / "results"
    out_dir             = out_root / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_input_dim = (2 * _latent_dim(cfg) if mom_type != "lewm"
                          else _latent_dim(cfg))

    # Does this dataset vary the rod length? Decides how many target channels a
    # probe is SUPPOSED to have, and therefore whether a cached one is stale.
    dataset_varies_L = _episode_lengths(h5_path) is not None

    def _load_cached_probe(key: str) -> Probe | None:
        p_path = out_dir / PROBE_FILENAMES[key]
        if not p_path.exists():
            return None
        with open(p_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.ridge.n_features_in_ != expected_input_dim:
            print(
                f"[3.a] Cached {key} probe has input dim {cached.ridge.n_features_in_} "
                f"but expected {expected_input_dim} — retraining."
            )
            return None
        # Target-side staleness. The k1 collectors gained the rod-length channel on
        # 2026-08-24; a pickle written before that predicts 3 channels, so L would be
        # silently dropped for that tier while the gt tiers report it.
        n_tgt = _probe_n_targets(cached.ridge)
        if dataset_varies_L and n_tgt < 4:
            print(
                f"[3.a] Cached {key} probe predicts {n_tgt} targets but this dataset "
                f"varies rod length (4 expected: sin, cos, omega, L) — retraining."
            )
            return None
        print(f"[3.a] Loaded cached {key} probe from {p_path}")
        return cached

    def _save_probe(key: str, trained: Probe) -> None:
        p_path = out_dir / PROBE_FILENAMES[key]
        with open(p_path, "wb") as fh:
            pickle.dump(trained, fh)
        print(f"[3.a] {key} probe saved to {p_path}")

    probes: dict[str, Probe] = {}

    if split_method == "clip":
        # Clip-manifest experiments only get the original ridge_gt probe — the
        # extra probes (knn_gt, ridge_k1, knn_k1) are only wired up for the
        # episode-manifest path the paper's experiments use.
        probe = _load_cached_probe("ridge_gt")
        if probe is None:
            print("\n[3.a] Training ridge_gt probe on clip-based train split...")
            probe = train_probe(
                model, cfg, h5_path, train_clips, num_steps, args.device, args.enc_batch,
                max_clips=args.n_probe_train,
            )
            _save_probe("ridge_gt", probe)
        probes["ridge_gt"] = probe
    else:
        train_frames = m_meta.get("train_frames", None)
        n_frames_msg = f"first {train_frames} frames" if train_frames else "all frames"

        probe = _load_cached_probe("ridge_gt")
        if probe is None:
            print(f"\n[3.a] Training ridge_gt probe on {len(train_eps)} train episodes "
                  f"({n_frames_msg} each)...")
            probe = _train_probe_episodes(
                model, cfg, h5_path, train_eps, args.device, args.enc_batch,
                train_frames=train_frames,
            )
            _save_probe("ridge_gt", probe)
        probes["ridge_gt"] = probe

        if not args.skip_extra_probes:
            from train_probe_physics import (
                train_probe_knn_gt, collect_k1_probe_data,
                fit_ridge_probe, fit_knn_probe, fit_mlp_probe,
                train_probe_mlp_gt,
            )

            knn_gt = _load_cached_probe("knn_gt")
            if knn_gt is None:
                print(f"\n[3.a] Training knn_gt probe on {len(train_eps)} train episodes "
                      f"({n_frames_msg} each)...")
                knn_gt = train_probe_knn_gt(
                    model, cfg, h5_path, train_eps, args.device, args.enc_batch,
                    train_frames=train_frames, n_neighbors=args.knn_neighbors,
                )
                _save_probe("knn_gt", knn_gt)
            probes["knn_gt"] = knn_gt

            # Third readout tier. Ridge is linear and kNN is local, so a quantity
            # that is globally nonlinear but smooth is invisible to both while
            # remaining perfectly decodable — which is exactly the ambiguity the
            # omega result turned on. Without this tier a low R² cannot distinguish
            # "not in the representation" from "not readable by these two".
            mlp_gt = _load_cached_probe("mlp_gt")
            if mlp_gt is None:
                print(f"\n[3.a] Training mlp_gt probe on {len(train_eps)} train episodes "
                      f"({n_frames_msg} each)...")
                mlp_gt = train_probe_mlp_gt(
                    model, cfg, h5_path, train_eps, args.device, args.enc_batch,
                    train_frames=train_frames,
                )
                _save_probe("mlp_gt", mlp_gt)
            probes["mlp_gt"] = mlp_gt

            ridge_k1 = _load_cached_probe("ridge_k1")
            knn_k1   = _load_cached_probe("knn_k1")
            mlp_k1   = _load_cached_probe("mlp_k1")
            if ridge_k1 is None or knn_k1 is None or mlp_k1 is None:
                print(f"\n[3.a] Collecting k=1 rollout output on {len(train_eps)} train "
                      f"episodes ({n_frames_msg} each)...")
                X_k1, Y_k1 = collect_k1_probe_data(
                    model, cfg, h5_path, train_eps, args.device, args.enc_batch,
                    train_frames=train_frames,
                )
                has_p_k1 = mom_type != "lewm"
                if ridge_k1 is None:
                    ridge = fit_ridge_probe(X_k1, Y_k1)
                    r2_tr = float(r2_score(Y_k1, ridge.predict(X_k1)))
                    print(f"[probe:ridge_k1] fit on {len(X_k1)} samples  "
                          f"input ∈ ℝ^{X_k1.shape[1]}  train R²={r2_tr:.4f}")
                    ridge_k1 = Probe(ridge=ridge, latent_dim=_latent_dim(cfg), uses_p=has_p_k1)
                    _save_probe("ridge_k1", ridge_k1)
                if knn_k1 is None:
                    knn = fit_knn_probe(X_k1, Y_k1, n_neighbors=args.knn_neighbors)
                    r2_tr = float(r2_score(Y_k1, knn.predict(X_k1)))
                    print(f"[probe:knn_k1] fit on {len(X_k1)} samples  "
                          f"input ∈ ℝ^{X_k1.shape[1]}  k={args.knn_neighbors}  "
                          f"train R²={r2_tr:.4f}")
                    knn_k1 = Probe(ridge=knn, latent_dim=_latent_dim(cfg), uses_p=has_p_k1)
                    _save_probe("knn_k1", knn_k1)
                if mlp_k1 is None:
                    mlp = fit_mlp_probe(X_k1, Y_k1)
                    r2_tr = float(r2_score(Y_k1, mlp.predict(X_k1)))
                    print(f"[probe:mlp_k1] fit on {len(X_k1)} samples  "
                          f"input ∈ ℝ^{X_k1.shape[1]}  train R²={r2_tr:.4f}")
                    mlp_k1 = Probe(ridge=mlp, latent_dim=_latent_dim(cfg), uses_p=has_p_k1)
                    _save_probe("mlp_k1", mlp_k1)
            probes["ridge_k1"] = ridge_k1
            probes["knn_k1"]   = knn_k1
            probes["mlp_k1"]   = mlp_k1

    # ── 3.b + 3.c: rollout evaluation ────────────────────────────────────────
    rollout_rows: list[dict] = []
    probe_rows:   list[dict] = []
    eval_clips:   list[tuple[int, int]] | None = None
    strat_rollout_rows: list[dict] = []
    strat_probe_rows:   list[dict] = []
    energy_bin_edges:   dict[str, list[float]] = {}

    if split_method == "clip":
        _clip_eval_fn = (evaluate_rollout_clips_lewm
                         if mom_type == "lewm" else evaluate_rollout_clips)
        print(
            f"\n[3.b+3.c] Clip-based rollout — test split "
            f"(up to {args.n_eval_clips} clips, seed={args.rollout_seed})..."
        )
        rr, pr, eval_clips = _clip_eval_fn(
            model, cfg, h5_path, test_clips, probes["ridge_gt"],
            args.horizon, args.device,
            n_eval=args.n_eval_clips, rng_seed=args.rollout_seed,
            enc_batch=args.enc_batch,
        )
        rollout_rows.extend(rr)
        probe_rows.extend(pr)
    else:
        eval_fn = evaluate_rollout_lewm if mom_type == "lewm" else evaluate_rollout
        for split_name, episodes in [("train", train_eps), ("test", test_eps)]:
            print(f"\n[3.b+3.c] Rollout — {split_name} split ({len(episodes)} episodes)...")
            rr, pr = eval_fn(
                model, cfg, h5_path, episodes, probes,
                args.horizon, args.device, split_name, args.enc_batch,
                zero_action=args.zero_action,
            )
            rollout_rows.extend(rr)
            probe_rows.extend(pr)

        # ── energy-stratified pass: same eval, per energy bin ────────────────
        if args.energy_bins > 0:
            for split_name, episodes in [("train", train_eps), ("test", test_eps)]:
                bins, edges, labels = _episode_energy_bins(
                    h5_path, episodes, args.energy_bins)
                energy_bin_edges[split_name] = edges
                print(f"\n[energy-strat] {split_name} bin edges (max|ω|): "
                      + ", ".join(f"{e:.2f}" for e in edges))
                for lab in labels:
                    eps_b = bins[lab]
                    if not eps_b:
                        continue
                    print(f"[energy-strat] {split_name}/{lab}: {len(eps_b)} episodes")
                    rr, pr = eval_fn(
                        model, cfg, h5_path, eps_b, probes,
                        args.horizon, args.device, split_name, args.enc_batch,
                    )
                    for row in rr + pr:
                        row["energy_bin"] = lab
                        row["n_episodes"] = len(eps_b)
                    strat_rollout_rows.extend(rr)
                    strat_probe_rows.extend(pr)

    # ── save ──────────────────────────────────────────────────────────────────
    _save_results(
        args.exp_name, args.checkpoint, args.manifest, cfg,
        rollout_rows, probe_rows, args.horizon, out_root, probes,
        eval_clips=eval_clips,
    )

    if strat_rollout_rows:
        out_dir = out_root / args.exp_name
        _write_rows_csv(out_dir / "rollout_metrics_by_energy.csv", strat_rollout_rows)
        _write_rows_csv(out_dir / "probe_metrics_by_energy.csv", strat_probe_rows)
        summ_path = out_dir / "summary.json"
        summ = json.loads(summ_path.read_text())
        summ["energy_bin_edges"] = energy_bin_edges
        summ_path.write_text(json.dumps(summ, indent=2))
        print("[energy-strat] bin edges recorded in summary.json")


if __name__ == "__main__":
    main()
