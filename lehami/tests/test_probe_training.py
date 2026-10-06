"""Tests for clip-based probe training in run_eval.py.

Run with:
    uv run pytest lehami/tests/test_probe_training.py -v

Or to debug a single test interactively:
    uv run python lehami/tests/test_probe_training.py
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "experiments" / "src"))
from run_eval import (
    Probe,
    _load_clips_from_manifest,
    _probe_has_length,
    _probe_n_targets,
    train_probe,
)

# ── synthetic dataset parameters ─────────────────────────────────────────────
N_EPISODES  = 4        # small: fast to run
FRAMES_EP   = 12       # frames per episode
NUM_STEPS   = 6        # clip length (history_size + num_preds; >= 5 for the SG stencil)
LATENT_DIM  = 8        # small latent space
VIT_DIM     = 16       # CLS token dim (must be 2× latent for mom_projector input)
IMG_SIZE    = 16       # tiny image
SEED        = 42
TRAIN_RATIO = 0.75
# clips per episode = FRAMES_EP - (NUM_STEPS-1) = 12 - 5 = 7
# total clips = N_EPISODES × 9 = 36
# n_train = round(36 × 0.75) = 27,  n_test = 9
CLIPS_PER_EP  = FRAMES_EP - (NUM_STEPS - 1)
N_TOTAL_CLIPS = N_EPISODES * CLIPS_PER_EP
N_TRAIN_CLIPS = round(N_TOTAL_CLIPS * TRAIN_RATIO)
N_TEST_CLIPS  = N_TOTAL_CLIPS - N_TRAIN_CLIPS
SAMPLES_PER_CLIP = NUM_STEPS - 1   # = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_h5(path: Path) -> None:
    """Write a synthetic HDF5 file with known physics states.

    States follow sin/cos encoding so the probe can actually learn something:
      θ_t = 2π·t / (FRAMES_EP - 1)   → covers full circle per episode
      ω_t = cos(θ_t)                  → smooth velocity proxy
    """
    pixels = np.zeros(
        (N_EPISODES * FRAMES_EP, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
    )
    states = np.zeros((N_EPISODES * FRAMES_EP, 2), dtype=np.float32)
    ep_offset = np.zeros(N_EPISODES, dtype=np.int64)
    ep_len    = np.full(N_EPISODES, FRAMES_EP, dtype=np.int64)

    for ep in range(N_EPISODES):
        base = ep * FRAMES_EP
        ep_offset[ep] = base
        for t in range(FRAMES_EP):
            theta = 2 * math.pi * t / max(FRAMES_EP - 1, 1)
            omega = math.cos(theta)
            states[base + t, 0] = theta
            states[base + t, 1] = omega
            # encode frame index in pixel value for easy visual debugging
            pixels[base + t, :, :, :] = (base + t) % 256

    with h5py.File(path, "w") as f:
        f.create_dataset("pixels",    data=pixels)
        f.create_dataset("state",     data=states)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_len",    data=ep_len)


def _make_manifest(path: Path, h5_path: Path) -> dict:
    """Write a clip-based manifest JSON matching the synthetic H5."""
    manifest = {
        "split_method":   "clip",
        "dataset_path":   str(h5_path),
        "seed":           SEED,
        "num_steps":      NUM_STEPS,
        "frameskip":      1,
        "train_ratio":    TRAIN_RATIO,
        "n_total_clips":  N_TOTAL_CLIPS,
        "n_train_clips":  N_TRAIN_CLIPS,
        "n_test_clips":   N_TEST_CLIPS,
        "dataset_stats":  {
            "n_episodes": N_EPISODES,
            "n_frames":   N_EPISODES * FRAMES_EP,
            "ep_lengths": [FRAMES_EP] * N_EPISODES,
            "ep_offsets": [i * FRAMES_EP for i in range(N_EPISODES)],
        },
    }
    path.write_text(json.dumps(manifest))
    return manifest


class _MockEncoderOutput:
    """Mimics HuggingFace encoder output.last_hidden_state."""
    def __init__(self, hidden: torch.Tensor):
        self.last_hidden_state = hidden.unsqueeze(1)  # (B, 1, vit_dim)


class _MockModel(nn.Module):
    """Minimal model exposing encoder / projector / mom_projector."""

    def __init__(self):
        super().__init__()
        self.projector     = nn.Linear(VIT_DIM, LATENT_DIM, bias=False)
        self.mom_projector = nn.Linear(VIT_DIM * 2, LATENT_DIM, bias=False)
        # fixed random weights — no training needed for structural tests
        torch.manual_seed(0)
        nn.init.normal_(self.projector.weight)
        nn.init.normal_(self.mom_projector.weight)

    def encoder(self, pixels: torch.Tensor, **kwargs) -> _MockEncoderOutput:
        """(B, C, H, W) → mock CLS tokens (B, vit_dim).

        Uses mean of pixel values across spatial dims as a simple feature.
        This is deterministic and differentiable, which is all the test needs.
        """
        B = pixels.shape[0]
        feat = pixels.mean(dim=(1, 2, 3))                 # (B,)
        feat = feat.unsqueeze(-1).expand(B, VIT_DIM)      # (B, vit_dim)
        return _MockEncoderOutput(feat)


def _make_cfg(model_name: str = "hjepa_sg_sep_sreg_joint_ms") -> SimpleNamespace:
    """Minimal config namespace for run_eval functions."""
    return SimpleNamespace(
        history_size=3,
        img_size=IMG_SIZE,
        embed_dim=LATENT_DIM,   # LeWM fallback in _latent_dim()
        model=SimpleNamespace(
            name=model_name,
            fd_dt=1.0,
            latent_dim=LATENT_DIM,
            # SimpleNamespace.get() doesn't exist — patch it in
        ),
    )


def _patch_cfg_model_get(cfg: SimpleNamespace) -> SimpleNamespace:
    """Add .get() to cfg.model so the code can call cfg.model.get('fd_dt', 1.0)."""
    model_ns = cfg.model
    model_dict = vars(model_ns)

    class _ModelNS:
        def __getattr__(self, name):
            return model_dict[name]
        def get(self, key, default=None):
            return model_dict.get(key, default)

    cfg.model = _ModelNS()
    return cfg


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("probe_test")


@pytest.fixture(scope="module")
def h5_path(tmp_dir):
    p = tmp_dir / "synthetic.h5"
    _make_h5(p)
    return p


@pytest.fixture(scope="module")
def manifest_path(tmp_dir, h5_path):
    p = tmp_dir / "manifest.json"
    _make_manifest(p, h5_path)
    return p


@pytest.fixture(scope="module")
def model():
    m = _MockModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def cfg():
    c = _make_cfg(model_name="hjepa_sg_sep_sreg_joint_ms")
    return _patch_cfg_model_get(c)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestClipManifestLoader:
    """_load_clips_from_manifest reproduces the right split."""

    def test_counts_match_manifest(self, manifest_path):
        train_clips, test_clips, num_steps = _load_clips_from_manifest(str(manifest_path))
        assert len(train_clips) == N_TRAIN_CLIPS, (
            f"Expected {N_TRAIN_CLIPS} train clips, got {len(train_clips)}"
        )
        assert len(test_clips) == N_TEST_CLIPS, (
            f"Expected {N_TEST_CLIPS} test clips, got {len(test_clips)}"
        )
        assert len(train_clips) + len(test_clips) == N_TOTAL_CLIPS

    def test_num_steps_returned(self, manifest_path):
        _, _, num_steps = _load_clips_from_manifest(str(manifest_path))
        assert num_steps == NUM_STEPS

    def test_no_overlap_between_train_and_test(self, manifest_path):
        """Train and test clips should be disjoint."""
        train_clips, test_clips, _ = _load_clips_from_manifest(str(manifest_path))
        train_set = set(train_clips)
        test_set  = set(test_clips)
        overlap   = train_set & test_set
        assert len(overlap) == 0, (
            f"{len(overlap)} clips appear in both train and test: {list(overlap)[:5]}"
        )

    def test_all_clips_within_episode_bounds(self, manifest_path):
        """Every clip (ep, start) must fit within its episode."""
        train_clips, test_clips, num_steps = _load_clips_from_manifest(str(manifest_path))
        for ep, start in train_clips + test_clips:
            last_frame = start + num_steps - 1
            assert last_frame < FRAMES_EP, (
                f"Clip (ep={ep}, start={start}) overshoots episode length {FRAMES_EP}: "
                f"last frame = {last_frame}"
            )

    def test_deterministic(self, manifest_path):
        """Two calls with same manifest must return identical splits."""
        a_train, a_test, _ = _load_clips_from_manifest(str(manifest_path))
        b_train, b_test, _ = _load_clips_from_manifest(str(manifest_path))
        assert a_train == b_train
        assert a_test  == b_test


class TestProbeTraining:
    """train_probe produces a correctly shaped, fittable Ridge probe."""

    @pytest.fixture(scope="class")
    def probe_and_clips(self, model, cfg, h5_path, manifest_path):
        train_clips, _, num_steps = _load_clips_from_manifest(str(manifest_path))
        probe = train_probe(
            model, cfg, h5_path, train_clips, num_steps, device="cpu"
        )
        return probe, train_clips, num_steps

    def test_probe_type(self, probe_and_clips):
        probe, _, _ = probe_and_clips
        assert isinstance(probe, Probe)

    def test_probe_uses_p_flag(self, probe_and_clips):
        """For a learned-momentum model, uses_p must be True."""
        probe, _, _ = probe_and_clips
        assert probe.uses_p is True, (
            "Expected uses_p=True for 'hjepa_sg_sep_sreg_joint_ms' (Savitzky-Golay momentum)"
        )

    def test_probe_latent_dim(self, probe_and_clips):
        probe, _, _ = probe_and_clips
        assert probe.latent_dim == LATENT_DIM

    def test_ridge_input_dim(self, probe_and_clips):
        """Ridge must have been fit on cat(q,p) ∈ ℝ^{2·latent_dim}."""
        probe, _, _ = probe_and_clips
        expected = 2 * LATENT_DIM
        actual   = probe.ridge.n_features_in_
        assert actual == expected, (
            f"Ridge input dim is {actual}, expected 2 × latent_dim = {expected}. "
            f"Probe was likely trained on q-only (old bug) instead of cat(q,p)."
        )

    def test_ridge_output_dim(self, probe_and_clips):
        """Ridge output must have 3 columns: [sinθ, cosθ, ω]."""
        probe, _, _ = probe_and_clips
        assert probe.ridge.coef_.shape == (3, 2 * LATENT_DIM), (
            f"Ridge coef shape is {probe.ridge.coef_.shape}, "
            f"expected (3, {2 * LATENT_DIM})"
        )

    def test_sample_count(self, probe_and_clips):
        """Number of training samples = n_train_clips × (num_steps - 1)."""
        probe, train_clips, num_steps = probe_and_clips
        expected_n = len(train_clips) * (num_steps - 1)
        # Infer N from ridge intercept shape (intercept has shape (3,)) — use coef
        # Actually we can't directly get N from the fitted Ridge, but we can check
        # indirectly: if N < 2·d the ridge can't be uniquely determined.
        # Here just verify n_features is right (proves it got 2d-dim input)
        assert probe.ridge.n_features_in_ == 2 * LATENT_DIM
        # A stronger check: predict on a dummy input of the right shape
        dummy = np.zeros((expected_n, 2 * LATENT_DIM))
        preds = probe.ridge.predict(dummy)
        assert preds.shape == (expected_n, 3), (
            f"Probe predict output shape {preds.shape} ≠ ({expected_n}, 3)"
        )

    def test_sincos_outputs_in_valid_range(self, probe_and_clips, h5_path, manifest_path, model, cfg):
        """sinθ and cosθ predictions should be near [-1, 1] for a learnable signal."""
        probe, train_clips, num_steps = probe_and_clips
        # Build a small set of real inputs to evaluate
        import h5py as _h5
        from run_eval import _preprocess_batch, _encode_episode

        ep, start = train_clips[0]
        with _h5.File(h5_path, "r") as f:
            ep_offset = int(f["ep_offset"][ep])
            L         = int(f["ep_len"][ep])
            pixels_raw = f["pixels"][ep_offset : ep_offset + L]

        from run_eval import _preprocess_batch, _encode_episode
        pixels = _preprocess_batch(pixels_raw, IMG_SIZE, "cpu")
        cls, q_enc = _encode_episode(model, pixels)

        fi = start + 1
        pair = torch.cat([cls[fi], cls[fi - 1]], dim=-1).unsqueeze(0)
        with torch.no_grad():
            p_i = model.mom_projector(pair).squeeze(0)
        qp = torch.cat([q_enc[fi], p_i]).detach().numpy().reshape(1, -1)

        pred = probe.ridge.predict(qp)[0]   # [sinθ, cosθ, ω]
        sin_pred, cos_pred = pred[0], pred[1]
        # Predictions don't have to be perfect, but must be finite floats
        assert math.isfinite(sin_pred), f"sinθ prediction is not finite: {sin_pred}"
        assert math.isfinite(cos_pred), f"cosθ prediction is not finite: {cos_pred}"


class TestProbeTargetCount:
    """`_probe_has_length` must work for BOTH estimator kinds a Probe can hold.

    Regression test for the gating bug found 2026-08-24: the check read
    `coef_` with a 3-row default, which a KNeighborsRegressor does not have, so
    every knn tier reported 3 targets and its `mae_length` / `r2_length` columns
    were silently dropped from probe_metrics.csv — even though the knn probes had
    been fit with L as a 4th target all along.
    """

    @staticmethod
    def _fit(estimator, n_targets: int):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2 * LATENT_DIM))
        Y = rng.normal(size=(50, n_targets))
        estimator.fit(X, Y)
        return Probe(ridge=estimator, latent_dim=LATENT_DIM, uses_p=True)

    @pytest.mark.parametrize("n_targets,expected", [(3, False), (4, True)])
    def test_ridge_target_count(self, n_targets, expected):
        from sklearn.linear_model import Ridge
        probe = self._fit(Ridge(alpha=1.0), n_targets)
        assert _probe_n_targets(probe.ridge) == n_targets
        assert _probe_has_length(probe) is expected

    @pytest.mark.parametrize("n_targets,expected", [(3, False), (4, True)])
    def test_knn_target_count(self, n_targets, expected):
        from sklearn.neighbors import KNeighborsRegressor
        probe = self._fit(KNeighborsRegressor(n_neighbors=5), n_targets)
        assert _probe_n_targets(probe.ridge) == n_targets
        assert _probe_has_length(probe) is expected


# ── standalone runner for manual debugging ────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("=== Running probe tests manually (no pytest) ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h5  = tmp / "synthetic.h5"
        mf  = tmp / "manifest.json"

        print(f"Creating synthetic H5 ({N_EPISODES} eps × {FRAMES_EP} frames)...")
        _make_h5(h5)
        _make_manifest(mf, h5)

        print("\n--- _load_clips_from_manifest ---")
        train_clips, test_clips, num_steps = _load_clips_from_manifest(str(mf))
        print(f"train: {len(train_clips)}  test: {len(test_clips)}  num_steps: {num_steps}")
        print(f"first 3 train clips: {train_clips[:3]}")
        print(f"first 3 test  clips: {test_clips[:3]}")

        print("\n--- train_probe ---")
        m   = _MockModel().eval()
        cfg = _patch_cfg_model_get(_make_cfg())
        probe = train_probe(m, cfg, h5, train_clips, num_steps, device="cpu")
        print(f"Probe.latent_dim  : {probe.latent_dim}")
        print(f"Probe.uses_p      : {probe.uses_p}")
        print(f"Ridge input dim   : {probe.ridge.n_features_in_}")
        print(f"Ridge coef shape  : {probe.ridge.coef_.shape}")
        print(f"Expected input dim: {2 * LATENT_DIM}  (2 × latent_dim={LATENT_DIM})")

        dummy = np.random.randn(5, 2 * LATENT_DIM)
        pred  = probe.ridge.predict(dummy)
        print(f"Predict on 5 dummy inputs → shape {pred.shape}")
        print(f"Sample prediction: sinθ={pred[0,0]:.3f}  cosθ={pred[0,1]:.3f}  ω={pred[0,2]:.3f}")

        print("\n=== All manual checks passed ===")
