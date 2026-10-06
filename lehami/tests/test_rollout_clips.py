"""Tests for evaluate_rollout_clips in run_eval.py.

Run with:
    uv run pytest lehami/tests/test_rollout_clips.py -v

Or standalone (no pytest required):
    uv run python lehami/tests/test_rollout_clips.py
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
    _load_clips_from_manifest,
    train_probe,
    evaluate_rollout_clips,
)

# ── synthetic dataset parameters ─────────────────────────────────────────────
N_EPISODES  = 4
FRAMES_EP   = 12       # frames per episode
NUM_STEPS   = 6        # clip length (= history_size + num_preds; >= 5 for the SG stencil)
LATENT_DIM  = 8
VIT_DIM     = 16       # CLS token dim (= 2 × LATENT_DIM for mom_projector input)
IMG_SIZE    = 16
SEED        = 42
TRAIN_RATIO = 0.75
ACTION_DIM  = 2
PHASE_DIM   = 2 * LATENT_DIM   # (q, p) concatenated

# clips per episode = 12 - 3 = 9;  total = 36;  train = 27;  test = 9
CLIPS_PER_EP  = FRAMES_EP - (NUM_STEPS - 1)
N_TOTAL_CLIPS = N_EPISODES * CLIPS_PER_EP
N_TRAIN_CLIPS = round(N_TOTAL_CLIPS * TRAIN_RATIO)
N_TEST_CLIPS  = N_TOTAL_CLIPS - N_TRAIN_CLIPS

# For horizon=HORIZON and our dataset, clips with start >= FRAMES_EP-HORIZON-1
# have k_max < HORIZON, so n_clips decreases at the boundary.
HORIZON = 5


# ── synthetic data helpers ────────────────────────────────────────────────────

def _make_h5(path: Path) -> None:
    """Write a synthetic HDF5 file.

    States: θ_t = 2π·t/(T-1) per episode (covers full circle), ω_t = cos(θ_t).
    Pixels: each frame's pixel value encodes the absolute frame index (so
    mock encoder outputs are distinct across frames).
    """
    pixels   = np.zeros((N_EPISODES * FRAMES_EP, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    states   = np.zeros((N_EPISODES * FRAMES_EP, 2), dtype=np.float32)
    ep_off   = np.zeros(N_EPISODES, dtype=np.int64)
    ep_len   = np.full(N_EPISODES, FRAMES_EP, dtype=np.int64)
    for ep in range(N_EPISODES):
        base = ep * FRAMES_EP
        ep_off[ep] = base
        for t in range(FRAMES_EP):
            theta = 2 * math.pi * t / max(FRAMES_EP - 1, 1)
            states[base + t, 0] = theta
            states[base + t, 1] = math.cos(theta)
            pixels[base + t, :, :, :] = (base + t) % 256
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels",    data=pixels)
        f.create_dataset("state",     data=states)
        f.create_dataset("ep_offset", data=ep_off)
        f.create_dataset("ep_len",    data=ep_len)


def _make_manifest(path: Path, h5_path: Path) -> None:
    path.write_text(json.dumps({
        "split_method":  "clip",
        "dataset_path":  str(h5_path),
        "seed":          SEED,
        "num_steps":     NUM_STEPS,
        "frameskip":     1,
        "train_ratio":   TRAIN_RATIO,
        "n_total_clips": N_TOTAL_CLIPS,
        "n_train_clips": N_TRAIN_CLIPS,
        "n_test_clips":  N_TEST_CLIPS,
        "dataset_stats": {
            "n_episodes": N_EPISODES,
            "n_frames":   N_EPISODES * FRAMES_EP,
            "ep_lengths": [FRAMES_EP] * N_EPISODES,
            "ep_offsets": [i * FRAMES_EP for i in range(N_EPISODES)],
        },
    }))


# ── mock model ────────────────────────────────────────────────────────────────

class _MockEncoderOutput:
    def __init__(self, hidden: torch.Tensor):
        self.last_hidden_state = hidden.unsqueeze(1)  # (B, 1, vit_dim)


class _MockFlow:
    """H(q, p) = 0.5 * ||p||²  (kinetic-only mock)."""
    def eval_H(self, q: torch.Tensor, p: torch.Tensor, u_emb: torch.Tensor) -> torch.Tensor:
        return (0.5 * (p ** 2).sum()).reshape(1)


class _MockPredictor(nn.Module):
    """Identity predictor — returns qp unchanged so we can verify frame mapping."""
    def __init__(self):
        super().__init__()
        self.flow = _MockFlow()

    def forward(self, qp: torch.Tensor, u_emb: torch.Tensor) -> torch.Tensor:
        # (B, T, phase_dim), (B, T, phase_dim) → (B, T, phase_dim)
        return qp.clone()


class _MockActionEncoder(nn.Module):
    def forward(self, act: torch.Tensor) -> torch.Tensor:
        B, T, _ = act.shape
        return torch.zeros(B, T, PHASE_DIM)


class _MockRolloutModel(nn.Module):
    """Minimal model supporting the full evaluate_rollout_clips interface."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.projector      = nn.Linear(VIT_DIM, LATENT_DIM, bias=False)
        self.mom_projector  = nn.Linear(VIT_DIM * 2, LATENT_DIM, bias=False)
        self.action_encoder = _MockActionEncoder()
        self.predictor      = _MockPredictor()
        nn.init.normal_(self.projector.weight)
        nn.init.normal_(self.mom_projector.weight)

    def encoder(self, pixels: torch.Tensor, **kwargs) -> _MockEncoderOutput:
        """(B, C, H, W) → mock CLS tokens (B, vit_dim).  Deterministic per frame."""
        B    = pixels.shape[0]
        feat = pixels.mean(dim=(1, 2, 3))
        feat = feat.unsqueeze(-1).expand(B, VIT_DIM)
        return _MockEncoderOutput(feat)


def _make_cfg() -> SimpleNamespace:
    model_dict = {
        "name":       "hjepa_sg_sep_sreg_joint_ms",
        "fd_dt":      1.0,
        "latent_dim": LATENT_DIM,
    }

    class _ModelNS:
        def __getattr__(self, name):
            return model_dict[name]
        def get(self, key, default=None):
            return model_dict.get(key, default)

    return SimpleNamespace(
        history_size=NUM_STEPS - 1,
        img_size=IMG_SIZE,
        embed_dim=LATENT_DIM,
        data=SimpleNamespace(action_dim=ACTION_DIM),
        model=_ModelNS(),
    )


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rollout_test")


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
def rollout_model():
    m = _MockRolloutModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def cfg():
    return _make_cfg()


@pytest.fixture(scope="module")
def test_clips(manifest_path):
    _, tc, _ = _load_clips_from_manifest(str(manifest_path))
    return tc


@pytest.fixture(scope="module")
def trained_probe(rollout_model, cfg, h5_path, manifest_path):
    train_clips, _, num_steps = _load_clips_from_manifest(str(manifest_path))
    return train_probe(rollout_model, cfg, h5_path, train_clips, num_steps, device="cpu")


@pytest.fixture(scope="module")
def rollout_result(rollout_model, cfg, h5_path, test_clips, trained_probe):
    """Full rollout run shared across all tests (expensive fixture, run once)."""
    rr, pr, ec = evaluate_rollout_clips(
        rollout_model, cfg, h5_path, test_clips, trained_probe,
        horizon=HORIZON,
        device="cpu",
        n_eval=N_TEST_CLIPS,   # use every test clip
        rng_seed=0,
    )
    return rr, pr, ec


# ── tests: clip sampling ──────────────────────────────────────────────────────

class TestRolloutClipsSampling:
    """evaluate_rollout_clips samples the right clips, deterministically."""

    def test_eval_clips_count(self, rollout_result):
        _, _, eval_clips = rollout_result
        assert len(eval_clips) == min(N_TEST_CLIPS, N_TEST_CLIPS)

    def test_eval_clips_all_from_test_split(self, rollout_result, test_clips):
        """Every sampled clip must come from the test split."""
        _, _, eval_clips = rollout_result
        test_set = set(map(tuple, test_clips))
        for clip in eval_clips:
            assert tuple(clip) in test_set, (
                f"Sampled clip {clip} is not a test clip"
            )

    def test_eval_clips_no_duplicates(self, rollout_result):
        _, _, eval_clips = rollout_result
        as_tuples = [tuple(c) for c in eval_clips]
        assert len(as_tuples) == len(set(as_tuples)), (
            "eval_clips contains duplicate (ep, start) pairs"
        )

    def test_eval_clips_deterministic(self, rollout_model, cfg, h5_path,
                                       test_clips, trained_probe):
        """Same rng_seed must reproduce the exact same clip selection."""
        kwargs = dict(horizon=2, device="cpu", n_eval=5, rng_seed=77)
        _, _, ec1 = evaluate_rollout_clips(
            rollout_model, cfg, h5_path, test_clips, trained_probe, **kwargs)
        _, _, ec2 = evaluate_rollout_clips(
            rollout_model, cfg, h5_path, test_clips, trained_probe, **kwargs)
        assert ec1 == ec2, "Same rng_seed produced different clip selections"

    def test_eval_clips_json_serializable(self, rollout_result):
        """eval_clips must be a plain list of (ep, start) pairs for easy storage."""
        _, _, eval_clips = rollout_result
        reloaded = json.loads(json.dumps(eval_clips))
        assert len(reloaded) == len(eval_clips)
        for clip in reloaded:
            assert len(clip) == 2
            assert all(isinstance(v, int) for v in clip)

    def test_different_seeds_give_different_clips(self, rollout_model, cfg,
                                                   h5_path, test_clips, trained_probe):
        n = min(5, N_TEST_CLIPS - 1)
        _, _, ec_a = evaluate_rollout_clips(
            rollout_model, cfg, h5_path, test_clips, trained_probe,
            horizon=2, device="cpu", n_eval=n, rng_seed=0)
        _, _, ec_b = evaluate_rollout_clips(
            rollout_model, cfg, h5_path, test_clips, trained_probe,
            horizon=2, device="cpu", n_eval=n, rng_seed=1)
        # With 5 out of 9 clips, two different seeds are very unlikely to agree
        assert ec_a != ec_b, (
            "Different rng seeds produced identical clip selections — check the RNG path"
        )


# ── tests: rollout row structure ──────────────────────────────────────────────

class TestRolloutRowsStructure:
    """Rollout metric rows have required columns and correct semantics."""

    def test_rollout_rows_exist(self, rollout_result):
        rr, _, _ = rollout_result
        assert len(rr) > 0, "No rollout rows returned"

    def test_required_columns(self, rollout_result):
        rr, _, _ = rollout_result
        required = {"split", "k", "n_clips", "rmse_norm_q", "r2_q",
                    "energy_mean", "energy_std"}
        for row in rr:
            missing = required - set(row.keys())
            assert not missing, f"Rollout row k={row['k']} missing: {missing}"

    def test_p_columns_present_for_hjepa(self, rollout_result):
        """HJEPA models (has_p=True) must report rmse_norm_p and r2_p."""
        rr, _, _ = rollout_result
        for row in rr:
            assert "rmse_norm_p" in row, f"Missing rmse_norm_p at k={row['k']}"
            assert "r2_p"        in row, f"Missing r2_p at k={row['k']}"

    def test_all_rows_split_test(self, rollout_result):
        rr, _, _ = rollout_result
        for row in rr:
            assert row["split"] == "test"

    def test_k_values_start_at_zero(self, rollout_result):
        rr, _, _ = rollout_result
        ks = {r["k"] for r in rr}
        assert 0 in ks, f"k=0 row missing; ks present: {sorted(ks)}"

    def test_k_values_within_horizon(self, rollout_result):
        rr, _, _ = rollout_result
        for row in rr:
            assert 0 <= row["k"] <= HORIZON, (
                f"k={row['k']} out of [0, {HORIZON}]"
            )

    def test_k0_anchor_perfect_q(self, rollout_result):
        """At k=0, the predicted q IS q_enc[1] (anchor) and so is the GT.
        RMSE must be exactly 0 and R² exactly 1.
        """
        rr, _, _ = rollout_result
        k0 = next(r for r in rr if r["k"] == 0)
        assert k0["rmse_norm_q"] == pytest.approx(0.0, abs=1e-7), (
            f"rmse_norm_q at k=0 should be 0 (pred==gt==anchor), "
            f"got {k0['rmse_norm_q']}"
        )
        assert k0["r2_q"] == pytest.approx(1.0, abs=1e-7), (
            f"r2_q at k=0 should be 1.0, got {k0['r2_q']}"
        )

    def test_k0_anchor_perfect_p(self, rollout_result):
        """At k=0, GT p is also set to the anchor p.  Same perfect result."""
        rr, _, _ = rollout_result
        k0 = next(r for r in rr if r["k"] == 0)
        assert k0["rmse_norm_p"] == pytest.approx(0.0, abs=1e-7)
        assert k0["r2_p"]        == pytest.approx(1.0, abs=1e-7)

    def test_identity_predictor_nonzero_rmse_at_k1(self, rollout_result):
        """With identity predictor q stays at anchor, but GT advances to frame start+2.
        The two frames have different encodings, so RMSE must be > 0.
        """
        rr, _, _ = rollout_result
        k1 = next((r for r in rr if r["k"] == 1), None)
        if k1 is None:
            pytest.skip("No k=1 row (all test clips have k_max=0)")
        assert k1["rmse_norm_q"] > 0, (
            "RMSE at k=1 is 0 — identity predictor must diverge from advancing GT"
        )

    def test_n_clips_nonincreasing(self, rollout_result):
        """Clips near episode end have shorter k_max, so n_clips must not grow with k."""
        rr, _, _ = rollout_result
        rows_by_k = {r["k"]: r for r in rr}
        ks = sorted(rows_by_k)
        for k_a, k_b in zip(ks, ks[1:]):
            n_a, n_b = rows_by_k[k_a]["n_clips"], rows_by_k[k_b]["n_clips"]
            assert n_b <= n_a, (
                f"n_clips grew from k={k_a} ({n_a}) to k={k_b} ({n_b}) — "
                "clips should drop out, not appear, as horizon extends"
            )

    def test_n_clips_drops_at_high_k(self, rollout_result):
        """At the maximum horizon (k=HORIZON=5), some test clips with high 'start'
        will have k_max < HORIZON, so n_clips at k=HORIZON < n_clips at k=0.

        For our dataset: k_max for start=8 is min(5, 12-9-1)=2, for start=7 is
        min(5,3)=3.  Any test clip with start >= 6 drops out before k=5.
        """
        rr, _, _ = rollout_result
        rows_by_k = {r["k"]: r for r in rr}
        if HORIZON not in rows_by_k:
            pytest.skip(f"No k={HORIZON} row — all clips had k_max < {HORIZON}")
        n_k0   = rows_by_k[0]["n_clips"]
        n_kmax = rows_by_k[HORIZON]["n_clips"]
        assert n_kmax < n_k0, (
            f"n_clips did not decrease from k=0 ({n_k0}) to k={HORIZON} ({n_kmax}). "
            "At least some test clips should have k_max < HORIZON."
        )

    def test_energy_finite(self, rollout_result):
        rr, _, _ = rollout_result
        for row in rr:
            assert math.isfinite(row["energy_mean"]), (
                f"energy_mean at k={row['k']} is not finite"
            )
            assert math.isfinite(row["energy_std"]), (
                f"energy_std at k={row['k']} is not finite"
            )


# ── tests: probe row structure ────────────────────────────────────────────────

class TestProbeRowsStructure:
    """Probe metric rows have correct columns, units, and finite values."""

    def test_probe_rows_exist(self, rollout_result):
        _, pr, _ = rollout_result
        assert len(pr) > 0, "No probe rows returned"

    def test_required_columns(self, rollout_result):
        _, pr, _ = rollout_result
        required = {"split", "k", "n_clips", "mae_theta", "r2_theta",
                    "mae_omega", "r2_omega"}
        for row in pr:
            missing = required - set(row.keys())
            assert not missing, f"Probe row k={row['k']} missing: {missing}"

    def test_all_rows_split_test(self, rollout_result):
        _, pr, _ = rollout_result
        for row in pr:
            assert row["split"] == "test"

    def test_probe_metrics_finite(self, rollout_result):
        """All four probe scalars must be finite at every k."""
        _, pr, _ = rollout_result
        for row in pr:
            for field in ("mae_theta", "r2_theta", "mae_omega", "r2_omega"):
                v = row[field]
                assert math.isfinite(v), (
                    f"probe_rows k={row['k']} {field}={v} is not finite"
                )

    def test_mae_theta_in_radian_range(self, rollout_result):
        """Angle MAE must lie in [0, π] rad — catches unit errors (e.g. degrees)."""
        _, pr, _ = rollout_result
        for row in pr:
            assert 0.0 <= row["mae_theta"] <= math.pi, (
                f"mae_theta={row['mae_theta']:.4f} at k={row['k']} is outside [0, π]. "
                "Check that θ is in radians, not degrees."
            )

    def test_rollout_and_probe_rows_same_ks(self, rollout_result):
        """Rollout and probe rows must cover the same set of k values."""
        rr, pr, _ = rollout_result
        rr_ks = {r["k"] for r in rr}
        pr_ks = {r["k"] for r in pr}
        assert rr_ks == pr_ks, (
            f"rollout k-set {rr_ks} ≠ probe k-set {pr_ks}"
        )

    def test_probe_rows_n_clips_matches_rollout(self, rollout_result):
        """n_clips in probe rows must agree with the corresponding rollout row."""
        rr, pr, _ = rollout_result
        rr_nc = {r["k"]: r["n_clips"] for r in rr}
        for row in pr:
            assert row["n_clips"] == rr_nc[row["k"]], (
                f"n_clips mismatch at k={row['k']}: "
                f"rollout={rr_nc[row['k']]} probe={row['n_clips']}"
            )


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== Running rollout tests manually (no pytest) ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h5  = tmp / "synthetic.h5"
        mf  = tmp / "manifest.json"

        print(f"Synthetic data: {N_EPISODES} eps × {FRAMES_EP} frames = "
              f"{N_TOTAL_CLIPS} clips  (train={N_TRAIN_CLIPS}, test={N_TEST_CLIPS})")
        _make_h5(h5)
        _make_manifest(mf, h5)

        train_clips, test_clips, num_steps = _load_clips_from_manifest(str(mf))
        print(f"Loaded manifest: {len(train_clips)} train / {len(test_clips)} test clips")

        model = _MockRolloutModel().eval()
        cfg   = _make_cfg()

        print("\nTraining probe on train clips...")
        probe = train_probe(model, cfg, h5, train_clips, num_steps, device="cpu")
        print(f"  Probe latent_dim={probe.latent_dim}, uses_p={probe.uses_p}, "
              f"ridge input dim={probe.ridge.n_features_in_}")

        print(f"\nRunning evaluate_rollout_clips (horizon={HORIZON}, "
              f"n_eval={N_TEST_CLIPS})...")
        rr, pr, ec = evaluate_rollout_clips(
            model, cfg, h5, test_clips, probe,
            horizon=HORIZON, device="cpu",
            n_eval=N_TEST_CLIPS, rng_seed=0,
        )
        print(f"  eval_clips selected: {len(ec)}")
        print(f"  rollout rows: {len(rr)}")
        print(f"  probe rows  : {len(pr)}")

        print("\nRollout metrics per k:")
        for r in rr:
            print(f"  k={r['k']:2d}  n={r['n_clips']:3d}  "
                  f"rmse_q={r['rmse_norm_q']:.4f}  r2_q={r['r2_q']:.4f}  "
                  f"rmse_p={r.get('rmse_norm_p', float('nan')):.4f}  "
                  f"energy={r['energy_mean']:.4f}")

        print("\nProbe metrics per k:")
        for r in pr:
            print(f"  k={r['k']:2d}  n={r['n_clips']:3d}  "
                  f"mae_θ={r['mae_theta']:.4f} rad  r2_θ={r['r2_theta']:.4f}  "
                  f"mae_ω={r['mae_omega']:.4f} rad/s  r2_ω={r['r2_omega']:.4f}")

        print("\n=== Assertions ===")
        # k=0 perfect anchor
        k0r = next(r for r in rr if r["k"] == 0)
        assert k0r["rmse_norm_q"] < 1e-6, f"Expected 0 rmse at k=0, got {k0r['rmse_norm_q']}"
        assert k0r["r2_q"] > 0.9999, f"Expected r2=1 at k=0, got {k0r['r2_q']}"
        print("  k=0 anchor perfect q: OK")

        # n_clips nonincreasing
        nc = [r["n_clips"] for r in sorted(rr, key=lambda x: x["k"])]
        for i in range(1, len(nc)):
            assert nc[i] <= nc[i - 1], f"n_clips grew at step {i}: {nc[i-1]} → {nc[i]}"
        print("  n_clips nonincreasing: OK")

        # mae_theta in [0, π]
        for row in pr:
            assert 0.0 <= row["mae_theta"] <= math.pi, (
                f"mae_theta out of range at k={row['k']}: {row['mae_theta']}")
        print("  mae_theta in [0, π]: OK")

        print("\n=== All manual checks passed ===")
