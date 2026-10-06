"""Shape assertions for the frozen-DINOv2 arms (EXP-098).

Covers the three things that would otherwise fail silently or deep inside a
training step:

  1. The encoder contract. DinoEncoder must emit exactly what the projector
     expects -- cfg.embed_dim wide -- for both feature keys, and must stay
     frozen and in eval() even after Lightning calls .train().
  2. The causal mask. Upstream pins it to 'cuda' and leaves it a plain
     attribute; make_vit_predictor must give back a real buffer that follows
     .to(device), or a CPU forward dies on a device mismatch.  The block-causal
     structure is checked too: frame t must not see frame t+1, which is the
     whole point of the mask.
  3. The NUM_PATCHES/NUM_FRAMES globals. vit.py stores them at module level, so
     building a second predictor with different dimensions must not corrupt the
     first one's masks, and the patched Attention class must not be left
     installed on the upstream module.

Everything runs on CPU and none of it needs the torch.hub weights EXCEPT the
encoder tests, which are skipped when the cache is cold so the suite still
passes on a machine that has never warmed TORCH_HOME.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lehami" / "models"))

from hjepa.dino_wm_adapter import (  # noqa: E402
    _DWM,
    DinoEncoder,
    _vit_module,
    make_vit_predictor,
)

# DINO-WM is cloned, not vendored (see README), so on a fresh checkout it is
# simply absent. Skip rather than fail: every test below builds their vit.py.
# Reuse the adapter's own _DWM so this check cannot drift from the real path.
pytestmark = pytest.mark.skipif(
    not (_DWM / "vit.py").exists(),
    reason=f"DINO-WM clone missing at {_DWM.parent}; see README",
)

EMB_DIM = 384          # dinov2_vits14
IMG_SIZE = 224         # 16x16 grid at patch size 14


def _hub_cache_cold() -> bool:
    return not (Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main").exists()


needs_weights = pytest.mark.skipif(
    _hub_cache_cold(),
    reason="DINOv2 torch.hub cache cold; warm TORCH_HOME on a networked machine",
)


# ── predictor (no weights needed) ─────────────────────────────────────────────

@pytest.mark.parametrize("num_patches,num_frames", [(1, 3), (1, 1), (4, 3)])
def test_vit_predictor_shape_and_cpu_forward(num_patches, num_frames):
    dim = 64
    pred = make_vit_predictor(
        num_patches=num_patches, num_frames=num_frames, dim=dim,
        depth=2, heads=4, mlp_dim=128, pool="mean",
    )
    n_tokens = num_frames * num_patches
    out = pred(torch.randn(2, n_tokens, dim))
    assert out.shape == (2, n_tokens, dim)


def test_causal_mask_is_a_buffer_that_follows_device():
    """Upstream's `.to('cuda')` would make this fail on CPU."""
    pred = make_vit_predictor(
        num_patches=1, num_frames=3, dim=64, depth=1, heads=4, mlp_dim=128, pool="mean",
    )
    attn = pred.transformer.layers[0][0]
    assert "bias" in dict(attn.named_buffers()), "causal mask is not a registered buffer"
    assert attn.bias.device.type == "cpu"
    # non-persistent: it is derived from config, so it must not bloat checkpoints
    assert not any(k.endswith("transformer.layers.0.0.bias") for k in pred.state_dict())


def test_mask_is_block_causal():
    dwm_vit = _vit_module()
    n_patch, n_frame = 4, 3
    m = dwm_vit.generate_mask_matrix(n_patch, n_frame)[0, 0]
    assert m.shape == (n_patch * n_frame, n_patch * n_frame)
    # frame 0 must not attend to frames 1..2; frame 1 must see frame 0 but not 2
    assert (m[:n_patch, n_patch:] == 0).all(), "frame 0 attends to the future"
    assert (m[n_patch:2 * n_patch, :n_patch] == 1).all(), "frame 1 cannot see frame 0"
    assert (m[n_patch:2 * n_patch, 2 * n_patch:] == 0).all(), "frame 1 attends to the future"


def test_second_predictor_does_not_corrupt_the_first():
    """vit.py keeps NUM_PATCHES/NUM_FRAMES as module globals (see the adapter)."""
    dwm_vit = _vit_module()
    original_attention = dwm_vit.Attention

    a = make_vit_predictor(num_patches=1, num_frames=3, dim=64,
                           depth=1, heads=4, mlp_dim=128, pool="mean")
    before = a(torch.randn(2, 3, 64))
    make_vit_predictor(num_patches=16, num_frames=3, dim=64,
                       depth=1, heads=4, mlp_dim=128, pool="mean")
    after = a(torch.randn(2, 3, 64) * 0 + 1)   # shape is what matters here

    assert before.shape == (2, 3, 64) and after.shape == (2, 3, 64)
    assert a.transformer.layers[0][0].bias.shape == (1, 1, 3, 3), "first mask was resized"
    assert dwm_vit.Attention is original_attention, "patched Attention left installed"


# ── encoder (needs the torch.hub cache) ───────────────────────────────────────

@needs_weights
@pytest.mark.parametrize("feature_key,n_patches", [
    ("x_norm_clstoken", 1),
    ("x_norm_patchtokens", (IMG_SIZE // 14) ** 2),
])
def test_dino_encoder_contract(feature_key, n_patches):
    enc = DinoEncoder(feature_key=feature_key, img_size=IMG_SIZE).eval()
    assert enc.emb_dim == EMB_DIM
    assert enc.n_patches == n_patches
    assert enc.out_dim == n_patches * EMB_DIM
    # 64x64 pendulum frames, upsampled inside the encoder
    out = enc(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, enc.out_dim)


@needs_weights
def test_dino_encoder_stays_frozen():
    enc = DinoEncoder(feature_key="x_norm_clstoken", img_size=IMG_SIZE)
    assert not any(p.requires_grad for p in enc.dino.parameters())
    enc.train()                      # Lightning does this every epoch
    assert not enc.dino.training, "frozen backbone must not re-enter train mode"
    out = enc(torch.randn(2, 3, 64, 64))
    assert not out.requires_grad, "frozen encoder must not build an autograd graph"


@needs_weights
def test_cls_encoder_matches_configured_embed_dim():
    """cfg.embed_dim: 384 in the experiment configs must equal out_dim, or the
    projector's matmul fails deep inside the first training step."""
    enc = DinoEncoder(feature_key="x_norm_clstoken", img_size=IMG_SIZE)
    assert enc.out_dim == EMB_DIM


def test_img_size_must_be_a_multiple_of_the_patch_size():
    with pytest.raises(ValueError, match="multiple of 14"):
        DinoEncoder(feature_key="x_norm_clstoken", img_size=64)


# ── the (x, c) predictor contract ─────────────────────────────────────────────
# REGRESSION. Arm B0D trained fine and then died at eval with
#   TypeError: ViTPredictor.forward() takes 2 positional arguments but 3 were given
# because upstream's predictor takes only the token stream while JEPA.predict --
# used at three sites in run_eval's rollout -- passes (emb, act_emb). Registering
# the model name in run_eval._LEWM was necessary but not sufficient. A second
# latent bug sat behind it: JEPA.predict also applies pred_proj unconditionally,
# so the builder's original pred_proj=None would have crashed straight after.

def test_action_concat_wrapper_signature_and_width():
    from hjepa.dino_wm_adapter import ActionConcatViTPredictor

    latent_dim, act_dim, frames = 16, 4, 3
    vit = make_vit_predictor(
        num_patches=1, num_frames=frames, dim=latent_dim + act_dim,
        depth=1, heads=4, mlp_dim=64, pool="mean",
    )
    pred = ActionConcatViTPredictor(vit, latent_dim=latent_dim)
    out = pred(torch.randn(2, frames, latent_dim), torch.randn(2, frames, act_dim))
    # must come back at the INPUT width so an autoregressive rollout can re-feed it
    assert out.shape == (2, frames, latent_dim)


def test_action_concat_wrapper_rejects_misaligned_actions():
    from hjepa.dino_wm_adapter import ActionConcatViTPredictor

    vit = make_vit_predictor(num_patches=1, num_frames=3, dim=20,
                             depth=1, heads=4, mlp_dim=64, pool="mean")
    pred = ActionConcatViTPredictor(vit, latent_dim=16)
    with pytest.raises(ValueError, match="does not align"):
        pred(torch.randn(2, 3, 16), torch.randn(2, 2, 4))   # 2 action steps, 3 tokens


@needs_weights
def test_b0d_supports_the_jepa_predict_path_run_eval_uses():
    """Build arm B0D from its real config and make the call run_eval makes."""
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, str(REPO / "third_party" / "leworldmodel"))
    from hjepa.builders import REGISTRY

    with initialize_config_dir(config_dir=str(REPO / "lehami" / "training" / "configs"),
                               version_base=None):
        cfg = compose(config_name="config",
                      overrides=["+experiment=phase0_dino_wm_cls_exp098_b0d"])
    model, _, _ = REGISTRY[cfg.model.name](cfg)

    d, a, h = cfg.model.latent_dim, cfg.model.action_emb_dim, cfg.history_size
    out = model.predict(torch.randn(2, h, d), torch.randn(2, h, a))
    assert out.shape == (2, h, d)
    assert model.pred_proj is not None, "JEPA.predict applies pred_proj unconditionally"
