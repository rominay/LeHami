"""Adapter over the DINO-WM clone in third_party/dino_wm (MIT license).

WHY AN ADAPTER RATHER THAN EDITS. third_party/ is a pinned clone of
https://github.com/gaoyuezhou/dino_wm at 0a9492fa12044b852ae9e001cc74604b79c8bb0c
and is never edited, matching how third_party/leworldmodel is treated.
Everything this project needs to change about their modules therefore lives
HERE, so `git -C third_party/dino_wm diff` stays empty and the comparison is
auditably against their code.

WHAT WE USE FROM THEM
  models/dino.py :: DinoV2Encoder  -- frozen DINOv2 via torch.hub. Returns
      (B, P, D): P = 256 patch tokens for feature_key="x_norm_patchtokens",
      or P = 1 for "x_norm_clstoken" (they unsqueeze a dummy patch axis, so
      both feature keys share one interface -- which is what lets the pooled
      and token arms differ only in `num_patches`).
  models/vit.py :: ViTPredictor    -- transformer over FLATTENED SPACE-TIME
      tokens (B, T*P, D) with a block-causal mask, i.e. frame t attends to all
      patches of frames <= t. Actions are NOT passed as a conditioning tensor;
      upstream concatenates them into the token stream (visual_world_model.py),
      unlike leworldmodel's ARPredictor which uses AdaLN via ConditionalBlock.

THE ONE FIX WE HAVE TO MAKE. Upstream Attention.__init__ (vit.py:39-58) ends with
    self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')
which hard-codes CUDA and leaves the causal mask a plain attribute, so it does
not follow .to(device) and is absent from state_dict. Measured consequence: a
predictor built on CPU has parameters on cpu and its mask on cuda:0, and the
forward dies with "expected self and mask to be on the same device". `_Attention`
below replicates that __init__ verbatim except for registering the mask as a
NON-PERSISTENT buffer -- identical mask tensor, identical numerics, but it moves
with the module and stays out of checkpoints.

UPSTREAM QUIRK WE MUST RESPECT. NUM_FRAMES/NUM_PATCHES are module-level globals
in vit.py, written by ViTPredictor.__init__ and read by Attention.__init__. That
only works because Attention is constructed (through Transformer) *inside*
ViTPredictor.__init__, after the globals are set. `make_vit_predictor` therefore
swaps the class in and out around a single construction and never leaves the
patched class installed -- two arms with different num_patches stay independent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# The upstream clones live in lehami/models/third_party/ -- the same
# location builders.py uses for _LEWM.
_DWM = Path(__file__).parents[1] / "third_party" / "dino_wm" / "models"
if str(_DWM) not in sys.path:
    sys.path.insert(0, str(_DWM))

DINO_WM_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
PATCH_SIZE = 14  # DINOv2; img_size must be a multiple of it


def _require_clone() -> None:
    if not (_DWM / "vit.py").exists():
        raise FileNotFoundError(
            f"DINO-WM clone missing at {_DWM.parent}. Restore it with:\n"
            f"  git clone https://github.com/gaoyuezhou/dino_wm.git third_party/dino_wm\n"
            f"  git -C third_party/dino_wm checkout {DINO_WM_COMMIT}"
        )


# ── predictor (their models/vit.py) ───────────────────────────────────────────

def _vit_module():
    _require_clone()
    import vit as dwm_vit  # noqa: PLC0415 — path is set above
    return dwm_vit


def make_vit_predictor(*, num_patches: int, num_frames: int, dim: int, **kwargs):
    """DINO-WM's ViTPredictor, with the causal mask as a device-following buffer.

    kwargs are theirs: depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout.
    conf/predictor/vit.yaml upstream is depth=6, heads=16, mlp_dim=2048,
    dropout=0.1, pool='mean' -- identical to the predictor block of this repo's
    lewm_ms_conv.yaml, so the two baselines differ in architecture, not in
    capacity knobs.
    """
    dwm_vit = _vit_module()

    class _Attention(dwm_vit.Attention):
        """vit.py:39-58 verbatim, except the mask is a buffer, not `.to('cuda')`."""

        def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
            nn.Module.__init__(self)
            inner_dim = dim_head * heads
            project_out = not (heads == 1 and dim_head == dim)
            self.heads = heads
            self.scale = dim_head ** -0.5
            self.norm = nn.LayerNorm(dim)
            self.attend = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = (
                nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                if project_out else nn.Identity()
            )
            self.register_buffer(
                "bias",
                dwm_vit.generate_mask_matrix(dwm_vit.NUM_PATCHES, dwm_vit.NUM_FRAMES),
                persistent=False,
            )

    original = dwm_vit.Attention
    dwm_vit.Attention = _Attention
    try:
        return dwm_vit.ViTPredictor(
            num_patches=num_patches, num_frames=num_frames, dim=dim, **kwargs,
        )
    finally:
        dwm_vit.Attention = original  # never leave the patched class installed


class ActionConcatViTPredictor(nn.Module):
    """DINO-WM's ViTPredictor behind this repo's (x, c) predictor signature.

    WHY THIS EXISTS. Upstream `ViTPredictor.forward(x)` takes ONLY the token
    stream: actions never reach it as an argument, they are concatenated into the
    stream by visual_world_model.py (conf/train.yaml `concat_dim=1` -- the action
    embedding is tiled across patches and concatenated on the FEATURE axis).
    Every predictor call site in this repo passes (x, c) instead, including
    `JEPA.predict`, which run_eval's rollout uses at three places. Without this
    wrapper the arm TRAINS FINE and then dies at eval with
        TypeError: ViTPredictor.forward() takes 2 positional arguments but 3 were given
    which is the worst possible time to find out. Centralising the concat here
    means the training forward and every eval path share one convention.

    The wrapper widens each token by the action embedding, runs the predictor,
    and slices the action channels back off so the output is the same width as
    the input -- what the autoregressive rollout needs to re-feed its own output.
    """

    def __init__(self, vit: nn.Module, latent_dim: int):
        super().__init__()
        self.vit = vit
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, latent_dim), c: (B, T, action_emb_dim) -> (B, T, latent_dim)."""
        if c.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"action embedding {tuple(c.shape)} does not align with tokens "
                f"{tuple(x.shape)} on the batch/time axes"
            )
        return self.vit(torch.cat([x, c], dim=-1))[..., : self.latent_dim]


# ── encoder (their models/dino.py) ────────────────────────────────────────────

class DinoEncoder(nn.Module):
    """Frozen DINO-WM DinoV2Encoder → (B, P*D), the ConvEncoder contract.

    Flattened rather than (B, P, D) so _encode_sequence's `.view(B, T, -1)` keeps
    working untouched; callers that need the token axis reshape to
    (B, T, n_patches, emb_dim) using the attributes below. For feature_key
    "x_norm_clstoken" n_patches == 1 and the flattening is a no-op, so the
    Hamiltonian arm sees a plain (B, T, 384) exactly like ConvEncoder's output.

    RESIZE. Frames arrive at cfg.img_size (64 for the pendulum) already
    ImageNet-normalised -- train.py:57-60 applies dt.dataset_stats.ImageNet in the
    dataloader, which is the same normalisation DINOv2 expects, so this must NOT
    normalise again. Upsampling happens HERE on GPU rather than by raising
    cfg.img_size, which would push (224/64)^2 = 12x the bytes through the CPU
    dataloader for an identical result: bicubic weights sum to 1, so resize and
    an affine normalisation commute.

    The backbone is pinned to eval() whenever frozen, overriding Lightning's
    train(), so dropout/drop-path never fire inside a fixed feature extractor.
    """

    FEATURE_KEYS = ("x_norm_clstoken", "x_norm_patchtokens")

    def __init__(
        self,
        name: str = "dinov2_vits14",
        feature_key: str = "x_norm_clstoken",
        img_size: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        _require_clone()
        if feature_key not in self.FEATURE_KEYS:
            raise ValueError(f"feature_key must be one of {self.FEATURE_KEYS}, got {feature_key!r}")
        if img_size % PATCH_SIZE:
            raise ValueError(f"img_size must be a multiple of {PATCH_SIZE}, got {img_size}")
        self._check_hub_cache(name)

        import dino as dwm_dino  # noqa: PLC0415 — path is set above

        self.dino      = dwm_dino.DinoV2Encoder(name=name, feature_key=feature_key)
        self.img_size  = img_size
        self.freeze    = freeze
        self.emb_dim   = self.dino.emb_dim
        self.n_patches = 1 if feature_key == "x_norm_clstoken" else (img_size // PATCH_SIZE) ** 2
        self.out_dim   = self.n_patches * self.emb_dim
        if freeze:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _check_hub_cache(name: str) -> None:
        """Fail fast and loudly instead of hanging on a compute node with no network.

        torch.hub.load git-clones facebookresearch/dinov2 and downloads an 84 MB
        checkpoint (89 MB of TORCH_HOME in total). On an offline compute node that
        is a long silent timeout, not an error, so check before calling it.
        """
        hub = Path(torch.hub.get_dir())
        if (hub / "facebookresearch_dinov2_main").exists():
            return
        raise FileNotFoundError(
            f"DINOv2 torch.hub cache not found under {hub}.\n"
            f"Download the weights ONCE on a machine with network access (~89 MB),\n"
            f"with the same TORCH_HOME that training will use:\n"
            f"  bash lehami/experiments/download_dinov2.sh"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(
                x, size=(self.img_size,) * 2, mode="bicubic", align_corners=False,
            )
        with torch.set_grad_enabled(not self.freeze):
            tokens = self.dino(x)                    # (B, P, D)
        return tokens.reshape(tokens.shape[0], -1)   # (B, P*D)
