"""Model builders and registry.

Each builder takes a Hydra DictConfig and returns (model, sigreg, forward_fn); the
forward_fn signature is (self, batch, stage) -> dict, matching spt.Module.
`REGISTRY` maps the `model.name` of a config to its builder.

The models are assembled from LeWorldModel components (`jepa.JEPA`, `module.MLP`,
`module.Embedder`, `module.SIGReg`, `module.ARPredictor`), imported from the clone
in lehami/models/third_party/leworldmodel/, which this module puts on sys.path.
The Hamiltonian models replace the predictor with `ham_flow.HamiltonianPredictor`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf

# Make the LeWorldModel clone importable (jepa, module).
_LEWM = Path(__file__).parents[1] / "third_party" / "leworldmodel"
if str(_LEWM) not in sys.path:
    sys.path.insert(0, str(_LEWM))


# ── shared helpers ────────────────────────────────────────────────────────────

def _vit(cfg: DictConfig):
    return spt.backbone.utils.vit_hf(
        "tiny", patch_size=cfg.model.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )


class ConvEncoder(torch.nn.Module):
    """Four-block ConvNet: (B, 3, H, W) → (B, embed_dim).

    Each block: Conv2d → BatchNorm2d → GELU → MaxPool2d(2).
    Final block uses AdaptiveAvgPool2d(1) instead of MaxPool so the network
    accepts any input resolution (not just multiples of 16).
    Channels: 3 → 32 → 64 → 128 → embed_dim.

    `window` / `lag` are the per-frame (W=1) values of the temporal-encoder
    contract shared with TemporalConvEncoder (see `_encode_sequence`); they are
    plain class constants and change nothing about this module's behaviour.
    """

    window: int = 1      # frames consumed per output embedding
    lag: float = 0.0     # centroid lag of the output w.r.t. its newest frame

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3,   32,       3, padding=1), torch.nn.BatchNorm2d(32),       torch.nn.GELU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,  64,       3, padding=1), torch.nn.BatchNorm2d(64),       torch.nn.GELU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,  128,      3, padding=1), torch.nn.BatchNorm2d(128),      torch.nn.GELU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, embed_dim, 3, padding=1), torch.nn.BatchNorm2d(embed_dim), torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvEncoder(torch.nn.Module):
    """(2+1)D encoder: shared per-frame stem + ONE symmetric temporal mix at stage 1.

    Maps a clip (B, T, 3, H, W) → (B, T-window+1, embed_dim).  Block 1 of
    `ConvEncoder` is run **once per frame** and shared across overlapping
    windows (no recomputation); the temporal mixing happens after it, at
    32×32 resolution over 32 channels, before block 2.  Blocks 2-4, the
    pooling and the projector are byte-identical to `ConvEncoder`.

    Why the temporal kernel is *structurally* symmetric
    ---------------------------------------------------
    `q = e(theta)` is a point transformation, and point transformations lift to
    canonical transformations of the cotangent bundle automatically — that free
    guarantee is the only reason (q, p) is a meaningful canonical pair, and a
    velocity-dependent `q` gives it up.  Expanding an EVEN temporal filter about
    its own centroid,

        q_t = e(theta_{t-lag}) + 0.5 sigma^2 [e''(theta) omega^2
                                              + e'(theta) omega_dot] + O(sigma^4)

    — a pure lag, plus an omega^2 term (even under t → -t), plus
    omega_dot = -(g/L) sin(theta), a function of theta alone and hence
    absorbable.  **No odd-in-omega component appears**, so `q` stays even,
    `p = SG(q)` stays odd, and the parity assignment survives.  An ODD
    (differencing) filter instead injects e'(theta) omega into `q` directly and
    breaks both separability and the q/p distinction.  Symmetry is therefore
    imposed by construction (`cat([w, w_c, w.flip(-1)])`), never by a penalty:
    a penalty would make the parity claim empirical rather than structural.

    `lag`
    -----
    A symmetric filter over a *trailing* window is symmetric about its own
    centroid, so it is causal but carries a pure lag of (window-1)/2 frames.
    That is harmless internally (targets come from the same encoder), but
    **probes are not internal**: comparing q_t against the true theta_t injects
    an error ~ omega * lag * dt that grows with speed and would look like an
    energy-dependent position error.  Eval/probe code must shift its targets by
    `lag`.

    variant
    -------
    "full"       Conv1d(32, 32, window) per spatial location, symmetric-tied.
    "depthwise"  the same with groups=32 (cheap control).
    "free"       UNTIED Conv1d(32, 32, window): "full" with the flip-tie
                 removed, so the kernel is a free (a, b, c).  For the LeWM
                 baseline only (arm b3f).  Decomposing a kernel about its
                 centre, the ODD part w- = (w2 - w0)/2 is the only component
                 that multiplies a term linear in omega; the tie sets w- = 0
                 identically, which is why a tied encoder cannot represent the
                 SIGN of a velocity.  Untying makes the sign REPRESENTABLE, not
                 guaranteed: the delta init starts at w- = 0 and nothing in the
                 baseline's objective asks for it to move.
                 `lag` is a DECLARED convention here, not (W-1)/2, because an
                 untied kernel's centroid is learned.
                 Do NOT use it for a Hamiltonian arm: an odd component injects
                 e'(theta) omega into q, destroying the point transformation
                 and with it the canonical pair (see the parity note above).
    "diff"       Conv2d(6→32) fed [x_t ; x_t - x_{t-1}]: a deliberately
                 parity-violating control, expected to degrade the Hamiltonian
                 diagnostics (p → -p round trip, energy drift, dt-refinement).
                 Not used by any experiment in the paper.
    """

    def __init__(self, embed_dim: int = 192, window: int = 3, variant: str = "full",
                 lag: int | None = None):
        super().__init__()
        if variant not in ("full", "depthwise", "diff", "free"):
            raise ValueError(
                f"variant must be full|depthwise|diff|free, got {variant!r}")
        if variant == "diff":
            if window != 2:
                raise ValueError(f"variant='diff' requires window=2, got {window}")
        elif window < 3 or window % 2 == 0:
            raise ValueError(f"variant={variant!r} requires an odd window >= 3, got {window}")

        self.window  = window
        self.variant = variant
        # Centroid lag of the output w.r.t. its newest frame.  For the TIED
        # variants this is a theorem -- a symmetric filter over a trailing
        # window is symmetric about its own centroid -- so `lag` may not
        # override it.  For "free" the centroid is LEARNED and no longer pinned
        # to the middle of the window, so the lag is a declared convention that
        # eval and the planner must be told; measure the realised centroid from
        # `temporal_weight()` and check it against this before trusting a probe.
        if variant == "diff":
            if lag not in (None, 0):
                raise ValueError(f"variant='diff' has lag 0 by construction, got {lag}")
            self.lag = 0.0
        elif variant == "free":
            lag = (window - 1) // 2 if lag is None else int(lag)
            if not 0 <= lag <= window - 1:
                raise ValueError(
                    f"lag must be an integer in [0, {window - 1}] for window={window}, "
                    f"got {lag}")
            self.lag = float(lag)
        else:
            if lag is not None and float(lag) != (window - 1) / 2.0:
                raise ValueError(
                    f"variant={variant!r} is symmetric-tied, so its lag is fixed at "
                    f"{(window - 1) / 2.0}; got lag={lag}. Use variant='free' to "
                    f"choose one.")
            self.lag = (window - 1) / 2.0

        in_ch = 6 if variant == "diff" else 3
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 3, padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.GELU(), torch.nn.MaxPool2d(2),
        )
        self.rest = torch.nn.Sequential(
            torch.nn.Conv2d(32,  64,        3, padding=1), torch.nn.BatchNorm2d(64),        torch.nn.GELU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,  128,       3, padding=1), torch.nn.BatchNorm2d(128),       torch.nn.GELU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, embed_dim, 3, padding=1), torch.nn.BatchNorm2d(embed_dim), torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

        if variant == "diff":
            self.w_half = self.w_centre = self.w_free = self.t_bias = None
        else:
            groups   = 32 if variant == "depthwise" else 1
            self.groups = groups
            in_per_g = 32 // groups
            half     = window // 2
            # delta init: the tap the latent REPRESENTS is the identity and every
            # other tap is 0, so at init the module is EXACTLY the per-frame
            # ConvEncoder and any gain is attributable to the window rather than
            # to a different init.  That tap is index W-1-lag, which is the
            # centre for the tied variants and for "free" at its default lag --
            # so a free arm starts byte-identical to the "full" arm it controls.
            if variant == "free":
                w_f   = torch.zeros(32, in_per_g, window)
                delta = window - 1 - int(self.lag)
                for c in range(32):
                    w_f[c, 0 if groups == 32 else c, delta] = 1.0
                self.w_free   = torch.nn.Parameter(w_f)
                self.w_half   = self.w_centre = None
            else:
                w_c = torch.zeros(32, in_per_g, 1)
                for c in range(32):
                    w_c[c, 0 if groups == 32 else c, 0] = 1.0
                self.w_centre = torch.nn.Parameter(w_c)
                self.w_half   = torch.nn.Parameter(torch.zeros(32, in_per_g, half))
                self.w_free   = None
            self.t_bias   = torch.nn.Parameter(torch.zeros(32))

    def temporal_weight(self) -> torch.Tensor:
        """The temporal kernel.

        The tied variants materialise (a, …, b, …, a) from half of it — never
        stored, so the symmetry cannot drift.  "free" returns its parameter
        unchanged: no flip and no tie, so a non-zero odd component is reachable.
        """
        if self.variant == "free":
            return self.w_free
        return torch.cat([self.w_half, self.w_centre, self.w_half.flip(-1)], dim=-1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T-window+1, embed_dim).

        Output index j corresponds to input frame j+window-1 (the newest frame
        of window j … j+window-1), and represents time j+window-1-`lag`.
        Valid padding only: temporal padding would fabricate frames at clip
        edges and corrupt the lag/parity accounting.
        """
        B, T = x.shape[:2]
        W    = self.window
        if T < W:
            raise ValueError(f"clip too short for temporal encoder: T={T} < window={W}")

        if self.variant == "diff":
            pair = torch.cat([x[:, 1:], x[:, 1:] - x[:, :-1]], dim=2)   # (B, T-1, 6, H, W)
            n    = T - 1
            f    = self.stem(pair.reshape(B * n, *pair.shape[2:]))
            return self.rest(f).view(B, n, -1)

        # shared per-frame stem — run ONCE per frame, not once per window
        f = self.stem(x.reshape(B * T, *x.shape[2:]))                   # (B*T, 32, H/2, W/2)
        C, Hs, Ws = f.shape[1:]
        # (B, T, C, Hs, Ws) → per spatial location, a length-T signal of C channels
        f = f.view(B, T, C, Hs, Ws).permute(0, 3, 4, 2, 1).reshape(B * Hs * Ws, C, T)
        f = torch.nn.functional.conv1d(
            f, self.temporal_weight(), self.t_bias, groups=self.groups)  # (·, C, T-W+1)
        Te = f.shape[-1]
        f = f.view(B, Hs, Ws, C, Te).permute(0, 4, 3, 1, 2).reshape(B * Te, C, Hs, Ws)
        return self.rest(f).view(B, Te, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "TemporalConvEncoder consumes a clip; call forward_sequence("
            "(B, T, 3, H, W)) — or go through _encode_sequence(), which "
            "dispatches on encoder.window."
        )


def _conv_encoder(cfg: DictConfig):
    """Single encoder factory — both the Hamiltonian and the LeWM conv builders
    call this, so the two arms get the same encoder by construction.
    """
    window  = int(OmegaConf.select(cfg, "model.enc_window", default=1) or 1)
    variant = str(OmegaConf.select(cfg, "model.enc_variant", default="full") or "full")
    # enc_lag is meaningful for variant='free' only, whose kernel centroid is
    # learned; the tied variants raise on anything but their own fixed value.
    lag     = OmegaConf.select(cfg, "model.enc_lag", default=None)
    if window == 1:
        return ConvEncoder(embed_dim=cfg.embed_dim)
    return TemporalConvEncoder(embed_dim=cfg.embed_dim, window=window, variant=variant,
                               lag=None if lag is None else int(lag))


def _encoder(cfg: DictConfig):
    """Encoder factory for every builder, so the Hamiltonian and baseline arms of
    an experiment always share the same encoder.

    `model.encoder` is "conv" (default; `_conv_encoder`) or "dino" (a frozen
    DINOv2, via `dino_wm_adapter.DinoEncoder`).
    """
    kind = str(OmegaConf.select(cfg, "model.encoder", default="conv") or "conv")
    if kind == "conv":
        return _conv_encoder(cfg)
    if kind != "dino":
        raise ValueError(f"model.encoder must be conv|dino, got {kind!r}")

    from lehami.dino_wm_adapter import DinoEncoder

    enc = DinoEncoder(
        name=str(OmegaConf.select(cfg, "model.dino_name", default="dinov2_vits14")),
        feature_key=str(OmegaConf.select(
            cfg, "model.dino_feature_key", default="x_norm_clstoken")),
        img_size=int(OmegaConf.select(cfg, "model.dino_img_size", default=224)),
        freeze=bool(OmegaConf.select(cfg, "model.freeze_encoder", default=True)),
    )
    # cfg.embed_dim feeds the projector's input_dim in every builder, so a
    # mismatch would surface as an opaque matmul error deep in the first step.
    # out_dim == emb_dim only for the CLS key (n_patches == 1); the patch-token
    # key would need a builder that bypasses the projector, which none does.
    if enc.out_dim != cfg.embed_dim:
        raise ValueError(
            f"cfg.embed_dim={cfg.embed_dim} but the encoder emits {enc.out_dim}-d "
            f"features ({enc.n_patches} patches x {enc.emb_dim}); set "
            f"embed_dim: {enc.out_dim} in the experiment config."
        )
    return enc


def _encode_sequence(encoder, pixels: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, H, W) → (B, T-window+1, enc_dim), for either encoder.

    For window == 1 this is the flat per-frame `encoder(pixels.reshape(B*T, ...))`.
    """
    B, T = pixels.shape[:2]
    if getattr(encoder, "window", 1) == 1:
        return encoder(pixels.reshape(B * T, *pixels.shape[2:])).view(B, T, -1)
    return encoder.forward_sequence(pixels)


def _episode_h(batch: dict, cfg: DictConfig) -> torch.Tensor | None:
    """Per-sample integrator step h = dt_episode / dt_ref, or None.

    Returns None unless the batch carries a `dt_frame` column AND the config
    sets `model.dt_ref`; otherwise the scalar `ham_flow.dt` is used.

    WHY A RATIO. Rescaling time t -> λt with H -> H/λ leaves every trajectory
    invariant, so only the ratio of steps ACROSS episodes is identifiable; the
    absolute unit is a gauge freedom the learned H absorbs. `dt_ref` picks where
    to put it, and putting it at h = 1 on the reference arm keeps the network
    outputs O(1) — the leapfrog multiplies dH by h, so an h of 0.05 s would
    force dH ~20x larger against an init and a gradient clip tuned for h = 1.

    WHY IT IS THE FIX FOR MIXED RATES. In frame units the dynamics carry a dt^2
    factor, so a V(q) fitted at one rate is wrong by (dt/dt')^2 at another.
    Dividing the step by dt_ref removes that factor from the target and makes
    ONE V valid across every arm.  The momentum must ride the same clock —
    p = dq/dh, i.e. the frame-unit stencil divided by h — or K(p) is handed an
    inconsistent scale.
    """
    dt_ref = OmegaConf.select(cfg, "model.dt_ref", default=None)
    if dt_ref is None or "dt_frame" not in batch:
        return None
    if float(dt_ref) <= 0:
        raise ValueError(f"model.dt_ref must be > 0, got {dt_ref}")
    dt_col = batch["dt_frame"].float()
    # (B, T, 1) or (B, T) -> (B,). dt is constant within an episode, hence within
    # a clip; take frame 0 and assert the clip does not straddle a boundary.
    dt_ep = dt_col.reshape(dt_col.shape[0], dt_col.shape[1], -1)[:, :, 0]
    if not torch.allclose(dt_ep, dt_ep[:, :1].expand_as(dt_ep)):
        raise ValueError("dt varies within a clip — clips must not cross episodes")
    return dt_ep[:, 0] / float(dt_ref)


def _participation_rank(x: torch.Tensor) -> torch.Tensor:
    """Effective rank as the participation ratio of the singular values.

    (sum s)^2 / sum s^2 on the column-centred matrix -- the definition
    `_encoder_diagnostics` has always used for eff_rank_q, so eff_rank_p is
    directly comparable to it within a run.  Accepts (N, d) or (B, T, d).
    """
    xf = x.reshape(-1, x.shape[-1]).float()
    sv = torch.linalg.svdvals(xf - xf.mean(0, keepdim=True))
    return (sv.sum() ** 2) / (sv.pow(2).sum() + 1e-12)


def _encoder_diagnostics(encoder, z: torch.Tensor) -> dict:
    """Per-step encoder diagnostics, logged during training.

    eff_rank_q  participation ratio of q's singular values (effective rank of
                the position embedding).
    tap_ratio   |off-centre taps| / |centre tap| of the temporal kernel.  Starts
                at exactly 0 (delta init) and says how much of the window the
                model actually chose to use — an arm that ends near 0 has
                reverted to a per-frame encoder.
    odd_ratio   ||w - flip(w)|| / ||w + flip(w)||, the odd/even energy about the
                kernel centre.  EXACTLY 0 at every step for a tied kernel, so
                the "full" arms log a hard zero and the metric is its own
                control.  For "free" it is the headline diagnostic: the odd part
                is the ONLY component that can carry the SIGN of a velocity, so
                an arm ending near 0 declined the capacity it was given.

    Only eff_rank_q for a per-frame encoder.
    """
    out = {}
    with torch.no_grad():
        out["eff_rank_q"] = _participation_rank(z)
        if getattr(encoder, "variant", None) in ("full", "depthwise", "free"):
            w = encoder.temporal_weight()
            half = w.shape[-1] // 2
            centre = w[..., half].abs().sum()
            off = w[..., :half].abs().sum() + w[..., half + 1:].abs().sum()
            out["tap_ratio"] = off / (centre + 1e-12)
            out["odd_ratio"] = ((w - w.flip(-1)).norm()
                                / ((w + w.flip(-1)).norm() + 1e-12))
    return out


_ENC_DIAG_KEYS = ("eff_rank_q", "tap_ratio", "odd_ratio", "std_p", "std_q",
                  "eff_rank_p")


def _make_sigreg(cfg: DictConfig, latent_dim: int):
    from module import SIGReg  # third_party/leworldmodel/module.py
    if latent_dim < 2:
        raise ValueError(f"SIGReg's random projections need latent_dim >= 2, got {latent_dim}")
    return SIGReg(**cfg.loss.sigreg.kwargs)


# ── builders ──────────────────────────────────────────────────────────────────


def _action_emb(batch, model, cfg, device, B, start, n):
    """Action embedding for `n` consecutive LATENT frames starting at `start`.

    Datasets with no `action` column (the conservative pendulum) get a constant
    zero action token; only the actuated dataset carries torque.

    Indexing lives in LATENT time.  At W = 1 latent frame == pixel frame; at
    W > 1 the encoder consumes W-1 lead-in frames, so the action index is shifted
    by the latent->pixel offset computed below.
    """
    a = batch.get("action")
    if a is None:
        if not getattr(_action_emb, "_warned", False):
            _action_emb._warned = True
            print("[action] NO `action` column in the batch — feeding the zero "
                  "token (this run is UNACTUATED). Add `action` to "
                  "cfg.data.keys_to_load if that is not intended.", flush=True)
        return model.action_encoder(
            torch.zeros(B, n, cfg.data.action_dim, device=device))
    # LATENT TIME -> PIXEL TIME.  A window-W encoder turns T pixel frames into
    # T-W+1 latents, so latent i is built from pixels [i, i+W-1] -- but WHICH of
    # those frames it stands for depends on the stencil, and the encoder is the
    # authority on that, not the window size.  `TemporalConvEncoder.lag` is the
    # centroid lag w.r.t. the newest frame in the window, so latent i represents
    # pixel frame  i + (W-1) - lag:
    #     full / depthwise (symmetric, odd W):  lag = (W-1)/2  ->  offset (W-1)/2
    #     diff             (W = 2, causal)   :  lag = 0        ->  offset 1
    #     per-frame        (W = 1)           :  offset 0
    # Assuming a causal stencil and using W-1 unconditionally is WRONG for the
    # symmetric variants -- it would read the action a full (W-1)/2 frames late,
    # and the error is silent at W=1 (the offset is zero either way).
    _w = int(getattr(model.encoder, "window", 1))
    _lag = float(getattr(model.encoder, "lag", 0.0))
    off_f = (_w - 1) - _lag
    assert float(off_f).is_integer(), (
        f"non-integral latent->pixel offset {off_f} for window={_w}, lag={_lag}; "
        "an action cannot be read between frames")
    off = int(off_f)
    if not getattr(_action_emb, "_logged", False):
        _action_emb._logged = True
        print(f"[action] LIVE: batch['action'] {tuple(a.shape)}, "
              f"|u| mean {a.abs().mean().item():.3f}, latent->pixel offset {off} "
              f"— V(q, u) is conditioned.", flush=True)
    assert start + off + n <= a.shape[1], (
        f"action index {start + off + n} exceeds the clip's {a.shape[1]} frames: "
        f"T must be T_eff + (W-1); check num_preds in the experiment config")
    a = a[:, start + off : start + off + n].to(device=device, dtype=torch.float32)
    if a.dim() == 2:                       # (B, n) -> (B, n, 1)
        a = a.unsqueeze(-1)
    return model.action_encoder(a)


def _build_lewm_ms(cfg: DictConfig):
    """LeWM baseline with a ViT-tiny encoder, trained by multi-step rollout.

    LeWorldModel's ViT encoder + ARPredictor (causal transformer, history_size=3
    context), with a latent_dim=64 projector bottleneck so its latent matches the
    Hamiltonian model's phase-space dimension (2 x 32). The predictor, pred_proj
    and action encoder all work in this latent space.

    Training unrolls autoregressively for n_unroll steps: a fixed-size context
    window slides forward and the model's own most recent prediction is fed back
    in as context (full BPTT, no stop-gradient) -- the same self-fed rollout used
    at evaluation, so there is no teacher-forcing / exposure-bias mismatch. SIGReg
    is applied to the encoder output over the whole clip, never to the model's own
    predictions; its weight is set in the experiment config.
    """
    import torch.nn.functional as F
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder, SIGReg

    vit_dim    = cfg.embed_dim        # ViT's native output width (192)
    latent_dim = cfg.model.latent_dim  # projector bottleneck (64)

    model = JEPA(
        encoder=_vit(cfg),
        predictor=ARPredictor(
            num_frames=cfg.history_size, input_dim=latent_dim,
            hidden_dim=latent_dim, output_dim=latent_dim,
            **OmegaConf.to_container(cfg.model.predictor, resolve=True),
        ),
        action_encoder=Embedder(input_dim=cfg.data.action_dim, emb_dim=latent_dim),
        projector=MLP(input_dim=vit_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(input_dim=latent_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
    )
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    h = cfg.history_size   # sliding context window length (ARPredictor's positional embedding is sized to this)

    def _forward(self, batch, stage):
        B, T = batch["pixels"].shape[:2]
        device = batch["pixels"].device
        n_unroll = T - h
        assert n_unroll >= 1, f"clip too short: T={T}, history_size={h}"

        batch["action"] = torch.zeros(B, T, cfg.data.action_dim, device=device)
        output = self.model.encode(batch)
        emb = output["emb"]   # (B, T, d) -- act_emb unused: we build a constant zero_act below instead

        # constant zero action embedding, reused every step -- matches
        # utils.py:_rollout_lewm (no actions in this dataset, nothing to slide)
        zero_act = self.model.action_encoder(
            torch.zeros(B, h, cfg.data.action_dim, device=device))

        ctx_emb = emb[:, :h]    # (B, h, d) -- sliding window, starts as real GT context
        pos_loss = 0.0
        for step in range(n_unroll):
            pred_seq = self.model.predict(ctx_emb, zero_act)   # (B, h, d)
            z_pred = pred_seq[:, -1]                            # next-frame prediction (full-context output)
            tgt_idx = h + step
            pos_loss = pos_loss + F.mse_loss(z_pred, emb[:, tgt_idx])
            ctx_emb = torch.cat([ctx_emb[:, 1:], z_pred.unsqueeze(1)], dim=1)   # slide window
        pos_loss = pos_loss / n_unroll

        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pos_loss + cfg.loss.sigreg.weight * sigreg_loss
        output_d = {"loss": loss, "pred_loss": pos_loss, "sigreg_loss": sigreg_loss}
        self.log_dict({f"{stage}/{k}": v.detach() for k, v in output_d.items() if "loss" in k},
                      on_step=True, on_epoch=True, sync_dist=True)
        return output_d

    return model, sigreg, _forward


def _build_lewm_ms_conv(cfg: DictConfig):
    """LeWM baseline with the ConvNet encoder, trained by multi-step rollout.

    Identical to `_build_lewm_ms` -- same ARPredictor, latent_dim=64 bottleneck,
    self-fed rollout with full BPTT, SIGReg placement and history_size=3 -- except
    the ViT-tiny encoder is replaced by `_conv_encoder` (ConvEncoder, ~315K params
    vs ~5.4M, or its temporal-window variant), the same encoder the Hamiltonian
    builder `_build_lehami_sg_sep_sreg_joint_ms_conv` uses.

    JEPA.encode() cannot be reused: it hardcodes the ViT call signature
    (`interpolate_pos_encoding=True`, then `.last_hidden_state[:, 0]`), while the
    conv encoders return plain embeddings, so the encode step is inlined here.
    """
    import torch.nn.functional as F
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder, SIGReg

    enc_dim    = cfg.embed_dim         # ConvEncoder output channels (192)
    latent_dim = cfg.model.latent_dim  # projector bottleneck (64)

    model = JEPA(
        encoder=_conv_encoder(cfg),
        predictor=ARPredictor(
            num_frames=cfg.history_size, input_dim=latent_dim,
            hidden_dim=latent_dim, output_dim=latent_dim,
            **OmegaConf.to_container(cfg.model.predictor, resolve=True),
        ),
        action_encoder=Embedder(input_dim=cfg.data.action_dim, emb_dim=latent_dim),
        projector=MLP(input_dim=enc_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(input_dim=latent_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
    )
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    h = cfg.history_size

    def _forward(self, batch, stage):
        B, T = batch["pixels"].shape[:2]
        device = batch["pixels"].device
        # A temporal encoder consumes W-1 lead-in frames, so every index below
        # lives in LATENT time (length T_eff), not pixel time.  Holding
        # n_unroll = 8 fixed therefore requires T = 11 + (W-1) here, against
        # 14 + (W-1) for the Hamiltonian arm, whose anchor needs a 4-frame causal
        # momentum stencil and whose last target needs 2 frames after it.
        W     = getattr(self.model.encoder, "window", 1)
        T_eff = T - W + 1
        n_unroll = T_eff - h
        assert n_unroll >= 1, f"clip too short: T={T}, enc_window={W}, history_size={h}"

        # ConvEncoder returns (B*T, enc_dim) directly -- no .last_hidden_state
        enc = _encode_sequence(self.model.encoder, batch["pixels"].float())
        emb = self.model.projector(enc.reshape(B * T_eff, -1)).view(B, T_eff, latent_dim)

        ctx_emb = emb[:, :h]
        pos_loss = 0.0
        for step in range(n_unroll):
            # the h actions driving the context window that predicts frame h+step
            act = _action_emb(batch, self.model, cfg, device, B, step, h)
            pred_seq = self.model.predict(ctx_emb, act)        # (B, h, d)
            z_pred = pred_seq[:, -1]
            tgt_idx = h + step
            pos_loss = pos_loss + F.mse_loss(z_pred, emb[:, tgt_idx])
            ctx_emb = torch.cat([ctx_emb[:, 1:], z_pred.unsqueeze(1)], dim=1)
        pos_loss = pos_loss / n_unroll

        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pos_loss + cfg.loss.sigreg.weight * sigreg_loss
        output_d = {"loss": loss, "pred_loss": pos_loss, "sigreg_loss": sigreg_loss}
        diag = _encoder_diagnostics(self.model.encoder, emb)
        self.log_dict({f"{stage}/{k}": v.detach()
                       for k, v in {**output_d, **diag}.items()
                       if "loss" in k or k in _ENC_DIAG_KEYS},
                      on_step=True, on_epoch=True, sync_dist=True)
        return output_d

    return model, sigreg, _forward


def _build_dino_wm_cls(cfg: DictConfig):
    """DINO-WM's predictor on a frozen DINOv2 encoder (the baseline of the DINOv2
    comparison).

    Encoder, projector and latent_dim are IDENTICAL to the Hamiltonian arm
    (`_build_lehami_sg_sep_sreg_joint_ms_conv` with `model.encoder: dino`); the only
    difference is the predictor. With the encoder frozen it cannot co-adapt with
    either predictor, so a difference between the arms is attributable to the
    predictor.

    WHAT IS FAITHFUL TO DINO-WM (third_party/dino_wm @ 0a9492f)
      * models/vit.py::ViTPredictor itself, via make_vit_predictor -- a
        block-causal transformer over FLATTENED space-time tokens, frame t
        attending to all patches of frames <= t.
      * conf/encoder/dino_cls.yaml: the CLS feature key, i.e. num_patches == 1.
        This is one of their two shipped encoder configs; their DEFAULT is
        x_norm_patchtokens, so this arm is DINO-WM's CLS variant.
      * conf/train.yaml concat_dim=1: the action embedding is concatenated on the
        FEATURE axis (tiled across patches, a no-op at num_patches=1), giving
        predictor width latent_dim + action_emb_dim. Upstream ViTPredictor.forward
        takes no conditioning tensor at all, unlike leworldmodel's ARPredictor,
        which uses AdaLN via ConditionalBlock -- so this must not be routed
        through JEPA.predict().

    WHAT IS DELIBERATELY NOT FAITHFUL
      * Upstream predicts in raw 384-d DINO space with no projector. This arm
        keeps the MLP(enc_dim -> latent_dim) bottleneck, because _build_lewm_ms_conv
        and the Hamiltonian builders have one, and dropping it would make the two
        arms differ in the bottleneck as well as the predictor. Parity was chosen
        over fidelity.
      * The multi-step self-fed rollout and SIGReg placement are this repo's
        (matching _build_lewm_ms_conv), not upstream's single-step objective.
    """
    import torch.nn.functional as F
    from jepa import JEPA
    from module import MLP, Embedder, SIGReg

    from lehami.dino_wm_adapter import ActionConcatViTPredictor, make_vit_predictor

    enc_dim     = cfg.embed_dim          # DINOv2 CLS width (384 for vits14)
    latent_dim  = cfg.model.latent_dim   # projector bottleneck, shared with the Hamiltonian arm
    act_emb_dim = int(OmegaConf.select(cfg, "model.action_emb_dim", default=10))
    h           = cfg.history_size
    pred_dim    = latent_dim + act_emb_dim   # concat_dim=1 widens the token

    model = JEPA(
        encoder=_encoder(cfg),
        # Wrapped so the predictor honours this repo's (x, c) signature: upstream
        # ViTPredictor.forward takes the token stream ONLY, and JEPA.predict --
        # which run_eval's rollout calls -- passes (emb, act_emb). Unwrapped, the
        # arm trains fine and then dies at eval.
        predictor=ActionConcatViTPredictor(
            make_vit_predictor(
                num_patches=1, num_frames=h, dim=pred_dim,
                **OmegaConf.to_container(cfg.model.predictor, resolve=True),
            ),
            latent_dim=latent_dim,
        ),
        action_encoder=Embedder(input_dim=cfg.data.action_dim, emb_dim=act_emb_dim),
        projector=MLP(input_dim=enc_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        # Identity, not None: JEPA.predict unconditionally applies pred_proj, so
        # None crashes the eval rollout. Identity keeps DINO-WM's architecture --
        # they have no post-predictor projection -- while satisfying that call.
        pred_proj=torch.nn.Identity(),
    )
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    def _forward(self, batch, stage):
        B, T = batch["pixels"].shape[:2]
        device = batch["pixels"].device
        W     = getattr(self.model.encoder, "window", 1)   # DinoEncoder is per-frame
        T_eff = T - W + 1
        n_unroll = T_eff - h
        assert n_unroll >= 1, f"clip too short: T={T}, enc_window={W}, history_size={h}"

        enc = _encode_sequence(self.model.encoder, batch["pixels"].float())
        emb = self.model.projector(enc.reshape(B * T_eff, -1)).view(B, T_eff, latent_dim)

        ctx_emb = emb[:, :h]
        pos_loss = 0.0
        for step in range(n_unroll):
            act = _action_emb(batch, self.model, cfg, device, B, step, h)  # (B, h, a)
            # ActionConcatViTPredictor does the concat_dim=1 widening and slices
            # the action channels back off, so training and every eval path go
            # through exactly one implementation of that convention.
            pred_seq = self.model.predictor(ctx_emb, act)         # (B, h, d)
            z_pred   = pred_seq[:, -1]                            # (B, d)
            pos_loss = pos_loss + F.mse_loss(z_pred, emb[:, h + step])
            ctx_emb  = torch.cat([ctx_emb[:, 1:], z_pred.unsqueeze(1)], dim=1)
        pos_loss = pos_loss / n_unroll

        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pos_loss + cfg.loss.sigreg.weight * sigreg_loss
        output_d = {"loss": loss, "pred_loss": pos_loss, "sigreg_loss": sigreg_loss}
        diag = _encoder_diagnostics(self.model.encoder, emb)
        self.log_dict({f"{stage}/{k}": v.detach()
                       for k, v in {**output_d, **diag}.items()
                       if "loss" in k or k in _ENC_DIAG_KEYS},
                      on_step=True, on_epoch=True, sync_dist=True)
        return output_d

    return model, sigreg, _forward


def _build_lehami_sg_sep_sreg_joint_ms(cfg: DictConfig):
    """Hamiltonian flow with a ViT-tiny encoder (the single-rod pendulum model).

    The encoder gives positions q_t = z_t; momenta are the time derivative of the
    latent trajectory, estimated with least-squares quadratic (Savitzky-Golay)
    stencils from `derivative.py`:
      - anchor p     : causal stencil over clip frames 0..3 (anchor at frame 3)
      - mom targets  : centered stencil at each target frame (frames exist on
                       both sides inside the clip)
      - SIGReg       : on cat([q_t, p_t]) with centered p, t = 2 .. T-3
    Both stencils estimate dq/dt AT frame t (no half-step lag). From the anchor
    (q_3, p_3) the separable Hamiltonian flow (`ham_flow.HamiltonianPredictor`,
    leapfrog) is unrolled autoregressively; each step is scored against the
    encoder's q and the stencil's p. Requires clip length T >= 9; T = 14 gives
    8 unroll steps.
    """
    import torch.nn.functional as F
    from jepa import JEPA
    from module import MLP, Embedder

    from lehami.derivative import (
        ANCHOR_OFFSET,
        CAUSAL_WINDOW,
        CENTERED_HALFWIDTH,
        p_causal_last,
        p_interp_series,
    )
    from lehami.ham_flow import HamiltonianPredictor

    latent_dim      = cfg.model.latent_dim
    phase_dim       = 2 * latent_dim
    vit_dim         = cfg.embed_dim
    fd_dt           = float(cfg.model.fd_dt)
    momentum_weight = float(cfg.model.momentum_weight)

    model = JEPA(
        encoder=_vit(cfg),
        predictor=HamiltonianPredictor(
            embed_dim=phase_dim, latent_dim=latent_dim, action_emb_dim=phase_dim,
            **OmegaConf.to_container(cfg.model.ham_flow, resolve=True),
        ),
        action_encoder=Embedder(input_dim=cfg.data.action_dim, emb_dim=phase_dim),
        projector=MLP(input_dim=vit_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        pred_proj=None,
    )
    sigreg = _make_sigreg(cfg, latent_dim)

    A  = ANCHOR_OFFSET        # anchor at clip frame 3 (full causal window 0..3)
    HW = CENTERED_HALFWIDTH   # centered targets valid for t in [HW, T-1-HW]

    def _forward(self, batch, stage):
        B, T = batch["pixels"].shape[:2]
        device = batch["pixels"].device
        assert T >= CAUSAL_WINDOW + 2 * HW + 1, f"clip too short for sg momentum: T={T}"
        n_unroll = T - 1 - HW - A   # T=14 -> 8 steps, targets at frames A+1 .. T-1-HW

        pixels_flat = batch["pixels"].float().reshape(B * T, *batch["pixels"].shape[2:])
        cls_flat    = self.model.encoder(
            pixels_flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        z           = self.model.projector(cls_flat).view(B, T, latent_dim)

        # interpolation momentum: centered on the interior (all targets), causal anchor
        p_ref = p_interp_series(z, fd_dt)                       # (B, T, d)

        # SRNN autoregressive rollout from anchor (q_A, p_A)
        q_pred = z[:, A]
        p_pred = p_causal_last(z[:, :A + 1], fd_dt)
        act_emb = self.model.action_encoder(
            torch.zeros(B, 1, cfg.data.action_dim, device=device))
        pos_loss = mom_loss = 0.0
        for step in range(n_unroll):
            qp_next = self.model.predictor(
                torch.cat([q_pred, p_pred], dim=-1).unsqueeze(1), act_emb
            ).squeeze(1)
            q_pred  = qp_next[..., :latent_dim]
            p_pred  = qp_next[..., latent_dim:]
            f = A + 1 + step
            pos_loss = pos_loss + F.mse_loss(q_pred, z[:, f])
            mom_loss = mom_loss + F.mse_loss(p_pred, p_ref[:, f])
        pos_loss = pos_loss / n_unroll
        mom_loss = mom_loss / n_unroll

        # SIGReg on joint GT phase space with centered p, t = HW .. T-1-HW
        qp_gt     = torch.cat([z[:, HW:T - HW], p_ref[:, HW:T - HW]], dim=-1)
        sreg_loss = self.sigreg(qp_gt.transpose(0, 1))

        loss = pos_loss + momentum_weight * mom_loss + cfg.loss.sigreg.weight * sreg_loss
        output = {"loss": loss, "pred_loss": pos_loss + mom_loss,
                  "pos_loss": pos_loss, "mom_loss": mom_loss, "sigreg_loss": sreg_loss}
        self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
                      on_step=True, on_epoch=True, sync_dist=True)
        return output

    return model, sigreg, _forward


def _build_lehami_sg_sep_sreg_joint_ms_conv(cfg: DictConfig):
    """Hamiltonian flow (LeHami) with the ConvNet or frozen-DINOv2 encoder.

    The model behind every conservative multi-rod, actuated and DINOv2 Hamiltonian
    arm in the paper. Identical to `_build_lehami_sg_sep_sreg_joint_ms` --
    Savitzky-Golay momentum, separable Hamiltonian flow, multi-step rollout --
    except that the encoder comes from `_encoder` (ConvEncoder, ~315K params vs
    ~5.7M for ViT-tiny; its temporal-window variant; or a frozen DINOv2), the
    action conditions V(q, u), a per-episode step size is supported
    (`_episode_h`), and the SIGReg placement is configurable (`model.sigreg_mode`).
    """
    import torch.nn.functional as F
    from jepa import JEPA
    from module import MLP, Embedder

    from lehami.derivative import (
        ANCHOR_OFFSET,
        CAUSAL_WINDOW,
        CENTERED_HALFWIDTH,
        p_causal_last,
        p_interp_series,
    )
    from lehami.ham_flow import HamiltonianPredictor

    latent_dim      = cfg.model.latent_dim
    phase_dim       = 2 * latent_dim
    enc_dim         = cfg.embed_dim       # encoder output channels
    fd_dt           = float(cfg.model.fd_dt)
    momentum_weight = float(cfg.model.momentum_weight)

    model = JEPA(
        # _encoder, not _conv_encoder: `model.encoder: dino` selects the frozen
        # DINOv2 encoder. The DINO-WM baseline resolves its encoder through the
        # same factory, so the two arms of that comparison share it exactly.
        encoder=_encoder(cfg),
        predictor=HamiltonianPredictor(
            embed_dim=phase_dim, latent_dim=latent_dim, action_emb_dim=phase_dim,
            **OmegaConf.to_container(cfg.model.ham_flow, resolve=True),
        ),
        action_encoder=Embedder(input_dim=cfg.data.action_dim, emb_dim=phase_dim),
        projector=MLP(input_dim=enc_dim, hidden_dim=2048, output_dim=latent_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        pred_proj=None,
    )
    sigreg = _make_sigreg(cfg, latent_dim)
    # SIGReg PLACEMENT, selected by cfg.model.sigreg_mode (default "joint"):
    #   joint : SIGReg([z || p_ref])  -- one statistic over the 2d-wide phase vector
    #   q     : SIGReg(z)             -- position only, p left unregularised
    #   split : SIGReg(z) + SIGReg(p_ref) -- the two coordinates independently
    # Why it matters: the joint statistic can be satisfied while p carries almost
    # nothing, because near-dead p dimensions hide behind healthy q ones (the SG
    # stencil of a near-constant q is numerical noise).
    sigreg_mode = str(OmegaConf.select(cfg, "model.sigreg_mode", default="joint") or "joint")
    if sigreg_mode not in ("joint", "q", "split"):
        raise ValueError(f"model.sigreg_mode must be joint|q|split, got {sigreg_mode!r}")
    sigreg_p_weight = float(OmegaConf.select(cfg, "model.sigreg_p_weight",
                                             default=None) or cfg.loss.sigreg.weight)

    A  = ANCHOR_OFFSET        # anchor at clip frame 3 (full causal window 0..3)
    HW = CENTERED_HALFWIDTH   # centered targets valid for t in [HW, T-1-HW]

    def _forward(self, batch, stage):
        B, T = batch["pixels"].shape[:2]
        device = batch["pixels"].device
        # A temporal encoder consumes W-1 lead-in frames, so every index below
        # lives in LATENT time (length T_eff), not pixel time.  Holding
        # n_unroll = 8 fixed therefore requires T = 14 + (W-1).
        W     = getattr(self.model.encoder, "window", 1)
        T_eff = T - W + 1
        assert T_eff >= CAUSAL_WINDOW + 2 * HW + 1, (
            f"clip too short for sg momentum: T={T}, enc_window={W}, T_eff={T_eff}")
        n_unroll = T_eff - 1 - HW - A   # T_eff=14 -> 8 steps, targets A+1 .. T_eff-1-HW

        enc = _encode_sequence(self.model.encoder, batch["pixels"].float())
        z   = self.model.projector(enc.reshape(B * T_eff, -1)).view(B, T_eff, latent_dim)

        # interpolation momentum: centered on the interior (all targets), causal
        # anchor.  Latents are one pixel-frame apart whatever W is, so the SG
        # stencil and fd_dt are unchanged by the temporal encoder.
        # Per-episode clock: every stencil is linear in 1/dt, so p at step h is
        # the frame-unit p divided by h; h=None keeps the scalar fd_dt.
        h = _episode_h(batch, cfg)
        p_ref = p_interp_series(z, fd_dt)                       # (B, T_eff, d)
        p_anchor = p_causal_last(z[:, :A + 1], fd_dt)
        if h is not None:
            p_ref = p_ref / h[:, None, None]
            p_anchor = p_anchor / h[:, None]

        # SRNN autoregressive rollout from anchor (q_A, p_A)
        q_pred = z[:, A]
        p_pred = p_anchor
        pos_loss = mom_loss = 0.0
        for step in range(n_unroll):
            # transition into latent frame A+1+step is driven by the action AT
            # frame A+step.  `h` below is the per-episode integrator step size
            # from _episode_h, NOT the history size -- do not conflate them.
            act_emb = _action_emb(batch, self.model, cfg, device, B, A + step, 1)
            qp_next = self.model.predictor(
                torch.cat([q_pred, p_pred], dim=-1).unsqueeze(1), act_emb, h
            ).squeeze(1)
            q_pred  = qp_next[..., :latent_dim]
            p_pred  = qp_next[..., latent_dim:]
            f = A + 1 + step
            pos_loss = pos_loss + F.mse_loss(q_pred, z[:, f])
            mom_loss = mom_loss + F.mse_loss(p_pred, p_ref[:, f])
        pos_loss = pos_loss / n_unroll
        mom_loss = mom_loss / n_unroll

        # SIGReg on joint GT phase space with centered p, t = HW .. T_eff-1-HW
        # (T_eff, not T: a temporal encoder shortens latent time by W-1, and
        # slicing to T here would run past the end of z and silently drop the
        # last W-1 valid frames from the SIGReg batch.)
        z_win = z[:, HW:T_eff - HW]
        p_win = p_ref[:, HW:T_eff - HW]
        extra = {}
        if sigreg_mode == "joint":
            sreg_loss = self.sigreg(torch.cat([z_win, p_win], dim=-1).transpose(0, 1))
            sreg_term = cfg.loss.sigreg.weight * sreg_loss
        elif sigreg_mode == "q":
            sreg_loss = self.sigreg(z_win.transpose(0, 1))
            sreg_term = cfg.loss.sigreg.weight * sreg_loss
        else:  # split — q and p each get their own statistic and their own weight
            sreg_q = self.sigreg(z_win.transpose(0, 1))
            sreg_p = self.sigreg(p_win.transpose(0, 1))
            sreg_loss = sreg_q + sreg_p
            sreg_term = cfg.loss.sigreg.weight * sreg_q + sigreg_p_weight * sreg_p
            extra = {"sigreg_loss_q": sreg_q, "sigreg_loss_p": sreg_p}

        loss = pos_loss + momentum_weight * mom_loss + sreg_term
        output = {"loss": loss, "pred_loss": pos_loss + mom_loss,
                  "pos_loss": pos_loss, "mom_loss": mom_loss, "sigreg_loss": sreg_loss,
                  **extra}
        diag = _encoder_diagnostics(self.model.encoder, z)
        # Collapse watch on the momentum TARGET.  p_ref is a stencil derivative of
        # the encoder's own positions, so raw mom_loss can be lowered for free by
        # shrinking it -- and under sigreg_mode=q nothing pins its scale.  A run
        # whose std_p drifts down through training has a moving MSE(q)/MSE(p)
        # ratio, so std_p / std_q / eff_rank_p are logged to catch that collapse.
        with torch.no_grad():
            diag = {**diag, "std_p": p_ref.std(), "std_q": z.std(),
                    "eff_rank_p": _participation_rank(p_ref)}
        self.log_dict({f"{stage}/{k}": v.detach()
                       for k, v in {**output, **diag}.items()
                       if "loss" in k or k in _ENC_DIAG_KEYS},
                      on_step=True, on_epoch=True, sync_dist=True)
        return output

    return model, sigreg, _forward


# ── registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, callable] = {
    # LeWorldModel baselines
    "lewm_ms":                          _build_lewm_ms,
    "lewm_ms_conv":                     _build_lewm_ms_conv,
    # DINO-WM predictor on a frozen DINOv2 encoder
    "dino_wm_cls":                      _build_dino_wm_cls,
    # the Hamiltonian flow (LeHami); the SIGReg placement is read from the model config
    "lehami_sg_sep_sreg_joint_ms":       _build_lehami_sg_sep_sreg_joint_ms,
    "lehami_sg_sep_sreg_joint_ms_conv":  _build_lehami_sg_sep_sreg_joint_ms_conv,
    "lehami_sg_sep_sreg_q_ms_conv":      _build_lehami_sg_sep_sreg_joint_ms_conv,
    "lehami_sg_sep_sreg_split_ms_conv":  _build_lehami_sg_sep_sreg_joint_ms_conv,
    "lehami_sg_sep_sreg_q_ms_dino":      _build_lehami_sg_sep_sreg_joint_ms_conv,
}
