"""Swing-up planning on the actuated pendulum: CEM-MPC in LATENT space.

    *** PREPARED, NOT RUN. ***  This script has been smoke-tested end to end
    against 1-epoch checkpoints to prove the plumbing (load -> encode -> latent
    rollout -> CEM -> plant), but it has NOT been run as an experiment and no
    number it prints has been recorded.  Run it only after EXP-071/072 have
    trained and their prediction metrics have been read.

The comparison this exists to make: LeHami (EXP-071) versus LeWM (EXP-072), same
data, same encoder, same planner, on a task where the model has to be right about
*velocity* to succeed.

── WHAT IS OPTIMISED ────────────────────────────────────────────────────────
A torque sequence u_{0..H-1}, |u| <= u_max, scored by a LATENT goal cost

    C(u) = || z_H(u) - z_goal ||^2

where z_H is the model's own latent after H self-fed steps and z_goal is the
encoding of a rendered goal frame.  This is LeWorldModel's actual planning cost
and is used deliberately in preference to a state-space cost: a state cost would
measure the probe, not the world model, and would make the comparison a probing
result wearing a planner's clothes.

TWO THINGS ARE KNOWN IN ADVANCE and are built in rather than discovered:

  1. A ONE-FRAME GOAL CANNOT SPECIFY VELOCITY, and only one arm has a slot for
     it.  State this precisely, because the loose version is wrong.  A single
     render determines theta but not omega -- (theta, 0) and (theta, 5) are
     pixel-identical -- so an encoding of ONE goal image pins theta and nothing
     else.  That does NOT mean either model is blind to velocity: the change is
     visible ACROSS frames and both arms use it (LeWM infers omega from its
     3-frame latent context; LeHami builds p by a finite difference on consecutive
     latents).  The asymmetry is in what the GOAL can express.  LeHami's state is
     explicitly (q, p), so "upright AND at rest" is representable as
     (q_goal, p_goal = 0) -- built below.  LeWM's z has no momentum block, so
     there is no coordinate to attach "at rest" to; the measured omega-AUC of
     0.499 follows from that, not from an inability to perceive.

     The cost does NOT try to repair that.  It is the squared distance between the
     goal latent and the current one, in whatever state the MODEL itself defines --
     (q, p) for LeHami, z for LeWM -- so each arm is scored in its own coordinates
     and nothing is bolted onto either.  An earlier `--goal-frames 2` synthesised a
     velocity channel for LeWM out of a latent difference; it is gone, because it
     scored the baseline on a quantity its state does not contain.

  2. LeHami carries omega but places ~91.5% of it in the q block.  So how the two
     blocks are weighted is a real choice -- and it is made by the CHECKPOINT, not
     at the command line: the default is `cfg.model.momentum_weight`, the same
     weight the training loss put on the p block.  `--p-weight` overrides it, which
     turns the choice into a deliberate ablation ("is the information there but
     mis-scaled?") rather than an unexamined default.

── THE PROTOCOL, FIXED BEFORE ANY NUMBER IS LOOKED AT ───────────────────────
  * u_max = 4.0.  From the catch-basin analysis, NOT inherited from Gymnasium's
    2.0: at u_max = 2 the basin admits only 0.42x of the measured latent velocity
    error, so the outcome would be decided by the probe tier rather than by the
    model.
  * H = 25 planning steps (2.5 s at dt = 0.1 s); budget 60 env steps.
  * Success: |wrap(theta - pi)| < 0.2 rad AND |omega| < 1.0 rad/s, satisfied at
    ANY step inside the budget.
  * >= 5 planner seeds x 16 initial conditions, reported PAIRED (same IC, same
    planner seed, only the model differs) and tested with McNemar.  An unpaired
    single-seed standard error is ~0.12 here and wrongly dismisses the effect --
    that error was made once already on the pilot; do not repeat it.

── GOAL MODES (`--goal-mode`) ───────────────────────────────────────────────
`upright` is the protocol above and remains the default.  `dataset` is
LeWorldModel's OWN evaluation construction, ported so the two can be read side by
side (third_party/leworldmodel/config/eval/pusht.yaml, swm/world.py:791):

    sample `--n-eval` episodes from `--split`; in each pick a GOAL frame; the
    START frame is `--goal-offset` steps EARLIER.

The difference is not difficulty, it is what each can claim.  Under `upright` the
goal is one fixed state two horizons away behind the separatrix, reachable only on
a WITNESS (an energy-shaping controller, 51 steps); no single plan can reach it,
measured -- 16.97 J is injectable over a horizon against a 37.31 J deficit.  Under
`dataset` reachability holds BY CONSTRUCTION and needs no verifying: `si` and `gi`
index the same recorded episode, so the actions between them are exactly what
produced `state[gi]` from `state[si]`, and `--goal-offset` is one horizon.  The
index arithmetic is checked once, offline, by
`tests/test_swingup_plan_matches_training.py`; the only way the guarantee can
break at RUN time is a clock mismatch, so `--dt-mult` is asserted against the
dataset's own `dt_mult` before anything loads.

One model-free control is computed per episode and written to the CSV:

  zero_success    the u = 0 floor.  A free pendulum reaches turning points on its
                  own, so some dataset goals need no torque at all -- measured at
                  34% on the real dataset at offset 25.  It is REPORTED rather than
                  filtered out: dropping those goals would select the evaluation
                  set on a property of the plant, and the natural criterion (an
                  energy gap) is the favoured arm's own coordinate.  Read every
                  planner number as lift over this floor, and split on it.

Success generalises to the goal state -- |wrap(theta - theta_goal)| < 0.2 AND
|omega - omega_goal| < 1.0 -- which reduces to the pre-registered criterion under
`upright`, where omega_goal is 0.

── USAGE ────────────────────────────────────────────────────────────────────
    python lehami/experiments/src/plan_pendulum_swingup.py \
        --checkpoints "a0s_s0=<ckpt>,b0s_s0=<ckpt>,..." \
        --n-ic 16 --planner-seeds 5 --out outputs/exp071_072_swingup

    # LeWorldModel's construction (goal first, start = goal - 25)
    python .../plan_pendulum_swingup.py --checkpoints "..." \
        --goal-mode dataset --n-eval 50 --goal-offset 25 --split test

    # plumbing check only (2 ICs, 1 seed, tiny CEM budget)
    python .../plan_pendulum_swingup.py --checkpoints "..." --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lehami" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pendulum_swingup import (  # noqa: E402
    BASE_DT,
    G,
    SUBSTEP,
    _rk4_step,
    bob_radius,
    render_frame,
)
from run_eval import (  # noqa: E402
    _anchor_frame,
    _build_anchor,
    _enc_shift,
    _enc_window,
    _encode_episode,
    _encode_z_lewm,
    _latent_dim,
    _mom_type,
    _preprocess_batch,
    load_model,
)

OUT_DIR = ROOT / "outputs" / "plan_pendulum_swingup"

#: LeWorldModel's reported CEM settings, which are also `CEMSolver`'s defaults.
CEM_DEFAULTS = dict(num_samples=300, n_steps=30, topk=30)

#: Success criterion. Fixed here, before any run, on purpose.
THETA_TOL = 0.2      # rad from upright
OMEGA_TOL = 1.0      # rad/s


def wrap(a):
    """Signed angle into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ── the true plant ────────────────────────────────────────────────────────────


def plant_step(theta, omega, L, m, u, dt):
    """One control step of the REAL pendulum, integrated at SUBSTEP.

    Identical integrator and substep to `pendulum_swingup.simulate`, so the plant
    the planner acts on is the same physics the training data came from.
    """
    n_sub = int(round(dt / SUBSTEP))
    for _ in range(n_sub):
        theta, omega = _rk4_step(theta, omega, L, m, float(u), SUBSTEP)
    return theta, omega


# ── latent encoding and rollout ───────────────────────────────────────────────


def encode_frames(model, cfg, thetas, L, m, device, return_cls=False):
    """Render and encode a list of angles -> latents, matching run_eval's path.

    `return_cls` also hands back the ViT/conv tokens, which `_build_anchor` needs
    for the `learned` and `freemom` momentum paths (it ignores them for `sg`/`fd`).

    PREPROCESSING IS NOT OPTIONAL and must be run_eval's, not a local `/255`.
    Training builds its pixels with `ToImage(**dataset_stats.ImageNet)`
    (`lehami/training/train.py:59`), which is
    `to_image -> ToDtype(scale=True) -> Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])` -- the scale AND the ImageNet shift.
    `_preprocess_batch` is the same operation and is what every eval number came
    from.  This function used to divide by 255 and stop there, handing the encoder
    inputs on [0, 1] where it was trained on roughly [-2.1, 2.6].  Measured on a
    recorded clip: ||q|| 35.9 against 5.0, cos(q_planner, q_eval) = -0.25 per
    frame, cos on the anchor p = +0.06.  The latents were not merely shifted, they
    were unrelated -- so every plan scored a goal distance between two vectors the
    model never learned to produce.
    """
    rod_px, bob_px = _rod_px(L), bob_radius(m)
    imgs = np.stack([render_frame(float(t), rod_px, bob_px) for t in thetas])
    x = _preprocess_batch(imgs, int(cfg.img_size), device)
    if _mom_type(cfg.model.name) == "lewm":
        return _encode_z_lewm(model, x)
    cls, q = _encode_episode(model, x)
    return (cls, q) if return_cls else q


def anchor_from_angles(model, cfg, thetas, L, m, device, mom_type):
    """(q, p) for an LeHami arm, built by run_eval's OWN anchor routine.

    `thetas` must be exactly `_anchor_frame(cfg, mom_type) + 1` angles, oldest
    first: `_build_anchor` indexes the anchor at that offset, so the anchor lands
    on the LAST angle given and the earlier ones are the stencil's history.

    Reusing `_build_anchor` rather than differencing by hand is the whole point.
    The momentum stencil is a property of the MODEL FAMILY (4-frame causal
    Savitzky-Golay for `sg`, 2-frame backward FD for `fd`), not of
    `cfg.history_size` -- which describes the ENCODER's frame window and is 1 for
    every swing-up run.  Reading the window length off `history_size` collapses
    the context to a single frame, and a single frame cannot express motion, so p
    silently became 0 at every replan.  `_anchor_frame` refuses to consult
    `history_size` for `sg` for exactly this reason; so must we.
    """
    cls, q = encode_frames(model, cfg, thetas, L, m, device, return_cls=True)
    q0, p0 = _build_anchor(model, cls, q, mom_type, cfg, device)
    return q0.unsqueeze(0), p0.unsqueeze(0)          # (1, d) each, for _expand_ctx


def _rod_px(L, L_max=1.4):
    from pendulum_swingup import MAX_ROD_PX
    return MAX_ROD_PX * L / L_max


def _flow_h(cfg, dt_frame):
    """The predictor's per-episode integrator step, exactly as training computes it.

    Training does NOT call the Hamiltonian predictor with two arguments. It calls
    `predictor(qp, act_emb, h)` with `h = _episode_h(batch, cfg)` (builders.py),
    which is the episode's own frame interval divided by a reference:

        h = dt_frame / cfg.model.dt_ref,   or None when `dt_ref` is unset.

    `h` is the LEAPFROG STEP. Getting it wrong is invisible -- same shapes, same
    dtypes, no error -- and simply integrates the learned Hamiltonian at a
    different rate than the one it was fitted at. So the planner reads it off the
    checkpoint rather than defaulting, and `latent_rollout` refuses to run a
    `dt_ref` model without it.

    Returns None for every current swing-up checkpoint (none sets `dt_ref`), which
    is what keeps those runs bit-identical to the two-argument call.
    """
    dt_ref = OmegaConf.select(cfg, "model.dt_ref", default=None)
    if dt_ref is None:
        return None
    if float(dt_ref) <= 0:
        raise ValueError(f"model.dt_ref must be > 0, got {dt_ref}")
    return float(dt_frame) / float(dt_ref)


def latent_rollout(model, cfg, z_ctx, actions, device, return_all=False,
                   past_actions=None, flow_h=None):
    """Self-fed latent rollout under a torque sequence.

    Returns the FINAL (value, velocity) pair, or -- with `return_all` -- the list of
    such pairs, one per horizon step.  The list is what a RUNNING cost needs; see
    `latent_cost` for why the terminal-only variant is broken under receding execution.

    Two branches, mirroring exactly what the training forwards do -- if these
    ever diverge from `builders.py`, the planner is scoring a different model
    than the one that was trained, so keep them in sync.  `tests/
    test_swingup_plan_matches_training.py` asserts that equality on random
    weights, which is the only thing that keeps this comment honest.

      LeHami: state is the phase vector (q, p); one predictor call per step, with
             the action embedded through the same `action_encoder`.
      LeWM:  state is a sliding context of `history_size` latents; one `predict`
             per step, action window slid alongside.

    ── THE ACTION WINDOW (LeWM only) ─────────────────────────────────────────
    `past_actions`: (h-1,) torques ALREADY EXECUTED at the context frames, oldest
    first, or None on the first solve of an episode (nothing has been applied).

    The pairing is positional and it is not negotiable: context slot i carries the
    action applied AT frame i.  Three independent implementations agree --
    training (`builders.py`: `ctx_emb = emb[:, step:step+h]` against
    `actions[step:step+h]`), eval (`run_eval.py:1512`), and LeWorldModel's own
    inference rollout (`third_party/leworldmodel/jepa.py:61`, which grows `emb`
    and `act` as parallel buffers and truncates BOTH with `[-HS:]`).  So the
    window ENDS at the newest context frame; its last slot is the action driving
    the predicted step, and the earlier slots are actions already baked into the
    context.

    This code sliced `u[:, t:t+h]` -- the window STARTING at t -- which put the
    driving action in the OLDEST slot and filled the history slots with torques
    not yet applied, a shift of h-1.  It also zero-padded the tail, so under a
    terminal cost the single scored latent came from `[u_last, 0, 0]`.

    DELIBERATE DEVIATION from `jepa.py`, stated because it is a real one.  LeWM
    splits the candidate as `act_0, act_future = split(actions, [H, T-H])` and
    spends the first H candidate torques on the history frames, giving T-H+1
    predictor calls.  It does that because its policy does not track what it
    executed.  We do, so we front-pad with the TRUE past actions instead: every
    candidate torque then drives a real step, and both arms roll out exactly
    `horizon` steps -- without which a terminal cost would score the two arms at
    different look-ahead depths (23 vs 25) and the paired comparison would be
    between two different questions.
    """
    is_lehami = _mom_type(cfg.model.name) != "lewm"
    n = actions.shape[0]
    # ── ACTION DIMENSION ─────────────────────────────────────────────────────
    # Branch on the CONFIG's declared action_dim, not on the tensor shape and not on
    # the dataset name. That value is what the model's own `action_encoder` was
    # built with (`builders.py`: Embedder(input_dim=cfg.data.action_dim, ...)), so
    # it is the only source of truth about what this checkpoint expects.
    #
    # A dataset-name branch would need a registry, and a new environment missing
    # from it would fall through to the scalar path and be silently mangled -- the
    # exact failure this guards against.
    #
    # SCALAR PATH IS THE ORIGINAL LINE, UNCHANGED. Every pendulum result rests on
    # it, so it is left byte-identical rather than replaced by an equivalent.
    a_dim = int(OmegaConf.select(cfg, "data.action_dim", default=1) or 1)
    u = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if a_dim == 1:
        # The old bug lived here: `.view` would happily fold a (n, H, 2) input into
        # (n, 2H, 1) -- twice the timesteps, half an action each, no error. With
        # action_dim 1 such an input is a caller bug, so check the element count
        # before reshaping. Costs nothing and cannot change a correct call.
        if u.numel() != n * (u.shape[1] if u.dim() > 1 else 1):
            raise ValueError(
                f"cfg.data.action_dim is 1 but actions have shape {tuple(u.shape)}; "
                f"the reshape would silently redistribute them across timesteps")
        u = u.view(n, -1, 1)
    else:
        if u.dim() != 3 or u.shape[-1] != a_dim:
            raise ValueError(
                f"cfg.data.action_dim is {a_dim}, so actions must be "
                f"(n, H, {a_dim}); got {tuple(u.shape)}")

    # A `dt_ref` model integrates at h, not at the flow's constructed scalar. Refuse
    # rather than default: the two agree in shape and differ only in the dynamics.
    if is_lehami and flow_h is None and \
            OmegaConf.select(cfg, "model.dt_ref", default=None) is not None:
        raise ValueError(
            f"{cfg.model.name} sets model.dt_ref={cfg.model.dt_ref}, so its predictor "
            f"takes a per-episode step h (see `_flow_h`). Pass flow_h=_flow_h(cfg, dt).")

    # ── THE CONTEXT'S SHAPE, ASSERTED ────────────────────────────────────────
    # The contract between whoever BUILDS the context and this function was only
    # ever written in comments, and both branches have now been violated by a new
    # environment: the LeHami branch was handed a raw encoder `q` instead of a
    # (q, p) pair ("unpacks as too many values, which is how this was caught"),
    # and the LeWM branch was handed `encode_states`' per-frame output, (n_frames,
    # d), whose frame axis `_expand_ctx` then tried to broadcast over candidates
    # (job 16571293).
    #
    # The LeWM check reads `h` from cfg, NOT from the tensor. Below, `h` is taken
    # as `ctx.shape[1]`, so a context carrying the wrong number of frames does not
    # fail -- it rolls with a window that disagrees with the one the checkpoint was
    # trained on and returns numbers. That is the worse failure, and it is the one
    # this catches: (n, 5, d) under history_size=3 crashes here instead of silently
    # scoring a different model.
    if is_lehami:
        if not (isinstance(z_ctx, tuple) and len(z_ctx) == 2):
            raise ValueError(
                f"{cfg.model.name} is an LeHami arm: z_ctx must be a (q, p) tuple of "
                f"tensors, each (n, d). Got {type(z_ctx).__name__}"
                + (f" of shape {tuple(z_ctx.shape)}" if hasattr(z_ctx, "shape") else "")
                + ". Build it with the model's own anchor routine, not the raw encoder."
            )
        for name, z in zip("qp", z_ctx):
            if z.dim() != 2 or z.shape[0] != n:
                raise ValueError(
                    f"z_ctx[{name}] must be (n={n}, d); got {tuple(z.shape)}")
    else:
        h_cfg = int(cfg.history_size)
        if z_ctx.dim() != 3 or z_ctx.shape[0] != n or z_ctx.shape[1] != h_cfg:
            raise ValueError(
                f"{cfg.model.name} is a LeWM arm: z_ctx must be "
                f"(n={n}, history_size={h_cfg}, d); got {tuple(z_ctx.shape)}. "
                f"A per-frame encoder output is (n_frames, d) -- trim it to the "
                f"newest {h_cfg} frames and add a batch axis before expanding."
            )

    # Training hands the predictor a PER-SAMPLE step of shape (B,) (`_episode_h`
    # returns `dt_ep[:, 0]`), and `SeparableHamiltonianFlow._resolve_dt` indexes
    # `dt.shape[0]` -- a scalar reaches it as a 0-dim tensor and raises IndexError.
    # Every candidate here shares the episode's dt, so it is one value expanded to
    # the batch: same contract, same code path as training.
    h_t = None if flow_h is None else torch.full((n,), float(flow_h), device=device)

    traj = []
    if is_lehami:
        q, p = z_ctx                                   # each (n, d)
        for t in range(u.shape[1]):
            a_emb = model.action_encoder(u[:, t : t + 1])          # (n, 1, phase)
            # Third argument, as in training: `predictor(qp, act_emb, h)`.
            qp = model.predictor(torch.cat([q, p], dim=-1).unsqueeze(1), a_emb, h_t)
            qp = qp.squeeze(1)
            d = _latent_dim(cfg)
            q, p = qp[..., :d], qp[..., d:]
            if return_all:
                traj.append((q, p))
        return traj if return_all else (q, p)

    ctx = z_ctx                                        # (n, h, d)
    h = ctx.shape[1]

    # ── THE ENCODER LAG, AND WHY IT SHOWS UP HERE ────────────────────────────
    # A window-W symmetric encoder emits a latent for instant j+W-1-lag from pixel
    # frames [j, j+W-1], i.e. it is centred on the instant it represents and that
    # instant sits `enc_lag = (W-1)/2` frames behind its newest frame. So a
    # controller holding frames up to t can encode instants up to t - enc_lag, and
    # no further: encoding "now" would need frame t+enc_lag, which has not happened.
    #
    # Nothing here looks into the future -- rollout latents come from `predict`,
    # never from the encoder, and the anchor is built from observed frames only.
    # The lag is not a leak; it is a LATENCY, and it costs exactly two things:
    #
    #   1. the executed prefix is longer by enc_lag (more already-applied torques
    #      fall inside the context window), and
    #   2. the rollout must run enc_lag steps further to reach the goal's instant.
    #
    # Both are handled below. At W = 1, enc_lag = 0 and this whole block reduces
    # EXACTLY to the per-frame code, which `test_swingup_plan_matches_training`
    # pins. Do not re-derive the offset from W: `diff` (W=2) has lag 0, so W and
    # lag are independent -- ask the encoder (`_enc_shift`), as training does.
    enc_lag = _enc_window(model) - 1 - _enc_shift(model)

    # `u_all` is a timeline in LATENT time: index j is the action paired with the
    # j-th latent, counting from the OLDEST context slot. The executed prefix is
    # h-1+enc_lag long, not h-1 -- a lagged encoder's newest context latent sits
    # enc_lag instants behind the plant, so that many extra already-applied torques
    # fall inside the context. Zeros only when there is genuinely no history: at
    # the first solve nothing has been applied to the padded frames, so zero is the
    # truth rather than a fallback. At enc_lag = 0 this is h-1 and everything below
    # is bit-identical to the per-frame case.
    n_past = h - 1 + enc_lag
    # The padding must carry the SAME action width as `u`. Both lines below used a
    # literal 1, which is the scalar torque this was written for; a 2-D thrust made
    # `torch.cat` raise on the width mismatch. This is the third appearance of the
    # hardcoded-scalar assumption in this function, so it is derived from `u` here
    # rather than restated. At a_dim == 1 both expressions are the original ones.
    if past_actions is None:
        pad = torch.zeros(n, n_past, a_dim, device=device)
    else:
        pad = torch.as_tensor(past_actions, dtype=torch.float32, device=device)
        pad = pad.reshape(1, -1, a_dim).expand(n, -1, -1)
        if pad.shape[1] != n_past:
            raise ValueError(
                f"past_actions must hold h-1+enc_lag={n_past} actions "
                f"(history_size {h}, enc_lag {enc_lag}), got {pad.shape[1]}")
    u_all = torch.cat([pad, u], dim=1)                 # (n, n_past+H, a_dim)

    # ROLL TO THE GOAL'S INSTANT, not merely H times. The newest context latent
    # represents (plant frame - enc_lag) while the goal latent is centred on the
    # goal frame, so the two instants are H + enc_lag apart. Rolling only H stops
    # one instant short per unit of lag and leaves the last executed torques out of
    # the cost -- the arm would optimise fewer torques than it is graded on.
    # At enc_lag = 0 this is H, exactly as before.
    n_steps = u.shape[1] + enc_lag

    for t in range(n_steps):
        a_win = u_all[:, t : t + h]                    # ENDS at latent slot t+h-1
        a_emb = model.action_encoder(a_win)
        z_next = model.predict(ctx, a_emb)[:, -1]
        ctx = torch.cat([ctx[:, 1:], z_next.unsqueeze(1)], dim=1)
        if return_all:
            traj.append(ctx[:, -1])
    # The newest latent, and only that: LeWM's state IS z, so the cost is a
    # distance in z. (This used to hand back the last TWO latents so their
    # difference could stand in for a velocity; that also made the arm silently
    # require history_size >= 2, since ctx[:, -2] has nothing to point at at 1.)
    return traj if return_all else ctx[:, -1]


def _train_dataset(cfg) -> Path:
    """The h5 the checkpoint was trained on, named by its own config."""
    import os
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    return home / f"{cfg.data.name}.h5"


def _check_plannable(cfg, args, key) -> None:
    """Refuse to plan a checkpoint whose training setup this planner does not match.

    Every check here guards a difference that produces a NUMBER rather than an
    error, or an error far from its cause. They run per arm, right after
    `load_model`, because each is a property of that checkpoint's config.

      1. enc_window  A window-W encoder turns W pixel frames into ONE latent.
                     SUPPORTED FOR LeWM SINCE 2026-08-23 (EXP-076 b3s): the pixel
                     context widens to `h_ctx + W - 1` and `latent_rollout` shifts
                     the action window by `_enc_shift` = (W-1) - lag, matching
                     `builders._action_emb`.  Still REFUSED for LeHami: the momentum
                     stencil's `_anchor_offset` does not account for W, so the
                     anchor would read the wrong frames and p would be a plausible
                     wrong number rather than an error.
      2. u_max       CEM clips candidates to +-u_max. Above the torque the DATA was
                     generated with, every sample is an extrapolation for the model
                     and the plan is scored on dynamics nobody fitted.
      3. dataset     The dt_mult guard in `main` compares the plant's clock to the
                     dataset's; this compares the dataset to the MODEL's. p is
                     latent-per-FRAME (fd_dt=1.0), so a wrong-clock file plans
                     silently with every velocity off by the ratio.
    """
    # `model.enc_window`, NOT `enc_window`.  train.py:290 saves the whole nested
    # Hydra cfg and every experiment config sets the key under `model:`, so the
    # top-level lookup selected the default 1 for EVERY arm and silently disabled
    # all four guards below -- the LeHami refusal included.  The live paths read
    # the encoder itself (`_enc_shift`, lines 431/989) and were always right, so
    # no reported number is affected; this restores the pre-flight.
    w = int(OmegaConf.select(cfg, "model.enc_window",
                             default=OmegaConf.select(cfg, "enc_window", default=1)) or 1)
    if w != 1 and _mom_type(cfg.model.name) != "lewm":
        raise NotImplementedError(
            f"{key} ({cfg.model.name}) was trained with enc_window={w}. The planner "
            f"supports W>1 for the LeWM family only: `_anchor_offset` does not "
            f"account for W, so the momentum stencil would read the wrong frames "
            f"and p would be a plausible wrong number. NOT IMPLEMENTED for LeHami -- "
            f"do not work around it by widening the context window.")
    # The two windows have separate requirements, so they get separate checks --
    # a wide-W arm can outgrow the START window while the GOAL window still fits
    # (b3s: needs 5 at the start, 3 at the goal, against 5 and 4). Refuse rather
    # than let `run_episode`'s `[-h_pix:]` / `[-enc_w:]` slices silently return a
    # window shorter than they asked for, which encodes fine and plans wrong.
    if w > GOAL_CTX:
        raise ValueError(
            f"{key}: enc_window={w} needs {w} pixel frames to encode ONE goal "
            f"latent, but GOAL_CTX is {GOAL_CTX}. Widening GOAL_CTX alone does not "
            f"change the goal draw (`lo` is set by START_CTX), so this one is cheap.")
    # NOT (w-1)//2: that is a theorem for the TIED kernels only.  `free` stores
    # its kernel outright, so its centroid is learned and its lag is a declared
    # convention (`model.enc_lag`) -- deriving it from W here would silently
    # accept an arm whose goal latent lands a frame off the scored instant.  The
    # two live paths already ask the encoder via `_enc_shift`; this pre-flight
    # runs before the model is built, so it reads the same field the encoder will.
    if w == 1:
        enc_lag = 0
    elif str(OmegaConf.select(cfg, "model.enc_variant", default="full")) == "diff":
        enc_lag = 0
    else:
        _declared = OmegaConf.select(cfg, "model.enc_lag", default=None)
        enc_lag = (w - 1) // 2 if _declared is None else int(_declared)
    if enc_lag > GOAL_LEAD:
        raise ValueError(
            f"{key}: enc_window={w} has lag {enc_lag}, so a goal latent centred on "
            f"the goal frame needs {enc_lag} frames after it, but GOAL_LEAD is "
            f"{GOAL_LEAD}. Without them the terminal latent would fall short of the "
            f"scored instant and the last {enc_lag} torques would be unscored.")
    h_pix = int(cfg.history_size) + w - 1
    if h_pix > START_CTX:
        raise ValueError(
            f"{key}: enc_window={w} with history_size={cfg.history_size} needs "
            f"history_size + W - 1 = {h_pix} pixel frames for the start context, "
            f"but START_CTX is {START_CTX}. Widening START_CTX RE-DRAWS EVERY GOAL "
            f"(`lo = K + START_CTX - 1`), so every arm must be re-run together.")

    h5_path = _train_dataset(cfg)
    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as f:
            u_train = float(f.attrs["u_max"]) if "u_max" in f.attrs else None
        if u_train is not None and args.u_max > u_train:
            raise SystemExit(
                f"--u-max {args.u_max} exceeds the {u_train} the training data for "
                f"{key} was generated with ({h5_path.name}). CEM would search torques "
                f"the model has never seen, so its predictions there are pure "
                f"extrapolation and the plan would be scored on unfitted dynamics.")
    else:
        print(f"[warn] {h5_path} not found; cannot check --u-max against the torque "
              f"range {key} was trained on.")

    if args.goal_mode == "dataset":
        assert Path(args.dataset).stem == cfg.data.name, (
            f"{key} was trained on {cfg.data.name} but is being planned against "
            f"{Path(args.dataset).stem}")


def _p_weight(cfg, override=None):
    """How much the p block counts in the goal distance -- taken from the MODEL.

    Training minimises `pos_loss + momentum_weight * mom_loss` (builders.py),
    an MSE over the q block plus a weighted MSE over the p block. `latent_cost` has
    that same form, so reading the weight off `cfg.model.momentum_weight` scores a
    plan in the metric the checkpoint was actually trained under, instead of one
    chosen at the command line. Same reasoning as the action block: if the number
    describes the model, the model should be the one to supply it.

    Sum-vs-mean is not a discrepancy: `F.mse_loss` averages over `latent_dim` and
    `latent_cost` sums, but q and p have the SAME dimension, so the two differ by a
    factor 1/d applied to both blocks alike -- which cannot move an argmin.

    Falls back to 1.0 -- the plain Euclidean distance in (q, p) -- when a checkpoint
    carries no `momentum_weight`. Every current swing-up config sets it to 1.0, so
    the fallback and the config agree today; the point of reading it anyway is that
    a model RETRAINED with a different weight is then planned under that weight
    automatically, with no flag to remember.
    """
    if override is not None:
        return float(override)
    return float(cfg.model.get("momentum_weight", 1.0))


def latent_cost(z_pred, z_goal, is_lehami, p_weight=1.0):
    """Squared distance to the goal, in each arm's OWN state.

    The state is whatever the model says it is, and nothing is synthesised on top
    of it:

      LeHami  (q, p)  -- momentum is an explicit latent coordinate, so the goal can
                        say "at this pose AND moving this way", and the distance
                        covers both blocks.
      LeWM   z       -- there is no momentum block, so the distance is over z and
                        the goal says only "look like this frame".

    That asymmetry is the finding, not a flaw in the cost: it is what "an explicit
    (q, p) state" buys, measured rather than argued.  Writing LeWM a velocity term
    out of a latent difference (the former `--goal-frames 2`) scored it on a
    quantity its state does not carry, so it is gone.

    `p_weight` rescales LeHami's p block. It is NOT a free knob: it comes from the
    checkpoint, as `cfg.model.momentum_weight` -- see `_p_weight`.
    """
    if isinstance(z_pred, list):        # RUNNING cost: sum the per-step cost
        return sum(latent_cost(zp, z_goal, is_lehami, p_weight) for zp in z_pred)
    if is_lehami:
        q, p = z_pred
        qg, pg = z_goal
        return ((q - qg) ** 2).sum(-1) + p_weight * ((p - pg) ** 2).sum(-1)
    return ((z_pred - z_goal) ** 2).sum(-1)


# ── CEM ───────────────────────────────────────────────────────────────────────


def cem_plan(model, cfg, z_ctx, z_goal, horizon, u_max, rng, device, args,
             past_actions=None):
    """Return the elite-mean plan, as `horizon` ENV-STEP actions.

    Mirrors `stable_worldmodel.solver.CEMSolver.solve` as closely as an unbatched
    numpy port can:

      * the search variable is `horizon // action_block` torques, each REPEATED
        `action_block` times -- CEMSolver's frameskip, where one horizon slot
        carries `action_block` env actions (`action_dim * config.action_block`);
      * the incumbent mean is forced in as candidate 0 (`candidates[:, 0] = mean`),
        so the elite set can never get worse than the previous iterate;
      * unweighted top-k elites by cost ORDER only, fixed iteration count;
      * mean/std re-estimated from the elites each iteration.

    The variance is re-initialised every solve. That matches the library: `policy`
    warm-starts from `actions[:, receding_horizon:]`, which is EMPTY whenever
    `receding_horizon == horizon`, as it is in every LeWM eval config.

    Deviation, deliberate: a 1e-6 floor on the elite std (CEMSolver uses a bare
    `.std()`). With 300 continuous samples the elites are never bit-identical, so
    this is a numerical guard rather than a behavioural difference.
    """
    is_lehami = _mom_type(cfg.model.name) != "lewm"
    N, iters, topk = args.cem_samples, args.cem_iters, args.cem_elites
    blk = args.action_block
    # CEILING, not floor: under `--replan` the horizon shrinks toward the goal
    # instant and is generally NOT a whole number of blocks. The last block is then
    # truncated -- on the SCORED rollout and on the returned plan alike, so the cost
    # is evaluated at the goal instant and not one or two steps past it (fixed
    # 2026-09-12; a no-op when blk divides horizon). `expand_blocks` in the three-body
    # planner does the same.
    n_blk = -(-horizon // blk)
    mean = np.zeros(n_blk, dtype=np.float32)
    std = np.full(n_blk, u_max / 2.0, dtype=np.float32)

    for _ in range(iters):
        cand = rng.normal(mean, std, size=(N, n_blk)).astype(np.float32)
        cand[0] = mean                                  # CEMSolver: candidates[:, 0] = mean
        cand = np.clip(cand, -u_max, u_max)
        # Truncated to `horizon`, exactly as the RETURNED plan is truncated below.
        # Without the slice the rollout is ceil(h/blk)*blk steps, so whenever blk does
        # not divide h the terminal cost is read 1-2 steps PAST the goal instant while
        # the executed plan stops at h: the planner is then scored on arriving late.
        # A no-op when blk divides h, which is every rung of the K=30 ladder
        # (30/24/18/12/6, 30/21/12/3, 30/15), so those results are bit-identical.
        steps = np.repeat(cand, blk, axis=1)[:, :horizon]   # frameskip -> env steps
        with torch.no_grad():
            ctx = _expand_ctx(z_ctx, N)
            z_end = latent_rollout(model, cfg, ctx, steps, device,
                                   return_all=(args.cost == "running"),
                                   past_actions=past_actions,
                                   flow_h=_flow_h(cfg, BASE_DT * args.dt_mult))
            c = latent_cost(z_end, _expand_goal(z_goal, N), is_lehami,
                            _p_weight(cfg, args.p_weight) if is_lehami else 1.0)
            c = c.detach().cpu().numpy()
        elite = cand[np.argsort(c)[:topk]]
        mean, std = elite.mean(0), elite.std(0) + 1e-6
    return np.repeat(mean, blk)[:horizon]               # (horizon,) env-step actions


def _expand_ctx(z_ctx, n):
    if isinstance(z_ctx, tuple):
        return tuple(z.expand(n, *z.shape[1:]) for z in z_ctx)
    return z_ctx.expand(n, *z_ctx.shape[1:])


def _expand_goal(z_goal, n):
    if isinstance(z_goal, tuple):
        return tuple(z.expand(n, *z.shape[1:]) for z in z_goal)
    return z_goal.expand(n, *z_goal.shape[1:])


# ── goals ─────────────────────────────────────────────────────────────────────
#
# `--goal-mode` selects how the target is built. The two differ in what they can
# claim, not merely in difficulty:
#
#   upright   The task goal: theta = pi, at rest. ONE fixed target, ~50 steps and
#             two horizons away, behind the separatrix. Reachability is witnessed
#             by an energy-shaping controller, not by construction, and no single
#             plan can reach it (measured: 16.97 J injectable over a horizon
#             against a 37.31 J deficit).
#
#   dataset   LeWorldModel's own evaluation construction, ported. Sample episodes;
#             in each, pick a GOAL frame; the START frame is `goal_offset` steps
#             EARLIER. Reachability is then not an assumption at all -- the torques
#             recorded between the two frames drive the plant from one to the other
#             exactly, and `replay_success` in the output CSV asserts it every run.
#             See third_party/leworldmodel/config/eval/pusht.yaml
#             (`goal_offset_steps: 25`, `eval_budget: 50`) and swm/world.py:791
#             (`end_steps = start_steps + goal_offset_steps`).
#
# Two more modes are planned and this structure is shaped to take them: a turning
# point (omega = 0 exactly, so the goal states a velocity a single frame cannot
# show), and upright-at-rest reached by the same backwards construction.


#: Two windows, two requirements. Both are FIXED rather than per-arm: every arm
#: must be handed the identical episode list or the paired test is comparing
#: different tasks. Each arm slices the tail it needs out of them.
#:
#: The quantity in each case is "pixel frames required to ENCODE this end", taken
#: as the max over the arms in the registry:
#:
#:                       LeHami (W=1)      LeWM W=1     LeWM W=3     max
#:   goal, one state     4 (sg stencil)   1 (=W)       3 (=W)        4
#:   start context       4 (stencil)      3 (=hist)    5 (hist+W-1)  5
#:
#: They differ because the goal is ONE encoded state while the start is a context
#: of `h_ctx` of them: a window-W encoder eats W-1 lead-in frames to produce each,
#: so the start needs history_size + W - 1 while the goal needs only W.
#:
#: GOAL_CTX: frames before (and including) the goal frame. 4 covers the widest
#: goal encoder in the registry — the sg momentum stencil (CAUSAL_WINDOW = 4) —
#: and it already covers LeWM W=3, whose goal latent needs W = 3.
GOAL_CTX = 4

#: START_CTX: frames before (and including) the START frame, enough to fill the
#: planner's context at the first solve. Raised 4 -> 5 on 2026-08-23 for LeWM W=3
#: (EXP-076 b3s): history_size 3 + W 3 - 1 = 5.
#:
#: ⚠ THIS ONE SETS THE GOAL DRAW. `dataset_goals` builds `lo = K + START_CTX - 1`
#: (the start needs START_CTX-1 frames of run-up BEFORE si = gi - K), and `lo` is
#: both the episode-pool filter and the low bound of the `gi` draw. So changing
#: START_CTX re-draws every goal, and runs made at different values are NOT
#: paired — the McNemar test needs every arm from one value. The h30 results of
#: 2026-08-22 (a0q 0.207 / b0s 0.044) were produced at 4; to compare b3s against
#: them, re-run those arms here rather than pooling across the change.
#:
#: Note it is the START, not the goal, that forces this: b3s's goal fits in 4.
START_CTX = 5

#: GOAL_LEAD: frames fetched AFTER the goal frame. A symmetric encoder's latent for
#: instant g is centred on g, so it needs g+lag as its newest frame. Without these
#: the newest goal latent an arm could build would represent g-lag, and rolling to
#: it would leave the last executed torque out of the cost entirely (b3s scored 29
#: of its 30 candidates). 1 covers every lag in the registry (W=3 -> lag 1).
#:
#: This is GOAL-side only, so unlike START_CTX it does not enter `lo`. It does cap
#: the draw from above (`gi <= ep_len-1-GOAL_LEAD`) so g+lag exists, which moves
#: the draw the same way — done in the same edit as the START_CTX change so there
#: is ONE re-draw, not two.
#:
#: Fetching them is not "looking into the future": the goal is recorded data from a
#: stored episode, never an observation the controller has to wait for. The START
#: side gets no such lead, which is exactly the asymmetry — a controller at frame t
#: can encode instants up to t-lag and no further.
GOAL_LEAD = 1


def _goal_success(theta, omega, g, tol=(THETA_TOL, OMEGA_TOL)):
    """Reached `g`? Pose AND velocity, both relative to the goal state.

    For `upright` the velocity clause reduces to |omega| < OMEGA_TOL because
    omega_goal is 0, so this generalises the pre-registered criterion rather than
    replacing it.

    WHEN this is evaluated is the caller's business and it is not a detail --
    see `--success-at`.
    """
    tt, ot = tol
    return (abs(float(wrap(theta - g["theta"]))) < tt
            and abs(float(omega - g["omega"])) < ot)


def _success_steps(args):
    """Which env steps the success test is applied at (1-indexed).

    'any'     every step in the budget. LeWorldModel's criterion -- swm/world.py:961
              accumulates `episode_successes = logical_or(..., terminateds)` over
              the rollout. Safe in THEIR environments: a random policy cannot
              stumble a T-block into a 20px/20deg pose.
    'target'  the goal's OWN timestamp, i.e. step `goal_offset`.

    WHY THE DEFAULT DIFFERS BY MODE, measured on 135 real test goals at offset 25:

        criterion            u=0 floor   best-of-300-random
        any    (0.2, 1.0)        0.496              1.000
        any    (0.05, 0.2)       0.067              0.983
        target (0.1, 0.5)        0.050              0.650

    Under 'any' the instrument is SATURATED at every tolerance tried: 300 random
    torque sequences -- one CEM iteration's worth, against a 300x30 search -- solve
    essentially every goal with exact dynamics, so no world model can be shown to be
    better or worse than any other. A pendulum under random torque sweeps its whole
    reachable set, so "pass near this state at some point in 50 steps" is nearly
    free however tight the band is. Tightening tolerances does not fix it; scoring
    at the goal's own timestamp does.

    'target' is also the more faithful reading of the construction: the goal IS the
    state the trajectory occupied at start+offset.

    `upright` keeps 'any' because there the question genuinely is "does it ever get
    to the top", and there is no timestamp attached to the goal.
    """
    if args.success_at == "any":
        return None                      # every step
    return {args.goal_offset}


def upright_goals(args, h_ctx=GOAL_CTX):
    """The original protocol: one fixed upright target per sampled (L, m, IC)."""
    rng = np.random.default_rng(args.ic_seed)
    eps = []
    for _ in range(args.n_ic):
        L = float(rng.choice(args.lengths))
        m = float(rng.choice(args.masses))
        eps.append(dict(
            L=L, m=m, theta0=float(rng.uniform(-0.6, 0.6)),
            omega0=float(rng.uniform(-0.5, 0.5)),
            goal=dict(theta=float(np.pi), omega=0.0, thetas=None),
            replay=None, ep=-1, start_idx=-1, goal_idx=-1))
    return eps


def dataset_goals(args, h_ctx=GOAL_CTX):
    """LeWorldModel's construction: goal first, start = goal - `goal_offset`.

    `thetas` in the returned goal is the GOAL_CTX-long window of RECORDED angles
    ending at the goal frame. That window -- not a repeated render of a synthetic
    pose -- is what the momentum stencil differentiates, so `p_goal` is the true
    momentum of a real observed state and stays in-distribution.

    `start_thetas` / `start_actions` mirror that at the START frame, and they exist
    so the two ends are built the SAME way. Without them `run_episode` seeds its
    history with one angle and pads, the stencil sees a constant sequence, and
    p_start is 0 however fast the pendulum is moving -- scored against a p_goal
    that is real. Under --success-at target there is exactly one CEM solve per
    episode, so that first anchor is the whole plan.

    The two windows are fetched at their OWN widths (START_CTX >= GOAL_CTX): the
    goal is one encoded state, the start is a context of several, so a window-W
    encoder needs history_size + W - 1 frames at the start but only W at the goal.
    Conflating them is what made a W=3 arm look like it needed a wider GOAL window.
    """
    # `h_ctx` is accepted so a caller may pass one, and then DELIBERATELY ignored:
    # both widths are the same for every arm, because handing LeHami and LeWM
    # different episode lists would void the paired test. Pinned by
    # `test_goal_window_is_arm_independent`.
    import h5py

    man = json.loads(Path(args.manifest).read_text())
    eps_pool = man[f"{args.split}_episode_indices"]
    # `goal_offset` used to do TWO jobs: set the goal DRAW (lo, the pool filter and
    # gi) and set the START (si = gi - K). That coupling is why shortening the reach
    # distance re-drew every goal and made the two K's incomparable. K_draw now
    # carries the draw and K_plan the start, and they are equal unless --draw-offset
    # says otherwise -- so every pre-existing run is reproduced bit for bit.
    # getattr, not attribute access: callers that predate --draw-offset (the
    # tests' _Args namespace among them) construct args without the field, and
    # they must keep the old single-offset behaviour untouched.
    _draw = getattr(args, "draw_offset", None)
    K_draw = args.goal_offset if _draw is None else _draw
    K = args.goal_offset                      # K_plan: the reach distance
    if K > K_draw:
        raise SystemExit(
            f"--goal-offset ({K}) > --draw-offset ({K_draw}): the start would sit "
            f"before the run-up reserved by the draw, so si could fall outside the "
            f"episode. Draw at the LONGER offset and plan at the shorter one.")

    with h5py.File(args.dataset, "r") as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        Ls, ms = f["length"][:], f["mass"][:]

        # A goal frame needs START_CTX-1 frames before the START, then K steps of
        # run-up to the goal, and the goal's own window is contained in that run-up
        # (GOAL_CTX <= START_CTX + K holds by a wide margin). Those distances
        # COMPOSE along the episode, so they add -- this was `max(K, h_ctx - 1)`,
        # which reserved the run-up and the goal's stencil but let the start sit at
        # frame 0 with nothing behind it. `eval.py:110` applies the same
        # feasibility filter from the other end (`max_start_idx = episode_len -
        # offset - 1`).
        #
        #   si-4 .. si-1   si ---- K steps ---- gi-3  gi-2  gi-1   gi
        #   |_ START_CTX _|                     |____ GOAL_CTX ____|
        #
        # It is the START window that binds: `lo` is driven by START_CTX, and the
        # goal window is comfortably inside the run-up. NOTE that goals can no
        # longer be drawn from the first K+START_CTX-1 frames of an episode, so
        # numbers are not comparable across a change to START_CTX.
        lo = K_draw + START_CTX - 1        # the DRAW: unchanged when K == K_draw
        pool = [e for e in eps_pool if ep_len[e] > lo + 1 + GOAL_LEAD]
        if len(pool) < args.n_eval:
            raise ValueError(f"only {len(pool)} usable episodes in "
                             f"{args.split}, need {args.n_eval}")

        rng = np.random.default_rng(args.ic_seed)
        chosen = rng.choice(len(pool), size=args.n_eval, replace=False)
        eps = []
        for c in sorted(chosen):                     # sorted: HDF5 likes it
            e = int(pool[c])
            # Upper bound leaves GOAL_LEAD frames after the goal, so a centred
            # goal latent exists for every arm.
            gi = int(rng.integers(lo, int(ep_len[e]) - GOAL_LEAD))
            si = gi - K
            o = int(ep_off[e])
            # A negative slice start would NOT raise: h5py follows numpy's
            # wrap-around convention, so `o + si - START_CTX + 1 < 0` would
            # silently return frames from the tail of a DIFFERENT episode and
            # p_start would be a plausible-looking number computed across a
            # discontinuity. `lo` above makes this impossible; the assert is what
            # keeps it that way.
            assert si >= START_CTX - 1, \
                f"si={si} < START_CTX-1={START_CTX - 1} (lo={lo}, K={K})"
            st = f["state"][o + si : o + gi + 1]              # (K+1, 2)
            # GOAL_CTX frames ending at gi, PLUS GOAL_LEAD after it. The goal frame
            # therefore sits at index GOAL_CTX-1 of `gth`, not at the end -- each
            # arm indexes off that, so a causal arm takes the tail up to it and a
            # symmetric one takes the window centred on it.
            gth = f["state"][o + gi - GOAL_CTX + 1 : o + gi + 1 + GOAL_LEAD, 0]
            # The START's own window: START_CTX real, distinct, consecutive angles
            # ENDING AT the start frame. Without it `run_episode` pads with copies
            # of theta0, and a constant sequence has zero slope -- so p_start was 0
            # however fast the pendulum was actually moving, while p_goal was the
            # true momentum. Wider than `gth` because it must fill a CONTEXT of
            # latents, not produce a single one.
            sth = f["state"][o + si - START_CTX + 1 : o + si + 1, 0]
            # actions[t] drives t -> t+1, so si..gi-1 is exactly the run-up.
            act = f["action"][o + si : o + gi, 0]
            # ...and si-START_CTX+1..si-1 are the torques that PRODUCED the start
            # window. The LeWM arm pairs one action per context frame, so it needs
            # these for the same reason LeHami needs `sth`, and it needs as many as
            # `sth` has frames minus one.
            sact = f["action"][o + si - START_CTX + 1 : o + si, 0]
            eps.append(dict(
                L=float(Ls[e]), m=float(ms[e]),
                theta0=float(st[0, 0]), omega0=float(st[0, 1]),
                goal=dict(theta=float(st[-1, 0]), omega=float(st[-1, 1]),
                          thetas=[float(a) for a in gth]),
                start_thetas=[float(a) for a in sth],
                start_actions=[float(a) for a in sact],
                replay=[float(a) for a in act], ep=e, start_idx=si, goal_idx=gi))
    return eps


GOAL_MODES = {"upright": upright_goals, "dataset": dataset_goals}


# ── one episode ───────────────────────────────────────────────────────────────


def run_episode(model, cfg, ep, planner_seed, args, device):
    """Closed-loop MPC on the true plant. Returns a result row.

    `ep` is one entry from `upright_goals` / `dataset_goals`: the start state, the
    goal, and (dataset mode) the recorded torques that provably connect them.
    """
    L, m = ep["L"], ep["m"]
    theta0, omega0, g = ep["theta0"], ep["omega0"], ep["goal"]
    dt = BASE_DT * args.dt_mult
    rng = np.random.default_rng(planner_seed)
    mom_type = _mom_type(cfg.model.name)
    is_lehami = mom_type != "lewm"
    # Frames of context per solve.  For LeHami this is the MOMENTUM STENCIL's span
    # (`_anchor_frame` + 1: 4 for sg, 2 for fd), NOT cfg.history_size -- see
    # `anchor_from_angles`.  Only the LeWM branch, whose state genuinely is a
    # sliding window of `history_size` latents, reads history_size.
    h_ctx = (_anchor_frame(cfg, mom_type) + 1) if is_lehami else int(cfg.history_size) ## a number of how many frames of context to use for the planner
    # PIXEL frames needed to produce `h_ctx` LATENT frames. A window-W encoder eats
    # W-1 lead-in frames, so the two differ for LeWM W>1 and are equal everywhere
    # else (LeHami is refused at W>1 by `_check_plannable`).
    enc_w = _enc_window(model)
    h_pix = h_ctx + enc_w - 1
    # How far behind the newest FRAME the newest latent sits. A controller holding
    # frames up to t can encode instants up to t - enc_lag, and no further.
    enc_lag = enc_w - 1 - _enc_shift(model)

    # Goal latent. The window of angles it is built from is what distinguishes the
    # modes, and it is the ONLY thing that does:
    #   upright  h_ctx renders of one static pose -> identical images, so every
    #            velocity estimator returns exactly zero on them (the SG stencil
    #            included: it is linear, and a constant sequence has zero slope).
    #            "Arrive at the top with no motion."
    #   dataset  the recorded angles ending at the goal frame -> p_goal is that
    #            state's TRUE momentum, read by the same stencil training used.
    # Each arm's goal is its own state read off the goal frame(s): LeHami needs the
    # whole stencil window, because p is part of that state; LeWM needs the goal
    # frame alone, because z is its whole state.
    goal_th = (g["thetas"] if g["thetas"] is not None
               else [g["theta"]] * (GOAL_CTX + GOAL_LEAD))
    # The goal frame's index inside `goal_th` (see `dataset_goals`): GOAL_CTX-1
    # real angles precede it and GOAL_LEAD follow, so it is NOT the last element.
    g_at = GOAL_CTX - 1

    def _goal_latent(window, at):
        """The arm's own latent target for the frame at index `at` of `window`."""
        with torch.no_grad():
            if is_lehami:
                # The stencil is causal: h_ctx frames ENDING at the goal frame. It
                # must not run past `at` or the anchor would read frames after it.
                return anchor_from_angles(model, cfg, window[at + 1 - h_ctx: at + 1],
                                          L, m, device, mom_type)
            # W frames CENTRED on the goal frame, so the latent represents that
            # frame itself rather than frame - lag. At W=1 (lag 0) this is the single
            # goal frame, i.e. exactly the previous behaviour. At W=3 it is
            # (gi-1, gi, gi+1) -- the lead frame GOAL_LEAD exists for this.
            lo_i = at - enc_lag
            return encode_frames(model, cfg, window[lo_i: lo_i + enc_w],
                                 L, m, device)[-1:]

    z_goal = _goal_latent(goal_th, g_at)

    theta, omega = float(theta0), float(omega0)
    # Seeded with the h_ctx-1 angles BEFORE the start, so the momentum stencil sees
    # real motion at the FIRST solve -- which, under --success-at target, is the only
    # solve there is. Seeding with [theta] alone made the pad on the context line
    # below fire with h_ctx copies of one angle: zero slope, hence p_start = 0
    # whatever omega0 was, against a p_goal that carried the true momentum.
    # `.get`: only `dataset_goals` builds these. `upright_goals` still starts from a
    # single frame and therefore still plans from p_start = 0 -- a known limitation
    # of that mode now, not a silent one.
    hist = list(ep.get("start_thetas") or [theta])[-h_pix:]
    assert abs(float(wrap(hist[-1] - theta))) < 1e-9, (
        f"start window must END at theta0: {hist[-1]} vs {theta}")
    # Torques actually applied, index-aligned with the transitions out of `hist`.
    # Only the LeWM arm reads it (its predictor takes an action per context slot);
    # LeHami consumes one action per step and needs no history. See `latent_rollout`.
    # Seeded alongside `hist` for the same reason: fixing only the angles would give
    # LeHami a true p_start while still telling LeWM that no torque was applied at
    # its context frames, which breaks the pairing from the other side.
    act_hist = list(ep.get("start_actions") or [])[-(h_pix - 1):] if h_pix > 1 else []
    # Distances are now measured to THE GOAL, not to upright. For --goal-mode
    # upright the goal IS upright, so this reduces to the pre-registered criterion.
    best_dth = abs(float(wrap(theta - g["theta"])))
    best_om, hit, t_hit = abs(float(omega - g["omega"])), False, -1
    ok_steps, tol = _success_steps(args), (args.theta_tol, args.omega_tol)
    # BOTH criteria are tracked on every run, whichever one `--success-at` scores, so
    # a single run reports "reached it at the deadline" and "reached it at all"
    # without choosing at launch time. They are not interchangeable -- `any` is a
    # SATURATED instrument (see `_success_steps`), so it is a diagnostic and needs
    # its own floor and ceiling quoted beside it, never a headline on its own.
    hit_any, t_hit_any, hit_target = False, -1, False
    th_at_t = om_at_t = float("nan")

    # `budget` counts ENV STEPS. It previously counted replans, so an episode was
    # budget*receding steps, 5x more than intended at receding=5. Counting steps
    # fixes that.
    # Runtime accounting, reported alongside the result: `plan_s` isolates the CEM
    # solves from encoding and the plant, so the cost of a planning arm can be quoted
    # per solve rather than per episode (episodes end early on a hit, so wall_s alone
    # conflates "fast planner" with "solved sooner").
    done, n_solves, plan_s = 0, 0, 0.0
    # Under `--replan` the horizon shrinks to the frames left before the goal
    # instant, so there is nothing to solve once the deadline is reached: the
    # episode stops there rather than at `--budget`. The single-shot case keeps the
    # budget as the stopping rule, which is the historical behaviour.
    stop = args.goal_offset if args.replan else args.budget
    while done < stop:
        # context: the last h frames. The pad is now dead code for `dataset` goals --
        # `hist` is seeded h_ctx long -- and fires only for `upright`, which has no
        # recorded run-up to seed from.
        ctx_th = ([hist[0]] * max(0, h_pix - len(hist)) + hist)[-h_pix:]
        with torch.no_grad():
            if is_lehami:
                # (q, p) via run_eval's anchor -- the same stencil training used, on
                # the same construction the GOAL anchor gets: h_ctx real, distinct,
                # consecutive angles. That symmetry is the point; see `hist` above.
                z_ctx = anchor_from_angles(model, cfg, ctx_th, L, m, device,
                                           mom_type)
            else:
                z = encode_frames(model, cfg, ctx_th, L, m, device)
                z_ctx = z.unsqueeze(0)                     # (1, h, d)

        # The h-1 torques applied at the context frames, oldest first. For `dataset`
        # goals these are the RECORDED torques from the first solve onward. The zero
        # front-pad now fires only for `upright`, where the context frames really are
        # the repeated initial frame and nothing had been applied to them yet.
        past = None
        n_past = h_ctx - 1 + enc_lag          # must match `latent_rollout`
        if not is_lehami and n_past > 0:
            recent = act_hist[-n_past:]
            past = [0.0] * (n_past - len(recent)) + list(recent)

        # WHERE THE TERMINAL COST AIMS, per solve.
        #
        # Without --replan: the fixed `--horizon`, aimed at the fixed goal. One solve
        # per episode when receding == horizon == goal_offset, which is what every
        # result before 2026-09-01 used, reproduced here unchanged.
        #
        # With --replan the horizon SHRINKS to `goal_offset - done`, so every solve
        # aims at the SAME absolute goal instant and `--horizon` is only an upper
        # bound. That is what EXP-085 got wrong: a FIXED horizon under receding
        # execution compares the rollout's endpoint -- frame `done + horizon` -- to a
        # goal sitting at frame `goal_offset`, so the planner is scored on reaching
        # the goal LATE and the goal stays perpetually H steps away
        # (predicted-at-goal 12/12, actual 0/12).
        #
        # Neither branch changes what counts as success: that is `--success-at`, and
        # both `success_target` and `success_any` are written on every run either way.
        #
        # An earlier `--replan-aim subgoal` aimed each solve at the RECORDED state
        # `--horizon` frames ahead. Removed 2026-09-01: it handed the planner the
        # demonstrator's route as waypoints, which is strictly more information than
        # the task provides, so it could never be quoted as a task result -- and it
        # has no deployment analogue, since a real caller has a goal but no
        # demonstration of how to reach it.
        if args.replan:
            h_now = args.goal_offset - done
        else:
            h_now = args.horizon
        z_goal_now = z_goal

        # A non-positive horizon makes `cem_plan` return an EMPTY plan: the inner
        # loop then executes nothing, `done` never advances and the `while` spins
        # forever. That is not hypothetical -- it is what the removed
        # `--replan-aim subgoal` did once `done` reached the deadline, because its
        # `stop` was `--budget` while its `h_now` was capped at the frames
        # remaining. Assert rather than hang: a stalled 8-GPU-hour job looks
        # exactly like a slow one.
        assert h_now > 0, (
            f"empty plan: h_now={h_now} at done={done}/{stop} "
            f"(receding={args.receding}, goal_offset={args.goal_offset}). The "
            f"stopping rule and the horizon disagree.")
        _t0 = time.time()
        plan = cem_plan(model, cfg, z_ctx, z_goal_now, h_now, args.u_max,
                        rng, device, args, past_actions=past)
        if device.type == "cuda":
            torch.cuda.synchronize()      # else the launch queue is timed, not the work
        plan_s += time.time() - _t0
        n_solves += 1
        for u in plan[: args.receding]: ## the rest is discarded, only the first `receding` steps are applied
            theta, omega = plant_step(theta, omega, L, m, u, dt)
            hist.append(theta)
            act_hist.append(float(u))
            done += 1
            dth = abs(float(wrap(theta - g["theta"])))
            om = abs(float(omega - g["omega"]))
            if dth < best_dth:
                best_dth = dth
            ok_now = _goal_success(theta, omega, g, tol)
            if ok_now and not hit_any:
                hit_any, t_hit_any = True, done          # FIRST pass, at any step
            if done == args.goal_offset:
                # The state AT THE DEADLINE, recorded so success at any other
                # tolerance can be recomputed from the CSV without re-running. The
                # tolerance is then a reporting choice made in the open, rather than
                # a number baked into 8 GPU-hours.
                th_at_t, om_at_t = float(theta), float(omega)
                hit_target = ok_now                      # AT the deadline, not before
            if (ok_steps is None or done in ok_steps) and ok_now and not hit:
                hit, t_hit, best_om = True, done, om     # t_hit is now an ENV STEP
            if done >= args.budget:
                break
        if hit:
            break

    return dict(L=L, m=m, theta0=theta0, omega0=omega0, planner_seed=planner_seed,
                theta_goal=g["theta"], omega_goal=g["omega"],
                theta_at_target=th_at_t, omega_at_target=om_at_t,
                err_theta_at_target=abs(float(wrap(th_at_t - g["theta"]))),
                err_omega_at_target=abs(float(om_at_t - g["omega"])),
                ep=ep["ep"], start_idx=ep["start_idx"], goal_idx=ep["goal_idx"],
                success=int(hit), t_hit=t_hit, best_theta_err=best_dth,
                # Both criteria, every run. `success_target` is NaN when the episode
                # ended before the deadline (only possible under --success-at any,
                # which breaks on the first hit): the question was never asked, and
                # scoring that as a 0 would understate the target rate.
                success_any=int(hit_any), t_hit_any=t_hit_any,
                success_target=(int(hit_target) if done >= args.goal_offset
                                else float("nan")),
                omega_at_hit=best_om if hit else float("nan"),
                n_solves=n_solves, steps_done=done,
                plan_s=round(plan_s, 2),
                s_per_solve=round(plan_s / max(n_solves, 1), 3))


# ── model-free controls, computed once per episode and reported alongside ─────


def open_loop_result(ep, actions, args):
    """Run a fixed torque sequence on the plant; did it reach the goal?

    Used for the DO-NOTHING floor (`actions = []`). A free pendulum reaches turning
    points on its own, so some sampled goals need no torque at all -- measured at
    34% on this dataset at offset 25. The floor is REPORTED, never filtered out:
    dropping those goals would select the evaluation set on a property of the
    plant, and the natural criterion for doing so (an energy gap) is the favoured
    arm's own conserved quantity. Read planner numbers as lift over this floor.

    Also takes `ep["replay"]`, the recorded torques, which is how
    `test_recorded_actions_reach_their_own_goal` checks the index arithmetic in
    `dataset_goals` once, offline, rather than on every episode of every run.
    """
    dt = BASE_DT * args.dt_mult
    th, om, g = ep["theta0"], ep["omega0"], ep["goal"]
    ok_steps, tol = _success_steps(args), (args.theta_tol, args.omega_tol)
    best = abs(float(wrap(th - g["theta"])))
    for i in range(args.budget):
        u = actions[i] if i < len(actions) else 0.0
        th, om = plant_step(th, om, ep["L"], ep["m"], u, dt)
        best = min(best, abs(float(wrap(th - g["theta"]))))
        if (ok_steps is None or i + 1 in ok_steps) and _goal_success(th, om, g, tol):
            return 1, i + 1, best
    return 0, -1, best


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help="comma-separated key=path list, e.g. a0s_s0=...,b0s_s0=...")
    ap.add_argument("--n-ic", type=int, default=16, help="initial conditions")
    ap.add_argument("--planner-seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=30,
                    help="planning horizon in ENV STEPS = 10 searched torques x "
                         "action_block 3. Must divide by --action-block, and in dataset "
                         "mode must EQUAL --goal-offset: the goal is the state at an "
                         "absolute frame, so a terminal cost that aims H steps ahead is "
                         "aiming at the deadline only when H is the distance to it. "
                         "(Was 25, which no longer divides by the block of 3.)")
    ap.add_argument("--action-block", type=int, default=3,
                    help="frameskip: each searched torque is held this many env steps. "
                         "LeWM sets action_block=5 in every eval config, which is also "
                         "what gives its plans temporally coherent torques.")
    ap.add_argument("--receding", type=int, default=None,
                    help="env steps executed per solve. DEFAULT = --horizon, i.e. execute "
                         "the whole plan, which is what every LeWM eval config uses "
                         "(receding_horizon == horizon, 5 and 5). A TERMINAL cost is only "
                         "sound under that equality -- see --cost. Pass a smaller value "
                         "only to re-measure the broken corner deliberately.")
    ap.add_argument("--replan", action="store_true",
                    help="Turn on receding-horizon MPC: solve, execute --receding env "
                         "steps, re-encode the state ACTUALLY reached, solve again. OFF "
                         "by default, which keeps the single-shot open-loop behaviour "
                         "every result before 2026-09-01 was produced with, bit for bit. "
                         "It also lifts the --cost terminal / --receding == --horizon "
                         "restriction, because the horizon SHRINKS to the frames left "
                         "before the goal instant (goal_offset - done), so every solve "
                         "aims at the one true goal and --horizon is only an upper "
                         "bound; the tunable is then --receding alone. It does NOT "
                         "change the success criterion -- that is --success-at, a "
                         "separate axis.")
    ap.add_argument("--budget", type=int, default=None,
                    help="episode length in ENV STEPS (300 x dt=0.1s = 30 s). NOTE: this "
                         "counted REPLANS before 2026-08-20, so the old default 60 meant "
                         "60x5 = 300 steps -- the same episode length, now stated directly.")
    ap.add_argument("--u-max", type=float, default=4.0)
    ap.add_argument("--dt-mult", type=int, default=2)
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.7, 1.0, 1.4])
    ap.add_argument("--masses", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--cost", choices=["terminal", "running"], default="terminal",
                    help="'terminal' scores only the last horizon step -- LeWorldModel's "
                         "own criterion (third_party/leworldmodel/jepa.py:112). It is sound "
                         "ONLY when receding == horizon, which LeWM's configs enforce and "
                         "which is now the default here. 'running' sums over all horizon "
                         "steps and is sound at any receding. Oracle 2x2 (16 ICs): "
                         "terminal+receding5 = 0/16, all three other corners 16/16.")
    ap.add_argument("--p-weight", type=float, default=None,
                    help="weight on the p block of the latent cost (LeHami only). "
                         "DEFAULT = the checkpoint's own cfg.model.momentum_weight, "
                         "i.e. the weighting the model was TRAINED under. Pass a "
                         "value only to run the reweighting ablation deliberately.")
    ap.add_argument("--goal-mode", choices=list(GOAL_MODES), default="upright",
                    help="'upright' = the fixed task goal (theta=pi, at rest), one "
                         "target for every episode, two horizons away and behind the "
                         "separatrix. 'dataset' = LeWorldModel's own construction: "
                         "sample episodes, pick a GOAL frame in each, and take the "
                         "START frame --goal-offset steps EARLIER, so reachability "
                         "holds by construction rather than by witness.")
    ap.add_argument("--goal-offset", type=int, default=30,
                    help="dataset mode: steps from start to goal, i.e. the reach "
                         "distance -- this is the difficulty knob. LeWM uses 25 "
                         "(goal_offset_steps); 30 is the nearest setting that is also "
                         "one whole planning horizon of 10 blocks x 3. Keep it EQUAL "
                         "to --horizon.")
    ap.add_argument("--draw-offset", type=int, default=None,
                    help="dataset mode: the offset used to DRAW the goals, decoupled "
                         "from --goal-offset which sets the START and the reach "
                         "distance. Default None = same as --goal-offset, which "
                         "reproduces every earlier run bit for bit. Pass "
                         "`--draw-offset 30 --goal-offset 15` to keep the SAME goal "
                         "frames as a K=30 sweep while starting 15 steps out, so the "
                         "two conditions are PAIRED row by row and McNemar applies. "
                         "Without it, changing --goal-offset moves `lo` and therefore "
                         "re-draws every goal, and the runs are not comparable.")
    ap.add_argument("--n-eval", type=int, default=50,
                    help="dataset mode: episodes to evaluate (LeWM: num_eval 50)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="dataset mode: which split to draw goals from. LeWM draws "
                         "from its TRAIN split (pusht_expert_train); test is the "
                         "defensible choice and is the default here.")
    ap.add_argument("--dataset", default=None,
                    help="dataset mode: path to pendulum_swingup.h5 "
                         "(default $STABLEWM_HOME/pendulum_swingup.h5)")
    ap.add_argument("--manifest",
                    default=str(ROOT / "lehami" / "data" /
                                "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json"),
                    help="dataset mode: the episode split manifest")
    ap.add_argument("--success-at", choices=["any", "target"], default=None,
                    help="WHEN success is tested. 'any' = at any step in the budget "
                         "(LeWorldModel's criterion, swm/world.py:961). 'target' = at "
                         "the goal's own timestamp, step --goal-offset. Default: "
                         "'target' for --goal-mode dataset, 'any' for upright. See "
                         "`_success_steps` for the measurements behind that split -- "
                         "under 'any' the dataset task is SATURATED (300 random "
                         "sequences solve it with exact dynamics).")
    ap.add_argument("--theta-tol", type=float, default=None,
                    help="angle tolerance (rad). Default 0.2 for upright (the "
                         "pre-registered 'balanced at the top'), 0.1 for dataset.")
    ap.add_argument("--omega-tol", type=float, default=None,
                    help="velocity tolerance (rad/s). Default 1.0 for upright, 0.5 "
                         "for dataset. NOTE: no LeWorldModel environment has a "
                         "velocity term in its success criterion at all (PushT tests "
                         "positions and a block angle; TwoRoom a distance). This "
                         "clause is ours, and it is what makes the task about "
                         "momentum rather than pose.")
    ap.add_argument("--cem-samples", type=int, default=CEM_DEFAULTS["num_samples"])
    ap.add_argument("--cem-iters", type=int, default=CEM_DEFAULTS["n_steps"])
    ap.add_argument("--cem-elites", type=int, default=CEM_DEFAULTS["topk"])
    ap.add_argument("--ic-seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: 2 ICs, 1 planner seed, tiny CEM budget")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    # BEFORE the derived settings below, because it overrides `horizon`.
    if args.smoke:
        args.n_ic, args.planner_seeds = 2, 1
        args.cem_samples, args.cem_iters, args.cem_elites = 16, 2, 4
        args.horizon, args.action_block = 9, 3
        if args.budget is None:
            args.budget = 20

    # Default: execute the whole plan (LeWM: receding_horizon == horizon). Resolved
    # after --smoke because that overrides `horizon`.
    if args.receding is None:
        args.receding = args.horizon

    # Criterion defaults, resolved per mode and printed below so a run is
    # self-describing. Fixed on the ORACLE pre-flight (floors and the random-search
    # ceiling), before any checkpoint was loaded.
    if args.success_at is None:
        args.success_at = "target" if args.goal_mode == "dataset" else "any"
    if args.theta_tol is None:
        args.theta_tol = 0.1 if args.goal_mode == "dataset" else THETA_TOL
    if args.omega_tol is None:
        args.omega_tol = 0.5 if args.goal_mode == "dataset" else OMEGA_TOL

    if args.goal_mode == "dataset":
        if args.dataset is None:
            import os
            args.dataset = str(Path(os.environ.get(
                "STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
                / "pendulum_swingup.h5")
        if args.smoke:
            # The deadline must stay INSIDE the budget, or --success-at target can
            # never fire and the smoke reports a uniform 0 that looks like a result.
            # 9, not 10: it must equal the smoke `horizon` (terminal cost aims at the
            # deadline) and divide by the action block.
            args.n_eval, args.goal_offset = 4, 9
        # The ONE way the dataset's start->goal guarantee can break at runtime.
        # Reachability is a property of the recorded episode, so nothing needs
        # re-verifying per run -- but only if the plant is stepped on the clock the
        # episode was recorded on. `--dt-mult 1` would integrate at half the
        # recorded rate and silently make every goal unreachable.
        import h5py
        with h5py.File(args.dataset, "r") as _f:
            _dm = int(_f["dt_mult"][0]) if "dt_mult" in _f else None
        if _dm is None:
            print(f"[warn] {args.dataset} carries no dt_mult column; cannot verify "
                  f"the plant clock matches the recording. Assuming "
                  f"--dt-mult {args.dt_mult} is right.")
        elif _dm != args.dt_mult:
            raise SystemExit(
                f"--dt-mult {args.dt_mult} but {args.dataset} was recorded at "
                f"dt_mult={_dm}. The recorded actions would no longer connect "
                f"start to goal, so the goals would not be reachable.")
    # LeWM's eval_budget is 2x goal_offset_steps (50 vs 25): one horizon to reach
    # the goal, one more of slack. The upright mode keeps 300 (30 s), which it
    # needs -- its goal is two horizons away and has to be pumped to.
    if args.budget is None:
        if args.goal_mode != "dataset":
            args.budget = 300
        elif args.success_at == "target":  # noqa: SIM114
            # Only step `goal_offset` is scored, so steps beyond it cannot change the
            # outcome. budget == offset == horizon == receding: exactly one CEM solve,
            # executed in full -- which is what LeWM's terminal cost assumes anyway.
            args.budget = args.goal_offset
        else:
            args.budget = 2 * args.goal_offset          # LeWM's eval_budget

    # A dataset goal is the state at an ABSOLUTE frame, si+K, while a terminal cost
    # asks to be at the goal H steps from now. If H != K the planner optimises for a
    # moment nothing is scored at: too early on the first solve, past the deadline on
    # the second. Same family of error as the receding/terminal bug below.
    # `--replan` keeps H == K on the first solve and shrinks from there, so it is
    # never optimising for a moment nothing is scored at; its own check below pins
    # --horizon >= --goal-offset, which together with this one forces equality.
    if (args.goal_mode == "dataset" and args.cost == "terminal"
            and args.horizon != args.goal_offset):
        ap.error(f"--cost terminal on dataset goals requires --horizon == "
                 f"--goal-offset (got {args.horizon} vs {args.goal_offset}): the goal "
                 f"is the state at step {args.goal_offset}, so a plan scored at step "
                 f"{args.horizon} is aiming at the wrong instant. Use --cost running "
                 f"to search a horizon shorter than the reach.")

    if args.horizon % args.action_block:
        ap.error(f"--horizon {args.horizon} must be divisible by "
                 f"--action-block {args.action_block}")

    # The failure that voided EXP-085. A terminal cost says nothing about the states
    # before step H, so if you execute only a prefix you reward plans that arrive at
    # exactly H and defer forever: measured 0/16 with perfect dynamics, against 16/16
    # for every other corner of the {terminal,running} x {prefix,whole-plan} square.
    if args.cost == "terminal" and args.receding != args.horizon and not args.replan:
        ap.error(
            f"--cost terminal requires --receding == --horizon (got {args.receding} vs "
            f"{args.horizon}). This is the EXP-085 bug: it scored 0/480, and an ORACLE "
            f"with exact dynamics scored 0/80 in the same configuration. Either set "
            f"--receding {args.horizon} (LeWorldModel's setting), use --cost running, "
            f"or pass --replan, which re-aims the terminal cost every solve so the "
            f"goal stops receding."
        )

    # --replan's own preconditions. They are what make a terminal cost sound under a
    # receding execution, so they are checked here rather than trusted.
    if args.replan:
        if not 1 <= args.receding <= args.goal_offset:
            ap.error(f"--replan needs 1 <= --receding <= --goal-offset (got "
                     f"{args.receding} vs {args.goal_offset}).")
        if args.horizon < args.goal_offset:
            ap.error(
                f"--replan needs --horizon >= --goal-offset ({args.horizon} "
                f"< {args.goal_offset}): the first solve aims at the goal "
                f"{args.goal_offset} steps away, so a shorter horizon cannot reach it. "
                f"Set --horizon {args.goal_offset} -- it then shrinks on its own.")
        # `--budget` is not the stopping rule under --replan (`stop` is
        # --goal-offset), but it still breaks the inner execution loop. A budget
        # short of the deadline would truncate the episode before the goal instant
        # and write success_target=NaN rather than 0, silently shrinking the
        # denominator the target rate is averaged over.
        if args.budget < args.goal_offset:
            ap.error(f"--replan needs --budget >= --goal-offset ({args.budget} < "
                     f"{args.goal_offset}): the episode would be cut off before the "
                     f"goal instant and success_target would be NaN, not 0.")

    # A deadline outside the budget is unreachable by construction, which would
    # look exactly like a model failure.
    if args.success_at == "target" and args.budget < args.goal_offset:
        ap.error(f"--success-at target scores step {args.goal_offset} but --budget "
                 f"is {args.budget}: the deadline is outside the episode, so no run "
                 f"can ever succeed. Set --budget >= {args.goal_offset}.")

    print(f"success: at={args.success_at}  theta_tol={args.theta_tol}  "
          f"omega_tol={args.omega_tol}")
    print(f"plan: horizon={args.horizon} env steps ({args.horizon // args.action_block} "
          f"torques x action_block {args.action_block}), receding={args.receding}, "
          f"cost={args.cost}, budget={args.budget} env steps "
          f"({args.budget * BASE_DT * args.dt_mult:.0f} s)")

    device = torch.device(args.device)
    specs = [tuple(kv.split("=", 1)) for kv in args.checkpoints.split(",") if kv]

    # Episodes are built ONCE and shared across arms and planner seeds -> the
    # comparison is PAIRED, which is the whole point. `GOAL_CTX` / `START_CTX`
    # rather than any arm's own widths for the same reason.
    eps = GOAL_MODES[args.goal_mode](args)
    print(f"goal-mode={args.goal_mode}  {len(eps)} episodes"
          + (f"  (split={args.split}, offset={args.goal_offset})"
             if args.goal_mode == "dataset" else ""))

    # The model-free controls. Computed here, once, because they depend on the
    # episode and not on the model -- and printed before any arm runs, so the floor
    # is on the record before a planner number exists to compare it against.
    #
    #   zero_*    the u = 0 FLOOR. A free pendulum reaches turning points on its own,
    #             so some sampled goals need no torque at all. Reported, never
    #             filtered: dropping them would select the evaluation set on a
    #             property of the plant. Read every planner number as LIFT over it.
    #   replay_*  the RECORDED torques, dataset mode only. This is the by-construction
    #             reference -- the control sequence that produced the goal state --
    #             so `replay_success` is the reachability guarantee re-checked on the
    #             plant every run rather than asserted once in a docstring. Anything
    #             below 1.0 here means the index arithmetic or the clock is wrong and
    #             NO planner number from the run is interpretable.
    ctrl = {}
    for e in eps:
        zs, zt, zb = open_loop_result(e, [], args)
        c = dict(zero_success=zs, zero_t_hit=zt, zero_best_err=zb)
        if e.get("replay") is not None:
            rs, rt, rb = open_loop_result(e, e["replay"], args)
            c |= dict(replay_success=rs, replay_t_hit=rt, replay_best_err=rb)
        ctrl[id(e)] = c
    print(f"controls: u=0 floor "
          f"{np.mean([c['zero_success'] for c in ctrl.values()]):.3f}")
    if any("replay_success" in c for c in ctrl.values()):
        rep = np.mean([c["replay_success"] for c in ctrl.values()])
        print(f"          replay (recorded torques) {rep:.3f}"
              + ("" if rep == 1.0 else "   <-- BELOW 1.0: the goals are NOT reachable "
                                       "as constructed; do not read the arms"))

    t_cell = time.time()
    rows = []
    for key, path in specs:
        model, cfg = load_model(Path(path), str(device))
        model.eval()
        _check_plannable(cfg, args, key)
        print(f"\n=== {key}   ({cfg.model.name})")
        # The frame arithmetic this arm will actually run, printed once per arm.
        # A W>1 encoder changes the context width, the goal window and the rollout
        # length, and every one of those fails SILENTLY when wrong: the run still
        # produces plausible successes. Printing them is what makes a smoke test of
        # a lagged arm (b3s) readable, and costs one line per arm.
        _mt = _mom_type(cfg.model.name)
        _hctx = (_anchor_frame(cfg, _mt) + 1) if _mt != "lewm" else int(cfg.history_size)
        _w, _lag = _enc_window(model), _enc_window(model) - 1 - _enc_shift(model)
        print(f"    encoder: window={_w} lag={_lag} | context {_hctx + _w - 1} pixel "
              f"frames -> {_hctx} latent(s) | goal window {_w} frame(s) "
              f"{'centred on' if _lag else 'at'} the goal | rollout "
              f"{args.horizon}+{_lag} steps to reach the goal instant")
        for e in eps:
            for ps in range(args.planner_seeds):
                t0 = time.time()
                r = run_episode(model, cfg, e, ps, args, device)
                # `cost`, `receding` and `goal_mode` ride along so CSVs from
                # separate submissions concatenate into one self-describing frame.
                r |= dict(model=key, arm=key.split("_")[0], cost=args.cost,
                          goal_mode=args.goal_mode, goal_offset=args.goal_offset,
                          horizon=args.horizon,
                          receding=args.receding, success_at=args.success_at,
                          replan=bool(args.replan),
                          # "single_shot" rather than "n/a": pandas reads "n/a" as NaN by default,
                          # which would erase the distinction on the way back in.
                          replan_aim=("goal" if args.replan else "single_shot"),
                          theta_tol=args.theta_tol, omega_tol=args.omega_tol,
                          **ctrl[id(e)],
                          wall_s=round(time.time() - t0, 1))
                rows.append(r)
                print(f"  L={e['L']:.1f} m={e['m']:.1f} th0={e['theta0']:+.2f} "
                      f"seed={ps}  success={r['success']}  "
                      f"(u=0 {r['zero_success']})  "
                      f"best_err={r['best_theta_err']:.3f}  {r['wall_s']:.1f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "swingup_episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(
        {"args": vars(args), "theta_tol": THETA_TOL, "omega_tol": OMEGA_TOL,
         "n_rows": len(rows), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
         # Runtime, so the paper can quote planning cost per solve rather than a
         # wall time that depends on how many episodes ended early.
         "runtime": {
             "cell_wall_s": round(time.time() - t_cell, 1),
             "device": (torch.cuda.get_device_name(0) if device.type == "cuda"
                        else "cpu"),
             "cem": {"samples": args.cem_samples, "iters": args.cem_iters,
                     "elites": args.cem_elites},
             "predictor_steps_per_solve": args.horizon // args.action_block,
         }},
        indent=1, default=str))

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 66)
    print(df.groupby("arm")[["success", "best_theta_err"]].mean().round(3).to_string())
    print("=" * 66)
    # s_per_solve is the comparable runtime number: it does not depend on how many
    # episodes ended early on a hit, unlike wall_s.
    print("runtime (" + (torch.cuda.get_device_name(0) if device.type == "cuda"
                         else "cpu") + f", cell {time.time() - t_cell:.0f}s):")
    print(df.groupby("arm")[["s_per_solve", "n_solves", "wall_s"]]
            .mean().round(3).to_string())
    print("=" * 66)
    print("Compare the arms PAIRED on (L, m, theta0, omega0, planner_seed) with "
          "McNemar; an unpaired s.e. here is ~0.12 and hides the effect.")


if __name__ == "__main__":
    main()
