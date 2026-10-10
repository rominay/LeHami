"""The swing-up planner must drive each model the way that model was trained.

`plan_pendulum_swingup.py` re-implements the forward pass of two different model
families so it can roll them out under candidate torques.  Two bugs came out of
that duplication, both silent -- no crash, no shape error, just wrong numbers:

1.  `p_0` was read with a 2-point difference over `cfg.history_size` frames.
    `history_size` is 1 for every swing-up run, so the fallback branch fired on
    EVERY replan and LeHami planned from `p = 0`, i.e. "the pendulum is at rest",
    which on a swing-up task is the one quantity the experiment is about.
    `history_size` describes the ENCODER's frame window; the momentum stencil is
    a property of the model family (4-frame causal Savitzky-Golay for `sg`).

2.  The LeWM action window was sliced `u[t : t+h]` -- starting at `t` instead of
    ending there -- so the torque driving the predicted step sat in the OLDEST
    context slot, shifted by `h-1`, with the history slots holding torques not
    yet applied.

Neither is detectable from a metric.  These tests pin the invariants instead, on
stub modules: no checkpoint, no GPU, no dataset.  The stubs make the action
embedding the identity, so what a test reads back IS the torque that was handed
over and an index error cannot hide -- the same trick
`test_eval_action_alignment.py` uses.

Run with:
    uv run pytest lehami/tests/test_swingup_plan_matches_training.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "lehami" / "models",
           ROOT / "lehami" / "data",
           ROOT / "lehami" / "experiments" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_eval import _anchor_frame, _mom_type  # noqa: E402

D = 4          # latent_dim in these tests; small enough to eyeball
H_CTX = 3      # LeWM history_size


# ── stubs ─────────────────────────────────────────────────────────────────────


class _StubEncoder:
    """Just the two attributes the latent-time arithmetic reads.

    `window` / `lag` are the TemporalConvEncoder contract (`builders.py`): a
    per-frame encoder is window=1, lag=0; the symmetric variants are lag=(W-1)/2.
    `_enc_shift` reads them to get the latent->pixel offset, so a stub without an
    encoder cannot exercise the W>1 path at all.
    """

    def __init__(self, window=1, lag=None):
        self.window = window
        self.lag = (window - 1) / 2.0 if lag is None else lag


class _LeWMStub:
    """`predict` is sensitive to WHICH slot each action landed in.

    Multiplying the action by a per-dimension coefficient makes any permutation of
    the window change the output, so an alignment test cannot pass by accident.
    """

    def __init__(self, window=1, lag=None):
        self.coef = torch.arange(1.0, D + 1.0)
        self.seen = []                       # every action window it was handed
        self.encoder = _StubEncoder(window, lag)

    @staticmethod
    def action_encoder(a):                   # identity: embedding == torque
        return a

    def predict(self, ctx, act_emb):         # (n,h,D), (n,h,1) -> (n,h,D)
        self.seen.append(act_emb.clone())
        return ctx + act_emb * self.coef


class _LeHamiStub:
    """One action per step; `predictor` records the action it was given."""

    def __init__(self):
        self.seen = []

    @staticmethod
    def action_encoder(a):
        return a

    def predictor(self, qp, a_emb, dt=None):  # (n,1,2D), (n,1,1) -> (n,1,2D)
        self.seen.append(a_emb.clone())
        return qp + a_emb


def _cfg(name, history_size=H_CTX):
    # `embed_dim` is the LeWM-side name for the same quantity. It must be present
    # even for an lehami cfg: `_latent_dim` is `cfg.model.get("latent_dim",
    # cfg.embed_dim)`, and OmegaConf evaluates that default eagerly.
    return OmegaConf.create(
        {"model": {"name": name, "latent_dim": D, "fd_dt": 1.0},
         "embed_dim": D, "history_size": history_size}
    )


# ── 1. the momentum stencil, not history_size, sets the context length ────────


@pytest.mark.parametrize("history_size", [1, 2, 3])
def test_sg_anchor_ignores_history_size(history_size):
    """The regression guard for bug 1.

    `history_size: 1` must NOT collapse the momentum context.  An `sg` model
    anchors at clip frame 3 whatever the encoder window is, because that is where
    the 4-frame causal stencil first has a full window (`builders.py`: `A =
    ANCHOR_OFFSET`).  The planner derives its context length from this, so if it
    ever starts tracking `history_size` again, `p_0` silently becomes 0.
    """
    from lehami.derivative import ANCHOR_OFFSET

    cfg = _cfg("lehami_sg_sep_sreg_q_ms_conv", history_size)
    assert _mom_type(cfg.model.name) == "sg"
    assert _anchor_frame(cfg, "sg") == ANCHOR_OFFSET == 3
    assert _anchor_frame(cfg, "sg") + 1 == 4, "sg needs 4 frames of context"


def test_sg_momentum_of_a_constant_window_is_zero():
    """`p_goal = 0` is DERIVED, not imposed.

    The goal is rendered `h_ctx` times from one upright pose, so the stencil sees a
    constant sequence.  A Savitzky-Golay fit is linear, so its slope is exactly
    zero -- which is what makes "upright AND at rest" expressible without any
    special-casing of the goal.
    """
    from lehami.derivative import CAUSAL_WINDOW, p_causal_last

    const = torch.randn(1, 1, D).repeat(1, CAUSAL_WINDOW, 1)
    assert p_causal_last(const, 1.0).abs().max() < 1e-6


def test_sg_momentum_recovers_a_known_slope():
    """The counterpart: the stencil is not zero on everything."""
    from lehami.derivative import CAUSAL_WINDOW, p_causal_last

    ramp = torch.arange(CAUSAL_WINDOW, dtype=torch.float32).view(1, -1, 1) * 3.0
    assert torch.allclose(p_causal_last(ramp, 1.0), torch.tensor(3.0), atol=1e-5)


# ── 2. the LeWM action window ends at the current frame ──────────────────────


def _reference_lewm_rollout(model, ctx0, u, past, h):
    """LeWorldModel's OWN formulation, from `jepa.py:61`.

    Deliberately NOT written as a slice.  `jepa.rollout` grows `emb` and `act` as
    two parallel buffers and truncates BOTH with `[-HS:]`; that is the invariant,
    and expressing it this way means the test cannot pass merely by restating the
    planner's arithmetic.  Context slot i carries the action applied at frame i,
    so the window always ends at the newest context frame.
    """
    H = u.shape[1]
    emb = ctx0.clone()                                   # frames c-h+1 .. c
    act = torch.cat([past, u[:, :1]], dim=1)             # actions at those frames
    for t in range(H):
        a_emb = model.action_encoder(act[:, -h:])
        pred = model.predict(emb[:, -h:], a_emb)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        if t + 1 < H:
            act = torch.cat([act, u[:, t + 1 : t + 2]], dim=1)
    return emb[:, -1]                                    # LeWM's state IS z


def test_lewm_window_matches_leworldmodels_parallel_buffers():
    """The regression guard for bug 2."""
    from plan_pendulum_swingup import latent_rollout

    torch.manual_seed(0)
    n, H = 2, 7
    ctx0 = torch.randn(n, H_CTX, D)
    u = torch.randn(n, H, 1)
    past = torch.randn(1, H_CTX - 1, 1).expand(n, -1, -1)

    ref = _reference_lewm_rollout(_LeWMStub(), ctx0, u, past, H_CTX)
    got = latent_rollout(_LeWMStub(), _cfg("lewm_ms_conv"), ctx0, u, torch.device("cpu"),
                         past_actions=past[0, :, 0].tolist())

    assert torch.allclose(got, ref, atol=1e-6)


def test_lewm_driving_action_is_the_last_slot():
    """Stated directly, so a failure says WHICH slot is wrong.

    At step t the window must be [u_{t-2}, u_{t-1}, u_t]: the torque driving the
    predicted transition is the newest one.  The pre-fix code produced
    [u_t, u_{t+1}, u_{t+2}], which is this assertion's exact negation.
    """
    from plan_pendulum_swingup import latent_rollout

    n, H = 1, 6
    u = torch.arange(1.0, H + 1.0).view(1, H, 1)          # 1, 2, ... — traceable
    past = [-2.0, -1.0]
    m = _LeWMStub()
    latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(n, H_CTX, D), u,
                   torch.device("cpu"), past_actions=past)

    full = past + [float(x) for x in u[0, :, 0]]
    for t, win in enumerate(m.seen):
        assert win[0, -1, 0].item() == full[t + H_CTX - 1], f"step {t}: wrong driver"
        assert [round(v.item(), 5) for v in win[0, :, 0]] == full[t : t + H_CTX]


def test_lewm_first_solve_pads_with_zeros_not_with_future_torques():
    """On the first solve nothing has been applied, so zero is the TRUTH.

    The bug zero-padded the TAIL, which under a terminal cost meant the single
    scored latent came from [u_last, 0, 0].  Padding belongs at the front.
    """
    from plan_pendulum_swingup import latent_rollout

    u = torch.arange(1.0, 5.0).view(1, 4, 1)
    m = _LeWMStub()
    latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D), u,
                   torch.device("cpu"), past_actions=None)

    assert [v.item() for v in m.seen[0][0, :, 0]] == [0.0, 0.0, 1.0]
    # and the LAST window is fully real -- no trailing zeros
    assert [v.item() for v in m.seen[-1][0, :, 0]] == [2.0, 3.0, 4.0]


def test_lewm_rejects_wrong_length_history():
    from plan_pendulum_swingup import latent_rollout

    with pytest.raises(ValueError, match="past_actions"):
        latent_rollout(_LeWMStub(), _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D),
                       torch.zeros(1, 4, 1), torch.device("cpu"),
                       past_actions=[0.0])          # needs h-1 = 2


# ── 3. LeHami takes exactly one action per step, and never pads ───────────────


def test_lehami_uses_action_t_at_step_t():
    """One action per step, no window, so no alignment to get wrong -- assert it.

    This is also what makes both arms roll out `horizon` steps: LeHami consumes 25
    torques in 25 calls, and the LeWM arm now does too because its history comes
    from executed actions rather than from the candidate's first h-1 slots (which
    is where `jepa.rollout` takes them, giving T-H+1 steps).  Equal depth is what
    a terminal cost needs to be comparing the same question on both arms.
    """
    from plan_pendulum_swingup import latent_rollout

    H = 5
    u = torch.arange(1.0, H + 1.0).view(1, H, 1)
    m = _LeHamiStub()
    latent_rollout(m, _cfg("lehami_sg_sep_sreg_q_ms_conv"),
                   (torch.zeros(1, D), torch.zeros(1, D)), u, torch.device("cpu"))

    assert len(m.seen) == H, "one predictor call per horizon step"
    assert [w.item() for w in m.seen] == [float(t) for t in range(1, H + 1)]


def test_both_arms_roll_out_the_same_number_of_steps():
    from plan_pendulum_swingup import latent_rollout

    H = 9
    u = torch.arange(1.0, H + 1.0).view(1, H, 1)

    hj = _LeHamiStub()
    latent_rollout(hj, _cfg("lehami_sg_sep_sreg_q_ms_conv"),
                   (torch.zeros(1, D), torch.zeros(1, D)), u, torch.device("cpu"))
    lw = _LeWMStub()
    latent_rollout(lw, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D), u,
                   torch.device("cpu"), past_actions=[0.0, 0.0])

    assert len(hj.seen) == len(lw.seen) == H


def test_running_cost_returns_one_state_per_step():
    """`--cost running` sums per-step costs, so the list length is load-bearing."""
    from plan_pendulum_swingup import latent_rollout

    H = 6
    u = torch.zeros(1, H, 1)
    traj = latent_rollout(_LeHamiStub(), _cfg("lehami_sg_sep_sreg_q_ms_conv"),
                          (torch.zeros(1, D), torch.zeros(1, D)), u,
                          torch.device("cpu"), return_all=True)
    assert len(traj) == H and all(len(s) == 2 for s in traj)


# ── 4. LeWorldModel's goal construction: reachability by construction ────────


def _synthetic_dataset(tmp, n_ep=6, n_frames=40, seed=0):
    """A miniature pendulum_swingup.h5 with the real generator's own physics.

    Written with `simulate`, so the recorded `action` column genuinely drives
    `state` -- which is the whole property `dataset_goals` leans on.
    """
    import h5py
    import numpy as np
    from pendulum_swingup import simulate, torque_sequence

    rng = np.random.default_rng(seed)
    dt = 0.1
    st, ac, Ls, ms = [], [], [], []
    for i in range(n_ep):
        L, m = float(rng.choice([0.7, 1.0, 1.4])), float(rng.choice([0.5, 1.0, 2.0]))
        u = torque_sequence(rng, n_frames, 4.0, 3)
        st.append(simulate(rng.uniform(-np.pi, np.pi), rng.uniform(-2, 2),
                           L, m, dt, n_frames, u))
        ac.append(u[:, None])
        Ls.append(L)
        ms.append(m)
    ep_len = np.full(n_ep, n_frames, dtype=np.int64)
    path = tmp / "mini.h5"
    with h5py.File(path, "w") as f:
        f["ep_len"] = ep_len
        f["ep_offset"] = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(np.int64)
        f["state"] = np.concatenate(st).astype(np.float32)
        f["action"] = np.concatenate(ac).astype(np.float32)
        f["length"] = np.array(Ls, dtype=np.float32)
        f["mass"] = np.array(ms, dtype=np.float32)
        f["dt_mult"] = np.full(n_ep, 2, dtype=np.int64)   # dt = BASE_DT * 2 = 0.1 s
    man = tmp / "mini.json"
    man.write_text(json.dumps({"test_episode_indices": list(range(n_ep))}))
    return path, man


class _Args:
    """argparse stand-in: only the fields the goal builders read."""
    def __init__(self, **kw):
        self.__dict__.update(dict(
            goal_offset=10, n_eval=4, split="test", ic_seed=1234,
            dt_mult=2, budget=20, n_ic=4,
            # mirrors the --goal-mode dataset defaults resolved in main()
            success_at="target", theta_tol=0.1, omega_tol=0.5,
            lengths=[0.7, 1.0, 1.4], masses=[0.5, 1.0, 2.0]), **kw)


def test_dataset_goal_start_is_offset_steps_before_the_goal(tmp_path):
    """The construction itself: start = goal - offset, per swm/world.py:791."""
    from plan_pendulum_swingup import GOAL_CTX, GOAL_LEAD, START_CTX, dataset_goals

    ds, man = _synthetic_dataset(tmp_path)
    a = _Args(dataset=str(ds), manifest=str(man))
    eps = dataset_goals(a)

    assert len(eps) == a.n_eval
    for e in eps:
        assert e["goal_idx"] - e["start_idx"] == a.goal_offset
        assert len(e["replay"]) == a.goal_offset      # actions[start .. goal-1]
        # GOAL_CTX frames up to and including the goal, plus GOAL_LEAD after it,
        # so a symmetric arm can centre its goal latent ON the goal frame.
        assert len(e["goal"]["thetas"]) == GOAL_CTX + GOAL_LEAD
        # The start window is wider than the goal window and is what bounds `si`.
        assert len(e["start_thetas"]) == START_CTX
        assert e["start_idx"] >= START_CTX - 1


def test_recorded_actions_reach_their_own_goal(tmp_path):
    """Reachability is by CONSTRUCTION, and this is the assertion of it.

    `--goal-mode dataset` earns its claim only if replaying the recorded torques
    from the start state lands on the goal state. If this ever fails, the plant
    has drifted from the dataset generator and every planning number built on it
    is measuring that drift.
    """
    from plan_pendulum_swingup import dataset_goals, open_loop_result

    ds, man = _synthetic_dataset(tmp_path)
    a = _Args(dataset=str(ds), manifest=str(man))
    for e in dataset_goals(a):
        ok, t_hit, _ = open_loop_result(e, e["replay"], a)
        assert ok == 1, f"replay missed its own goal: ep {e['ep']}"
        # Under success_at='target' this is an equality: the recorded torques put
        # the pendulum on the goal state at exactly step `goal_offset`, which is
        # the only step scored.
        assert t_hit == a.goal_offset, f"reached at {t_hit}, not {a.goal_offset}"


def test_goal_window_is_arm_independent(tmp_path):
    """Every arm must be handed the SAME episodes or the paired test is void.

    The goal context is GOAL_CTX (4, the widest stencil), never the calling arm's
    h_ctx -- LeHami/sg wants 4 and LeWM 3, and letting that leak into the sampler
    would silently give the two arms different tasks.
    """
    from plan_pendulum_swingup import dataset_goals

    ds, man = _synthetic_dataset(tmp_path)
    a = _Args(dataset=str(ds), manifest=str(man))
    keys = ("ep", "start_idx", "goal_idx", "theta0", "omega0")
    for h in (2, 3, 4):
        assert [tuple(e[k] for k in keys) for e in dataset_goals(a, h_ctx=h)] == \
               [tuple(e[k] for k in keys) for e in dataset_goals(a)]


def test_upright_mode_is_unchanged(tmp_path):
    """The pre-registered protocol must survive the refactor bit-for-bit."""
    from plan_pendulum_swingup import upright_goals

    a = _Args(dataset=None, manifest=None)
    eps = upright_goals(a)
    rng = np.random.default_rng(a.ic_seed)
    for e in eps:
        L = float(rng.choice(a.lengths))
        m = float(rng.choice(a.masses))
        assert (e["L"], e["m"]) == (L, m)
        assert e["theta0"] == float(rng.uniform(-0.6, 0.6))
        assert e["omega0"] == float(rng.uniform(-0.5, 0.5))
        assert e["goal"]["theta"] == np.pi and e["goal"]["omega"] == 0.0
        assert e["replay"] is None


def _record_encoded_windows(monkeypatch, enc_window, enc_shift, plan_value=0.5):
    """Run the loop with the encoder stubbed, keeping every window it was HANDED.

    The other loop tests record window LENGTHS; these two need the angles
    themselves, because the question is WHICH frames of the recorded episode the
    planner reaches for.
    """
    import plan_pendulum_swingup as P
    seen = []

    def fake_encode(model, cfg, thetas, L, m, device, return_cls=False):
        seen.append([float(t) for t in thetas])
        return torch.zeros(len(thetas), D)

    def fake_cem(model, cfg, z_ctx, z_goal, horizon, u_max, rng, device, args,
                 past_actions=None):
        return np.full(horizon, plan_value, dtype=np.float32)

    monkeypatch.setattr(P, "encode_frames", fake_encode)
    monkeypatch.setattr(P, "cem_plan", fake_cem)
    monkeypatch.setattr(P, "_mom_type", lambda name: "lewm")
    monkeypatch.setattr(P, "_enc_window", lambda model: enc_window)
    monkeypatch.setattr(P, "_enc_shift", lambda model: enc_shift)
    return seen


def _w3_run(tmp_path, monkeypatch, receding, enc_window=3, enc_shift=1):
    """One dataset goal, driven through `run_episode` with a W=3 LeWM stub."""
    import h5py

    import plan_pendulum_swingup as P

    ds, man = _synthetic_dataset(tmp_path)
    a = _Args(dataset=str(ds), manifest=str(man))
    ep = P.dataset_goals(a)[0]
    with h5py.File(ds, "r") as f:
        off = int(f["ep_offset"][ep["ep"]])
        theta = np.asarray(f["state"][off: off + int(f["ep_len"][ep["ep"]]), 0])

    seen = _record_encoded_windows(monkeypatch, enc_window, enc_shift)
    args = _loop_args(goal_offset=a.goal_offset, horizon=a.goal_offset,
                      receding=receding, budget=a.goal_offset, replan=True,
                      action_block=1)
    P.run_episode(None, _cfg("lewm_ms_conv", history_size=3), ep,
                  planner_seed=0, args=args, device=torch.device("cpu"))
    return ep, theta, seen


def test_w3_goal_window_is_the_three_frames_centred_on_the_goal(tmp_path, monkeypatch):
    """b3s's goal latent must represent the goal FRAME, not the frame before it.

    A symmetric W=3 encoder emits a latent for the centre of its window, so the
    goal window is (gi-1, gi, gi+1) and `GOAL_LEAD = 1` exists to supply the last
    of those. Taking the causal tail instead would aim the whole plan one frame
    (0.1 s) early, which at swing-up speeds is a real angle and reads as model
    error. The start window is the counterpart: START_CTX frames ending AT the
    start frame, never past it.
    """
    ep, theta, seen = _w3_run(tmp_path, monkeypatch, receding=5)
    gi, si = ep["goal_idx"], ep["start_idx"]

    goal_win, first_ctx = seen[0], seen[1]          # goal is encoded before the loop
    assert goal_win == pytest.approx([theta[gi - 1], theta[gi], theta[gi + 1]]), \
        "goal window is not the three frames centred on the goal frame"
    assert first_ctx == pytest.approx(list(theta[si - 4: si + 1])), \
        "start context is not the 5 recorded frames ending at the start frame"


@pytest.mark.parametrize("receding", [2, 5, 10])
def test_w3_never_encodes_a_frame_it_has_not_reached(tmp_path, monkeypatch, receding):
    """The leakage guard, stated in frames rather than in prose.

    Exactly two parts of the recorded episode may be encoded: the start window,
    which says where the plan begins, and the goal window, which says what it aims
    at. Every frame strictly between them is the trajectory the planner is
    supposed to PREDICT, so if any of those recorded angles ever reaches the
    encoder the arm is reading its own answer. Under replanning the contexts are
    rebuilt from the plant's own executed angles, which differ from the recorded
    ones because the plan is not the recorded torque sequence, so a recorded
    in-between angle can only appear by being fetched deliberately.
    """
    ep, theta, seen = _w3_run(tmp_path, monkeypatch, receding=receding)
    gi, si = ep["goal_idx"], ep["start_idx"]

    allowed = set(np.concatenate([theta[si - 4: si + 1], theta[gi - 1: gi + 2]]))
    forbidden = set(theta[si + 1: gi - 1])
    for window in seen:
        for t in window:
            assert t not in forbidden or t in allowed, (
                f"a recorded frame from inside (start, goal) was encoded: {t}")


def test_wrong_clock_is_refused(tmp_path, monkeypatch):
    """The one runtime way the start->goal guarantee can break.

    Reachability is a property of the recorded episode, so it needs no per-run
    verification -- but only if the plant is stepped on the clock the episode was
    recorded on. `--dt-mult 1` against a dt_mult=2 dataset integrates at half the
    recorded rate, and every goal silently stops being reachable.
    """
    import subprocess

    ds, man = _synthetic_dataset(tmp_path)
    script = ROOT / "lehami" / "experiments" / "src" / "plan_pendulum_swingup.py"
    r = subprocess.run(
        [sys.executable, str(script), "--checkpoints", "f=/nope.ckpt",
         "--goal-mode", "dataset", "--dataset", str(ds), "--manifest", str(man),
         "--n-eval", "2", "--dt-mult", "1", "--device", "cpu"],
        capture_output=True, text=True)
    assert "dt_mult" in r.stdout + r.stderr and r.returncode != 0


def test_success_at_target_scores_only_the_goal_timestamp(tmp_path):
    """The fix for a SATURATED instrument, pinned.

    Under 'any' (LeWM's criterion, swm/world.py:961) the dataset task is solved by
    300 random torque sequences with exact dynamics -- one CEM iteration against a
    300x30 search -- so no world model can be distinguished from any other. A
    pendulum under random torque sweeps its reachable set, so "pass near this state
    at some point" is nearly free. Scoring at the goal's own timestamp restores the
    range (measured: u=0 floor 0.050, best-of-300-random 0.650).
    """
    from plan_pendulum_swingup import _success_steps

    a = _Args(dataset=None, manifest=None)
    assert _success_steps(a) == {a.goal_offset}
    a.success_at = "any"
    assert _success_steps(a) is None


def test_a_lucky_pass_before_the_deadline_does_not_count(tmp_path):
    """The behavioural difference, not just the index set.

    A trajectory that touches the goal early and leaves must FAIL under 'target'
    and PASS under 'any'. That gap is the whole reason the instrument was blind.
    """
    from plan_pendulum_swingup import _goal_success, _success_steps

    g = dict(theta=0.0, omega=0.0)
    tol = (0.1, 0.5)
    # step 3 is on the goal, step 10 (the deadline) is far away
    traj = {3: (0.0, 0.0), 10: (2.0, 3.0)}
    for mode, expected in (("any", True), ("target", False)):
        a = _Args(dataset=None, manifest=None, success_at=mode)
        ok_steps = _success_steps(a)
        hit = any((ok_steps is None or k in ok_steps)
                  and _goal_success(th, om, g, tol) for k, (th, om) in traj.items())
        assert hit is expected, f"{mode} gave {hit}"


# ── 6. temporal encoder (W > 1), LeWM only ───────────────────────────────────
# Added 2026-08-23 for EXP-076 b3s. At W = 1 every assertion below is the
# identity the earlier tests already pin; the point is that the SAME code now
# also matches training's `builders._action_emb` when latent time and pixel time
# come apart.


def test_w3_pairs_each_latent_with_its_own_instant_and_rolls_to_the_goal():
    """The W=3 contract, in latent time, spelled out slot by slot.

    A symmetric W=3 encoder has lag 1: the newest context latent represents one
    instant BEFORE the plant's current frame. Two consequences, both asserted:

      1. the executed prefix is h-1+enc_lag = 3, not h-1 = 2 -- one extra
         already-applied torque falls inside the context;
      2. the rollout runs H+enc_lag steps, so its terminal latent reaches the
         goal's instant. Rolling only H would leave the last torque unscored,
         i.e. the arm would optimise 29 torques and be graded on 30.

    Windows are u_all[t : t+h] with NO offset in the slice -- the lag lives in the
    prefix length, not in the indexing.
    """
    from plan_pendulum_swingup import latent_rollout

    H = 6
    u = torch.arange(1.0, H + 1.0).view(1, H, 1)              # candidates 1..6
    past = [-3.0, -2.0, -1.0]                                 # h-1+enc_lag = 3
    m = _LeWMStub(window=3)
    latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D), u,
                   torch.device("cpu"), past_actions=past)

    assert len(m.seen) == H + 1, "must roll H+enc_lag steps to reach the goal instant"

    full = past + [float(x) for x in u[0, :, 0]]
    for t, win in enumerate(m.seen):
        got = [round(v.item(), 5) for v in win[0, :, 0]]
        assert got == full[t : t + H_CTX], f"step {t}: {got}"

    # First window is all history; the last consumes the FINAL candidate, which is
    # the property that makes the cost account for every executed torque.
    assert [v.item() for v in m.seen[0][0, :, 0]] == [-3.0, -2.0, -1.0]
    assert m.seen[-1][0, -1, 0].item() == float(H)


def test_w3_rolling_only_h_would_drop_the_last_torque():
    """The defect the H+enc_lag rollout fixes, stated as its own assertion.

    Guards against someone "simplifying" n_steps back to u.shape[1]: at enc_lag=1
    the final candidate would never appear in any action window, so changing it
    could not change the cost -- a silently unoptimisable control input.
    """
    from plan_pendulum_swingup import latent_rollout

    H = 6
    u = torch.arange(1.0, H + 1.0).view(1, H, 1)
    m = _LeWMStub(window=3)
    latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D), u,
                   torch.device("cpu"), past_actions=[-3.0, -2.0, -1.0])

    seen_vals = {v.item() for win in m.seen for v in win[0, :, 0]}
    assert float(H) in seen_vals, "final candidate never reached the predictor"
    # and it only appears in the step that the +enc_lag buys
    assert float(H) not in {v.item() for win in m.seen[:H] for v in win[0, :, 0]}


def test_w1_is_the_zero_shift_special_case_of_w3_code():
    """The W>1 path must not perturb W=1 -- the whole existing result set is W=1.

    Same inputs, one stub per-frame and one built with window=1 explicitly: the
    windows handed to `predict` must be bit-identical.
    """
    from plan_pendulum_swingup import latent_rollout

    u = torch.arange(1.0, 5.0).view(1, 4, 1)
    past = [-2.0, -1.0]
    seen = []
    for w in (1, 1):
        m = _LeWMStub(window=w)
        latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D), u,
                       torch.device("cpu"), past_actions=past)
        seen.append([[v.item() for v in win[0, :, 0]] for win in m.seen])
    assert seen[0] == seen[1] == [[-2.0, -1.0, 1.0], [-1.0, 1.0, 2.0],
                                  [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]


def test_w3_rejects_a_w1_length_history():
    """h-1+enc_lag, not h-1: a W=1-length history at W=3 is one torque short.

    Silently accepting it would left-pad the window with the wrong torques and
    return a number, which is the failure mode this whole file exists to prevent.
    """
    from plan_pendulum_swingup import latent_rollout

    m = _LeWMStub(window=3)
    with pytest.raises(ValueError, match=r"h-1\+enc_lag=3"):
        latent_rollout(m, _cfg("lewm_ms_conv"), torch.zeros(1, H_CTX, D),
                       torch.zeros(1, 4, 1), torch.device("cpu"),
                       past_actions=[-1.0, -2.0])


def test_lehami_is_still_refused_at_w3():
    """The guard was relaxed for LeWM ONLY; `_anchor_offset` still ignores W."""
    from plan_pendulum_swingup import _check_plannable

    cfg = _cfg("lehami_sg_sep_sreg_joint_ms_conv")
    cfg.enc_window = 3
    with pytest.raises(NotImplementedError, match="LeWM family only"):
        _check_plannable(cfg, _Args(dataset=None, manifest=None), "a3s_s0")


def test_goal_and_start_windows_have_separate_requirements():
    """The two windows are sized by different quantities, so they guard separately.

    A goal is ONE encoded state (W frames); a start is a CONTEXT of history_size
    of them (history_size + W - 1 frames). b3s is the case that separates them:
    5 frames at the start, 3 at the goal. Conflating the two is what made a W=3
    arm look like it needed a wider GOAL window -- and widening GOAL_CTX is free
    while widening START_CTX re-draws every goal, so the distinction is not
    cosmetic.
    """
    from plan_pendulum_swingup import GOAL_CTX, START_CTX, _check_plannable

    assert START_CTX >= GOAL_CTX, "the start context can never be the narrower one"

    # b3s as shipped: fits both, so neither guard fires (it gets past them and on
    # to the u_max/dataset checks, which need a real cfg -- hence the narrow match).
    cfg = _cfg("lewm_ms_conv", history_size=3)
    cfg.enc_window = 3
    assert 3 <= GOAL_CTX and 3 + 3 - 1 <= START_CTX

    # Outgrowing the START window only: history_size 3 + W 5 - 1 = 7 > 5, while
    # the goal still needs 5 <= ... only if GOAL_CTX allows, so use a case where
    # the goal fits and the start does not.
    cfg = _cfg("lewm_ms_conv", history_size=3)
    cfg.enc_window = GOAL_CTX          # goal needs exactly GOAL_CTX -> fits
    with pytest.raises(ValueError, match="RE-DRAWS EVERY GOAL"):
        _check_plannable(cfg, _Args(dataset=None, manifest=None), "wide_start")

    # Outgrowing the GOAL window: W > GOAL_CTX fires the cheap guard first.
    cfg = _cfg("lewm_ms_conv", history_size=1)
    cfg.enc_window = GOAL_CTX + 1
    with pytest.raises(ValueError, match="to encode ONE goal"):
        _check_plannable(cfg, _Args(dataset=None, manifest=None), "wide_goal")


# ── 7. run_episode's solve/execute loop ───────────────────────────────────────
#
# Ported from `test_three_body_planning.py`, which had these and the pendulum did
# not -- the asymmetry is why two loop bugs shipped here and neither shipped there:
#
#   * EXP-085: a FIXED horizon under a receding execution. Each solve compared its
#     rollout endpoint (frame `done + horizon`) to a goal sitting at frame
#     `goal_offset`, so the planner was scored on arriving LATE and the goal stayed
#     perpetually H steps away. Predicted-at-goal 12/12, actual 0/12.
#   * The `--replan-aim subgoal` hang (mode removed 2026-09-01): `stop` was
#     `--budget` while the aim point capped `h_now` at `goal_offset - done`, so at
#     the deadline `h_now` hit 0, `cem_plan` returned an EMPTY plan, `done` stopped
#     incrementing and the `while` never exited.
#
# Both are invisible to any metric -- the planner reports plausible numbers either
# way -- so they are pinned structurally: what horizon each solve asked for, what
# it aimed at, and how far the plant actually advanced.
#
# These drive the REAL `run_episode`: the loop, the plant (true `plant_step`), the
# context rebuild and the scoring. Only the model-touching calls are stubbed.


def _loop_args(**kw):
    """The argparse namespace `run_episode` reads, with the defaults under test."""
    from types import SimpleNamespace
    base = dict(replan=False, horizon=30, receding=30, goal_offset=30, budget=30,
                u_max=4.0, dt_mult=1.0, success_at="target",
                theta_tol=0.1, omega_tol=0.5, cost="terminal", p_weight=None,
                action_block=3, cem_samples=8, cem_iters=2, cem_elites=2)
    base.update(kw)
    return SimpleNamespace(**base)


def _loop_episode(goal_offset):
    """One `dataset_goals`-shaped entry: a start window, a goal, recorded torques."""
    from plan_pendulum_swingup import GOAL_CTX, GOAL_LEAD, START_CTX
    rng = np.random.default_rng(11)
    theta0, omega0 = 0.3, -0.2
    # The start window must END at theta0 -- run_episode asserts it.
    start = [theta0 - 0.05 * (START_CTX - 1 - i) for i in range(START_CTX)]
    return dict(
        L=1.0, m=1.0, theta0=theta0, omega0=omega0,
        # An unreachable goal, deliberately: these tests pin the SCHEDULE, and an
        # episode that succeeded early would break the loop and truncate it.
        goal=dict(theta=float(np.pi), omega=9.0,
                  thetas=[float(a) for a in rng.normal(size=GOAL_CTX + GOAL_LEAD)]),
        start_thetas=start,
        start_actions=[0.0] * (START_CTX - 1),
        replay=[0.0] * goal_offset, ep=0, start_idx=0, goal_idx=goal_offset)


def _instrument_loop(monkeypatch, plan_value=0.5, enc_window=1, enc_shift=0):
    """Replace the model-touching calls; record what each solve asked for.

    `enc_window` / `enc_shift` are the encoder's, asked for exactly as the planner
    asks (`_enc_window` / `_enc_shift`) rather than derived from W: `diff` (W=2)
    has shift 1 and therefore lag 0, so the two are independent.
    """
    import plan_pendulum_swingup as P
    log = {"h_now": [], "ctx_len": [], "n_past": [], "goal_calls": 0}

    def fake_encode(model, cfg, thetas, L, m, device, return_cls=False):
        log["ctx_len"].append(len(thetas))
        return torch.zeros(len(thetas), D)

    def fake_cem(model, cfg, z_ctx, z_goal, horizon, u_max, rng, device, args,
                 past_actions=None):
        log["h_now"].append(horizon)
        log["n_past"].append(None if past_actions is None else len(past_actions))
        return np.full(horizon, plan_value, dtype=np.float32)

    monkeypatch.setattr(P, "encode_frames", fake_encode)
    monkeypatch.setattr(P, "cem_plan", fake_cem)
    monkeypatch.setattr(P, "_mom_type", lambda name: "lewm")
    monkeypatch.setattr(P, "_enc_window", lambda model: enc_window)
    monkeypatch.setattr(P, "_enc_shift", lambda model: enc_shift)
    return log


def _run_loop(monkeypatch, goal_offset=30, enc_window=1, enc_shift=0,
              history_size=1, **kw):
    import plan_pendulum_swingup as P
    log = _instrument_loop(monkeypatch, enc_window=enc_window, enc_shift=enc_shift)
    ep = _loop_episode(goal_offset)
    args = _loop_args(goal_offset=goal_offset, **kw)
    res = P.run_episode(None, _cfg("lewm_ms_conv", history_size=history_size), ep,
                        planner_seed=0, args=args, device=torch.device("cpu"))
    # The goal latent is built by the same stubbed encoder, before the loop runs.
    log["goal_calls"] = 1
    return log, res


def test_run_episode_default_is_one_solve_over_the_whole_task(monkeypatch):
    """Without --replan: one solve at the full horizon. The historical path."""
    K = 30
    log, res = _run_loop(monkeypatch, goal_offset=K, replan=False,
                         horizon=K, receding=K, budget=K)
    assert res["n_solves"] == 1
    assert log["h_now"] == [K], "single-shot must solve the whole task at once"
    assert res["steps_done"] == K


@pytest.mark.parametrize("receding", [1, 3, 9, 15, 30])
def test_run_episode_replan_shrinks_the_horizon_to_the_goal_instant(monkeypatch,
                                                                    receding):
    """--replan: every solve's rollout must END on the goal instant.

    This is the EXP-085 regression guard. `h_now` must equal the frames REMAINING
    (`goal_offset - done`), never the fixed `--horizon`: solve i starts at frame
    i*receding, so a rollout of `goal_offset - i*receding` lands exactly on the
    goal frame. A fixed horizon would put every endpoint past it.
    """
    K = 30
    log, res = _run_loop(monkeypatch, goal_offset=K, replan=True,
                         horizon=K, receding=receding, budget=K)
    n = res["n_solves"]
    assert n == -(-K // receding), "one solve per receding segment"
    expect = [K - i * receding for i in range(n)]
    assert log["h_now"] == expect, (
        f"horizon must shrink {expect}, got {log['h_now']}")
    # Every solve's endpoint is the SAME absolute frame: start + h_now == K.
    ends = [i * receding + h for i, h in enumerate(log["h_now"])]
    assert ends == [K] * n, f"aim point drifted off the goal instant: {ends}"


@pytest.mark.parametrize("receding", [1, 4, 7, 9, 30])
def test_run_episode_executes_exactly_the_task_length(monkeypatch, receding):
    """However the segments fall, the plant advances exactly `goal_offset` steps.

    Overshooting would score a state past the goal instant, undershooting one short
    of it; both read as model quality. `receding` deliberately does NOT divide K
    here -- the last segment must be truncated by the shrinking horizon, not by the
    budget. A schedule that stalled (the empty-plan hang) fails this as a timeout.
    """
    K = 30
    log, res = _run_loop(monkeypatch, goal_offset=K, replan=True,
                         horizon=K, receding=receding, budget=K)
    assert res["steps_done"] == K, "the plant must land ON the goal instant"
    assert sum(min(receding, h) for h in log["h_now"]) == K
    # success_target is only meaningful if the episode reached the deadline.
    assert not np.isnan(res["success_target"])


def test_run_episode_context_width_is_constant_across_solves(monkeypatch):
    """Every solve re-encodes the same number of frames.

    A context that grew or shrank with `done` would feed the encoder a window it
    was never trained on, and for LeHami would change the momentum stencil's span
    mid-episode -- the failure mode `anchor_from_angles` exists to prevent.
    """
    K = 30
    log, _ = _run_loop(monkeypatch, goal_offset=K, replan=True,
                       horizon=K, receding=9, budget=K)
    widths = set(log["ctx_len"])
    assert len(widths) == 1, f"context width drifted across solves: {log['ctx_len']}"


# ── the W=3 arm INSIDE the replanning loop (EXP-076 b3s) ──────────────────────
# The W>1 code in `latent_rollout` and `_goal_latent` is covered above, but every
# loop test monkeypatches `_enc_window` to 1, so without these the combination
# "lagged encoder + shrinking horizon" would have NO coverage --
# the same gap that hid two bugs in this loop before. These pin the two quantities
# a lagged encoder changes per solve, both of which produce plausible numbers when
# wrong: the width of the re-encoded context, and how many already-applied torques
# are handed to the predictor.
#
# b3s: history_size 3, TemporalConvEncoder W=3 full -> shift 1, so lag 1. It sits
# EXACTLY at both budgets (h_pix = 3+3-1 = 5 = START_CTX, lag 1 = GOAL_LEAD).


@pytest.mark.parametrize("receding", [9, 15, 30])
def test_run_episode_w3_reencodes_history_plus_window_minus_one_frames(monkeypatch,
                                                                       receding):
    """Each solve re-encodes h_ctx + W - 1 PIXEL frames, constant across solves.

    A window-W encoder eats W-1 lead-in frames to emit its first latent, so a
    3-latent context needs 5 frames. Encoding 3 would silently hand the predictor
    a 1-latent context: no crash, a plausible plan, a wrong arm.
    """
    K = 30
    log, res = _run_loop(monkeypatch, goal_offset=K, replan=True, horizon=K,
                         receding=receding, budget=K,
                         enc_window=3, enc_shift=1, history_size=3)
    # The first encode is the GOAL latent, built before the loop from the W frames
    # CENTRED on the goal frame; every later one is a re-encoded context.
    goal_w, ctx_w = log["ctx_len"][0], set(log["ctx_len"][1:])
    assert goal_w == 3, f"goal window was {goal_w} frames, expected W=3"
    assert ctx_w == {5}, f"context widths seen: {sorted(ctx_w)}"
    assert res["steps_done"] == K, "the lag must not change where the episode ends"
    assert log["h_now"] == [K - i * receding for i in range(res["n_solves"])]


def test_run_episode_w3_hands_the_predictor_h_minus_one_plus_lag_torques(monkeypatch):
    """`past_actions` is h_ctx - 1 + enc_lag long, not h_ctx - 1.

    A lagged encoder's newest context latent sits `enc_lag` instants behind the
    plant, so that many extra already-applied torques fall inside the context
    window. `latent_rollout` raises on a wrong length, which is what makes this
    worth pinning here rather than only at W=1: at lag 0 the two expressions
    coincide and the bug is invisible.
    """
    K = 30
    log, _ = _run_loop(monkeypatch, goal_offset=K, replan=True, horizon=K,
                       receding=9, budget=K,
                       enc_window=3, enc_shift=1, history_size=3)
    assert set(log["n_past"]) == {3}, f"past_actions lengths: {sorted(set(log['n_past']))}"


def test_run_episode_w1_is_the_lag_zero_case_of_the_same_loop(monkeypatch):
    """The W=3 path must reduce to the W=1 one, not merely resemble it.

    Same schedule, goal window and context narrower by W-1, one fewer past torque.
    If these two
    ever diverge in anything but those two numbers, the lagged arm is running a
    different protocol and the b0s/b3s pairing is not a pairing.
    """
    K = 30
    kw = dict(goal_offset=K, replan=True, horizon=K, receding=9, budget=K)
    w1, r1 = _run_loop(monkeypatch, history_size=3, **kw)
    w3, r3 = _run_loop(monkeypatch, enc_window=3, enc_shift=1, history_size=3, **kw)
    assert w1["h_now"] == w3["h_now"]
    assert r1["steps_done"] == r3["steps_done"] == K
    assert (w1["ctx_len"][0], set(w1["ctx_len"][1:])) == (1, {3})
    assert (w3["ctx_len"][0], set(w3["ctx_len"][1:])) == (3, {5})
    assert set(w1["n_past"]) == {2} and set(w3["n_past"]) == {3}


def test_run_episode_replan_stops_at_the_deadline_not_the_budget(monkeypatch):
    """`stop` is `goal_offset` under --replan, so a longer budget adds no steps.

    Under a shrinking horizon there is nothing left to solve once the deadline is
    reached: `h_now` would be 0 and `cem_plan` would return an empty plan, which is
    exactly the hang the removed subgoal mode had. The loop must exit on the
    deadline instead.
    """
    K = 30
    log, res = _run_loop(monkeypatch, goal_offset=K, replan=True,
                         horizon=K, receding=9, budget=3 * K)
    assert res["steps_done"] == K, "a bigger budget must not extend the episode"
    assert all(h > 0 for h in log["h_now"]), "a zero horizon means an empty plan"


# ── 6. the CEM rollout must end ON the goal instant, not past it ──────────────


def _cem_steps(monkeypatch, horizon, action_block):
    """Run one `cem_plan` with the rollout stubbed, returning what it was scored on.

    Only the step COUNT matters here, so the stub returns a constant latent and a
    cost that ranks candidates by their first block -- enough for the elite refit
    to run for real.
    """
    from types import SimpleNamespace

    import plan_pendulum_swingup as P

    seen = []

    def fake_rollout(model, cfg, ctx, steps, device, return_all=False,
                     past_actions=None, flow_h=None):
        seen.append(np.asarray(steps).shape)
        return torch.zeros(len(steps), D)

    def fake_cost(z_end, z_goal, is_lehami, p_weight):
        return torch.arange(len(z_end), dtype=torch.float32)

    monkeypatch.setattr(P, "latent_rollout", fake_rollout)
    monkeypatch.setattr(P, "latent_cost", fake_cost)
    args = _loop_args(action_block=action_block, cem_samples=8, cem_iters=2,
                      cem_elites=2)
    plan = P.cem_plan(None, _cfg("lewm_ms_conv"), torch.zeros(1, D), torch.zeros(1, D),
                      horizon, args.u_max, np.random.default_rng(0),
                      torch.device("cpu"), args)
    return plan, seen


@pytest.mark.parametrize("horizon", [30, 24, 18, 12, 6, 3])
def test_cem_block_multiple_horizons_are_unchanged(horizon, monkeypatch):
    """The K=30 ladder's rungs are all whole blocks: the fix must not touch them."""
    plan, seen = _cem_steps(monkeypatch, horizon, action_block=3)
    assert {s[1] for s in seen} == {horizon}
    assert len(plan) == horizon


@pytest.mark.parametrize("horizon", [26, 22, 14, 10, 2, 25, 5])
def test_cem_scores_exactly_the_horizon_it_returns(horizon, monkeypatch):
    """A horizon that is NOT a whole number of blocks must still be scored at h.

    Under `--replan` the horizon shrinks toward a fixed goal instant, so whether it
    is a multiple of `action_block` is an accident of the rung: R=9 and R=6 give
    only multiples, R=4 and R=5 do not. `cem_plan` used to roll
    `ceil(h/blk)*blk` steps and read the terminal cost there -- 1-2 steps PAST the
    goal -- while returning a plan truncated to h, so the cost and the execution
    disagreed and the planner was optimised for arriving late. At these speeds one
    step is ~0.1 s, i.e. a real angle against a 0.1 rad tolerance.
    """
    plan, seen = _cem_steps(monkeypatch, horizon, action_block=3)
    assert {s[1] for s in seen} == {horizon}, "scored a rounded-up rollout"
    assert len(plan) == horizon, "returned plan and scored rollout must agree"


def test_cem_still_holds_each_torque_for_a_whole_block(monkeypatch):
    """Truncating the tail must not break the frameskip inside the horizon.

    The search variable stays `ceil(h/blk)` torques, each repeated `blk` times; only
    the LAST block is cut short. If the slice ever turned into a per-step search the
    action space would silently become 3x larger than CEMSolver's.
    """
    from types import SimpleNamespace  # noqa: F401

    import plan_pendulum_swingup as P

    seen_steps = []

    def fake_rollout(model, cfg, ctx, steps, device, return_all=False,
                     past_actions=None, flow_h=None):
        seen_steps.append(np.asarray(steps).copy())
        return torch.zeros(len(steps), D)

    monkeypatch.setattr(P, "latent_rollout", fake_rollout)
    monkeypatch.setattr(P, "latent_cost",
                        lambda z, g, h, w: torch.arange(len(z), dtype=torch.float32))
    args = _loop_args(action_block=3, cem_samples=8, cem_iters=1, cem_elites=2)
    P.cem_plan(None, _cfg("lewm_ms_conv"), torch.zeros(1, D), torch.zeros(1, D),
               26, args.u_max, np.random.default_rng(0), torch.device("cpu"), args)
    s = seen_steps[0]
    assert s.shape[1] == 26
    for b in range(8):                      # 8 whole blocks, then a 2-step remainder
        assert np.allclose(s[:, 3 * b:3 * b + 3], s[:, 3 * b:3 * b + 1])
    assert np.allclose(s[:, 24:26], s[:, 24:25])


# ── 5. the encoder-window pre-flight (_check_plannable) ──────────────────────
#
# These guards were DEAD from the day they were written: they read `enc_window`
# at the top level of the saved config, but train.py:290 writes the whole nested
# Hydra cfg and every experiment sets the key under `model:`, so `w` was 1 for
# every arm and none of the four checks could fire.  The live paths read the
# encoder itself and were always right, so no reported number moved -- but the
# LeHami refusal, the GOAL_CTX/GOAL_LEAD checks and the START_CTX check were all
# silently off.  These tests exist so that cannot happen again.


def _plan_cfg(name, enc_window, enc_variant="full", enc_lag=None, history_size=H_CTX):
    """A saved config shaped exactly as train.py writes it: enc_* under `model`."""
    model = {"name": name, "latent_dim": D, "fd_dt": 1.0,
             "enc_window": enc_window, "enc_variant": enc_variant}
    if enc_lag is not None:
        model["enc_lag"] = enc_lag
    return OmegaConf.create({"model": model, "embed_dim": D,
                             "history_size": history_size,
                             "data": {"name": "pendulum_swingup"}})


def _plan_args():
    from types import SimpleNamespace
    return SimpleNamespace(goal_mode="upright", dataset="pendulum_swingup.h5",
                           u_max=4.0)


@pytest.mark.parametrize("window", [2, 3])
def test_check_plannable_reads_the_window_from_the_model_block(window):
    """The regression guard: a W>1 LeHami arm must be REFUSED, not waved through."""
    import plan_pendulum_swingup as P

    cfg = _plan_cfg("lehami_sg_sep_sreg_q_ms_conv", window,
                    enc_variant="diff" if window == 2 else "full")
    with pytest.raises(NotImplementedError, match="LeWM family only"):
        P._check_plannable(cfg, _plan_args(), "a3s")


@pytest.mark.parametrize("window,variant,lag", [(2, "diff", None), (3, "full", None),
                                                (3, "free", 1), (3, "free", 0)])
def test_check_plannable_admits_every_supported_lewm_arm(window, variant, lag):
    """b2s, b3s and both lag conventions of b3f fit the fixed goal draw."""
    import plan_pendulum_swingup as P

    P._check_plannable(_plan_cfg("lewm_ms_conv", window, variant, lag),
                       _plan_args(), "b")


def test_check_plannable_refuses_a_window_that_outgrows_start_ctx():
    """W=4 needs history_size + W - 1 = 6 > START_CTX = 5.

    Admitting it would re-draw every goal (`lo = K + START_CTX - 1`) and un-pair
    every McNemar test, so this must fail loudly rather than be widened.
    """
    import plan_pendulum_swingup as P

    with pytest.raises(ValueError, match="START_CTX"):
        P._check_plannable(_plan_cfg("lewm_ms_conv", 4, "free", 0), _plan_args(), "b4")


def test_check_plannable_uses_the_declared_lag_not_half_the_window():
    """A `free` kernel's centroid is learned, so its lag is declared, not derived.

    At lag 2 the goal latent would need 2 frames after the goal frame but
    GOAL_LEAD is 1, so the last torques would go unscored.  Deriving the lag from
    W (= 2 for W=5) would reject it for the same reason by accident; the point is
    that a W=3 arm declaring lag 2 must ALSO be rejected.
    """
    import plan_pendulum_swingup as P

    with pytest.raises(ValueError, match="GOAL_LEAD"):
        P._check_plannable(_plan_cfg("lewm_ms_conv", 3, "free", 2), _plan_args(), "b3f2")
