#!/usr/bin/env python
"""Collect every input the paper's figures and tables need into one release folder.

    uv run python lehami/experiments/src/export_paper_data.py [--skip-models]

Writes paper_figures/data/, which together with paper_figures/paper_figures.ipynb
reproduces every figure and table in the paper.
Three kinds of input:

  1. COPIES of result files (probe/rollout CSVs, planning episode CSVs, FLOP counts).
  2. TRIMMED arrays from larger dumps: only the keys a figure or table reads, e.g.
     the k=0 test features of probe_features.npz rather than the whole file.
  3. MODEL-DERIVED arrays (latents, learned H on a grid) and dataset FRAMES, which need
     the checkpoints and the .h5 datasets. These are the only steps that load a model;
     --skip-models reuses whatever is already in paper_figures/data.

The notebook reads paper_figures/data only, so a reader needs neither the checkpoints, the
datasets nor the cluster results to rebuild the paper's figures.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "lehami" / "experiments"
RES = EXP / "results"
OUT = ROOT / "paper_figures" / "data"
DATA = ROOT / "lehami" / "data"
SWM = Path.home() / ".stable_worldmodel"
CKPT = SWM / "checkpoints"

CONS, ACT, DSET = OUT / "conservative_pendulum", OUT / "actuated_pendulum", OUT / "datasets"
SEEDS = (0, 1, 2)


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ── 1. copies ────────────────────────────────────────────────────────────────────
def copy_conservative():
    # rollout MAE curves, 3 seeds per arm (Fig. cons-physics, Tab. cons-knnk1)
    for stem in ("exp067_a0m_lehami_w1_multi", "exp068_b0m_lewm_w1_multi"):
        for i, suf in enumerate(("", "_s1", "_s2")):
            _copy(RES / f"{stem}{suf}" / "probe_metrics.csv",
                  CONS / "probe_metrics" / f"{stem}_s{i}.csv")
    # latent R^2 of the q and p blocks (Tab. cons-latent-r2); dt sweep and reversal
    for f in ("rollout_metrics.csv", "physics_dt.csv", "physics_reversal.csv"):
        _copy(RES / "exp067_a0m_lehami_w1_multi" / f, CONS / "exp067_a0m" / f)
    # Jacobian log-volume (Fig. cons-physics right, Fig./Tab. jacobian-volume)
    for f in ("jacobian_k0.csv", "jacobian_rollout.csv", "epsball.csv"):
        _copy(RES / "latent_volume" / f, CONS / "latent_volume" / f)
    _copy(RES / "latent_volume" / "jacobian_rollout_actuated.csv",
          ACT / "latent_volume" / "jacobian_rollout_actuated.csv")
    # frozen DINOv2 encoder, matched epoch 60 (Fig. dino-powered)
    for stem in ("exp098_a0d_ep059", "exp098_b0d_ep059"):
        _copy(RES / stem / "probe_metrics.csv", CONS / "dino" / f"{stem}.csv")
    # training compute (Tab. exp096-isocompute, Tab. multi-params)
    _copy(ROOT / "outputs/exp096/torch_l40s/training_compute_l40s.csv",
          CONS / "exp096" / "training_compute_l40s.csv")
    _copy(ROOT / "outputs/exp096/wandb_loss_history.csv",
          CONS / "exp096" / "wandb_loss_history.csv")
    rows = []
    for arm in ("a0m", "b0m", "b1m"):
        for ep in range(9, 100, 10):
            d = pd.read_csv(RES / f"exp096_{arm}_ep{ep:03d}" / "probe_metrics.csv")
            rows.append(d.assign(arm=arm, epoch=ep))
    pd.concat(rows).to_csv(CONS / "exp096" / "probe_metrics_per_epoch.csv", index=False)


def copy_actuated():
    for stem in ("exp079_a0q_lehami", "exp072_b0s_lewm", "exp076_b3s_lewm"):
        for s in SEEDS:
            _copy(RES / f"{stem}_s{s}" / "probe_metrics.csv",
                  ACT / "probe_metrics" / f"{stem}_s{s}.csv")
    # ridge / 25-NN / MLP at the anchor (Tab. physics_prediction_anchor, Tab. capacity)
    _copy(RES / "capacity_probes.csv", ACT / "capacity_probes.csv")
    # planning ladder: every rung x arm x model seed at CEM 300x30 (Tab. swingup-*)
    run = ROOT / "outputs" / "plan_replan_swingup" / "test"
    cells = ("cell1_single_shot_H30_R30", "cell2_goal_H30_R15", "cell3_goal_H30_R9",
             "cell6_goal_H30_R6", "cell7_goal_H30_R3")
    for c in cells:
        for d in sorted((run / c).glob("*_s[0-9]")):
            if d.name.split("_s")[0] not in ("a0q", "b0s", "b3s", "b3f"):
                continue
            for f in ("swingup_episodes.csv", "meta.json"):
                _copy(d / f, ACT / "planning" / c / d.name / f)
    for f in sorted((ROOT / "outputs" / "planning_compute").glob("*.csv")):
        _copy(f, ACT / "planning_compute" / f.name)


def copy_manifests():
    for name in ("split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json",
                 "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json",
                 "split_pendulum_episode_srnn_K100_seed42.json"):
        m = json.loads((DATA / name).read_text())
        m.pop("dataset_path", None)            # an absolute path on the author's machine
        (DSET / "manifests").mkdir(parents=True, exist_ok=True)
        (DSET / "manifests" / name).write_text(json.dumps(m, indent=1))


# ── 2. trimmed arrays ────────────────────────────────────────────────────────────
def _n_test(run_dir, split="test"):
    man = json.loads((run_dir / "summary.json").read_text())["manifest"]
    return int(json.loads((DATA / Path(man).name).read_text())[f"n_{split}_episodes"])


def trim_actuated():
    # Anchor features of the frozen encoder, test split (Fig. canonicality, Tab. sign-blind).
    # Tab. sign-blind fits its kNN on a 12k-row subsample of the TRAIN anchors drawn with
    # rng(0); exactly those rows are kept, so the table is reproduced bit for bit.
    for stem in ("exp079_a0q_lehami", "exp072_b0s_lewm", "exp076_b3s_lewm"):
        for s in SEEDS:
            d = RES / f"{stem}_s{s}"
            z = np.load(d / "probe_features.npz")
            idx = np.random.default_rng(0).choice(len(z["X_train_k0"]),
                                                  min(12000, len(z["X_train_k0"])),
                                                  replace=False)
            _savez(ACT / "probe_features_k0" / f"{stem}_s{s}.npz",
                   X_test_k0=z["X_test_k0"], Y_test_k0=z["Y_test_k0"],
                   X_train_k0_sub12k=z["X_train_k0"][idx], Y_train_k0_sub12k=z["Y_train_k0"][idx],
                   y_cols=z["y_cols"], uses_p=z["uses_p"], latent_dim=z["latent_dim"],
                   n_test_episodes=_n_test(d))
    # Encoder latents at every rollout step, LeHami arms (Tab. episode-variance)
    for stem in ("exp071_a0s_lehami", "exp079_a0q_lehami", "exp080_a0p_lehami",
                 "exp075_a3s_lehami", "exp081_a3q_lehami", "exp082_a3p_lehami"):
        for s in SEEDS:
            z = np.load(RES / f"{stem}_s{s}" / "rollout_latents.npz")
            ks = np.array([int(k) for k in z["ks"]])
            _savez(ACT / "encoder_latents" / f"{stem}_s{s}.npz", ks=ks,
                   gt_q=np.stack([z[f"gt_q_k{k}"] for k in ks]),
                   gt_p=np.stack([z[f"gt_p_k{k}"] for k in ks]))
    # Rod length and mass (Tab. param-conservation): train anchors to fit the probes,
    # the model's own predicted latents at k=0 and k=76 to apply them.
    import h5py
    import hdf5plugin  # noqa: F401
    man = json.loads((DATA / "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json")
                     .read_text())
    with h5py.File(SWM / "pendulum_swingup.h5", "r") as f:
        _savez(ACT / "episode_parameters.npz", mass=f["mass"][:], length=f["length"][:],
               train_episode_indices=np.array(man["train_episode_indices"]),
               test_episode_indices=np.array(man["test_episode_indices"]))
    for stem in ("exp079_a0q_lehami", "exp072_b0s_lewm"):
        for s in SEEDS:
            d = RES / f"{stem}_s{s}"
            z = np.load(d / "probe_features.npz")
            arr = dict(X_train_k0=z["X_train_k0"], L_train_k0=z["Y_train_k0"][:, 3])
            if (d / "rollout_latents.npz").exists():
                r = np.load(d / "rollout_latents.npz")
                for k in (0, 76):
                    q = r[f"pred_q_k{k}"]
                    p = r[f"pred_p_k{k}"] if f"pred_p_k{k}" in r else np.zeros((len(q), 0), q.dtype)
                    arr[f"pred_k{k}"] = np.concatenate([q, p], axis=1)
            _savez(ACT / "param_probe" / f"{stem}_s{s}.npz", **arr)


# ── 3. model-derived arrays and dataset frames ───────────────────────────────────
def _models():
    sys.path.insert(0, str(EXP / "src"))
    import torch
    import run_eval as RE
    from plot_latent_phase_portrait import encode_episode_qp
    return torch, RE, encode_episode_qp


def export_conservative_latents():
    """EXP-027 (LeHami) and EXP-030 (LeWM), single-rod pendulum, episodes 0..59.

    The same episodes as the manuscript's figures (te_w3_multi_MAIN cells 39, 47, 50, 53,
    55, 59 all loop over `range(60)`), so paper_figures.ipynb redraws those figures
    exactly. Encoded frames (the encoder's geometry) and each model's own rollout from its
    anchor to the end of the episode.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    torch, RE, encode_episode_qp = _models()
    eps = list(range(60))
    # Devices as in te_w3_multi_MAIN: LeHami's encoding and rollout (cells 38, 39, 47) and
    # LeWM's rollout (cell 55) ran on the GPU, LeWM's encoded-frame pool for the energy
    # readout (cell 53) on the CPU. Long autoregressive rollouts amplify float differences
    # between devices, so matching them is what makes the figures identical.
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # LeHami, EXP-027
    model, cfg = RE.load_model(CKPT / "lehami_sg_sep_sreg_joint_ms_pendulum_sg_sep_sreg_joint_ms_d32_exp027"
                               / "best.ckpt", dev)
    h5 = RE._h5_path(cfg)
    enc = {k: [] for k in ("q", "p", "state", "ep")}
    for ep in eps:
        q, p, st = encode_episode_qp(model, cfg, h5, ep, dev)
        enc["q"].append(q); enc["p"].append(p); enc["state"].append(st)
        enc["ep"].append(np.full(len(q), ep))
    mom = RE._mom_type(cfg.model.name)
    d, a = RE._latent_dim(cfg), RE._anchor_frame(cfg, mom)
    u0 = RE._zero_action_emb(model, cfg, dev)
    roll = {k: [] for k in ("q", "p", "H", "k", "ep", "state")}
    with h5py.File(h5, "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        for ep in eps:
            s0, L = int(off[ep]), int(ln[ep])
            px = RE._preprocess_batch(f["pixels"][s0:s0 + L], cfg.img_size, dev)
            st = RE._align_states(model, f["state"][s0:s0 + L])
            with torch.no_grad():
                cls, q_enc = RE._encode_episode(model, px, 50)
                q0, p0 = RE._build_anchor(model, cls, q_enc, mom, cfg, dev)
                qp = torch.cat([q0, p0], -1).unsqueeze(0)
                for k in range((L - RE._enc_window(model) + 1) - a):
                    if k > 0:
                        qp = model.predictor(qp.unsqueeze(1), u0.unsqueeze(1))[:, 0]
                    qk, pk = qp[0, :d], qp[0, d:]
                    H = model.predictor.flow.eval_H(qk[None], pk[None], u0).item()
                    roll["q"].append(qk.cpu().numpy()); roll["p"].append(pk.cpu().numpy())
                    roll["H"].append(H); roll["k"].append(k); roll["ep"].append(ep)
                    roll["state"].append(st[a + k])

    # LeWM, EXP-030
    lm, lc = RE.load_model(CKPT / "lewm_ms_pendulum_ms_srnn_k100_exp030" / "best.ckpt", dev)
    lm_cpu, _ = RE.load_model(CKPT / "lewm_ms_pendulum_ms_srnn_k100_exp030" / "best.ckpt", "cpu")
    h = lc.history_size
    lat2pix = RE._latent_to_pixel_offset(lm)
    ubase = lm.action_encoder(torch.zeros(1, h, lc.data.action_dim, device=dev))
    lenc = {k: [] for k in ("z", "state", "ep")}
    lrol = {k: [] for k in ("z", "k", "ep", "state")}
    with h5py.File(RE._h5_path(lc), "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        for ep in eps:
            s0, L = int(off[ep]), int(ln[ep])
            px = RE._preprocess_batch(f["pixels"][s0:s0 + L], lc.img_size, dev)
            st = RE._align_states(lm, f["state"][s0:s0 + L])
            acts = RE._episode_actions(f, s0, L, lc, dev)
            with torch.no_grad():
                z = RE._encode_z_lewm(lm, px, 50)                   # rollout start (GPU)
                z_pool = RE._encode_z_lewm(lm_cpu, px.cpu(), 50)     # readout pool (CPU)
            n = min(len(z_pool), len(st))
            lenc["z"].append(z_pool[:n].numpy()); lenc["state"].append(st[:n])
            lenc["ep"].append(np.full(n, ep))
            k_max = (L - RE._enc_window(lm) + 1) - h
            lrol["z"].append(z[h - 1].cpu().numpy()); lrol["k"].append(0)
            lrol["ep"].append(ep); lrol["state"].append(st[h - 1])
            buf = z[:h].unsqueeze(0)
            for k in range(1, k_max + 1):
                act = RE._act_emb_window(lm, acts, ubase, k - 1, h, lat2pix)
                with torch.no_grad():
                    zp = lm.predict(buf, act)[:, -1]
                buf = torch.cat([buf[:, 1:], zp.unsqueeze(1)], 1)
                lrol["z"].append(zp[0].cpu().numpy()); lrol["k"].append(k)
                lrol["ep"].append(ep); lrol["state"].append(st[h - 1 + k])

    cat = lambda v: np.concatenate(v) if np.ndim(v[0]) else np.array(v)
    _savez(CONS / "latents_exp027_exp030.npz",
           lehami_enc_q=cat(enc["q"]), lehami_enc_p=cat(enc["p"]),
           lehami_enc_state=cat(enc["state"]), lehami_enc_ep=cat(enc["ep"]),
           lehami_roll_q=np.array(roll["q"]), lehami_roll_p=np.array(roll["p"]),
           lehami_roll_H=np.array(roll["H"]), lehami_roll_k=np.array(roll["k"]),
           lehami_roll_ep=np.array(roll["ep"]), lehami_roll_state=np.array(roll["state"]),
           lewm_enc_z=cat(lenc["z"]), lewm_enc_state=cat(lenc["state"]),
           lewm_enc_ep=cat(lenc["ep"]),
           lewm_roll_z=np.array(lrol["z"]), lewm_roll_k=np.array(lrol["k"]),
           lewm_roll_ep=np.array(lrol["ep"]), lewm_roll_state=np.array(lrol["state"]),
           episodes=np.array(eps))


def export_actuated_landscape():
    """EXP-079 a0q seed 0: learned H on a grid in the (PC1 q, PC1 p) plane, per torque.

    The plane is fitted to the model's own rollouts over the test episodes, then mapped
    back to the full latent and H is evaluated at u in {-4, 0, 4}. Mirrors
    te_w3_multi_MAIN cells 66 and 71.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    torch, RE, _ = _models()
    dev = "cpu"
    model, cfg = RE.load_model(
        CKPT / "lehami_sg_sep_sreg_q_ms_conv_pendulum_swingup_lehami_sg_sep_conv_w1_swingup_exp079_a0q_s0"
        / "best.ckpt", dev)
    man = json.loads((DATA / "split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json")
                     .read_text())
    eps = sorted(man["test_episode_indices"])
    h5 = RE._h5_path(cfg)
    mom = RE._mom_type(cfg.model.name)
    d, a = RE._latent_dim(cfg), RE._anchor_frame(cfg, mom)
    u0 = RE._zero_action_emb(model, cfg, dev)
    lat2pix = RE._latent_to_pixel_offset(model)
    h_arr = RE._episode_dt_ratio(h5, cfg)
    Q, P = [], []
    with h5py.File(h5, "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        for ep in eps:
            s0, L = int(off[ep]), int(ln[ep])
            px = RE._preprocess_batch(f["pixels"][s0:s0 + L], cfg.img_size, dev)
            acts = RE._episode_actions(f, s0, L, cfg, dev)
            with torch.no_grad():
                cls, q_enc = RE._encode_episode(model, px, 50)
                q0, p0 = RE._build_anchor(model, cls, q_enc, mom, cfg, dev)
                if h_arr is not None:
                    p0 = p0 / float(h_arr[ep])
                qp = torch.cat([q0, p0], -1).unsqueeze(0)
                for k in range((L - RE._enc_window(model) + 1) - a):
                    u_k = RE._act_emb_at(model, acts, u0, a + (k - 1 if k else 0), lat2pix)
                    if k > 0:
                        qp = model.predictor(
                            qp.unsqueeze(1), u_k.unsqueeze(1),
                            None if h_arr is None else torch.full((1,), float(h_arr[ep])))[:, 0]
                    Q.append(qp[0, :d].numpy()); P.append(qp[0, d:].numpy())
    Q, P = np.array(Q), np.array(P)

    def basis(X):
        mu = X.mean(0, keepdims=True)
        vt = np.linalg.svd(X - mu, full_matrices=False)[2][0]
        return mu, vt * np.sign(vt[np.abs(vt).argmax()])

    muq, vq = basis(Q); mup, vp = basis(P)
    x, y = (Q - muq) @ vq, (P - mup) @ vp
    GX, GY = np.meshgrid(np.linspace(x.min(), x.max(), 40), np.linspace(y.min(), y.max(), 40))
    qg = torch.from_numpy(muq + GX.ravel()[:, None] * vq[None]).float()
    pg = torch.from_numpy(mup + GY.ravel()[:, None] * vp[None]).float()
    u_vals = np.array([-4.0, 0.0, 4.0])
    surfs = []
    for uv in u_vals:
        with torch.no_grad():
            ue = model.action_encoder(torch.full((1, 1, cfg.data.action_dim), float(uv)))[:, 0]
            surfs.append(model.predictor.flow.eval_H(qg, pg, ue.expand(len(qg), -1))
                         .squeeze(-1).numpy().reshape(GX.shape))
    _savez(ACT / "landscape_exp079_a0q_s0.npz", GX=GX, GY=GY, surfs=np.stack(surfs),
           u_vals=u_vals, x=x, y=y, test_episodes=np.array(eps))


def export_frames():
    """Dataset frames for the three figures that show raw observations."""
    import h5py
    import hdf5plugin  # noqa: F401
    sys.path.insert(0, str(EXP / "src"))
    import make_datasets_strip_pendula as strip

    # (a) the two-environment strip: the script's own episode choice
    with h5py.File(SWM / "pendulum_swingup.h5", "r") as f:
        stride = int(f.attrs["hold"])
        off_a, ep_a, _ = strip._pick_actuated(f, stride, strip.N_COL)
        rows = [off_a + i * stride for i in range(strip.N_COL)]
        frames_a = np.stack([f["pixels"][r] for r in rows])
        u_blocks = np.array([float(f["action"][r, 0]) for r in rows])
        u_max = float(f.attrs["u_max"])
    with h5py.File(SWM / "pendulum_multi_v2.h5", "r") as f:
        off_c, ep_c, _ = strip._pick_conservative(f, stride, strip.N_COL)
        frames_c = np.stack([f["pixels"][off_c + i * stride] for i in range(strip.N_COL)])
        # (b) three rod lengths at the same angle: the frame closest to theta = 0.6 rad
        # among the dt = 0.1 s episodes of each length
        L_ep, dtm = f["length"][:], f["dt_mult"][:]
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        rods, rod_frames, rod_theta = [], [], []
        for L in f.attrs["lengths"]:
            best = (np.inf, None, None)
            for e in np.where((np.isclose(L_ep, L)) & (dtm == 2))[0][:100]:
                th = f["state"][int(off[e]):int(off[e]) + int(ln[e]), 0]
                i = int(np.argmin(np.abs(th - 0.6)))
                if abs(th[i] - 0.6) < best[0]:
                    best = (abs(th[i] - 0.6), int(off[e]) + i, float(th[i]))
            rods.append(float(L)); rod_frames.append(f["pixels"][best[1]])
            rod_theta.append(best[2])
    _savez(DSET / "strip_frames.npz", frames_conservative=frames_c, frames_actuated=frames_a,
           u_blocks=u_blocks, stride=stride, u_max=u_max, episode_conservative=ep_c,
           episode_actuated=ep_a)
    _savez(DSET / "rod_length_frames.npz", frames=np.stack(rod_frames),
           lengths=np.array(rods), theta=np.array(rod_theta))
    # (c) fifteen evenly spaced frames of one single-rod episode (Fig. pendulum-frames)
    with h5py.File(SWM / "pendulum.h5", "r") as f:
        s0, L = int(f["ep_offset"][0]), int(f["ep_len"][0])
        t = np.linspace(0, L - 1, 15).round().astype(int)
        _savez(DSET / "single_rod_episode0_frames.npz", frames=f["pixels"][s0 + t],
               t=t, theta=f["state"][s0 + t, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-models", action="store_true",
                    help="skip the steps that load checkpoints or datasets")
    args = ap.parse_args()
    copy_conservative(); copy_actuated(); copy_manifests(); trim_actuated()
    if not args.skip_models:
        export_frames(); export_conservative_latents(); export_actuated_landscape()
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT.relative_to(ROOT)}  ({size / 1e6:.1f} MB, "
          f"{sum(1 for f in OUT.rglob('*') if f.is_file())} files)")


if __name__ == "__main__":
    main()
