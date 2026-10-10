#!/usr/bin/env python
"""Figure 1, actuated variant: the property that survives a control input is SYMPLECTICITY.

`make_concept_properties_figure.py` draws the same idea on a closed orbit riding one level
set of H_phi.  That picture is correct and it is also easy to misread: a reader who cares
about planning sees a conservative system and asks whether the method needs one.  It does
not.  With the action entering the potential, H_phi(q, p; u) = K_phi(p) + V_phi(q, u), the
Hamiltonian is TIME-DEPENDENT through u_t: its level sets move when the action changes, the
rollout crosses them, and energy is not conserved -- nor is it claimed to be.  What is
preserved for ANY open-loop action sequence is the symplectic form, because a leapfrog step
at fixed u is symplectic and a composition of symplectic maps is symplectic.

So the three right-hand panels are re-run under a NON-ZERO, time-varying action:

  (i)   VOLUME, under actuation.  The cloud is transported by a driven sequence and the
        area is measured with the shoelace formula at each snapshot.

  (ii)  STEP SIZE.  h is an argument to the integrator, so the same H_phi places states
        between observed frames; the driven case is no different.

  (iii) TIME REVERSAL, under actuation.  Run K steps with u_0..u_{K-1}, negate p, run K
        steps with the SAME actions in REVERSE order.  That is the time-reversed problem,
        and for a kinetic term even in p it closes exactly.  Replaying them forwards would
        not close, and the label says which is meant.

The main panel writes the Hamiltonian down, shows the level sets AT FIXED u as a reference
family, and draws a resonantly driven rollout that spirals across them: the visual statement
that actions do work.  The inset shows the mechanism, V_phi tilting with u.

Every number printed on the figure is computed here from the composition it draws.  The
drive is OPEN LOOP (a function of step index, not of the state), which is what a planner's
candidate sequence is, and is what makes the symplectic claim true -- a state-dependent
feedback law is not a time-dependent Hamiltonian flow.

    .venv/bin/python lehami/experiments/src/make_concept_properties_actuated_figure.py

Writes submission/figs/diagrams/concept_lehami_properties_actuated.pdf (+ .png).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the blosc filter the dataset was written with
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "submission" / "figs" / "diagrams"
H5 = Path.home() / ".stable_worldmodel" / "pendulum_swingup.h5"   # the ACTUATED dataset

INK, MUTED, FAINT = "#0b0b0b", "#5b5b57", "#9aa1ab"
BLUE, WARM, TEAL = "#2a78d6", "#eb6834", "#1a9e8f"
REV = "#c2185b"
LATENT_BG = "#f5f8fc"

# One knob for the whole figure.  Every fontsize below is written as its original
# value times FS, so the relative hierarchy is preserved and the absolute size is
# retuned in one place rather than in twenty.
FS = 1.45


def dK(p):
    """K(p) = p^2/2 -- EVEN in p, which is what makes the leapfrog exactly reversible."""
    return p


def dV(q, u):
    """V(q, u) = (1 - cos q) - u q.  The action enters the POTENTIAL, and only there."""
    return np.sin(q) - u


def leapfrog(q, p, h, n, u=0.0):
    """Symplectic step, driven by an OPEN-LOOP action: u scalar or one value per step."""
    us = np.full(n, float(u)) if np.isscalar(u) else np.asarray(u, dtype=float)
    assert us.shape == (n,), f"need one action per step, got {us.shape} for n={n}"
    qs, ps = [q.copy()], [p.copy()]
    for k in range(n):
        p = p - 0.5 * h * dV(q, us[k])
        q = q + h * dK(p)
        p = p - 0.5 * h * dV(q, us[k])
        qs.append(q.copy())
        ps.append(p.copy())
    return np.array(qs), np.array(ps)


def polygon_area(x, y):
    """Shoelace, so the area in the figure is measured and not asserted."""
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _style(a):
    a.set_facecolor(LATENT_BG)
    a.set(xticks=[], yticks=[])
    for sp in a.spines.values():
        sp.set_color("#c9cdd4")


def main():
    n_frames = 5
    with h5py.File(H5, "r") as f:
        off, ln = int(f["ep_offset"][0]), int(f["ep_len"][0])
        # One thumbnail per torque block: the dataset holds u constant for
        # `hold` frames (pendulum_swingup.py --hold), so each frame shown starts a
        # new block and the u printed under it is the torque over that interval.
        stride = int(f.attrs["hold"])
        rows = [off + i * stride for i in range(n_frames)]
        frames = [f["pixels"][r] for r in rows]
        u_frame = np.array([float(f["action"][r, 0]) for r in rows])
        th_frame = np.array([float(f["state"][r, 0]) for r in rows])
        u_traj = f["action"][off:off + n_frames * stride, 0].astype(float)

    # The screen convention is CHECKED, not assumed: an arrow drawn the wrong way round
    # would be a real error.  The bob is the red blob and the pivot the blue one, and the
    # image angle of bob-about-pivot, measured from straight down as atan2(dx, dy) with y
    # increasing downward, must equal theta.  Positive u raises theta
    # (theta'' = -(g/L) sin theta + u/(m L^2), pendulum_swingup.py:99), so a positive
    # torque is an arc in the direction of increasing screen angle.
    img = frames[0].astype(float)
    r_, g_, b_ = img[..., 0], img[..., 1], img[..., 2]
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    red, blue = (r_ > 120) & (g_ < 90) & (b_ < 90), (b_ > 120) & (r_ < 90) & (g_ < 90)
    dx, dy = xx[red].mean() - xx[blue].mean(), yy[red].mean() - yy[blue].mean()
    ang = np.arctan2(dx, dy)
    assert abs(np.angle(np.exp(1j * (ang - th_frame[0])))) < 0.12, (
        f"screen convention check failed: image angle {ang:.3f} vs theta {th_frame[0]:.3f}")
    print(f"screen check: image angle {np.degrees(ang):+.1f} deg vs theta "
          f"{np.degrees(th_frame[0]):+.1f} deg -> increasing theta is counter-clockwise")

    fig = plt.figure(figsize=(11.6, 6.0))
    ax = fig.add_axes([0.050, 0.430, 0.545, 0.525])          # latent space
    ax_vol = fig.add_axes([0.665, 0.665, 0.315, 0.290])      # (i)   volume
    ax_dt = fig.add_axes([0.665, 0.375, 0.315, 0.215])       # (ii)  step size
    ax_rev = fig.add_axes([0.665, 0.065, 0.315, 0.215])      # (iii) reversal
    for a in (ax, ax_vol, ax_dt, ax_rev):
        _style(a)

    # ── main panel: level sets AT FIXED u, and a driven rollout crossing them ───
    qq, pp = np.meshgrid(np.linspace(-3.4, 3.4, 500), np.linspace(-2.6, 2.6, 500))
    ax.contour(qq, pp, 0.5 * pp ** 2 + (1 - np.cos(qq)),
               levels=np.linspace(0.2, 3.2, 10), colors=MUTED, linewidths=0.6, alpha=0.38)

    # Resonant OPEN-LOOP drive: u_k = A sin(Omega k h), Omega at the small-oscillation
    # frequency of this V (omega_0 = 1), so the sequence pumps energy and the rollout
    # spirals ACROSS the level sets drawn above. A function of the step index only.
    h_main, n_main = 0.02, 1150
    A, OMEGA = 0.085, 1.0
    u_main = A * np.sin(OMEGA * h_main * np.arange(n_main))
    qs, ps = leapfrog(np.array([-0.55]), np.array([0.0]), h_main, n_main, u_main)
    qs, ps = qs[:, 0], ps[:, 0]
    E0 = 0.5 * ps[0] ** 2 + (1 - np.cos(qs[0]))
    E1 = 0.5 * ps[-1] ** 2 + (1 - np.cos(qs[-1]))
    ax.plot(qs, ps, color=BLUE, lw=1.7, zorder=3, solid_capstyle="round", alpha=0.95)

    marks = np.linspace(int(0.26 * n_main), n_main - 30, n_frames).astype(int)
    ax.scatter(qs[marks], ps[marks], s=40, color=BLUE, zorder=5,
               edgecolor="white", linewidth=1.1)
    tip = n_main - 2
    ax.annotate("", xy=(qs[tip], ps[tip]), xytext=(qs[tip - 30], ps[tip - 30]),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0, shrinkA=0,
                                shrinkB=0, mutation_scale=20), zorder=6)

    ax.set_title("latent representation space $(\\mathbf{q}, \\mathbf{p})$",
                 loc="left", fontsize=11.5 * FS, color=INK, pad=6)
    ax.text(0.015, 0.972,
            "$H_\\phi(\\mathbf{q}, \\mathbf{p}; u) = "
            "K_\\phi(\\mathbf{p}) + V_\\phi(\\mathbf{q}, u)$",
            transform=ax.transAxes, va="top", fontsize=11.0 * FS, color=INK, zorder=10,
            bbox=dict(fc=LATENT_BG, ec="none", pad=2.0))
    ax.text(0.015, 0.045, "level sets of $H_\\phi(\\cdot\\,; u)$ at fixed $u$",
            transform=ax.transAxes, ha="left", fontsize=9.0 * FS, color=INK, zorder=10,
            bbox=dict(fc=LATENT_BG, ec="none", pad=1.6))
    # The energy rise along the drawn curve is still COMPUTED (E0 -> E1) and printed to
    # stdout below; it is no longer annotated on the panel.
    ax.set_xlabel("$\\mathbf{q}$, the position-like block", fontsize=10.2 * FS, labelpad=2)
    ax.set_ylabel("$\\mathbf{p}$, the momentum-like block", fontsize=10.2 * FS)
    ax.set(xlim=(-3.4, 3.4), ylim=(-2.45, 2.45))

    # inset: the mechanism, V tilting with u
    ax_in = fig.add_axes([0.452, 0.790, 0.132, 0.130])
    _style(ax_in)
    qv = np.linspace(-3.2, 3.2, 300)
    ax_in.plot(qv, 1 - np.cos(qv), color=MUTED, lw=1.3, label="$u = 0$")
    ax_in.plot(qv, (1 - np.cos(qv)) - 0.6 * qv, color=TEAL, lw=1.3, ls=(0, (3, 2)),
               label="$u \\neq 0$")
    ax_in.set_title("$V_\\phi(\\mathbf{q}, u)$ tilts with $u$", fontsize=8.2 * FS,
                    color=INK, pad=3)
    ax_in.legend(fontsize=7.2 * FS, frameon=False, loc="upper center", handlelength=1.1,
                 handletextpad=0.4, borderpad=0.1, labelspacing=0.15,
                 bbox_to_anchor=(0.5, 0.02), ncol=2, columnspacing=0.8)

    # ── (i) a cloud of nearby states, transported by a DRIVEN sequence ─────────
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    cq, cp = 0.32 * np.cos(th) - 0.9, 0.26 * np.sin(th) + 1.05
    snaps = [0, 55, 110]
    u_vol = 0.9 * np.sin(0.06 * np.arange(max(snaps)))        # open loop, time-varying
    bq, bp = leapfrog(cq, cp, 0.02, max(snaps), u_vol)
    areas = [polygon_area(bq[s], bp[s]) for s in snaps]
    for j, s in enumerate(snaps):
        ax_vol.add_patch(Polygon(np.column_stack([bq[s], bp[s]]), closed=True,
                                 facecolor=WARM, edgecolor=WARM, alpha=0.55 - 0.12 * j,
                                 lw=1.1, zorder=4))
    ax_vol.annotate("", xy=(bq[snaps[-1]].mean() - 0.26, bp[snaps[-1]].mean() + 0.28),
                    xytext=(bq[0].mean() + 0.20, bp[0].mean() + 0.32),
                    arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.2,
                                    connectionstyle="arc3,rad=-0.25"))
    ax_vol.set_title("(i)  volume preservation",
                     loc="left", fontsize=10.0 * FS, color=INK, pad=4)
    # The measured areas are still computed by the shoelace above and printed to stdout
    # below; the on-panel readout is gone.
    vis = slice(0, max(snaps) + 1)
    x0, x1 = bq[vis].min(), bq[vis].max()
    y0, y1 = bp[vis].min(), bp[vis].max()
    pad = 0.14
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    bx, by, bw, bh = ax_vol.get_position().bounds
    box_aspect = (bw * fig.get_size_inches()[0]) / (bh * fig.get_size_inches()[1])
    xr, yr = x1 - x0, y1 - y0
    if xr / yr < box_aspect:
        grow = (yr * box_aspect - xr) / 2
        x0, x1 = x0 - grow, x1 + grow
    else:
        grow = (xr / box_aspect - yr) / 2
        y0, y1 = y0 - grow, y1 + grow
    ax_vol.set(xlim=(x0, x1), ylim=(y0, y1))

    # ── (ii) the step size is an argument, not a property of the weights ───────
    h_frame, n_frame, u_dt = 0.25, 4, 0.55
    z0 = (np.array([-1.55]), np.array([0.95]))
    cq_h, cp_h = leapfrog(*z0, h_frame, n_frame, u_dt)
    cq_q, cp_q = leapfrog(*z0, h_frame / 4, 4 * n_frame, u_dt)
    endpoint_gap = float(np.hypot(cq_h[-1, 0] - cq_q[-1, 0], cp_h[-1, 0] - cp_q[-1, 0]))
    frame_len = float(np.hypot(cq_h[1, 0] - cq_h[0, 0], cp_h[1, 0] - cp_h[0, 0]))

    ax_dt.plot(cq_q[:, 0], cp_q[:, 0], color=BLUE, lw=1.4, alpha=0.45, zorder=2)
    ax_dt.scatter(cq_q[1:-1, 0], cp_q[1:-1, 0], s=9, color=BLUE, alpha=0.75, zorder=3,
                  label="$h/4$: states between frames")
    ax_dt.scatter(cq_h[:, 0], cp_h[:, 0], s=46, color=BLUE, zorder=5,
                  edgecolor="white", linewidth=1.1, label="$h$: the observed frames")
    ax_dt.set_title("(ii)  predictor at continuous step size",
                    loc="left", fontsize=10.4 * FS, color=INK, pad=5)
    ax_dt.legend(loc="lower center", fontsize=8.4 * FS, frameon=False, handletextpad=0.35,
                 borderpad=0.2, labelspacing=0.25, bbox_to_anchor=(0.57, -0.02))
    # Both traces share one H_phi and one u -- the legend already says which is which,
    # so the sentence that used to state it here is gone.
    ax_dt.set_xlabel("$\\mathbf{q}$", fontsize=9.6 * FS, labelpad=1)
    ax_dt.set_ylabel("$\\mathbf{p}$", fontsize=9.6 * FS, labelpad=1)

    # ── (iii) reversal under actuation: the actions are replayed in REVERSE ────
    K, h_rev = 12, 0.20
    u_rev = 0.7 * np.sin(0.45 * np.arange(K) + 0.3)
    r0 = (np.array([-2.40]), np.array([0.70]))
    fq, fp = leapfrog(*r0, h_rev, K, u_rev)
    rq, rp = leapfrog(fq[-1], -fp[-1], h_rev, K, u_rev[::-1])      # reversed schedule
    closure = float(max(abs(rq[-1, 0] - r0[0][0]), abs(-rp[-1, 0] - r0[1][0])))
    # The control: the SAME actions replayed forwards do not undo the trajectory.
    wq, _ = leapfrog(fq[-1], -fp[-1], h_rev, K, u_rev)
    closure_wrong = float(abs(wq[-1, 0] - r0[0][0]))

    kk = np.arange(K + 1)
    ax_rev.plot(kk, fq[:, 0], color=BLUE, lw=3.4, zorder=3, solid_capstyle="round")
    ax_rev.plot(kk[::-1], rq[:, 0], color=REV, lw=2.0, ls=(0, (2.6, 4.6)), zorder=4,
                dash_capstyle="round")
    ax_rev.scatter([0, K], [fq[0, 0], fq[-1, 0]], s=42, zorder=6,
                   color=[BLUE, MUTED], edgecolor="white", linewidth=1.1)
    ax_rev.axhline(r0[0][0], color=FAINT, lw=0.8, ls=":", zorder=1)
    lo, hi = fq[:, 0].min(), fq[:, 0].max()
    ax_rev.set_ylim(lo - 0.34 * (hi - lo), hi + 0.22 * (hi - lo))
    ax_rev.set_title("(iii)  backward-running predictor",
                     loc="left", fontsize=10.0 * FS, color=INK, pad=5)
    a, b = int(0.14 * K), int(0.40 * K)
    c, d = int(0.86 * K), int(0.60 * K)
    ax_rev.annotate("", xy=(b, fq[b, 0]), xytext=(a, fq[a, 0]),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8, shrinkA=0,
                                    shrinkB=0))
    ax_rev.annotate("", xy=(d, fq[d, 0]), xytext=(c, fq[c, 0]),
                    arrowprops=dict(arrowstyle="-|>", color=REV, lw=1.9, shrinkA=0,
                                    shrinkB=0))
    ax_rev.text(0.030, 0.94, "forward, $u_0 \\ldots u_{K-1}$",
                transform=ax_rev.transAxes, ha="left", va="top",
                fontsize=8.8 * FS, color=BLUE, zorder=8)
    ax_rev.text(0.970, 0.06, "backward, $u_{K-1} \\ldots u_0$",
                transform=ax_rev.transAxes, ha="right", va="bottom",
                fontsize=8.8 * FS, color=REV, zorder=8)
    ax_rev.set_xlabel("time, in steps", fontsize=9.6 * FS, labelpad=1)
    ax_rev.set_ylabel("$\\mathbf{q}$", fontsize=9.6 * FS, labelpad=1)

    # ── the observation row, placed under the latent state it encodes ──────────
    w = 0.086
    lefts = np.linspace(0.055, 0.595 - w, n_frames)
    for i, (img, m) in enumerate(zip(frames, marks)):
        left = float(lefts[i])
        axi = fig.add_axes([left, 0.140, w, w * 11.6 / 6.0])
        axi.imshow(img)
        axi.set(xticks=[], yticks=[])
        axi.grid(False)
        for sp in axi.spines.values():
            sp.set_color("#c9cdd4")
        # the torque applied over [t, t+1), as an arc about the pivot: direction is the
        # sign of u, angular extent grows with |u|.  Drawn opposite the rod so it never
        # sits on top of it.
        H, W_ = img.shape[0], img.shape[1]
        cx, cy, rr = W_ / 2, H / 2, 0.30 * W_
        u_i = u_frame[i]
        span = np.radians(28 + 52 * min(abs(u_i) / 4.0, 1.0))
        phi_c = th_frame[i] + np.pi
        phi = np.linspace(phi_c - span / 2, phi_c + span / 2, 40)
        if u_i < 0:
            phi = phi[::-1]
        ax_arc_x, ax_arc_y = cx + rr * np.sin(phi), cy + rr * np.cos(phi)
        axi.plot(ax_arc_x, ax_arc_y, color=TEAL, lw=1.7, solid_capstyle="round", zorder=6)
        axi.annotate("", xy=(ax_arc_x[-1], ax_arc_y[-1]),
                     xytext=(ax_arc_x[-6], ax_arc_y[-6]), zorder=7,
                     arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.7, shrinkA=0,
                                     shrinkB=0, mutation_scale=11))
        axi.set_xlabel(f"$x_{{t+{i * stride}}}$\n$u = {u_i:+.1f}$",
                       fontsize=9.0 * FS, labelpad=2, linespacing=1.35)
        y_bot = 0.140 + w * 11.6 / 6.0
        ax.annotate("", xy=(qs[m], ps[m]), xycoords=ax.transData,
                    xytext=(left + w / 2, y_bot), textcoords="figure fraction",
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.2,
                                    linestyle=(0, (1.4, 2.2)), shrinkA=1, shrinkB=5))
        x_top, y_top = fig.transFigure.inverted().transform(
            ax.transData.transform((qs[m], ps[m])))
        y_lab = 0.335
        t = (y_lab - y_bot) / (y_top - y_bot)
        fig.text(left + w / 2 + t * (x_top - (left + w / 2)), y_lab, "encoder",
                 ha="center", va="center", fontsize=8.6 * FS, color=INK,
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#c9cdd4", lw=0.7))
    # The five thumbnails are a subsample; the torque changes between them, so the
    # sequence is drawn as well.  This is the object a planner searches over.
    ax_u = fig.add_axes([0.055, 0.000, 0.540, 0.058])
    _style(ax_u)
    kk = np.arange(len(u_traj) + 1)
    ax_u.step(kk, np.append(u_traj, u_traj[-1]), where="post", color=TEAL, lw=1.3)
    ax_u.scatter([i * stride for i in range(n_frames)], u_frame, s=22, color=TEAL,
                 zorder=5, edgecolor="white", linewidth=0.9)
    ax_u.axhline(0.0, color=FAINT, lw=0.8, ls=":")
    ax_u.set_xlim(0, len(u_traj))
    # Symmetric about zero, so the SIGN of the torque is what the strip shows; the
    # sampled frames are a minority of the sequence and a one-sided range hid that.
    m_u = 1.15 * float(np.abs(u_traj).max())
    ax_u.set_ylim(-m_u, m_u)
    ax_u.set_yticks([-round(m_u / 1.15, 1), 0, round(m_u / 1.15, 1)])
    ax_u.tick_params(axis="y", labelsize=7.2 * FS, length=2, pad=1)
    # No axis label on the strip: it sat over the step trace, and the frame row
    # directly above already reads "u = ..." under every thumbnail.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"concept_lehami_properties_actuated.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    print(f"frame strip: stride={stride} = the dataset's torque hold; "
          f"rows t+{[i * stride for i in range(n_frames)]}")
    print("driven cloud areas:", [f"{a:.6f}" for a in areas],
          f"  max relative drift {max(abs(a / areas[0] - 1) for a in areas):.2e}")
    print(f"main panel: H_phi {E0:.4f} -> {E1:.4f} under the drive "
          f"(A={A}, Omega={OMEGA}); energy is NOT conserved, area is")
    print(f"step-size panel: h={h_frame}, u={u_dt}, endpoint gap h vs h/4 = "
          f"{endpoint_gap:.3e} ({endpoint_gap / frame_len:.2%} of one frame)")
    print(f"reversal panel:  K={K}, h={h_rev}, closure with REVERSED actions = "
          f"{closure:.3e};  replaying them forwards instead = {closure_wrong:.3e}")


if __name__ == "__main__":
    main()
