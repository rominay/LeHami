"""The two pendulum environments, one block each, with the torque drawn underneath.

A narrowed redraw of `datasets_strip.png`: the three-body row is gone, the frame labels are
set large enough to read at column width, and each row of thumbnails now sits above the
action that drives it, as a piecewise-constant step plot in the style of the action strip in
`make_concept_properties_actuated_figure.py`.

The alignment is the point of the layout and is exact rather than approximate.  The actuated
dataset holds the torque constant for `hold` frames (pendulum_swingup.py --hold), so one
thumbnail is drawn per torque block and each thumbnail is centred over the time interval its
own action acts on.  Reading a column top to bottom therefore gives the frame, then the
torque applied from that frame onward.

The conservative row keeps its strip, flat at zero.  That row is the reason the comparison
reads: the two systems differ in whether the generator changes with an action, not in the
geometry of the step.

Writes submission/figs/diagrams/datasets_strip_pendula.{png,pdf}.
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the blosc filter the datasets were written with
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "submission" / "figs" / "diagrams"
H5_CONS = Path.home() / ".stable_worldmodel" / "pendulum_multi_v2.h5"  # conservative
H5_ACT = Path.home() / ".stable_worldmodel" / "pendulum_swingup.h5"  # actuated

INK, MUTED, FAINT = "#0b0b0b", "#5b5b57", "#9aa1ab"
TEAL = "#1a9e8f"

N_COL = 8  # thumbnails per row
GAP = 0.16  # fraction of a block left blank between thumbnails


def _pick_conservative(f, stride, n_col, dt_mult_want=2, span_want=3.2):
    """An episode on the same frame interval as the actuated set, with a legible swing.

    dt_mult is per-episode in this dataset, so the frame-to-frame interval is only
    comparable across the two rows if it is matched explicitly.  The target span is a
    readability constraint, not a physical one: the widest-swinging episode is a pendulum
    going over the top, and consecutive thumbnails of it jump far enough that the eye
    cannot follow the motion at all.  Roughly one swing across the strip reads.
    """
    need = stride * n_col
    offs, lens = f["ep_offset"][:], f["ep_len"][:]
    dtm = f["dt_mult"][:]  # per EPISODE, not per frame
    best, best_span, best_err = None, None, np.inf
    for e in range(len(lens)):
        off, ln = int(offs[e]), int(lens[e])
        if ln < need or int(dtm[e]) != dt_mult_want:
            continue
        th = f["state"][off : off + need, 0]
        span = float(np.ptp(th))  # how much of the swing the strip actually shows
        err = abs(span - span_want)
        if err < best_err:
            best, best_span, best_err = (off, e), span, err
        if e > 400:  # enough of the file scanned; this is a figure, not a search
            break
    assert best is not None, "no conservative episode long enough at the requested dt_mult"
    return best[0], best[1], best_span


def _pick_actuated(f, stride, n_col):
    """An episode whose torque blocks are not all the same sign, so the steps are visible."""
    need = stride * n_col
    offs, lens = f["ep_offset"][:], f["ep_len"][:]
    best, best_var = None, -1.0
    for e in range(min(len(lens), 300)):
        off, ln = int(offs[e]), int(lens[e])
        if ln < need:
            continue
        u = f["action"][off : off + need, 0]
        blocks = u[::stride]
        # want both signs present and a wide range, so the strip shows real structure
        score = float(np.ptp(blocks)) * (1.0 if (blocks.min() < 0 < blocks.max()) else 0.3)
        if score > best_var:
            best, best_var = (off, e), score
    assert best is not None, "no actuated episode long enough"
    return best[0], best[1], best_var


def _draw_block(fig, frames, u_steps, stride, y_img, h_img, y_act, h_act, x0, x1, *, label,
                actuated, show_times, u_lim):
    """One environment: a row of thumbnails over its action strip, on a shared time axis."""
    n = len(frames)
    t_end = n * stride
    # Every horizontal position is computed from the same time->figure mapping, so the
    # thumbnails and the step plot cannot drift apart.
    def tx(t):
        return x0 + (x1 - x0) * (t / t_end)

    for i, img in enumerate(frames):
        left, right = tx(i * stride + GAP / 2 * stride), tx((i + 1) * stride - GAP / 2 * stride)
        ax = fig.add_axes([left, y_img, right - left, h_img])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(FAINT)
            s.set_linewidth(0.8)
        if show_times:
            ax.set_title(f"$t{{+}}{i * stride}$", fontsize=26, color=INK, pad=8)

    # The row label spans the thumbnails AND the strip below them, because "Conservative
    # pendulum" set at a readable size is taller than a single 64-pixel thumbnail row.
    # ROTATED 90. The two words are given as two LINES of one rotated text: at
    # rotation=90 matplotlib offsets successive lines perpendicular to the text
    # direction, so on the page they sit as adjacent columns and read together
    # when the head is tilted. Stacking them along the reading direction instead
    # would need vertical room equal to their COMBINED length (~4.0in at 26pt)
    # against a block only 2.67in tall, so it cannot be done at this size.
    fig.text(0.035, (y_act + y_img + h_img) / 2, label.replace(" ", "\n"),
             rotation=90, ha="center", va="center", fontsize=30, color=INK,
             linespacing=1.15, multialignment="center")

    ax_u = fig.add_axes([tx(0), y_act, tx(t_end) - tx(0), h_act])
    kk = np.arange(n + 1) * stride
    uu = np.append(u_steps, u_steps[-1])
    colour = TEAL if actuated else FAINT
    ax_u.step(kk, uu, where="post", color=colour, lw=2.0)
    ax_u.axhline(0.0, color=FAINT, lw=0.7, ls=(0, (3, 3)), zorder=0)
    ax_u.set_xlim(0, t_end)
    # Both strips share the y range, so the flat row reads as "no torque" on the same
    # scale rather than as a different measurement.
    ax_u.set_ylim(-u_lim, u_lim)
    ax_u.set_xticks([])
    ax_u.set_yticks([-4, 0, 4])
    ax_u.set_ylabel("$u_t$", fontsize=30, color=INK, rotation=0, labelpad=34, va="center")
    ax_u.tick_params(labelsize=24, colors=MUTED, length=3.0, pad=3)
    if not actuated:
        ax_u.text(0.5, 0.60, "$u_t \\equiv 0$", transform=ax_u.transAxes,
                  ha="center", va="bottom", fontsize=24, color=MUTED)
    for side in ("top", "right"):
        ax_u.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_u.spines[side].set_edgecolor(FAINT)
        ax_u.spines[side].set_linewidth(0.8)
    return ax_u


def main():
    with h5py.File(H5_ACT, "r") as f:
        stride = int(f.attrs["hold"])  # one thumbnail per torque block
        off_a, ep_a, _ = _pick_actuated(f, stride, N_COL)
        rows = [off_a + i * stride for i in range(N_COL)]
        frames_a = [f["pixels"][r] for r in rows]
        u_blocks = np.array([float(f["action"][r, 0]) for r in rows])
        # the torque really is constant within a block, which is what lets one value per
        # block stand for the whole interval
        u_full = f["action"][off_a : off_a + N_COL * stride, 0].astype(float)
        assert np.allclose(u_full.reshape(N_COL, stride), u_blocks[:, None], atol=1e-6), (
            "actuated dataset is not piecewise-constant over `hold` frames")
        u_max = float(f.attrs["u_max"])

    with h5py.File(H5_CONS, "r") as f:
        off_c, ep_c, span_c = _pick_conservative(f, stride, N_COL)
        frames_c = [f["pixels"][off_c + i * stride] for i in range(N_COL)]
        assert "action" not in f, "conservative dataset unexpectedly carries an action"

    assert frames_a[0].shape == frames_c[0].shape == (64, 64, 3)
    assert np.abs(u_blocks).max() <= u_max + 1e-6

    fig = plt.figure(figsize=(12.4, 6.6))
    x0, x1 = 0.215, 0.990
    u_lim = u_max * 1.20
    _draw_block(fig, frames_c, np.zeros(N_COL), stride,
                y_img=0.620, h_img=0.290, y_act=0.505, h_act=0.095,
                x0=x0, x1=x1, label="Conservative pendulum", actuated=False,
                show_times=True, u_lim=u_lim)
    _draw_block(fig, frames_a, u_blocks, stride,
                y_img=0.175, h_img=0.290, y_act=0.060, h_act=0.095,
                x0=x0, x1=x1, label="Actuated pendulum", actuated=True,
                show_times=False, u_lim=u_lim)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"datasets_strip_pendula.{ext}", dpi=200,
                    facecolor="white")
    plt.close(fig)

    print(f"conservative: episode {ep_c}, offset {off_c}, theta range {span_c:.2f} rad")
    print(f"actuated:    episode {ep_a}, offset {off_a}, hold={stride}, u_max={u_max}")
    print(f"torque blocks shown: {np.array2string(u_blocks, precision=2)}")
    print(f"wrote {OUT_DIR / 'datasets_strip_pendula.png'}")


if __name__ == "__main__":
    main()
