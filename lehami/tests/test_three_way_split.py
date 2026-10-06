"""Tests for the train/val/test/ood episode split (`data_prep.py --val-ratio`).

These exist because the failure they guard against is silent and expensive: a
checkpoint selected on data it should not have seen makes every downstream
number optimistically biased, and nothing in the training logs looks wrong. The
original instance (`best.ckpt` monitored on a `validate` loader built from the
TEST episodes) survived dozens of runs undetected and was found only by reading
`_split_from_manifest`.

Three properties, in order of how badly getting them wrong would hurt:

  1. the held-out dt arm never reaches train or val -- otherwise checkpoints are
     selected on the very generalisation being claimed;
  2. train / val / test / ood are mutually disjoint and cover every episode;
  3. each split is balanced across (L, dt) design cells.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lehami" / "data"))

import data_prep  # noqa: E402

# The shipped design: 3 rod lengths x 4 sampling rates, dt x1 held out.
# x1 (not x3) because an interior hold-out tests blending rather than
# dt-dependence, and coarser than x4 aliases -- see the notebook's sec 2.
CELLS = [(0.7, 1), (0.7, 2), (0.7, 3), (0.7, 4),
         (1.0, 1), (1.0, 2), (1.0, 3), (1.0, 4),
         (1.4, 1), (1.4, 2), (1.4, 3), (1.4, 4)]
N_PER_CELL = 100
N_EPISODES = N_PER_CELL * len(CELLS)
HELD_OUT = [1]


def _stats(n_per_cell=N_PER_CELL, held_out=HELD_OUT, cells=CELLS):
    ep_cells, ep_dt = [], []
    for L, k in cells:
        ep_cells += [(L, k)] * n_per_cell
        ep_dt += [k] * n_per_cell
    n = len(ep_cells)
    return {
        "n_episodes": n,
        "ep_lengths": [int(round(8.0 / (0.05 * k))) for _, k in ep_cells],
        "cell_keys": ["length", "dt_mult"],
        "ep_cells": ep_cells,
        "ep_dt_mult": ep_dt,
        "held_out_dt": held_out,
    }


# ── property 1: the held-out arm is quarantined ───────────────────────────────

def test_held_out_dt_never_reaches_train_or_val():
    s = _stats()
    train, val, test, ood = data_prep._partition_episodes(s, 0.70, 0.15, seed=42)
    for name, idx in [("train", train), ("val", val), ("test", test)]:
        arms = {s["ep_dt_mult"][i] for i in idx}
        assert not (arms & set(HELD_OUT)), f"{name} contains held-out dt {arms & set(HELD_OUT)}"
    assert {s["ep_dt_mult"][i] for i in ood} == set(HELD_OUT)
    assert len(ood) == N_PER_CELL * 3        # one cell per rod length


def test_ood_is_independent_of_ratios_and_seed():
    """Quarantine is by design column, so it must not move with the split RNG."""
    base = data_prep._partition_episodes(_stats(), 0.70, 0.15, seed=42)[3]
    for ratios in [(0.5, 0.25), (0.9, 0.05), (0.7, 0.0)]:
        for seed in (0, 1, 42):
            assert data_prep._partition_episodes(_stats(), *ratios, seed=seed)[3] == base


# ── property 2: disjoint and exhaustive ───────────────────────────────────────

def test_splits_are_disjoint_and_cover_every_episode():
    s = _stats()
    train, val, test, ood = data_prep._partition_episodes(s, 0.70, 0.15, seed=42)
    sets = [set(train), set(val), set(test), set(ood)]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert not (sets[a] & sets[b]), f"splits {a} and {b} overlap"
    assert set().union(*sets) == set(range(s["n_episodes"]))


def test_val_is_disjoint_from_test():
    """The point of the third split: selection data is not reported data."""
    _, val, test, _ = data_prep._partition_episodes(_stats(), 0.70, 0.15, seed=42)
    assert val and test and not (set(val) & set(test))


# ── property 3: balance across design cells ───────────────────────────────────

@pytest.mark.parametrize("split_i,name,frac", [(0, "train", 0.70), (1, "val", 0.15),
                                               (2, "test", 0.15)])
def test_each_split_is_balanced_across_cells(split_i, name, frac):
    s = _stats()
    parts = data_prep._partition_episodes(s, 0.70, 0.15, seed=42)
    per_cell = Counter(s["ep_cells"][i] for i in parts[split_i])
    trained_cells = [c for c in CELLS if c[1] not in HELD_OUT]
    assert set(per_cell) == set(trained_cells), f"{name} is missing cells"
    assert set(per_cell.values()) == {round(N_PER_CELL * frac)}, \
        f"{name} unbalanced: {per_cell}"


# ── degenerate / legacy inputs ────────────────────────────────────────────────

def test_val_ratio_zero_leaves_no_val_and_keeps_two_way_behaviour():
    train, val, test, ood = data_prep._partition_episodes(_stats(), 0.75, 0.0, seed=42)
    assert val == []
    assert len(train) + len(test) + len(ood) == N_EPISODES


def test_dataset_without_design_columns_still_splits():
    """Plain pendulum.h5 has no length/dt_mult; it must degrade to a shuffle."""
    s = {"n_episodes": 500, "ep_lengths": [100] * 500}
    train, val, test, ood = data_prep._partition_episodes(s, 0.70, 0.15, seed=42)
    assert ood == []
    assert (len(train), len(val), len(test)) == (350, 75, 75)
    assert not (set(train) & set(val)) and not (set(val) & set(test))


def test_ratios_that_leave_no_test_are_rejected():
    with pytest.raises(ValueError, match="no episodes would be left"):
        data_prep._partition_episodes(_stats(), 0.90, 0.20, seed=42)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_val_ratio_is_rejected(bad):
    with pytest.raises(ValueError, match="val_ratio"):
        data_prep._partition_episodes(_stats(), 0.5, bad, seed=42)


def test_split_is_deterministic_in_the_seed():
    a = data_prep._partition_episodes(_stats(), 0.70, 0.15, seed=7)
    b = data_prep._partition_episodes(_stats(), 0.70, 0.15, seed=7)
    c = data_prep._partition_episodes(_stats(), 0.70, 0.15, seed=8)
    assert a == b and a[:3] != c[:3]


# ── the generated manifest, if it is present ──────────────────────────────────

MANIFEST = (ROOT / "lehami" / "data"
            / "split_pendulum_multi_v2_episode_srnn_K40_val.15_seed42.json")
SLICE_MANIFEST = (ROOT / "lehami" / "data"
                  / "split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json")


@pytest.mark.skipif(not MANIFEST.exists(), reason="v2 manifest not generated here")
def test_generated_manifest_has_a_real_three_way_split():
    import json
    m = json.loads(MANIFEST.read_text())
    tr = set(m["train_episode_indices"])
    va = set(m["val_episode_indices"])
    te = set(m["test_episode_indices"])
    oo = set(m["ood_episode_indices"])
    assert (len(tr), len(va), len(te), len(oo)) == (630, 135, 135, 300)
    assert not (tr & va) and not (va & te) and not (tr & te)
    assert not (oo & (tr | va | te))
    assert m["held_out_dt"] == [1]


@pytest.mark.skipif(not MANIFEST.exists(), reason="v2 manifest not generated here")
def test_train_py_selects_the_real_val_split_when_one_exists():
    import json
    sys.path.insert(0, str(ROOT / "lehami" / "training"))
    try:
        import train as train_mod
    except ImportError as e:
        pytest.skip(f"train.py not importable: {e}")
    m = json.loads(MANIFEST.read_text())
    eps, label = train_mod._val_episodes(m)
    assert eps == set(m["val_episode_indices"]) and label == "val"
    # ...and a legacy two-way manifest falls back to test, loudly.
    legacy = {k: v for k, v in m.items() if k != "val_episode_indices"}
    eps2, label2 = train_mod._val_episodes(legacy)
    assert eps2 == set(m["test_episode_indices"]) and "TEST" in label2


# ── restrict_dt_mults: the single-dt slice ────────────────────────────────────

def test_restrict_drops_other_arms_entirely():
    """A fixed-step predictor needs one dt; other arms must vanish, not migrate."""
    s = _stats()
    s["restrict_dt_mults"] = [2]
    train, val, test, ood = data_prep._partition_episodes(s, 0.70, 0.15, seed=42)
    for idx in (train, val, test):
        assert {s["ep_dt_mult"][i] for i in idx} == {2}
    assert ood == []                      # the held-out arm is x1, which is dropped too
    assert len(train) + len(val) + len(test) == N_PER_CELL * 3   # one cell per length


def test_restrict_to_an_absent_arm_raises():
    s = _stats()
    s["restrict_dt_mults"] = [9]
    with pytest.raises(ValueError, match="leaves no training episodes"):
        data_prep._partition_episodes(s, 0.70, 0.15, seed=42)


@pytest.mark.skipif(not SLICE_MANIFEST.exists(), reason="dt2 slice manifest not generated here")
def test_slice_manifest_is_single_dt_and_three_way():
    import json
    m = json.loads(SLICE_MANIFEST.read_text())
    tr, va, te = (set(m[k]) for k in ["train_episode_indices", "val_episode_indices",
                                      "test_episode_indices"])
    assert (len(tr), len(va), len(te)) == (210, 45, 45)
    assert not m.get("ood_episode_indices")
    assert not (tr & va) and not (va & te) and not (tr & te)
