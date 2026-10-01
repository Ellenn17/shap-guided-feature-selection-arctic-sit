# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def block_split(n, block_size=80, test_fraction=0.2, gap=10, seed=None):
    """
    Generates alternating train/test block indices with buffers.

    Parameters
    ----------
    n : int
        Total number of timesteps.
    block_size : int
        Length of each contiguous block.
    test_fraction : float
        Approximate fraction of blocks to assign to the test set.
    gap : int
        Number of points to discard at each train/test boundary.
    seed : int or None
        If provided, shuffles block assignments reproducibly
        instead of alternating them regularly (1 every K).

    Returns
    -------
    train_idx, test_idx, gap_idx : np.ndarray
        Indices (0-based) assigned to train, test, and gap sets, respectively.
    """
    n_blocks = int(np.ceil(n / block_size))
    block_bounds = [(i * block_size, min((i + 1) * block_size, n)) for i in range(n_blocks)]

    step = max(1, round(1 / test_fraction))
    is_test_block = np.zeros(n_blocks, dtype=bool)
    if seed is None:
        offset = step // 2
        is_test_block[offset::step] = True
    else:
        rng = np.random.default_rng(seed)
        n_test_blocks = max(1, round(n_blocks * test_fraction))
        test_block_ids = rng.choice(n_blocks, size=n_test_blocks, replace=False)
        is_test_block[test_block_ids] = True

    train_idx, test_idx, gap_idx = [], [], []
    for i, (start, end) in enumerate(block_bounds):
        idx = np.arange(start, end)
        if is_test_block[i]:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
  
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    gap_set = set()

    for i, (start, end) in enumerate(block_bounds):
        if not is_test_block[i]:
            continue
        for p in range(max(0, start - gap), start):
            if p in train_set:
                gap_set.add(p)
        for p in range(end, min(n, end + gap)):
            if p in train_set:
                gap_set.add(p)

    train_set -= gap_set

    train_idx = np.array(sorted(train_set))
    test_idx = np.array(sorted(test_set))
    gap_idx = np.array(sorted(gap_set))

    return train_idx, test_idx, gap_idx


def plot_split(y, train_idx, test_idx, gap_idx, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(y))
    ax.plot(x, y, color="tab:blue", linewidth=1, zorder=1, label="Target")
    ax.legend_done = False
    return ax


def _contiguous_runs(idx):
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return runs


def plot_split_with_trends(y, train_idx, test_idx, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(y))
    ax.plot(x, y, color="tab:blue", linewidth=1, zorder=1, label="Target")

    train_label_done = False
    for start, end in _contiguous_runs(train_idx):
        xs = np.arange(start, end + 1)
        if len(xs) < 2:
            continue
        coef = np.polyfit(xs, y[xs], 1)
        ax.plot(xs, np.polyval(coef, xs), color="orange", linewidth=2,
                 label="Train trend" if not train_label_done else None, zorder=2)
        train_label_done = True

    test_label_done = False
    for start, end in _contiguous_runs(test_idx):
        xs = np.arange(start, end + 1)
        if len(xs) < 2:
            continue
        coef = np.polyfit(xs, y[xs], 1)
        ax.plot(xs, np.polyval(coef, xs), color="red", linewidth=2,
                 label="Test trend" if not test_label_done else None, zorder=3)
        test_label_done = True

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Target Value")
    ax.legend(loc="upper right", fontsize=8)
    return ax


if __name__ == "__main__":
    df = pd.read_csv("combined_data6_tel12_t_1.csv")
    y = df["SIT_1(t)"].values
    n = len(y)

    train_idx, test_idx, gap_idx = block_split(
        n, block_size=50, test_fraction=0.2, gap=10
    )

    np.save("train_idx.npy", train_idx)
    np.save("test_idx.npy", test_idx)
    np.save("gap_idx.npy", gap_idx)
    print(f"Split saved: train={len(train_idx)} ({len(train_idx)/n:.1%}), "
          f"test={len(test_idx)} ({len(test_idx)/n:.1%}), "
          f"gap={len(gap_idx)} ({len(gap_idx)/n:.1%})")

    fig, ax = plt.subplots(figsize=(13, 4))
    plot_split_with_trends(
        y, train_idx, test_idx,
        title=(f"Split: block_size=50, test_fraction=0.2, gap=10 | "
               f"train={len(train_idx)} test={len(test_idx)} gap={len(gap_idx)}"),
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig("block_split_final.png", dpi=120)
    print("Plot saved in block_split_final.png")
