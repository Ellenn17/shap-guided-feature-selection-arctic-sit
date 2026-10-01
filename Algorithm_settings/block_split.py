# -*- coding: utf-8 -*-
"""
Split a blocchi alternati con gap/buffer, per confrontare in modo equo
feature selection (tutte le features vs features selezionate) su una
serie con trend non stazionario.

Schema:
- la serie viene divisa in blocchi contigui di lunghezza `block_size`
- ogni blocco e' assegnato a train o test alternando, secondo `test_fraction`
- un gap di `gap` punti viene scartato (non usato in train ne' in test)
  ad ogni confine train/test, per evitare leakage da autocorrelazione locale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def block_split(n, block_size=80, test_fraction=0.2, gap=10, seed=None):
    """
    Genera indici train/test a blocchi alternati con buffer.

    Parameters
    ----------
    n : int
        Numero totale di timestep.
    block_size : int
        Lunghezza di ciascun blocco contiguo.
    test_fraction : float
        Frazione approssimativa di blocchi da assegnare al test.
    gap : int
        Numero di punti da scartare ad ogni confine train/test.
    seed : int o None
        Se fornito, mescola l'assegnazione dei blocchi in modo riproducibile
        invece di alternarli in modo regolare (1 ogni K).

    Returns
    -------
    train_idx, test_idx, gap_idx : np.ndarray
        Indici (0-based) assegnati rispettivamente a train, test e gap.
    """
    n_blocks = int(np.ceil(n / block_size))
    block_bounds = [(i * block_size, min((i + 1) * block_size, n)) for i in range(n_blocks)]

    step = max(1, round(1 / test_fraction))
    is_test_block = np.zeros(n_blocks, dtype=bool)
    if seed is None:
        # ogni K-esimo blocco va in test, partendo da un offset centrale
        # per evitare che il primo o l'ultimo blocco (spesso piu' instabili)
        # finiscano sempre in test
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

    # applica il gap: rimuove da train i punti entro `gap` da un confine con un blocco di test
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    gap_set = set()

    for i, (start, end) in enumerate(block_bounds):
        if not is_test_block[i]:
            continue
        # buffer a sinistra del blocco di test
        for p in range(max(0, start - gap), start):
            if p in train_set:
                gap_set.add(p)
        # buffer a destra del blocco di test
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
    """Spezza un array di indici (eventualmente non contiguo) in run contigui."""
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
    """Plot in stile 'Time Series with Segmented Trends': serie blu + rette di
    regressione (arancio per train, rosso per test) su ciascun blocco contiguo."""
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

    # Split definitivo: block_size=50, test_fraction=0.2, gap=10
    # (scelto confrontando block_size=50 vs 80 sui trend locali, vedi
    # block_split_comparison.png)
    train_idx, test_idx, gap_idx = block_split(
        n, block_size=50, test_fraction=0.2, gap=10
    )

    np.save("train_idx.npy", train_idx)
    np.save("test_idx.npy", test_idx)
    np.save("gap_idx.npy", gap_idx)
    print(f"Split definitivo salvato: train={len(train_idx)} ({len(train_idx)/n:.1%}), "
          f"test={len(test_idx)} ({len(test_idx)/n:.1%}), "
          f"gap={len(gap_idx)} ({len(gap_idx)/n:.1%})")

    fig, ax = plt.subplots(figsize=(13, 4))
    plot_split_with_trends(
        y, train_idx, test_idx,
        title=(f"Split definitivo: block_size=50, test_fraction=0.2, gap=10 | "
               f"train={len(train_idx)} test={len(test_idx)} gap={len(gap_idx)}"),
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig("block_split_final.png", dpi=120)
    print("Plot salvato in block_split_final.png")
