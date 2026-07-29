"""Sliding-window dataset builder: turn a scaled series into (X[L,C], y[H]) pairs.

Windows are built *per split*, so a window never straddles a train/val/test
boundary — another piece of the leakage guardrail. The target is the last column
of the frame (guaranteed by data.load_raw), and only the target's future values
form y; X carries all C channels of the past window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame, target: str, L: int, H: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build all sliding windows from one contiguous frame.

    Returns:
        X: float32 array [N, L, C] — past L steps of every channel.
        Y: float32 array [N, H]    — next H steps of the target only.

    N = len(df) - L - H + 1. Raises if the frame is too short for one window.
    """
    values = df.to_numpy(dtype=np.float32)          # [T, C]
    target_idx = list(df.columns).index(target)
    T, C = values.shape
    n_windows = T - L - H + 1
    if n_windows <= 0:
        raise ValueError(
            f"Frame of length {T} too short for L={L}, H={H} (need >= {L + H})."
        )

    # Vectorized window extraction. Index arithmetic is explicit for clarity over
    # stride tricks: window i covers past rows [i, i+L) and future rows [i+L, i+L+H).
    X = np.empty((n_windows, L, C), dtype=np.float32)
    Y = np.empty((n_windows, H), dtype=np.float32)
    for i in range(n_windows):
        X[i] = values[i : i + L, :]
        Y[i] = values[i + L : i + L + H, target_idx]
    return X, Y


def target_index(df: pd.DataFrame, target: str) -> int:
    """Column position of the target channel within X's last axis."""
    return list(df.columns).index(target)


class WindowDataset:
    """torch Dataset over precomputed windows.

    torch is imported lazily so that `make_windows` (pure numpy) can be used —
    e.g. by the baseline sanity check — without requiring torch to be installed.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        import torch  # lazy: keeps numpy-only paths torch-free

        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.Y = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]
