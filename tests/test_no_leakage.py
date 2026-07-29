"""Leakage guardrail tests (SPEC.md §3, acceptance criteria).

Two failure modes most CV forecasting projects fall into, encoded as tests:
  1. The scaler sees val/test data (statistics leak backward in time).
  2. A sliding window straddles a split boundary (future leaks into a window).
"""

import numpy as np
import pytest
import yaml

from src.data import prepare_data
from src.windowing import make_windows

CONFIG = yaml.safe_load(open("config.yaml"))


@pytest.fixture(scope="module")
def data():
    return prepare_data(CONFIG)


def test_scaler_fit_on_train_only(data):
    """scaler.mean_/std_ must equal TRAIN statistics exactly..."""
    assert np.allclose(data.scaler.mean_, data.raw_train.mean().to_numpy())
    assert np.allclose(data.scaler.std_, data.raw_train.std(ddof=0).to_numpy())


def test_scaler_excludes_val_test(data):
    """...and must NOT equal the full-series statistics.

    If val/test had leaked into the fit, the scaler mean would match the mean over
    all splits. It must not — that mismatch is the proof of exclusion.
    """
    full = np.concatenate(
        [data.raw_train.to_numpy(), data.raw_val.to_numpy(), data.raw_test.to_numpy()],
        axis=0,
    )
    full_mean = full.mean(axis=0)
    assert not np.allclose(data.scaler.mean_, full_mean), "Scaler statistics leaked from val/test!"


def test_transform_uses_train_stats(data):
    """Val/test are transformed with TRAIN mean/std, not their own."""
    expected = (data.raw_val.to_numpy() - data.scaler.mean_) / data.scaler.std_
    assert np.allclose(data.val.to_numpy(), expected)


def test_windows_never_cross_split_boundary(data):
    """Per-split windowing must drop exactly the boundary-straddling windows.

    Building windows on each split separately yields fewer windows than building
    them over the concatenated series — and the difference must equal exactly the
    number of windows that WOULD have straddled the two internal boundaries,
    2*(L+H-1). That exact accounting proves no window crosses a boundary.
    """
    L, H = CONFIG["window"]["L"], CONFIG["window"]["H"]
    target = data.target

    per_split = sum(
        len(make_windows(split, target, L, H)[0])
        for split in (data.train, data.val, data.test)
    )

    import pandas as pd

    full = pd.concat([data.train, data.val, data.test], axis=0)
    full_windows = len(make_windows(full, target, L, H)[0])

    assert full_windows - per_split == 2 * (L + H - 1)
