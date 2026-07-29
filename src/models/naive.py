"""Naive persistence baseline: y_hat(t+h) = y(t) for all h.

The bar every other model must clear. It has no parameters and no training. A
model that cannot beat persistence is not worth its complexity — checking this is
the entire reason the baseline exists.
"""

from __future__ import annotations

import numpy as np


class NaiveForecaster:
    name = "naive"
    requires_training = False

    def __init__(self, target_idx: int, H: int) -> None:
        self.target_idx = target_idx
        self.H = H

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: [N, L, C] -> Y_hat: [N, H]. Repeat the last observed target value."""
        last_value = X[:, -1, self.target_idx]          # [N]
        return np.repeat(last_value[:, None], self.H, axis=1)  # [N, H]
