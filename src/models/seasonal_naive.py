"""Seasonal-naive baseline: y_hat(t+h) = y(t + h - season).

For hourly data with season=24 this predicts "same hour yesterday". It is often
surprisingly hard to beat on strongly cyclic series, which is exactly why it is a
fair, informative comparator. No parameters, no training.
"""

from __future__ import annotations

import numpy as np


class SeasonalNaiveForecaster:
    name = "seasonal_naive"
    requires_training = False

    def __init__(self, target_idx: int, H: int, season: int) -> None:
        self.target_idx = target_idx
        self.H = H
        self.season = season

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: [N, L, C] -> Y_hat: [N, H].

        Take the last `season` observed target values and repeat them cyclically to
        length H (h % season). For H <= season this is exactly "value one season
        ago"; for H > season it tiles the last seasonal cycle, the standard
        seasonal-naive multi-step behavior.
        """
        L = X.shape[1]
        if L < self.season:
            raise ValueError(f"Input window L={L} shorter than season={self.season}.")
        last_block = X[:, L - self.season : L, self.target_idx]  # [N, season]
        idx = np.arange(self.H) % self.season                    # [H]
        return last_block[:, idx]                                # [N, H]
