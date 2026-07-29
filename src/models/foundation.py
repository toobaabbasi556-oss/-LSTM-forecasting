"""Stub for a zero-shot foundation-model comparator (e.g. Amazon Chronos).

This is intentionally NOT wired to run — it documents the interface a zero-shot
forecaster would implement so it can be dropped into the same evaluation harness
later. The point of the stub is to prove the pipeline is model-agnostic: adding a
5th comparator means implementing `predict` and registering it in main.py, nothing
else changes.

To make it real later:
    pip install chronos-forecasting
    load a pretrained pipeline (e.g. "amazon/chronos-t5-small"), and for each
    window feed the target history X[:, :, target_idx] to `predict`, taking the
    median of the sampled trajectories as the point forecast.
"""

from __future__ import annotations

import numpy as np


class FoundationForecaster:
    name = "chronos_stub"
    requires_training = False

    def __init__(self, target_idx: int, H: int, model_name: str = "amazon/chronos-t5-small") -> None:
        self.target_idx = target_idx
        self.H = H
        self.model_name = model_name

    def predict(self, X: np.ndarray) -> np.ndarray:  # [N, L, C] -> [N, H]
        raise NotImplementedError(
            "FoundationForecaster is a stub. Install `chronos-forecasting`, load "
            f"'{self.model_name}', and implement zero-shot prediction over "
            "X[:, :, target_idx]. See the module docstring."
        )
