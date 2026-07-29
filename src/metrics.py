"""Forecast metrics, implemented from scratch (no sklearn/statsmodels).

All metrics operate on arrays of real-unit (inverse-scaled) values with shape
[N, H] — N test windows, each with an H-step forecast — or any broadcastable
shape; they reduce over every element unless noted. Writing MASE by hand is
deliberate: it is the headline metric and getting its scale factor right (the
in-sample *seasonal*-naive error on TRAIN) is the whole point.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (as a percentage, 0..200).

    sMAPE = mean( 2*|y - yhat| / (|y| + |yhat|) ) * 100

    The denominator can be zero when both values are ~0; we guard those elements
    (contribute 0 error) rather than emit NaN/inf.
    """
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0.0, 0.0, numerator / denominator)
    return float(np.mean(ratio) * 100.0)


def seasonal_naive_scale(y_train: np.ndarray, season: int) -> float:
    """In-sample scale factor for MASE: mean |y[t] - y[t-season]| over TRAIN.

    This is the denominator of MASE (Hyndman & Koehler). Using the seasonal-naive
    in-sample error makes MASE scale-free and interpretable: MASE < 1 means the
    model beats a seasonal-naive one-step forecast on the training distribution.

    `y_train` is the 1-D training target series in REAL units.
    """
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    if y_train.size <= season:
        raise ValueError(
            f"Training series length {y_train.size} must exceed season {season} "
            "to compute the MASE scale factor."
        )
    diffs = np.abs(y_train[season:] - y_train[:-season])
    scale = float(np.mean(diffs))
    if scale == 0.0:
        # Degenerate constant series; avoid divide-by-zero downstream.
        raise ValueError("MASE scale factor is zero (constant training series?).")
    return scale


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int,
) -> float:
    """Mean Absolute Scaled Error.

    MASE = MAE(test) / scale, where scale is the in-sample seasonal-naive MAE on
    TRAIN. Note the scale is computed from training data ONLY — it must never see
    the test set, or the metric leaks.
    """
    scale = seasonal_naive_scale(y_train, season)
    return mae(y_true, y_pred) / scale


def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int,
) -> dict[str, float]:
    """Convenience bundle returning every metric as a dict (for the results table)."""
    return {
        "MASE": mase(y_true, y_pred, y_train, season),
        "sMAPE": smape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean absolute error at each forecast step h = 1..H.

    Inputs are [N, H]; returns a length-H vector. Used for the error-by-horizon
    plot: a healthy multi-horizon model should show error growing with h.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.ndim == 2 and y_true.shape == y_pred.shape, "Expect matching [N, H]."
    return np.mean(np.abs(y_true - y_pred), axis=0)
