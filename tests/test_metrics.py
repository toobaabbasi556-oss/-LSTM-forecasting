"""Unit tests for the from-scratch metrics — especially MASE's scale factor."""

import numpy as np
import pytest

from src.metrics import mae, mase, rmse, seasonal_naive_scale, smape


def test_perfect_forecast_is_zero_error():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_mae_and_rmse_known_values():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 2.0])  # abs errs 1,2,2 -> MAE=5/3; sq 1,4,4 -> RMSE=sqrt(3)
    assert mae(y_true, y_pred) == pytest.approx(5 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(3.0))


def test_smape_bounds_and_zero_guard():
    # Both zero -> guarded to 0 contribution, not NaN.
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0
    # Opposite signs / far apart approaches the 200% ceiling.
    assert smape(np.array([1.0]), np.array([-1.0])) == pytest.approx(200.0)


def test_seasonal_naive_scale_matches_manual():
    # season=1 -> mean absolute first difference.
    y = np.array([1.0, 3.0, 6.0, 10.0])  # diffs 2,3,4 -> mean 3
    assert seasonal_naive_scale(y, season=1) == pytest.approx(3.0)


def test_mase_equals_one_when_error_matches_scale():
    """If test MAE equals the in-sample scale, MASE must be exactly 1.0."""
    y_train = np.array([1.0, 3.0, 6.0, 10.0])  # scale (season=1) = 3
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 3.0])               # MAE = 3
    assert mase(y_true, y_pred, y_train, season=1) == pytest.approx(1.0)


def test_mase_rejects_short_training_series():
    with pytest.raises(ValueError):
        seasonal_naive_scale(np.array([1.0, 2.0]), season=24)
