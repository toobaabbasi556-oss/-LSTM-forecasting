"""Evaluation: uniform prediction, metrics table, and the three required plots.

Every model — baseline or neural — is scored by the exact same code here, so any
difference in the results table is attributable to the model, not the measurement.
All metrics are computed on inverse-scaled (real-unit) test values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import all_metrics, error_by_horizon

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def predict_scaled(model, X: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Return SCALED predictions [N, H] for any model.

    Neural models (nn.Module) are run batched under no_grad; baselines expose a
    numpy `predict`. This single dispatch is why train/eval treat every model
    identically.
    """
    try:
        import torch
        import torch.nn as nn

        is_module = isinstance(model, nn.Module)
    except ImportError:  # torch absent (baseline-only runs)
        is_module = False

    if is_module:
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size])).to(device)
                outputs.append(model(xb).cpu().numpy())
        return np.concatenate(outputs, axis=0)
    return model.predict(X)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_train_real: np.ndarray, season: int
) -> dict[str, float]:
    """Wrapper so callers don't import metrics directly."""
    return all_metrics(y_true, y_pred, y_train_real, season)


def build_results_table(results: dict[str, dict[str, float]], out_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    """Assemble a per-model metrics table, sorted by MASE (best first), and write markdown."""
    df = pd.DataFrame(results).T  # rows = models, cols = metrics
    df = df[["MASE", "sMAPE", "RMSE", "MAE"]].sort_values("MASE")
    df.index.name = "model"

    out_dir.mkdir(parents=True, exist_ok=True)
    md = df.round(4).to_markdown()
    (out_dir / "results.md").write_text(md + "\n")
    df.round(6).to_csv(out_dir / "results.csv")
    return df


def save_predictions(name: str, y_pred: np.ndarray, out_dir: Path = REPORTS_DIR) -> None:
    """Persist a model's real-unit test predictions [N, H] to disk."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{name}.npy", y_pred.astype(np.float32))


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path = REPORTS_DIR,
    n_windows: int = 3,
) -> None:
    """(a) Predicted vs actual for a few evenly-spaced test windows."""
    H = y_true.shape[1]
    idxs = np.linspace(0, len(y_true) - 1, n_windows, dtype=int)
    steps = np.arange(1, H + 1)

    fig, axes = plt.subplots(n_windows, 1, figsize=(9, 3 * n_windows), squeeze=False)
    for ax, i in zip(axes[:, 0], idxs):
        ax.plot(steps, y_true[i], "k-o", linewidth=2, markersize=3, label="actual")
        for name, y_pred in predictions.items():
            ax.plot(steps, y_pred[i], "--", alpha=0.8, label=name)
        ax.set_title(f"Test window #{i}")
        ax.set_xlabel("forecast step h")
        ax.set_ylabel("target (real units)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "predictions_vs_actual.png", dpi=120)
    plt.close(fig)


def plot_error_by_horizon(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path = REPORTS_DIR,
) -> None:
    """(b) Mean absolute error at each horizon h — does accuracy decay with h?"""
    H = y_true.shape[1]
    steps = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, y_pred in predictions.items():
        ax.plot(steps, error_by_horizon(y_true, y_pred), "-o", markersize=3, label=name)
    ax.set_title("Error by forecast horizon")
    ax.set_xlabel("forecast step h (1 = nearest, H = farthest)")
    ax.set_ylabel("mean absolute error (real units)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_horizon.png", dpi=120)
    plt.close(fig)


def plot_metric_bars(results: dict[str, dict[str, float]], out_dir: Path = REPORTS_DIR) -> None:
    """(c) One bar panel per metric, models on the x-axis."""
    metrics = ["MASE", "sMAPE", "RMSE", "MAE"]
    names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        values = [results[n][metric] for n in names]
        bars = ax.bar(names, values, color="#4C72B0")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        if metric == "MASE":
            # Reference line: MASE = 1 is the "beats seasonal-naive" threshold.
            ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="MASE = 1")
            ax.legend(fontsize=8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Metrics across models (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=120)
    plt.close(fig)
