"""End-to-end benchmark: `python main.py` runs every model and regenerates reports/.

One data path, many models (see SPEC.md §4). We load + split + scale once, build
identical windows for every model, train the neural ones, score all of them with
the same evaluation code, and emit the table + plots into reports/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data import prepare_data
from src.evaluate import (
    build_results_table,
    compute_metrics,
    plot_error_by_horizon,
    plot_metric_bars,
    plot_predictions,
    predict_scaled,
    save_predictions,
)
from src.models.dlinear import DLinear
from src.models.lstm import LSTMForecaster
from src.models.naive import NaiveForecaster
from src.models.seasonal_naive import SeasonalNaiveForecaster
from src.train import set_seed, train_model
from src.windowing import make_windows, target_index


def build_model(name: str, *, C: int, L: int, H: int, season: int, target_idx: int, tcfg: dict):
    """Instantiate a model by name using hyperparameters from config."""
    if name == "naive":
        return NaiveForecaster(target_idx, H)
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(target_idx, H, season)
    if name == "dlinear":
        return DLinear(L, H, C, target_idx, kernel_size=tcfg["kernel_size"])
    if name == "lstm":
        return LSTMForecaster(
            C, H,
            hidden_size=tcfg["hidden_size"],
            num_layers=tcfg["num_layers"],
            dropout=tcfg["dropout"],
        )
    raise ValueError(f"Unknown model '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-series forecasting benchmark.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    set_seed(config["train"]["seed"])

    L = config["window"]["L"]
    H = config["window"]["H"]
    season = config["window"]["season"]

    # --- One data path: load, split chronologically, scale on train only ---
    print("Loading and preparing data...")
    data = prepare_data(config)
    C = data.train.shape[1]
    target_idx = target_index(data.train, data.target)
    print(
        f"  source={config['data']['source']}  target={data.target}  channels={C}\n"
        f"  rows: train={len(data.train)} val={len(data.val)} test={len(data.test)}"
    )

    # Identical windows for every model.
    X_train, Y_train = make_windows(data.train, data.target, L, H)
    X_val, Y_val = make_windows(data.val, data.target, L, H)
    X_test, Y_test = make_windows(data.test, data.target, L, H)
    print(f"  windows: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Real-unit series/arrays used by MASE and the plots.
    y_train_real = data.raw_train[data.target].to_numpy()
    y_test_real = data.scaler.inverse_transform_target(Y_test, data.target)

    results: dict[str, dict[str, float]] = {}
    predictions_real: dict[str, np.ndarray] = {}

    for name in config["models"]:
        print(f"\n=== {name} ===")
        model = build_model(
            name, C=C, L=L, H=H, season=season, target_idx=target_idx, tcfg=config["train"]
        )
        if getattr(model, "requires_training", False):
            model = train_model(
                model, X_train, Y_train, X_val, Y_val,
                data.scaler, data.target, y_train_real, config, device=args.device,
            )

        y_pred_scaled = predict_scaled(model, X_test, device=args.device)
        y_pred_real = data.scaler.inverse_transform_target(y_pred_scaled, data.target)

        metrics = compute_metrics(y_test_real, y_pred_real, y_train_real, season)
        results[name] = metrics
        predictions_real[name] = y_pred_real
        save_predictions(name, y_pred_real)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # --- Reports: table + three plots ---
    print("\nWriting reports/ ...")
    table = build_results_table(results)
    plot_predictions(y_test_real, predictions_real)
    plot_error_by_horizon(y_test_real, predictions_real)
    plot_metric_bars(results)

    print("\nResults (ranked by MASE):\n")
    print(table.round(4).to_markdown())
    best = table.index[0]
    print(f"\nBest model by MASE: {best} (MASE={table.loc[best, 'MASE']:.4f})")


if __name__ == "__main__":
    main()
