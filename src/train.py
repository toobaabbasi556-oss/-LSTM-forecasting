"""Training loop: Adam + MSE, early stopping and checkpointing on validation MASE.

We early-stop on validation MASE rather than raw MSE loss because MASE is the
metric we actually care about (scale-free, relative to naive). Optimizing MSE but
selecting on MASE keeps training stable while making model selection honest about
the real objective.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run, and force deterministic cuDNN."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _val_mase(
    model,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    scaler,
    target: str,
    y_train_real: np.ndarray,
    season: int,
    device: str,
) -> float:
    """MASE on the validation set in real units — the early-stopping signal."""
    from src.evaluate import predict_scaled
    from src.metrics import mase

    y_pred = scaler.inverse_transform_target(
        predict_scaled(model, X_val, device=device), target
    )
    y_true = scaler.inverse_transform_target(Y_val, target)
    return mase(y_true, y_pred, y_train_real, season)


def train_model(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    scaler,
    target: str,
    y_train_real: np.ndarray,
    config: dict,
    device: str = "cpu",
):
    """Train `model` in place; return it with best-validation-MASE weights loaded.

    Best weights (lowest val MASE) are checkpointed to disk and restored at the
    end, so the returned model is the selected one, not the last-epoch one.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tcfg = config["train"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=tcfg["batch_size"],
        shuffle=True,  # shuffling WINDOWS is fine; the chronological split already
        # happened, so no future data leaks into training.
        drop_last=False,
    )

    best_mase = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = tcfg["patience"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{model.name}.pt"

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_mase = _val_mase(
            model, X_val, Y_val, scaler, target, y_train_real,
            config["window"]["season"], device,
        )

        if val_mase < best_mase - 1e-6:
            best_mase = val_mase
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            patience_left = tcfg["patience"]
        else:
            patience_left -= 1

        print(
            f"[{model.name}] epoch {epoch:3d}/{tcfg['epochs']}  "
            f"val_MASE={val_mase:.4f}  best={best_mase:.4f}  patience={patience_left}"
        )

        if patience_left <= 0:
            print(f"[{model.name}] early stopping at epoch {epoch} (best MASE {best_mase:.4f}).")
            break

    model.load_state_dict(best_state)
    return model
