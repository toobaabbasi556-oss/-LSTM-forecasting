"""Stacked LSTM encoder with a direct multi-horizon linear head.

The LSTM encodes the L-step multivariate window; the final layer's last hidden
state is mapped by a single Linear to all H future steps AT ONCE (direct
multi-horizon, not autoregressive). Direct output avoids the compounding error of
feeding predictions back in, and keeps the comparison against DLinear apples-to-
apples. Unlike channel-independent DLinear, this model does mix channels, so it
can genuinely exploit the covariates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    name = "lstm"
    requires_training = True

    def __init__(
        self,
        C: int,
        H: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # nn.LSTM only applies dropout BETWEEN layers, so it is a no-op when
            # num_layers == 1; guard to silence the framework warning.
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, C] -> [B, H]
        # h_n: [num_layers, B, hidden]; take the top layer's final hidden state.
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]              # [B, hidden]
        return self.head(last_hidden)      # [B, H]
