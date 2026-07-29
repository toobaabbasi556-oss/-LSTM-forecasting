"""DLinear: decomposition + linear heads (Zeng et al., 2022).

"Are Transformers Effective for Time Series Forecasting?" showed a model this
simple often matches or beats heavy transformers. We decompose the input into a
trend (moving average) and a remainder, learn one Linear(L->H) per component, and
sum them. Channel-independent: the same linear weights are applied to every
channel, and we read out the target channel's H-step forecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving-average smoother that preserves sequence length via edge padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, C] -> [B, L, C]
        # Replicate the endpoints so the smoothed trend keeps length L (avoids the
        # shrinkage a naive pool would cause at the series edges).
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)          # [B, L+k-1, C]
        trend = self.avg(x_padded.transpose(1, 2)).transpose(1, 2)  # [B, L, C]
        return trend


class DLinear(nn.Module):
    name = "dlinear"
    requires_training = True

    def __init__(
        self, L: int, H: int, C: int, target_idx: int, kernel_size: int = 25
    ) -> None:
        super().__init__()
        self.target_idx = target_idx
        self.decomposition = MovingAvg(kernel_size)
        # Two independent linear maps over the time axis, shared across channels.
        self.linear_trend = nn.Linear(L, H)
        self.linear_seasonal = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, C] -> [B, H]
        trend = self.decomposition(x)
        seasonal = x - trend
        # Move time to the last axis so Linear(L->H) acts per channel: [B, C, L].
        trend_out = self.linear_trend(trend.transpose(1, 2))        # [B, C, H]
        seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2))  # [B, C, H]
        out = trend_out + seasonal_out                              # [B, C, H]
        return out[:, self.target_idx, :]                          # [B, H]
