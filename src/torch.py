"""
Torch implementation of Dynamic Tanh
Copyright (c) Meta Platforms, Inc. and affiliates.
Modifications Copyright (c) 2026 Khang
"""

import torch
import torch.nn as nn
from .function import DyTFunction

class DyT(nn.Module):
    def __init__(self, num_features: int, alpha_init_value: float = 0.5):
        super().__init__()
        self.num_features = num_features
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DyTFunction.apply(x, self.alpha, self.weight, self.bias)
