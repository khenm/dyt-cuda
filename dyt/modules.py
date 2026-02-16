"""
Torch implementation of Dynamic Tanh
Copyright (c) Meta Platforms, Inc. and affiliates.
Modifications Copyright (c) 2026 Khang
"""

import torch
import torch.nn as nn
from .functional import DyTFunctionCuda, DyTFunctionTriton

class DyT(nn.Module):
    def __init__(self, num_features: int, alpha_init_value: float = 0.5, backend: str = "torch"):
        super().__init__()
        self.num_features = num_features
        self.backend = backend.lower()

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "cuda":
            return DyTFunctionCuda.apply(x, self.alpha, self.weight, self.bias)
        elif self.backend == "triton":
            return DyTFunctionTriton.apply(x, self.alpha, self.weight, self.bias)
        elif self.backend == "torch":
            return torch.tanh(self.alpha * x) * self.weight + self.bias
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Option: 'cuda', 'triton', 'torch'.")
