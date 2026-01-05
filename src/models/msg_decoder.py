from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MessageDecoder(nn.Module):
    """Lightweight CNN head that predicts codeword bits and a confidence score."""

    def __init__(self, code_bits: int, base_channels: int = 64) -> None:
        super().__init__()
        self.code_bits = code_bits
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(base_channels * 4, code_bits)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        pooled = F.adaptive_avg_pool2d(feats, output_size=1).flatten(1)
        logits = self.head(pooled)
        confidence = torch.sigmoid(logits.abs().mean(dim=1, keepdim=True))
        return logits, confidence


__all__ = ["MessageDecoder"]
