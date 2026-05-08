"""Neural anti-spoofing model definitions."""

from __future__ import annotations

import torch
from torch import nn

from src.neural.aasist import AASISTLite


class MaxFeatureMap2d(nn.Module):
    """Max-Feature-Map activation used by LCNN-style models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError("MaxFeatureMap2d requires an even channel count")
        a, b = torch.chunk(x, 2, dim=1)
        return torch.maximum(a, b)


class MFMConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1, bias=False),
            MaxFeatureMap2d(),
            nn.BatchNorm2d(out_channels),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LogMelLCNN(nn.Module):
    """Compact LCNN-style countermeasure for log-mel spectrogram inputs."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            MFMConvBlock(1, 32, pool=True),
            MFMConvBlock(32, 64, pool=True),
            MFMConvBlock(64, 96, pool=True),
            MFMConvBlock(96, 128, pool=True),
            MFMConvBlock(128, 128, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


def build_model(model_type: str, dropout: float = 0.3, **kwargs) -> nn.Module:
    if model_type == "lcnn":
        return LogMelLCNN(dropout=dropout)
    if model_type == "aasist_lite":
        return AASISTLite(dropout=dropout, **kwargs)
    raise ValueError(f"unsupported neural model type: {model_type}")
