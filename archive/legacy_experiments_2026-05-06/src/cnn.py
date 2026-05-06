"""
Fix G: Lightweight CNN baseline for anti-spoofing.

Input: log-mel spectrogram [1 x 64 x T]
Architecture: 4 conv blocks → global average pool → FC → sigmoid
Comparable in complexity to classical MFCC+RF but end-to-end trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class AntiSpoofCNN(nn.Module):
    """
    Small CNN anti-spoof detector.
    Input shape: (batch, 1, n_mels, time_frames)
    Output: raw logit — apply sigmoid for probability, BCEWithLogitsLoss for training.
    """
    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1,  16, kernel=3, pool=2),
            ConvBlock(16, 32, kernel=3, pool=2),
            ConvBlock(32, 64, kernel=3, pool=2),
            ConvBlock(64, 128, kernel=3, pool=2),
        )
        # freq dim after 4 pools: n_mels // 16
        freq_out = max(1, n_mels // 16)
        self.fc = nn.Linear(128 * freq_out, 1)
        self._freq_out = freq_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.blocks(x)              # (B, 128, freq_out, T//16)
        x = x.mean(dim=-1)             # global avg pool over time → (B, 128, freq_out)
        x = x.flatten(start_dim=1)     # (B, 128 * freq_out)
        return self.fc(x).squeeze(-1)  # (B,)


def build_model(n_mels: int = 64) -> AntiSpoofCNN:
    return AntiSpoofCNN(n_mels=n_mels)
