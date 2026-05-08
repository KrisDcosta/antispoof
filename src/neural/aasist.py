"""AASIST-lite inspired raw-waveform graph-attention model."""

from __future__ import annotations

import torch
from torch import nn


class Residual2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride_time: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(1, stride_time),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        if in_channels == out_channels and stride_time == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, stride_time),
                bias=False,
            )
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.main(x) + self.skip(x))


class GraphAttention(nn.Module):
    """Lightweight node attention layer for homogeneous graph branches."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.input_drop = nn.Dropout(dropout)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        self.with_att = nn.Linear(in_dim, out_dim)
        self.without_att = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SELU(inplace=True)
        nn.init.xavier_normal_(self.att_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        pairwise = x.unsqueeze(2) * x.unsqueeze(1)
        att = torch.tanh(self.att_proj(pairwise)) @ self.att_weight
        att = torch.softmax(att.squeeze(-1) / self.temperature, dim=-2)
        out = self.with_att(torch.matmul(att, x)) + self.without_att(x)
        return self.act(self.norm(out))


class HeterogeneousGraphAttention(nn.Module):
    """Two-type graph attention with a master node for spectro-temporal fusion."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, temperature: float = 100.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.input_drop = nn.Dropout(dropout)
        self.proj_spectral = nn.Linear(in_dim, in_dim)
        self.proj_temporal = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        self.master_proj = nn.Linear(in_dim, out_dim)
        self.with_att = nn.Linear(in_dim, out_dim)
        self.without_att = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SELU(inplace=True)
        nn.init.xavier_normal_(self.att_weight)

    def forward(
        self,
        spectral: torch.Tensor,
        temporal: torch.Tensor,
        master: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectral = self.proj_spectral(spectral)
        temporal = self.proj_temporal(temporal)
        num_spectral = spectral.shape[1]
        x = torch.cat([spectral, temporal], dim=1)
        if master is None:
            master = x.mean(dim=1, keepdim=True)

        x = self.input_drop(x)
        pairwise = x.unsqueeze(2) * x.unsqueeze(1)
        att = torch.tanh(self.att_proj(pairwise)) @ self.att_weight
        att = torch.softmax(att.squeeze(-1) / self.temperature, dim=-2)
        out = self.with_att(torch.matmul(att, x)) + self.without_att(x)
        out = self.act(self.norm(out))

        master_context = torch.softmax((x * master).mean(dim=-1, keepdim=True), dim=1)
        master = self.act(self.master_proj((master_context * x).sum(dim=1, keepdim=True)))
        return out[:, :num_spectral], out[:, num_spectral:], master


class AASISTLite(nn.Module):
    """Compact AASIST-inspired countermeasure using raw waveform input."""

    def __init__(
        self,
        *,
        num_samples: int = 64_600,
        first_conv: int = 64,
        encoder_channels: tuple[int, int, int] = (16, 32, 64),
        graph_dim: int = 64,
        graph_hidden: int = 32,
        spectral_nodes: int = 16,
        temporal_nodes: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.spectral_nodes = spectral_nodes
        self.temporal_nodes = temporal_nodes
        c1, c2, c3 = encoder_channels
        self.frontend = nn.Sequential(
            nn.Conv1d(1, first_conv, kernel_size=128, stride=64, padding=32, bias=False),
            nn.BatchNorm1d(first_conv),
            nn.SELU(inplace=True),
        )
        self.encoder = nn.Sequential(
            Residual2dBlock(1, c1, stride_time=1),
            Residual2dBlock(c1, c2, stride_time=2),
            Residual2dBlock(c2, c3, stride_time=2),
        )
        self.node_proj = nn.Linear(c3, graph_dim)
        self.spectral_pos = nn.Parameter(torch.zeros(1, spectral_nodes, graph_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, temporal_nodes, graph_dim))
        self.spectral_gat = GraphAttention(graph_dim, graph_dim, dropout=dropout, temperature=2.0)
        self.temporal_gat = GraphAttention(graph_dim, graph_dim, dropout=dropout, temperature=2.0)
        self.fusion_1 = HeterogeneousGraphAttention(graph_dim, graph_hidden, dropout=dropout)
        self.fusion_2 = HeterogeneousGraphAttention(graph_hidden, graph_hidden, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_hidden * 5, 2),
        )
        nn.init.trunc_normal_(self.spectral_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"AASISTLite expects [batch, samples] or [batch, 1, samples], got {tuple(x.shape)}")

        x = self.frontend(x).unsqueeze(1)
        x = self.encoder(x)

        spectral = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (self.spectral_nodes, 1),
        ).squeeze(-1).transpose(1, 2)
        temporal = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (1, self.temporal_nodes),
        ).squeeze(2).transpose(1, 2)

        spectral = self.node_proj(spectral) + self.spectral_pos
        temporal = self.node_proj(temporal) + self.temporal_pos
        spectral = self.spectral_gat(spectral)
        temporal = self.temporal_gat(temporal)
        spectral, temporal, master = self.fusion_1(spectral, temporal)
        spectral, temporal, master = self.fusion_2(spectral, temporal, master)

        readout = torch.cat(
            [
                spectral.max(dim=1).values,
                spectral.mean(dim=1),
                temporal.max(dim=1).values,
                temporal.mean(dim=1),
                master.squeeze(1),
            ],
            dim=1,
        )
        return self.classifier(readout)
