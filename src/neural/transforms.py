"""Waveform and spectrogram transforms for neural countermeasures."""

from __future__ import annotations

import torch
import torchaudio


def crop_or_pad(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Return a mono waveform with exactly target_samples samples."""
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.flatten()
    if waveform.numel() > target_samples:
        return waveform[:target_samples]
    if waveform.numel() < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform


class LogMelTransform(torch.nn.Module):
    """Fixed log-mel front end returning a [1, n_mels, frames] tensor."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max or sample_rate / 2,
            n_mels=n_mels,
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.to_db(self.mel(waveform))
        spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)
        return spec.to(torch.float32)

