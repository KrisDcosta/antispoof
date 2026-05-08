"""PyTorch datasets for ASVspoof neural experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio.functional as AF

from src.dataset import Sample, load_split
from src.neural.transforms import crop_or_pad


@dataclass(frozen=True)
class NeuralSample:
    file_id: str
    path: str
    label: int
    system_id: str
    x: torch.Tensor


class ASVspoofSpectrogramDataset(Dataset):
    """Dataset that loads audio and returns fixed-size spectrogram features."""

    def __init__(
        self,
        samples: list[Sample],
        transform: Callable[[torch.Tensor], torch.Tensor],
        sample_rate: int,
        clip_seconds: float,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_samples = int(round(sample_rate * clip_seconds))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        audio, sr = sf.read(sample.path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(np.asarray(audio))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)
        waveform = crop_or_pad(waveform, self.target_samples)
        x = self.transform(waveform)
        return {
            "x": x,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "file_id": sample.file_id,
            "path": sample.path,
            "system_id": sample.system_id,
        }


class ASVspoofWaveformDataset(Dataset):
    """Dataset that loads audio and returns fixed-size raw waveforms."""

    def __init__(
        self,
        samples: list[Sample],
        sample_rate: int,
        num_samples: int,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        audio, sr = sf.read(sample.path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(np.asarray(audio))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)
        waveform = crop_or_pad(waveform, self.num_samples)
        return {
            "x": waveform.to(torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "file_id": sample.file_id,
            "path": sample.path,
            "system_id": sample.system_id,
        }


def load_limited_split(
    data_root: str,
    split: str,
    limit: int | None,
    seed: int = 42,
) -> list[Sample]:
    samples = load_split(data_root, split, limit=None)
    if limit is None or limit <= 0 or limit >= len(samples):
        return samples
    return class_balanced_limit(samples, limit, seed)


def class_balanced_limit(samples: list[Sample], limit: int, seed: int) -> list[Sample]:
    rng = np.random.default_rng(seed)
    by_label = {
        1: [sample for sample in samples if sample.label == 1],
        0: [sample for sample in samples if sample.label == 0],
    }
    if not by_label[1] or not by_label[0]:
        selected = rng.choice(samples, size=min(limit, len(samples)), replace=False).tolist()
        return selected

    per_class = limit // 2
    selected: list[Sample] = []
    for label in [1, 0]:
        take = min(per_class, len(by_label[label]))
        selected.extend(rng.choice(by_label[label], size=take, replace=False).tolist())
    remainder = limit - len(selected)
    if remainder > 0:
        selected_ids = {sample.file_id for sample in selected}
        remaining = [sample for sample in samples if sample.file_id not in selected_ids]
        if remaining:
            selected.extend(rng.choice(remaining, size=min(remainder, len(remaining)), replace=False).tolist())
    rng.shuffle(selected)
    return selected
