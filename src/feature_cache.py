"""On-disk cache for frame-level audio features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from src.dataset import Sample
from src.features import SR


FEATURE_CACHE_VERSION = 1
FrameFn = Callable[[np.ndarray, int], np.ndarray]


class FeatureCache:
    """Cache one feature matrix per utterance and feature configuration."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def load_or_extract(
        self,
        sample: Sample,
        feature_name: str,
        feature_fn: FrameFn,
    ) -> np.ndarray:
        path = self.path_for(sample, feature_name)
        if path.exists():
            return np.load(path).astype(np.float32, copy=False)

        audio, sr = librosa.load(sample.path, sr=SR, mono=True)
        frames = feature_fn(audio, sr)
        if frames.ndim != 2 or frames.shape[0] == 0:
            raise ValueError(f"empty frame matrix for {sample.file_id}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp.npy")
        np.save(tmp_path, frames.astype(np.float32, copy=False))
        os.replace(tmp_path, path)
        self._write_metadata(sample, feature_name, path)
        return frames.astype(np.float32, copy=False)

    def path_for(self, sample: Sample, feature_name: str) -> Path:
        stat = os.stat(sample.path)
        key_payload = {
            "version": FEATURE_CACHE_VERSION,
            "feature": feature_name,
            "file_id": sample.file_id,
            "path": os.path.abspath(sample.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        key = hashlib.sha1(
            json.dumps(key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.root / f"v{FEATURE_CACHE_VERSION}" / feature_name / f"{sample.file_id}_{key}.npy"

    def _write_metadata(self, sample: Sample, feature_name: str, feature_path: Path) -> None:
        metadata_path = feature_path.with_suffix(".json")
        if metadata_path.exists():
            return
        payload = {
            "cache_version": FEATURE_CACHE_VERSION,
            "feature": feature_name,
            "file_id": sample.file_id,
            "audio_path": os.path.abspath(sample.path),
            "label": sample.label,
            "system_id": sample.system_id,
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
