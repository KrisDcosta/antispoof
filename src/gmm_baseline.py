"""
Frame-level GMM log-likelihood-ratio baseline for ASVspoof countermeasures.

The model follows the classical anti-spoofing setup:
  - fit one GMM on bonafide frames
  - fit one GMM on spoof frames
  - score each utterance as mean log p(x | bonafide) - mean log p(x | spoof)

Scores are higher for bonafide speech and can be passed directly to EER
calculation with label 1 = bonafide, 0 = spoof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.dataset import Sample
from src.feature_cache import FeatureCache
from src.features import SR, cqcc_frames, lfcc_frames, mfcc_frames, wcqcc_frames


FrameFn = Callable[[np.ndarray, int], np.ndarray]


FRAME_FEATURES: dict[str, FrameFn] = {
    "lfcc": lfcc_frames,
    "mfcc": mfcc_frames,
    "cqcc": cqcc_frames,
    "wcqcc": wcqcc_frames,
}


@dataclass
class GMMConfig:
    feature: str
    n_components: int = 64
    covariance_type: str = "diag"
    max_iter: int = 100
    reg_covar: float = 1e-6
    seed: int = 42
    max_frames_per_class: int | None = 300_000
    standardize: bool = True


@dataclass
class GMMBundle:
    config: GMMConfig
    scaler: StandardScaler | None
    bonafide_gmm: GaussianMixture
    spoof_gmm: GaussianMixture


def load_frame_features(
    sample: Sample,
    feature_fn: FrameFn,
    feature_name: str | None = None,
    cache: FeatureCache | None = None,
) -> np.ndarray:
    if cache is not None:
        if feature_name is None:
            raise ValueError("feature_name is required when feature cache is enabled")
        frames = cache.load_or_extract(sample, feature_name, feature_fn)
        if frames.ndim != 2 or frames.shape[0] == 0:
            raise ValueError(f"empty frame matrix for {sample.file_id}")
        return frames.astype(np.float32, copy=False)

    audio, sr = librosa.load(sample.path, sr=SR, mono=True)
    frames = feature_fn(audio, sr)
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise ValueError(f"empty frame matrix for {sample.file_id}")
    return frames.astype(np.float32, copy=False)


def collect_training_frames(
    samples: Iterable[Sample],
    feature_name: str,
    feature_fn: FrameFn,
    max_frames_per_class: int | None,
    seed: int,
    cache: FeatureCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bonafide_parts: list[np.ndarray] = []
    spoof_parts: list[np.ndarray] = []
    failed = 0

    for sample in tqdm(list(samples), desc="extract train frames"):
        try:
            frames = load_frame_features(sample, feature_fn, feature_name, cache)
        except Exception:
            failed += 1
            continue

        if sample.label == 1:
            bonafide_parts.append(frames)
        else:
            spoof_parts.append(frames)

    if failed:
        print(f"  [!] skipped {failed} training files with extraction errors")
    if not bonafide_parts or not spoof_parts:
        raise ValueError("training data must contain both bonafide and spoof frames")

    bonafide = np.vstack(bonafide_parts)
    spoof = np.vstack(spoof_parts)
    return (
        _subsample_frames(bonafide, max_frames_per_class, seed),
        _subsample_frames(spoof, max_frames_per_class, seed + 1),
    )


def _subsample_frames(frames: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit <= 0 or len(frames) <= limit:
        return frames
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frames), size=limit, replace=False)
    return frames[idx]


def train_gmm(
    samples: Iterable[Sample],
    config: GMMConfig,
    cache_root: str | None = None,
) -> GMMBundle:
    if config.feature not in FRAME_FEATURES:
        raise ValueError(f"unsupported feature '{config.feature}'")

    feature_fn = FRAME_FEATURES[config.feature]
    cache = FeatureCache(cache_root) if cache_root else None
    bonafide_frames, spoof_frames = collect_training_frames(
        samples,
        config.feature,
        feature_fn,
        config.max_frames_per_class,
        config.seed,
        cache,
    )
    print(f"  Bonafide frames: {bonafide_frames.shape}")
    print(f"  Spoof frames   : {spoof_frames.shape}")

    scaler = None
    if config.standardize:
        scaler = StandardScaler()
        scaler.fit(np.vstack([bonafide_frames, spoof_frames]))
        bonafide_frames = scaler.transform(bonafide_frames)
        spoof_frames = scaler.transform(spoof_frames)

    bonafide_gmm = _fit_one_gmm(bonafide_frames, config, config.seed)
    spoof_gmm = _fit_one_gmm(spoof_frames, config, config.seed + 1)
    return GMMBundle(config, scaler, bonafide_gmm, spoof_gmm)


def _fit_one_gmm(frames: np.ndarray, config: GMMConfig, seed: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        max_iter=config.max_iter,
        reg_covar=config.reg_covar,
        random_state=seed,
        verbose=0,
    )
    return gmm.fit(frames)


def score_samples(
    samples: Iterable[Sample],
    bundle: GMMBundle,
    cache_root: str | None = None,
) -> list[dict[str, object]]:
    feature_fn = FRAME_FEATURES[bundle.config.feature]
    cache = FeatureCache(cache_root) if cache_root else None
    rows: list[dict[str, object]] = []
    failed = 0

    for sample in tqdm(list(samples), desc="score utterances"):
        try:
            frames = load_frame_features(sample, feature_fn, bundle.config.feature, cache)
            if bundle.scaler is not None:
                frames = bundle.scaler.transform(frames)
            bonafide_ll = float(bundle.bonafide_gmm.score(frames))
            spoof_ll = float(bundle.spoof_gmm.score(frames))
        except Exception:
            failed += 1
            continue

        rows.append({
            "file_id": sample.file_id,
            "path": sample.path,
            "label": sample.label,
            "label_name": "bonafide" if sample.label == 1 else "spoof",
            "system_id": sample.system_id,
            "score": bonafide_ll - spoof_ll,
            "bonafide_log_likelihood": bonafide_ll,
            "spoof_log_likelihood": spoof_ll,
            "n_frames": len(frames),
        })

    if failed:
        print(f"  [!] skipped {failed} scoring files with extraction errors")
    return rows
