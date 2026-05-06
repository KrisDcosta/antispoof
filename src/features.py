"""
Feature extraction for speech anti-spoofing.

Features:
  - MFCC  : baseline mel-frequency cepstral coefficients
  - LFCC  : linear-frequency cepstral coefficients with deltas
  - CQCC  : constant-Q cepstral coefficients
  - WCQCC : CQCC with linearly decreasing frequency weights (novel)
  - AZCR  : average zero-crossing rate on silence segments (TextGrid required)

All utterance-level functions return a 1-D mean vector suitable for
downstream classifiers. Frame-level arrays are also available.
"""

import numpy as np
import librosa
import scipy.fftpack

SR = 16000
HOP = 160       # 10 ms at 16 kHz
WIN = 400       # 25 ms at 16 kHz
N_FFT = 512
N_BINS = 72     # 6 octaves × 12 bins — 7 octaves from C2 exceeds Nyquist at 16kHz SR
N_COEFF = 20    # coefficients kept (excl. 0th)
N_LFCC_FILTERS = 70
PREEMPH = 0.97


# ---------------------------------------------------------------------------
# Frame-level feature matrices  [T × N_COEFF]
# ---------------------------------------------------------------------------

def mfcc_frames(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_COEFF, hop_length=HOP)
    return feat.T   # [T × N_COEFF]


def lfcc_frames(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Linear-frequency cepstral coefficients with delta and delta-delta terms.

    ASVspoof GMM baselines commonly use LFCC-style front ends with temporal
    derivatives. This implementation keeps the pipeline fully Python-native
    while preserving the important shape: frame-level linear filterbank
    cepstra, followed by first- and second-order deltas.
    """
    if audio.size == 0:
        return np.empty((0, N_COEFF * 3), dtype=np.float32)

    emphasized = np.append(audio[0], audio[1:] - PREEMPH * audio[:-1])
    spectrum = np.abs(
        librosa.stft(
            emphasized,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window="hann",
            center=True,
        )
    ) ** 2
    filters = linear_filterbank(sr=sr, n_fft=N_FFT, n_filters=N_LFCC_FILTERS)
    energies = filters @ spectrum
    log_energies = np.log(np.clip(energies, 1e-10, None))
    cepstra = scipy.fftpack.dct(log_energies, axis=0, type=2, norm="ortho")
    static = cepstra[1:N_COEFF + 1, :]
    delta = librosa.feature.delta(static, order=1, mode="nearest")
    delta_delta = librosa.feature.delta(static, order=2, mode="nearest")
    return np.vstack([static, delta, delta_delta]).T.astype(np.float32, copy=False)


def linear_filterbank(sr: int, n_fft: int, n_filters: int) -> np.ndarray:
    """Create triangular filters evenly spaced on the linear frequency axis."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    fbanks = np.zeros((n_filters, len(freqs)), dtype=np.float32)

    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        left_slope = (freqs - left) / max(center - left, 1e-12)
        right_slope = (right - freqs) / max(right - center, 1e-12)
        fbanks[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    enorm = 2.0 / np.maximum(edges[2:n_filters + 2] - edges[:n_filters], 1e-12)
    fbanks *= enorm[:, np.newaxis]
    return fbanks


def cqcc_frames(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=HOP,
                              fmin=librosa.note_to_hz("C2"),
                              n_bins=N_BINS, bins_per_octave=12))
    log_cqt = np.log(np.clip(cqt, 1e-10, None))
    cqcc = scipy.fftpack.dct(log_cqt, axis=0, type=2, norm="ortho")
    return cqcc[1:N_COEFF + 1, :].T   # [T × N_COEFF]


def wcqcc_frames(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    WCQCC: linearly decreasing weights across frequency bins before DCT.
    Higher-frequency bins carry less energy in natural speech;
    down-weighting them makes the cepstrum more discriminative for spoofed
    speech artifacts that concentrate in upper frequencies.
    """
    cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=HOP,
                              fmin=librosa.note_to_hz("C2"),
                              n_bins=N_BINS, bins_per_octave=12))
    log_cqt = np.log(np.clip(cqt, 1e-10, None))
    weights = np.linspace(1.0, 0.7, N_BINS)[:, np.newaxis]
    wcqcc = scipy.fftpack.dct(log_cqt * weights, axis=0, type=2, norm="ortho")
    return wcqcc[1:N_COEFF + 1, :].T   # [T × N_COEFF]


# ---------------------------------------------------------------------------
# Utterance-level vectors  (mean pooling over time)
# ---------------------------------------------------------------------------

def mfcc_vec(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    return mfcc_frames(audio, sr).mean(axis=0)


def lfcc_vec(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    return lfcc_frames(audio, sr).mean(axis=0)


def cqcc_vec(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    return cqcc_frames(audio, sr).mean(axis=0)


def wcqcc_vec(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    return wcqcc_frames(audio, sr).mean(axis=0)


# ---------------------------------------------------------------------------
# AZCR — average ZCR on silence gaps (requires TextGrid alignment)
# ---------------------------------------------------------------------------

def azcr_from_textgrid(audio: np.ndarray, sr: int, textgrid_path: str) -> np.ndarray:
    """
    Returns ZCR values measured in silence gaps between words.
    Falls back to empty array if TextGrid is missing.
    """
    import textgrid as tg_lib
    try:
        tg = tg_lib.TextGrid.fromFile(textgrid_path)
        intervals = list(tg.getFirst("Token"))
    except Exception:
        return np.array([])

    azcr_values = []
    for i in range(len(intervals) - 1):
        gap_start = intervals[i].maxTime
        gap_end = intervals[i + 1].minTime
        if gap_end - gap_start < 0.01:   # skip sub-10ms gaps
            continue
        s0 = int(gap_start * sr)
        s1 = int(gap_end * sr)
        if s1 > s0:
            zcr = librosa.feature.zero_crossing_rate(audio[s0:s1])[0]
            azcr_values.append(float(np.mean(zcr)))

    return np.array(azcr_values)


# ---------------------------------------------------------------------------
# Combined feature vector (WCQCC + scalar AZCR mean)
# ---------------------------------------------------------------------------

def wcqcc_azcr_vec(audio: np.ndarray, sr: int = SR,
                   textgrid_path: str = None) -> np.ndarray:
    wc = wcqcc_vec(audio, sr)
    if textgrid_path and __import__("os").path.exists(textgrid_path):
        az = azcr_from_textgrid(audio, sr, textgrid_path)
        azcr_mean = float(az.mean()) if len(az) > 0 else 0.0
    else:
        azcr_mean = 0.0
    return np.append(wc, azcr_mean)   # [N_COEFF + 1]
