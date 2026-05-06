"""
Feature extraction for speech anti-spoofing.

Features:
  - MFCC  : baseline mel-frequency cepstral coefficients
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
N_BINS = 72     # 6 octaves × 12 bins — 7 octaves from C2 exceeds Nyquist at 16kHz SR
N_COEFF = 20    # coefficients kept (excl. 0th)


# ---------------------------------------------------------------------------
# Frame-level feature matrices  [T × N_COEFF]
# ---------------------------------------------------------------------------

def mfcc_frames(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_COEFF, hop_length=HOP)
    return feat.T   # [T × N_COEFF]


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
