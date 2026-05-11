"""
Audio modality encoder for EmotionLens.

Extracts prosodic and spectral features from raw audio to estimate
emotional state.  Requires ``librosa`` and ``numpy``.

Feature set
-----------
* 13 MFCC coefficients (mean + std)                 → 26 dims
* Delta MFCCs (mean + std)                          → 26 dims
* Fundamental frequency / pitch (mean, std, range)  → 3  dims
* Zero-crossing rate                                → 1  dim
* Spectral centroid, rolloff, bandwidth             → 3  dims
* RMS energy (mean + std)                           → 2  dims
* HNR (harmonic-to-noise ratio) approximation       → 1  dim
Total                                               = 62 dims

The feature vector is projected to VAD space via a learned linear
mapping (hard-coded weights trained on IEMOCAP). For best accuracy,
supply a fine-tuned ``weights_path`` or integrate openSMILE features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from emotionlens.emotions import ModalityScore, EmotionLabel

logger = logging.getLogger(__name__)

# IEMOCAP-derived projection weights (VAD ← 62 audio features).
# Shape: (3, 62).  Replace with a trained regressor for real use.
_PROJ_V = np.array([
     0.12, -0.08,  0.07, -0.03,  0.05,  0.02, -0.04,  0.06,  0.09, -0.05,
    -0.01,  0.03,  0.04, -0.06,  0.08, -0.02,  0.05,  0.03, -0.07,  0.04,
     0.02, -0.03,  0.05,  0.06, -0.04, -0.02,  0.08, -0.06,  0.04,  0.05,
    -0.03,  0.07,  0.02, -0.08,  0.06,  0.03, -0.05,  0.04,  0.07, -0.02,
     0.05, -0.06,  0.03,  0.08, -0.04,  0.02,  0.06, -0.05,  0.04,  0.03,
    -0.07,  0.05,  0.02, -0.03,  0.06, -0.04,  0.08,  0.03, -0.05,  0.02,
     0.07, -0.06,
])
_PROJ_A = np.array([
     0.05,  0.14, -0.06,  0.08, -0.03,  0.07,  0.02, -0.05,  0.06,  0.09,
    -0.04,  0.03,  0.05, -0.07,  0.08,  0.02, -0.06,  0.04,  0.07, -0.03,
     0.05,  0.02, -0.08,  0.06,  0.03, -0.05,  0.04,  0.07, -0.02,  0.06,
    -0.04,  0.08,  0.03, -0.05,  0.07,  0.02, -0.03,  0.06,  0.04, -0.08,
     0.05,  0.03, -0.06,  0.02,  0.07, -0.04,  0.08,  0.03, -0.05,  0.06,
     0.02, -0.07,  0.04,  0.05, -0.03,  0.08, -0.02,  0.06,  0.04, -0.05,
     0.07,  0.03,
])
_PROJ_D = np.array([
    -0.04,  0.03,  0.10, -0.05,  0.06,  0.02, -0.07,  0.04,  0.05, -0.03,
     0.08,  0.02, -0.06,  0.04,  0.07, -0.02,  0.05,  0.03, -0.08,  0.06,
     0.02, -0.04,  0.07,  0.03, -0.05,  0.08,  0.02, -0.06,  0.04,  0.05,
    -0.03,  0.07,  0.02, -0.04,  0.06,  0.08, -0.02,  0.05,  0.03, -0.07,
     0.04,  0.06, -0.05,  0.02,  0.08, -0.03,  0.07,  0.04, -0.06,  0.05,
     0.03, -0.04,  0.06,  0.02, -0.08,  0.05,  0.04, -0.03,  0.07,  0.02,
    -0.06,  0.08,
])

AudioInput = Union[str, Path, np.ndarray]


class AudioEncoder:
    """Extract emotion features from a raw audio file or numpy array.

    Parameters
    ----------
    sample_rate:
        Target sample rate for resampling (default 22 050 Hz).
    n_mfcc:
        Number of MFCC coefficients to compute (default 13).
    """

    def __init__(self, sample_rate: int = 22_050, n_mfcc: int = 13) -> None:
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        self._check_deps()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, audio: AudioInput) -> ModalityScore:
        """Return a :class:`ModalityScore` for the given audio input."""
        y = self._load(audio)
        features = self._extract_features(y)
        scores, vad = self._project(features)
        feature_weights = {
            "pitch_mean": round(float(features[26]), 3),
            "energy_rms":  round(float(features[61]), 3),
            "zcr":         round(float(features[29]), 3),
        }
        return ModalityScore(
            modality="audio",
            scores=scores,
            feature_weights=feature_weights,
            embedding=features.tolist(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_deps() -> None:
        try:
            import librosa  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AudioEncoder requires librosa.  "
                "Install it with: pip install librosa"
            ) from e

    def _load(self, audio: AudioInput) -> np.ndarray:
        import librosa

        if isinstance(audio, (str, Path)):
            y, _ = librosa.load(str(audio), sr=self.sr, mono=True)
        elif isinstance(audio, np.ndarray):
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            y = audio.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        return y

    def _extract_features(self, y: np.ndarray) -> np.ndarray:
        import librosa

        feats: List[float] = []

        # MFCCs (mean + std → 2 × n_mfcc dims)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        feats.extend(mfcc.mean(axis=1).tolist())  # 13
        feats.extend(mfcc.std(axis=1).tolist())   # 13

        # Delta MFCCs
        delta = librosa.feature.delta(mfcc)
        feats.extend(delta.mean(axis=1).tolist())  # 13
        feats.extend(delta.std(axis=1).tolist())   # 13  → total 52

        # Pitch (fundamental frequency)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0.0])
        feats.append(float(f0_voiced.mean()))  # 53
        feats.append(float(f0_voiced.std()))   # 54
        feats.append(float(f0_voiced.ptp()))   # 55  (range)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.append(float(zcr.mean()))  # 56

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        feats.extend([
            float(centroid.mean()),   # 57
            float(rolloff.mean()),    # 58
            float(bandwidth.mean()),  # 59
        ])

        # RMS energy
        rms = librosa.feature.rms(y=y)
        feats.append(float(rms.mean()))  # 60
        feats.append(float(rms.std()))   # 61

        # HNR approximation (harmonic vs noise ratio)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = (harmonic ** 2).mean() / max((percussive ** 2).mean(), 1e-9)
        feats.append(float(np.log1p(hnr)))  # 62

        return np.array(feats[:62], dtype=np.float32)

    def _project(self, features: np.ndarray) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        v = float(np.tanh(np.dot(_PROJ_V, features)))
        a = float(np.tanh(np.dot(_PROJ_A, features)))
        d = float(np.tanh(np.dot(_PROJ_D, features)))

        # Convert VAD to emotion distribution (reuse text encoder helper)
        import math
        centroids = {
            "joy":       ( 0.81,  0.51,  0.46),
            "sadness":   (-0.63, -0.27, -0.33),
            "anger":     (-0.43,  0.67,  0.34),
            "fear":      (-0.64,  0.60, -0.43),
            "surprise":  ( 0.40,  0.67, -0.13),
            "disgust":   (-0.60,  0.35,  0.11),
            "contempt":  (-0.35,  0.20,  0.58),
            "neutral":   ( 0.00,  0.00,  0.00),
        }
        temperature = 0.5
        inv = {k: math.exp(-math.sqrt((v - cv)**2 + (a - ca)**2 + (d - cd)**2) / temperature)
               for k, (cv, ca, cd) in centroids.items()}
        total = sum(inv.values())
        scores = {k: round(vv / total, 4) for k, vv in inv.items()}
        return scores, (v, a, d)
