"""
EmotionPipeline — the single entry point for all emotion analysis.

Usage
-----
    from emotionlens import EmotionPipeline

    pipe = EmotionPipeline()

    # Text only
    result = pipe.predict(text="I'm so excited about this project!")
    print(result)

    # Multi-modal
    result = pipe.predict(
        text="This is terrible.",
        audio_path="clip.wav",
    )
    print(result.to_dict())
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from emotionlens.emotions import EmotionLabel, EmotionResult, ModalityScore, VADScore
from emotionlens.encoders.text_encoder import TextEncoder
from emotionlens.fusion import AdaptiveFusion, FusionStrategy
from emotionlens.explainer import EmotionExplainer

logger = logging.getLogger(__name__)


class EmotionPipeline:
    """High-level pipeline for multimodal emotion analysis.

    Parameters
    ----------
    fusion_strategy:
        One of ``"weighted_average"``, ``"confidence_gating"``, ``"attention"``.
    use_transformers:
        Load a HuggingFace transformer for text (requires the ``transformers`` package).
    modality_weights:
        Optional explicit weights for the ``weighted_average`` fusion strategy.
    explain:
        Whether to generate a natural-language explanation for each prediction.
    verbose:
        If True, log timing and per-modality scores.
    """

    def __init__(
        self,
        fusion_strategy: FusionStrategy = "confidence_gating",
        use_transformers: bool = False,
        modality_weights: Optional[Dict[str, float]] = None,
        explain: bool = True,
        verbose: bool = False,
    ) -> None:
        self._text_encoder = TextEncoder(use_transformers=use_transformers)
        self._fusion = AdaptiveFusion(
            strategy=fusion_strategy,
            modality_weights=modality_weights or {},
        )
        self._explainer = EmotionExplainer() if explain else None
        self._verbose = verbose
        self._explain = explain

        # Optional heavy encoders — lazy-loaded on first use
        self._audio_encoder = None
        self._image_encoder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        text: Optional[str] = None,
        audio_path: Optional[Union[str, Path]] = None,
        image_path: Optional[Union[str, Path]] = None,
        physiological: Optional[Dict] = None,
    ) -> EmotionResult:
        """Predict emotion from one or more input modalities.

        At least one of ``text``, ``audio_path``, or ``image_path``
        must be provided.

        Parameters
        ----------
        text:
            Raw text string to analyse.
        audio_path:
            Path to an audio file (WAV, MP3, FLAC …).
        image_path:
            Path to an image file (JPG, PNG …).
        physiological:
            Dict of physiological readings, e.g.
            ``{"heart_rate": 88, "eeg_alpha": 0.4}``.

        Returns
        -------
        EmotionResult
            Full prediction with confidence, VAD, and optional explanation.
        """
        t0 = time.perf_counter()

        if all(x is None for x in [text, audio_path, image_path, physiological]):
            raise ValueError("At least one input modality must be provided.")

        modality_scores: List[ModalityScore] = []

        # --- Text ---
        if text is not None:
            score = self._text_encoder.encode(text)
            modality_scores.append(score)
            if self._verbose:
                logger.info("Text → %s (%.0f%%)", score.top_emotion, score.confidence * 100)

        # --- Audio ---
        if audio_path is not None:
            enc = self._get_audio_encoder()
            score = enc.encode(audio_path)
            modality_scores.append(score)
            if self._verbose:
                logger.info("Audio → %s (%.0f%%)", score.top_emotion, score.confidence * 100)

        # --- Image ---
        if image_path is not None:
            enc = self._get_image_encoder()
            score = enc.encode(image_path)
            modality_scores.append(score)
            if self._verbose:
                logger.info("Image → %s (%.0f%%)", score.top_emotion, score.confidence * 100)

        # --- Physiological (placeholder) ---
        if physiological is not None:
            score = self._encode_physiological(physiological)
            modality_scores.append(score)

        # --- Fuse ---
        fused_scores, fusion_weights = self._fusion.fuse(modality_scores)

        # --- Derive VAD from fused scores (approximate reverse mapping) ---
        vad = self._scores_to_vad(fused_scores)

        # --- Pick winner ---
        top_label_str = max(fused_scores, key=fused_scores.get)
        label = EmotionLabel(top_label_str)
        confidence = fused_scores[top_label_str]

        result = EmotionResult(
            label=label,
            confidence=confidence,
            all_scores=fused_scores,
            vad=vad,
            modality_scores=modality_scores,
            fusion_weights=fusion_weights,
            metadata={
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "modalities": [m.modality for m in modality_scores],
                "fusion_strategy": self._fusion.strategy,
            },
        )

        if self._explain and self._explainer:
            result.explanation = self._explainer.explain(result)

        return result

    def batch_predict(
        self,
        texts: List[str],
        *,
        show_progress: bool = True,
    ) -> List[EmotionResult]:
        """Predict emotions for a list of texts (text-only, fast path)."""
        results = []
        for i, text in enumerate(texts):
            if show_progress and (i % 10 == 0):
                logger.info("Processing %d / %d …", i, len(texts))
            results.append(self.predict(text=text))
        return results

    def stream_predict(self, texts):
        """Generator version of :meth:`batch_predict`."""
        for text in texts:
            yield self.predict(text=text)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_audio_encoder(self):
        if self._audio_encoder is None:
            from emotionlens.encoders.audio_encoder import AudioEncoder
            self._audio_encoder = AudioEncoder()
        return self._audio_encoder

    def _get_image_encoder(self):
        if self._image_encoder is None:
            from emotionlens.encoders.image_encoder import ImageEncoder
            self._image_encoder = ImageEncoder()
        return self._image_encoder

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_physiological(data: Dict) -> ModalityScore:
        """Minimal rule-based physiological encoder (placeholder)."""
        hr = data.get("heart_rate", 70)
        eeg_alpha = data.get("eeg_alpha", 0.5)
        # High HR → arousal; high alpha → relaxation
        arousal_proxy = min(1.0, max(-1.0, (hr - 70) / 50))
        valence_proxy = eeg_alpha * 2 - 1  # 0.5 baseline → 0

        import math
        centroids = {
            "joy": (0.81, 0.51), "sadness": (-0.63, -0.27),
            "anger": (-0.43, 0.67), "fear": (-0.64, 0.60),
            "surprise": (0.40, 0.67), "disgust": (-0.60, 0.35),
            "contempt": (-0.35, 0.20), "neutral": (0.00, 0.00),
        }
        temp = 0.6
        inv = {
            k: math.exp(-math.sqrt((valence_proxy - cv)**2 + (arousal_proxy - ca)**2) / temp)
            for k, (cv, ca) in centroids.items()
        }
        total = sum(inv.values())
        scores = {k: round(v / total, 4) for k, v in inv.items()}
        return ModalityScore(
            modality="physiological",
            scores=scores,
            feature_weights={"heart_rate": hr, "eeg_alpha": eeg_alpha},
        )

    @staticmethod
    def _scores_to_vad(scores: Dict[str, float]) -> VADScore:
        """Recover a weighted-average VAD triple from a score distribution."""
        vad_centroids = {
            "joy":       ( 0.81,  0.51,  0.46),
            "sadness":   (-0.63, -0.27, -0.33),
            "anger":     (-0.43,  0.67,  0.34),
            "fear":      (-0.64,  0.60, -0.43),
            "surprise":  ( 0.40,  0.67, -0.13),
            "disgust":   (-0.60,  0.35,  0.11),
            "contempt":  (-0.35,  0.20,  0.58),
            "neutral":   ( 0.00,  0.00,  0.00),
        }
        v = sum(scores.get(lbl, 0) * vad_centroids.get(lbl, (0, 0, 0))[0]
                for lbl in scores)
        a = sum(scores.get(lbl, 0) * vad_centroids.get(lbl, (0, 0, 0))[1]
                for lbl in scores)
        d = sum(scores.get(lbl, 0) * vad_centroids.get(lbl, (0, 0, 0))[2]
                for lbl in scores)
        return VADScore(valence=round(v, 3), arousal=round(a, 3), dominance=round(d, 3))
