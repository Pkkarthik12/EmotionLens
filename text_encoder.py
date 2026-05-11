"""
Text modality encoder for EmotionLens.

Uses a lightweight transformer (or a TF-IDF fallback) to extract
lexical, syntactic, and semantic emotion signals from raw text.

Design goals
------------
* No mandatory GPU — runs on CPU out of the box.
* Optional HuggingFace transformers for higher accuracy.
* Always returns a ModalityScore with SHAP-compatible feature weights.
"""

from __future__ import annotations

import re
import math
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from emotionlens.emotions import ModalityScore, EmotionLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal VAD lexicon (subset of NRC-VAD, hard-coded for zero-dependency use)
# Replace with a full NRC-VAD load in production.
# ---------------------------------------------------------------------------
_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
    "love":      ( 0.91,  0.55,  0.52),
    "happy":     ( 0.81,  0.51,  0.46),
    "joy":       ( 0.83,  0.62,  0.54),
    "excited":   ( 0.72,  0.75,  0.43),
    "sad":       (-0.63, -0.27, -0.33),
    "angry":     (-0.43,  0.67,  0.34),
    "furious":   (-0.65,  0.85,  0.52),
    "fear":      (-0.64,  0.60, -0.43),
    "scared":    (-0.60,  0.72, -0.51),
    "surprised": ( 0.40,  0.67, -0.13),
    "disgust":   (-0.60,  0.35,  0.11),
    "contempt":  (-0.35,  0.20,  0.58),
    "bored":     (-0.25, -0.50, -0.10),
    "calm":      ( 0.35, -0.45,  0.15),
    "great":     ( 0.78,  0.40,  0.55),
    "terrible":  (-0.72,  0.50, -0.20),
    "awful":     (-0.70,  0.45, -0.15),
    "wonderful": ( 0.90,  0.50,  0.55),
    "hate":      (-0.80,  0.60,  0.35),
    "like":      ( 0.55,  0.20,  0.30),
    "dislike":   (-0.50,  0.15, -0.10),
    "worried":   (-0.50,  0.55, -0.35),
    "hopeful":   ( 0.60,  0.30,  0.20),
    "frustrated": (-0.52, 0.60,  0.20),
    "grateful":  ( 0.75,  0.25,  0.35),
    "lonely":    (-0.55, -0.20, -0.40),
    "proud":     ( 0.70,  0.45,  0.65),
    "ashamed":   (-0.58,  0.30, -0.50),
    "confident": ( 0.65,  0.40,  0.70),
    "nervous":   (-0.45,  0.65, -0.40),
}

_NEGATIONS = {"not", "no", "never", "neither", "nor", "nobody", "nothing",
               "nowhere", "without", "hardly", "barely", "scarcely"}

_INTENSIFIERS = {"very": 1.4, "extremely": 1.8, "really": 1.3, "quite": 1.1,
                 "somewhat": 0.7, "slightly": 0.5, "a bit": 0.6, "so": 1.3,
                 "totally": 1.5, "absolutely": 1.6, "utterly": 1.7}


class TextEncoder:
    """Encode raw text into a :class:`~emotionlens.emotions.ModalityScore`.

    Parameters
    ----------
    use_transformers:
        If True and ``transformers`` is installed, load a pretrained
        RoBERTa model fine-tuned on GoEmotions. Falls back to the
        lexicon-based approach on import failure.
    model_name:
        HuggingFace model identifier (default: cardiffnlp/twitter-roberta-base-emotion).
    """

    EMOTION_LABELS: List[str] = [e.value for e in EmotionLabel]

    def __init__(
        self,
        use_transformers: bool = False,
        model_name: str = "cardiffnlp/twitter-roberta-base-emotion",
    ) -> None:
        self._transformer_pipeline = None
        if use_transformers:
            self._try_load_transformer(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> ModalityScore:
        """Return a :class:`ModalityScore` for the given text string."""
        if self._transformer_pipeline is not None:
            return self._transformer_encode(text)
        return self._lexicon_encode(text)

    # ------------------------------------------------------------------
    # Transformer path
    # ------------------------------------------------------------------

    def _try_load_transformer(self, model_name: str) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            self._transformer_pipeline = hf_pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded transformer model: %s", model_name)
        except ImportError:
            logger.warning("transformers not installed — using lexicon fallback.")
        except Exception as exc:
            logger.warning("Could not load transformer (%s) — using lexicon fallback.", exc)

    def _transformer_encode(self, text: str) -> ModalityScore:
        raw = self._transformer_pipeline(text)[0]
        label_map = {r["label"].lower(): r["score"] for r in raw}
        scores = self._align_scores(label_map)
        return ModalityScore(
            modality="text",
            scores=scores,
            feature_weights={"transformer_confidence": max(scores.values())},
        )

    # ------------------------------------------------------------------
    # Lexicon path
    # ------------------------------------------------------------------

    def _lexicon_encode(self, text: str) -> ModalityScore:
        tokens = self._tokenize(text)
        vad_hits, feature_weights = self._extract_vad(tokens)

        if not vad_hits:
            # No lexicon matches — fall back to uniform distribution
            n = len(self.EMOTION_LABELS)
            scores = {lbl: 1.0 / n for lbl in self.EMOTION_LABELS}
            return ModalityScore(modality="text", scores=scores, feature_weights={})

        # hits: List[Tuple[word, (vv, av, dv), weight]]
        total_weight = sum(h[2] for h in vad_hits)
        v = sum(h[1][0] * h[2] for h in vad_hits) / total_weight
        a = sum(h[1][1] * h[2] for h in vad_hits) / total_weight
        d = sum(h[1][2] * h[2] for h in vad_hits) / total_weight

        scores = self._vad_to_scores(v, a, d)
        feature_weights["valence"] = round(v, 3)
        feature_weights["arousal"] = round(a, 3)
        feature_weights["dominance"] = round(d, 3)
        feature_weights["lexicon_hits"] = len(vad_hits)

        return ModalityScore(
            modality="text",
            scores=scores,
            feature_weights=feature_weights,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        return text.split()

    @staticmethod
    def _extract_vad(
        tokens: List[str],
    ) -> Tuple[List[Tuple[str, Tuple[float, float, float], float]], Dict[str, float]]:
        """Walk tokens respecting negation and intensifier windows."""
        hits: List[Tuple[str, Tuple[float, float, float], float]] = []
        feature_weights: Dict[str, float] = {}
        negated = False
        intensity = 1.0

        for i, tok in enumerate(tokens):
            if tok in _NEGATIONS:
                negated = True
                continue
            if tok in _INTENSIFIERS:
                intensity = _INTENSIFIERS[tok]
                continue

            if tok in _VAD_LEXICON:
                vv, av, dv = _VAD_LEXICON[tok]
                if negated:
                    vv = -vv * 0.8   # flip valence, dampen
                    av = av * 0.6
                    negated = False
                effective = intensity
                hits.append((tok, (vv, av, dv), effective))
                feature_weights[tok] = round(effective, 2)
                intensity = 1.0  # reset

        return hits, feature_weights

    @staticmethod
    def _vad_to_scores(v: float, a: float, d: float) -> Dict[str, float]:
        """Convert a VAD triple to a soft probability distribution."""
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
        distances = {}
        for label, (cv, ca, cd) in centroids.items():
            dist = math.sqrt((v - cv) ** 2 + (a - ca) ** 2 + (d - cd) ** 2)
            distances[label] = dist

        # Softmin: invert distances and softmax
        inv = {k: math.exp(-dist / temperature) for k, dist in distances.items()}
        total = sum(inv.values())
        return {k: round(vv / total, 4) for k, vv in inv.items()}

    @staticmethod
    def _align_scores(raw: Dict[str, float]) -> Dict[str, float]:
        """Align arbitrary model output labels to EmotionLens label set."""
        canonical: Dict[str, float] = {e.value: 0.0 for e in EmotionLabel}
        # Direct matches
        for label in list(canonical.keys()):
            if label in raw:
                canonical[label] = raw.pop(label)
        # Leftovers get pooled into neutral
        for v in raw.values():
            canonical["neutral"] += v
        # Renormalise
        total = sum(canonical.values()) or 1.0
        return {k: round(v / total, 4) for k, v in canonical.items()}
