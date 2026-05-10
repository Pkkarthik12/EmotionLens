"""
Emotion taxonomy and result types used throughout EmotionLens.

Implements the Ekman 8-emotion model alongside a continuous
Valence-Arousal-Dominance (VAD) space for nuanced predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class EmotionLabel(str, Enum):
    """Discrete emotion categories (extended Ekman model)."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"
    NEUTRAL = "neutral"

    @classmethod
    def from_vad(cls, valence: float, arousal: float, dominance: float) -> "EmotionLabel":
        """Map a VAD triple to the nearest discrete emotion label.

        Uses a simple nearest-centroid approach in 3-D VAD space.
        Centroids are derived from the NRC-VAD lexicon averages.
        """
        centroids: Dict["EmotionLabel", tuple] = {
            cls.JOY:       ( 0.81,  0.51,  0.46),
            cls.SADNESS:   (-0.63, -0.27, -0.33),
            cls.ANGER:     (-0.43,  0.67,  0.34),
            cls.FEAR:      (-0.64,  0.60, -0.43),
            cls.SURPRISE:  ( 0.40,  0.67, -0.13),
            cls.DISGUST:   (-0.60,  0.35,  0.11),
            cls.CONTEMPT:  (-0.35,  0.20,  0.58),
            cls.NEUTRAL:   ( 0.00,  0.00,  0.00),
        }
        v, a, d = valence, arousal, dominance
        best, best_dist = cls.NEUTRAL, float("inf")
        for label, (cv, ca, cd) in centroids.items():
            dist = (v - cv) ** 2 + (a - ca) ** 2 + (d - cd) ** 2
            if dist < best_dist:
                best, best_dist = label, dist
        return best


@dataclass
class ModalityScore:
    """Confidence scores contributed by a single modality."""

    modality: str
    scores: Dict[str, float]       # emotion label → confidence
    feature_weights: Dict[str, float] = field(default_factory=dict)  # SHAP-style
    embedding: Optional[List[float]] = None

    @property
    def top_emotion(self) -> str:
        return max(self.scores, key=self.scores.get)

    @property
    def confidence(self) -> float:
        return max(self.scores.values())


@dataclass
class VADScore:
    """Continuous emotion representation in Valence-Arousal-Dominance space."""

    valence: float     # [-1, 1] negative ↔ positive
    arousal: float     # [-1, 1] calm ↔ excited
    dominance: float   # [-1, 1] submissive ↔ dominant

    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @property
    def quadrant(self) -> str:
        """Return a human-readable quadrant name."""
        v_str = "positive" if self.valence >= 0 else "negative"
        a_str = "high-arousal" if self.arousal >= 0 else "low-arousal"
        return f"{v_str}, {a_str}"


@dataclass
class EmotionResult:
    """Full prediction result returned by :class:`EmotionPipeline`.

    Attributes
    ----------
    label:
        Winning discrete emotion.
    confidence:
        Probability of the winning label (0–1).
    all_scores:
        Full probability distribution over all labels.
    vad:
        Continuous VAD representation.
    modality_scores:
        Per-modality breakdown (populated only for multi-modal inputs).
    explanation:
        SHAP-based natural-language explanation of the prediction.
    fusion_weights:
        How much each modality contributed to the final fused score.
    metadata:
        Free-form dict for provenance, latency, model versions, etc.
    """

    label: EmotionLabel
    confidence: float
    all_scores: Dict[str, float]
    vad: VADScore
    modality_scores: List[ModalityScore] = field(default_factory=list)
    explanation: str = ""
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        bar = "█" * int(self.confidence * 20) + "░" * (20 - int(self.confidence * 20))
        lines = [
            f"Emotion : {self.label.value.upper()}",
            f"Confidence : [{bar}] {self.confidence:.1%}",
            f"VAD      : V={self.vad.valence:+.2f}  A={self.vad.arousal:+.2f}  D={self.vad.dominance:+.2f}",
        ]
        if self.fusion_weights:
            wts = "  ".join(f"{k}={v:.0%}" for k, v in self.fusion_weights.items())
            lines.append(f"Fusion   : {wts}")
        if self.explanation:
            lines.append(f"Why      : {self.explanation}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "vad": self.vad.to_dict(),
            "fusion_weights": self.fusion_weights,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }
