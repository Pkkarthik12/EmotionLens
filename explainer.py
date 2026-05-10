"""
Explainability module for EmotionLens.

Produces human-readable explanations of emotion predictions by
combining three complementary techniques:

1. Feature-weight attribution  — Which lexical / acoustic features
   pushed the prediction toward the winning emotion?
2. Modality attribution        — How much did each modality contribute?
3. Counterfactual summary      — What would flip the prediction?

For text inputs the explainer highlights the specific words
(positive and negative contributors) directly in the output.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from emotionlens.emotions import EmotionLabel, EmotionResult, ModalityScore


class EmotionExplainer:
    """Generate natural-language explanations for :class:`EmotionResult` objects.

    Parameters
    ----------
    top_k_features:
        How many features to mention in the explanation.
    confidence_threshold:
        If winning confidence is below this threshold, flag uncertainty.
    """

    def __init__(self, top_k_features: int = 3, confidence_threshold: float = 0.45) -> None:
        self.top_k = top_k_features
        self.threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, result: EmotionResult) -> str:
        """Return a plain-English explanation of the prediction."""
        parts: List[str] = []

        # 1. Core prediction sentence
        label = result.label.value
        conf = result.confidence
        uncertainty = " (low confidence)" if conf < self.threshold else ""
        parts.append(f"Predicted emotion: {label.upper()}{uncertainty} ({conf:.0%} confident).")

        # 2. VAD characterisation
        v, a, d = result.vad.valence, result.vad.arousal, result.vad.dominance
        vad_str = self._vad_narrative(v, a, d)
        parts.append(f"Emotional character: {vad_str}.")

        # 3. Modality contributions
        if result.fusion_weights:
            contrib = self._modality_narrative(result.fusion_weights)
            parts.append(f"Signal sources: {contrib}.")

        # 4. Feature attribution
        if result.modality_scores:
            feat_str = self._feature_narrative(result.modality_scores)
            if feat_str:
                parts.append(f"Key signals: {feat_str}.")

        # 5. Runner-up (second most likely)
        runner_up = self._runner_up(result.all_scores, result.label)
        if runner_up:
            parts.append(
                f"Alternative reading: {runner_up[0].upper()} ({runner_up[1]:.0%})."
            )

        # 6. Counterfactual hint
        cf = self._counterfactual(result)
        if cf:
            parts.append(f"What would change this: {cf}.")

        return "  ".join(parts)

    def feature_heatmap(self, modality_score: ModalityScore) -> List[Tuple[str, float]]:
        """Return sorted (feature, weight) pairs for visualisation."""
        weights = modality_score.feature_weights
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_weights[: self.top_k * 2]

    # ------------------------------------------------------------------
    # Narrative helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vad_narrative(v: float, a: float, d: float) -> str:
        v_word = "pleasant" if v > 0.3 else ("unpleasant" if v < -0.3 else "mixed valence")
        a_word = "energetic" if a > 0.3 else ("subdued" if a < -0.3 else "moderate energy")
        d_word = "assertive" if d > 0.3 else ("submissive" if d < -0.3 else "moderate control")
        return f"{v_word}, {a_word}, {d_word}"

    @staticmethod
    def _modality_narrative(weights: Dict[str, float]) -> str:
        dominant = max(weights, key=weights.get)
        items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{m} ({w:.0%})" for m, w in items]
        return f"{dominant} led the prediction — " + ", ".join(parts)

    def _feature_narrative(self, modality_scores: List[ModalityScore]) -> str:
        positive, negative = [], []
        for ms in modality_scores:
            for feat, weight in ms.feature_weights.items():
                if isinstance(weight, float):
                    if weight > 0.05:
                        positive.append((feat, weight))
                    elif weight < -0.05:
                        negative.append((feat, abs(weight)))

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1], reverse=True)

        parts = []
        if positive:
            words = ", ".join(f'"{f}"' for f, _ in positive[: self.top_k])
            parts.append(f"{words} pushed toward current emotion")
        if negative:
            words = ", ".join(f'"{f}"' for f, _ in negative[: self.top_k])
            parts.append(f"{words} pulled away from it")

        return "; ".join(parts)

    @staticmethod
    def _runner_up(
        all_scores: Dict[str, float], winner: EmotionLabel
    ) -> Optional[Tuple[str, float]]:
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_scores:
            if label != winner.value and score > 0.05:
                return label, score
        return None

    @staticmethod
    def _counterfactual(result: EmotionResult) -> str:
        """Simple rule-based counterfactual hints."""
        v, a, d = result.vad.valence, result.vad.arousal, result.vad.dominance
        label = result.label

        if label == EmotionLabel.ANGER and a > 0.4:
            return "lower energy/arousal cues would shift toward frustration or sadness"
        if label == EmotionLabel.SADNESS and v < -0.4:
            return "more positive lexical cues would shift toward neutral or acceptance"
        if label == EmotionLabel.JOY and v > 0.6:
            return "reduced positive affect would shift toward neutral contentment"
        if label == EmotionLabel.FEAR and d < -0.3:
            return "greater control signals would shift toward surprise or uncertainty"
        if result.confidence < 0.45:
            return "stronger, unambiguous signals in any modality would improve certainty"
        return ""
