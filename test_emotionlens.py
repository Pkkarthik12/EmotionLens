"""
Unit and integration tests for EmotionLens.

Run with:
    pytest tests/ -v
"""

import math
import pytest

# ---------------------------------------------------------------------------
# EmotionLabel tests
# ---------------------------------------------------------------------------

class TestEmotionLabel:
    def test_all_labels_present(self):
        from emotionlens.emotions import EmotionLabel
        expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "neutral"}
        assert {e.value for e in EmotionLabel} == expected

    def test_from_vad_joy(self):
        from emotionlens.emotions import EmotionLabel
        label = EmotionLabel.from_vad(valence=0.85, arousal=0.55, dominance=0.50)
        assert label == EmotionLabel.JOY

    def test_from_vad_neutral(self):
        from emotionlens.emotions import EmotionLabel
        label = EmotionLabel.from_vad(0.0, 0.0, 0.0)
        assert label == EmotionLabel.NEUTRAL

    def test_from_vad_anger(self):
        from emotionlens.emotions import EmotionLabel
        label = EmotionLabel.from_vad(-0.45, 0.70, 0.35)
        assert label == EmotionLabel.ANGER


# ---------------------------------------------------------------------------
# VADScore tests
# ---------------------------------------------------------------------------

class TestVADScore:
    def test_to_dict_keys(self):
        from emotionlens.emotions import VADScore
        vad = VADScore(0.5, -0.3, 0.1)
        d = vad.to_dict()
        assert set(d.keys()) == {"valence", "arousal", "dominance"}

    def test_quadrant_positive_high(self):
        from emotionlens.emotions import VADScore
        vad = VADScore(0.6, 0.6, 0.0)
        assert "positive" in vad.quadrant
        assert "high-arousal" in vad.quadrant

    def test_quadrant_negative_low(self):
        from emotionlens.emotions import VADScore
        vad = VADScore(-0.6, -0.6, 0.0)
        assert "negative" in vad.quadrant
        assert "low-arousal" in vad.quadrant


# ---------------------------------------------------------------------------
# TextEncoder tests
# ---------------------------------------------------------------------------

class TestTextEncoder:
    def setup_method(self):
        from emotionlens.encoders.text_encoder import TextEncoder
        self.enc = TextEncoder(use_transformers=False)

    def test_returns_modality_score(self):
        from emotionlens.emotions import ModalityScore
        result = self.enc.encode("I love this!")
        assert isinstance(result, ModalityScore)
        assert result.modality == "text"

    def test_scores_sum_to_one(self):
        result = self.enc.encode("This is a test sentence.")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-4

    def test_all_emotion_labels_present(self):
        result = self.enc.encode("Some text")
        from emotionlens.emotions import EmotionLabel
        assert set(result.scores.keys()) == {e.value for e in EmotionLabel}

    def test_joy_sentence(self):
        result = self.enc.encode("I am so happy and excited today!")
        # Joy should rank higher than sadness
        assert result.scores["joy"] > result.scores["sadness"]

    def test_sad_sentence(self):
        result = self.enc.encode("I feel so sad and lonely.")
        assert result.scores["sadness"] > result.scores["joy"]

    def test_negation_flips_valence(self):
        pos = self.enc.encode("I love this")
        neg = self.enc.encode("I do not love this")
        # Negated sentence should have lower joy
        assert pos.scores["joy"] > neg.scores["joy"]

    def test_empty_text_returns_uniform(self):
        result = self.enc.encode("   ")
        scores = list(result.scores.values())
        # All scores roughly equal
        assert max(scores) - min(scores) < 0.2

    def test_feature_weights_populated(self):
        result = self.enc.encode("I feel great and wonderful today!")
        assert len(result.feature_weights) > 0

    def test_confidence_between_0_and_1(self):
        result = self.enc.encode("Testing confidence bounds.")
        assert 0.0 <= result.confidence <= 1.0

    def test_top_emotion_is_argmax(self):
        result = self.enc.encode("I am furious!")
        assert result.top_emotion == max(result.scores, key=result.scores.get)


# ---------------------------------------------------------------------------
# AdaptiveFusion tests
# ---------------------------------------------------------------------------

class TestAdaptiveFusion:
    def _make_score(self, modality: str, top_emotion: str, confidence: float):
        from emotionlens.emotions import ModalityScore, EmotionLabel
        n = len(EmotionLabel)
        rest = (1.0 - confidence) / (n - 1)
        scores = {e.value: rest for e in EmotionLabel}
        scores[top_emotion] = confidence
        return ModalityScore(modality=modality, scores=scores)

    def test_single_modality_passthrough(self):
        from emotionlens.fusion import AdaptiveFusion
        fusion = AdaptiveFusion()
        ms = self._make_score("text", "joy", 0.8)
        fused, weights = fusion.fuse([ms])
        assert abs(fused["joy"] - 0.8) < 1e-3
        assert weights == {"text": 1.0}

    def test_empty_returns_uniform(self):
        from emotionlens.fusion import AdaptiveFusion
        from emotionlens.emotions import EmotionLabel
        fusion = AdaptiveFusion()
        fused, weights = fusion.fuse([])
        assert abs(sum(fused.values()) - 1.0) < 1e-4
        assert weights == {}

    def test_confidence_gating_weights_by_confidence(self):
        from emotionlens.fusion import AdaptiveFusion
        fusion = AdaptiveFusion(strategy="confidence_gating")
        high = self._make_score("text", "joy", 0.9)
        low  = self._make_score("audio", "anger", 0.3)
        _, weights = fusion.fuse([high, low])
        assert weights["text"] > weights["audio"]

    def test_weighted_average_equal_weights(self):
        from emotionlens.fusion import AdaptiveFusion
        fusion = AdaptiveFusion(strategy="weighted_average")
        ms1 = self._make_score("text", "joy", 0.8)
        ms2 = self._make_score("audio", "joy", 0.8)
        fused, _ = fusion.fuse([ms1, ms2])
        assert abs(fused["joy"] - 0.8) < 0.01

    def test_attention_fusion_sums_to_one(self):
        from emotionlens.fusion import AdaptiveFusion
        fusion = AdaptiveFusion(strategy="attention")
        ms1 = self._make_score("text", "joy", 0.7)
        ms2 = self._make_score("audio", "sadness", 0.6)
        fused, _ = fusion.fuse([ms1, ms2])
        assert abs(sum(fused.values()) - 1.0) < 5e-4

    def test_all_strategies_return_valid_distribution(self):
        from emotionlens.fusion import AdaptiveFusion
        for strategy in ["weighted_average", "confidence_gating", "attention"]:
            fusion = AdaptiveFusion(strategy=strategy)
            ms1 = self._make_score("text", "joy", 0.7)
            ms2 = self._make_score("audio", "anger", 0.5)
            fused, _ = fusion.fuse([ms1, ms2])
            assert abs(sum(fused.values()) - 1.0) < 5e-4
            assert all(0.0 <= v <= 1.0 for v in fused.values())


# ---------------------------------------------------------------------------
# EmotionPipeline tests
# ---------------------------------------------------------------------------

class TestEmotionPipeline:
    def setup_method(self):
        from emotionlens.pipeline import EmotionPipeline
        self.pipe = EmotionPipeline(explain=True)

    def test_text_only_prediction(self):
        result = self.pipe.predict(text="I'm so happy!")
        from emotionlens.emotions import EmotionResult, EmotionLabel
        assert isinstance(result, EmotionResult)
        assert isinstance(result.label, EmotionLabel)
        assert 0.0 <= result.confidence <= 1.0

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            self.pipe.predict()

    def test_result_has_vad(self):
        from emotionlens.emotions import VADScore
        result = self.pipe.predict(text="Life is beautiful")
        assert isinstance(result.vad, VADScore)

    def test_result_has_explanation(self):
        result = self.pipe.predict(text="Everything is terrible today.")
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10

    def test_result_metadata_has_latency(self):
        result = self.pipe.predict(text="Quick test")
        assert "latency_ms" in result.metadata

    def test_to_dict_serialisable(self):
        import json
        result = self.pipe.predict(text="Testing JSON output.")
        d = result.to_dict()
        s = json.dumps(d)
        assert len(s) > 10

    def test_batch_predict(self):
        texts = ["Happy day!", "Terrible loss.", "Just another day."]
        results = self.pipe.batch_predict(texts, show_progress=False)
        assert len(results) == 3

    def test_stream_predict(self):
        texts = ["A", "B", "C"]
        results = list(self.pipe.stream_predict(texts))
        assert len(results) == 3

    def test_physiological_input(self):
        result = self.pipe.predict(physiological={"heart_rate": 100, "eeg_alpha": 0.2})
        from emotionlens.emotions import EmotionResult
        assert isinstance(result, EmotionResult)

    def test_str_representation(self):
        result = self.pipe.predict(text="I feel great!")
        s = str(result)
        assert "EMOTION" in s.upper() or result.label.value.upper() in s.upper()


# ---------------------------------------------------------------------------
# Explainer tests
# ---------------------------------------------------------------------------

class TestEmotionExplainer:
    def test_explain_returns_string(self):
        from emotionlens.pipeline import EmotionPipeline
        pipe = EmotionPipeline(explain=True)
        result = pipe.predict(text="I am thrilled!")
        explanation = result.explanation
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explanation_mentions_label(self):
        from emotionlens.pipeline import EmotionPipeline
        pipe = EmotionPipeline(explain=True)
        result = pipe.predict(text="I am devastated.")
        # Explanation should mention the predicted label
        assert result.label.value.upper() in result.explanation.upper()
