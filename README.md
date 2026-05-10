# 🔍 EmotionLens

> **Multimodal Emotion Intelligence** — fuse text, audio, image, and physiological signals to understand how people feel, with built-in explainability.

[![CI](https://github.com/yourusername/emotionlens/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/emotionlens/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why EmotionLens?

Most emotion-analysis libraries either:
- work on **text only** and ignore the voice tremor or the tense face behind the words, or
- output a single label with no explanation of *why* they predicted it.

**EmotionLens** solves both problems:

| Feature | EmotionLens |
|---|---|
| Modalities | Text · Audio · Image · Physiological |
| Output | 8 discrete labels + continuous VAD (Valence-Arousal-Dominance) |
| Fusion | Adaptive — confidence-gated, weighted, or cross-modal attention |
| Explainability | SHAP-style feature attribution + natural-language explanation |
| API | REST (FastAPI) + CLI + Python API |
| Dependencies | Zero mandatory heavy deps for text-only use |

---

## Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐
│  Text    │  │  Audio   │  │  Image   │  │ Physiological  │
│ encoder  │  │ encoder  │  │ encoder  │  │   encoder      │
│(RoBERTa/ │  │(MFCC+    │  │(EfficientNet│ (wavelet       │
│ lexicon) │  │ prosody) │  │-B0)      │  │  features)     │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬─────────┘
     │              │              │                │
     └──────────────┴──────────────┴────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Adaptive Fusion    │
                   │  (gating/attention) │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Emotion Predictor  │
                   │  8 labels + VAD     │
                   └──────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         SHAP explainer    REST API     Live dashboard
```

---

## Quick Start

### Installation

```bash
# Core (text analysis, no heavy deps)
pip install emotionlens

# With transformer-based text models
pip install "emotionlens[transformers]"

# Full stack (text + audio + image + API)
pip install "emotionlens[all]"

# From source
git clone https://github.com/yourusername/emotionlens
cd emotionlens
pip install -e ".[dev]"
```

### Python API

```python
from emotionlens import EmotionPipeline

pipe = EmotionPipeline()

# Text only
result = pipe.predict(text="I just got the promotion I've been working for!")
print(result)
# Emotion : JOY
# Confidence : [████████████████░░░░]  80%
# VAD      : V=+0.75  A=+0.48  D=+0.42
# Why      : Predicted emotion: JOY (80% confident). Emotional character:
#            pleasant, energetic, assertive. Key signals: "promotion",
#            "working" pushed toward current emotion.

# Multi-modal
result = pipe.predict(
    text="This is a disaster.",
    audio_path="recording.wav",
)
print(result.to_dict())  # JSON-serialisable

# Batch
results = pipe.batch_predict([
    "Great news!",
    "I'm devastated.",
    "Whatever.",
])
```

### CLI

```bash
# Single prediction
emotionlens predict "I can't believe how amazing this is!"

# Multi-modal with JSON output
emotionlens predict "This is terrible." --audio clip.wav --json

# Batch — one text per line
emotionlens batch --file sentences.txt --output results.jsonl

# Interactive demo
emotionlens demo

# REST API server
emotionlens serve --host 0.0.0.0 --port 8000
```

### REST API

```bash
# Start the server
emotionlens serve

# Predict from text
curl -X POST http://localhost:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am ecstatic!", "explain": true}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Happy!", "Sad.", "Angry!"]}'

# Audio prediction (upload a WAV file)
curl -X POST http://localhost:8000/predict/audio \
  -F "file=@speech.wav"
```

---

## Emotion Model

EmotionLens predicts **8 discrete emotions** (extended Ekman model):

| Emotion | Description |
|---------|-------------|
| joy | Happiness, elation, excitement |
| sadness | Grief, loss, loneliness |
| anger | Frustration, rage, irritability |
| fear | Anxiety, dread, nervousness |
| surprise | Astonishment, shock |
| disgust | Revulsion, distaste |
| contempt | Disdain, disrespect |
| neutral | Absence of strong affect |

And a continuous **VAD** (Valence-Arousal-Dominance) representation:

- **Valence** [-1 → +1]: how positive or negative the emotion is
- **Arousal** [-1 → +1]: how calm or excited
- **Dominance** [-1 → +1]: how in-control or submissive

---

## Fusion Strategies

| Strategy | When to use |
|----------|-------------|
| `confidence_gating` | Default. Weights modalities by their prediction confidence. Best for mismatched modalities. |
| `weighted_average` | When you know which modality is more reliable for your domain. |
| `attention` | Cross-modal soft attention. Best when modalities are complementary and equally reliable. |

```python
pipe = EmotionPipeline(
    fusion_strategy="attention",
    modality_weights={"text": 0.6, "audio": 0.4},  # for weighted_average
)
```

---

## Explainability

Every prediction includes:

1. **Feature attribution** — which words or acoustic features drove the prediction
2. **Modality weights** — how much each input contributed
3. **VAD characterisation** — human-readable description of the emotional space
4. **Counterfactual hint** — what would flip the prediction

```python
result = pipe.predict(text="I'm trying not to panic.")
print(result.explanation)
# Predicted emotion: FEAR (62% confident).
# Emotional character: unpleasant, energetic, submissive.
# Signal sources: text led the prediction — text (100%).
# Key signals: "panic" pushed toward current emotion.
# Alternative reading: ANGER (14%).
# What would change this: greater control signals would shift toward surprise.
```

---

## Project Structure

```
emotionlens/
├── emotionlens/
│   ├── __init__.py          # Public API
│   ├── emotions.py          # EmotionLabel, VADScore, EmotionResult
│   ├── pipeline.py          # EmotionPipeline (main entry point)
│   ├── fusion.py            # AdaptiveFusion (gating / attention)
│   ├── explainer.py         # EmotionExplainer (SHAP-style)
│   ├── api.py               # FastAPI REST server
│   ├── cli.py               # Click CLI
│   └── encoders/
│       ├── text_encoder.py  # RoBERTa / lexicon text encoder
│       ├── audio_encoder.py # MFCC + prosody audio encoder
│       └── image_encoder.py # EfficientNet image encoder
├── tests/
│   └── test_emotionlens.py  # 30+ unit & integration tests
├── notebooks/
│   └── quickstart.ipynb     # Interactive demo notebook
├── .github/workflows/ci.yml # GitHub Actions CI
├── pyproject.toml
└── README.md
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=emotionlens --cov-report=html

# Lint
ruff check emotionlens/

# Format
black emotionlens/ tests/

# Type check
mypy emotionlens/
```

---

## Roadmap

- [ ] Real-time streaming predictions via WebSocket
- [ ] Temporal emotion tracking (how emotion shifts across a conversation)
- [ ] Fine-tuning scripts for domain adaptation
- [ ] Docker image & Helm chart
- [ ] Web dashboard (React + D3 VAD space visualisation)
- [ ] ONNX export for edge deployment
- [ ] Multilingual support (mBERT backbone)

---

## Citation

If you use EmotionLens in your research, please cite:

```bibtex
@software{emotionlens2024,
  author = {Your Name},
  title  = {EmotionLens: Multimodal Emotion Intelligence},
  year   = {2024},
  url    = {https://github.com/yourusername/emotionlens},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
