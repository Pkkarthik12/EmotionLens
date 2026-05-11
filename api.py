"""
EmotionLens REST API
====================

Run with:
    uvicorn emotionlens.api:app --reload

Or via the CLI:
    emotionlens serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "The API requires FastAPI and Uvicorn.\n"
        "Install with: pip install emotionlens[api]"
    ) from e

from emotionlens.pipeline import EmotionPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EmotionLens API",
    description=(
        "Multimodal emotion intelligence — analyse text, audio, "
        "and image inputs for emotional state prediction."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialised once on startup)
_pipeline: Optional[EmotionPipeline] = None


@app.on_event("startup")
async def startup() -> None:
    global _pipeline
    logger.info("Loading EmotionLens pipeline …")
    _pipeline = EmotionPipeline(explain=True)
    logger.info("Pipeline ready.")


def get_pipeline() -> EmotionPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(..., description="Text to analyse.", min_length=1, max_length=4096)
    explain: bool = Field(True, description="Include a natural-language explanation.")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyse.", min_items=1, max_items=100)


class VADResponse(BaseModel):
    valence: float
    arousal: float
    dominance: float


class EmotionResponse(BaseModel):
    label: str
    confidence: float
    all_scores: Dict[str, float]
    vad: VADResponse
    explanation: str
    fusion_weights: Dict[str, float]
    metadata: Dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    return {"name": "EmotionLens API", "version": "0.1.0", "status": "ok"}


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "healthy", "pipeline_loaded": _pipeline is not None}


@app.post("/predict/text", response_model=EmotionResponse, tags=["prediction"])
async def predict_text(req: TextRequest):
    """Predict emotion from a text string."""
    pipe = get_pipeline()
    try:
        result = pipe.predict(text=req.text)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    d = result.to_dict()
    return EmotionResponse(
        label=d["label"],
        confidence=d["confidence"],
        all_scores=d["all_scores"],
        vad=VADResponse(**d["vad"]),
        explanation=d["explanation"] if req.explain else "",
        fusion_weights=d["fusion_weights"],
        metadata=d["metadata"],
    )


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(req: BatchRequest):
    """Predict emotions for up to 100 texts in a single call."""
    pipe = get_pipeline()
    try:
        results = pipe.batch_predict(req.texts, show_progress=False)
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"results": [r.to_dict() for r in results], "count": len(results)}


@app.post("/predict/audio", tags=["prediction"])
async def predict_audio(file: UploadFile = File(...)):
    """Predict emotion from an uploaded audio file (WAV, MP3, FLAC)."""
    import tempfile, os

    pipe = get_pipeline()
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = pipe.predict(audio_path=tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return result.to_dict()


from pathlib import Path  # noqa: E402 (needed inside endpoint above)
