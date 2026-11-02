"""API de inferência: recebe o texto do laudo e devolve a classificação."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="Texto do laudo médico")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(settings.model_path)
    app.state.model = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Triagem de laudos médicos", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    loaded = app.state.model is not None
    return {"status": "ok" if loaded else "model_not_loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    start = time.perf_counter()
    proba = model.predict_proba([req.text])[0]
    idx = int(proba.argmax())
    return PredictResponse(
        label=str(model.classes_[idx]),
        confidence=round(float(proba[idx]), 4),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
