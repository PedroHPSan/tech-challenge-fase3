"""API de inferência: recebe o texto do laudo e devolve a classificação."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.config import settings

REQUESTS = Counter(
    "http_requests_total", "Requisições HTTP recebidas", ["method", "path", "status"]
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Tempo de resposta das requisições HTTP",
    ["method", "path"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTIONS = Counter("predictions_total", "Predições por classe", ["label"])


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
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    # Usa o template da rota para não criar uma série por URL arbitrária.
    path = route.path if route else "other"
    LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response


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
    label = str(model.classes_[idx])
    PREDICTIONS.labels(label).inc()
    return PredictResponse(
        label=label,
        confidence=round(float(proba[idx]), 4),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
