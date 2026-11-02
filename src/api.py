"""API de inferência: recebe o texto do laudo e devolve a classificação."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.config import settings
from src.features import shock_index
from src.train import load_onnx, predict_onnx

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
    text: str = Field(min_length=1, max_length=5000, description="Queixa principal do paciente")
    # Sinais medidos na chegada. Todos opcionais: o que faltar é preenchido pela mediana do treino.
    age: int | None = Field(None, ge=0, le=120, description="Idade em anos")
    sex: int | None = Field(None, ge=1, le=2, description="1 = feminino, 2 = masculino")
    arrival_mode: int | None = Field(None, ge=1, le=7, description="Meio de chegada (código KTAS)")
    injury: int | None = Field(None, ge=1, le=2, description="1 = não, 2 = trauma")
    mental: int | None = Field(
        None, ge=1, le=4, description="1 alerta, 2 voz, 3 dor, 4 sem resposta"
    )
    pain: int | None = Field(None, ge=0, le=1, description="1 = com dor")
    nrs_pain: float | None = Field(None, ge=0, le=10, description="Intensidade da dor (0-10)")
    sbp: float | None = Field(None, gt=0, le=300, description="Pressão sistólica (mmHg)")
    dbp: float | None = Field(None, gt=0, le=200, description="Pressão diastólica (mmHg)")
    hr: float | None = Field(None, gt=0, le=250, description="Frequência cardíaca (bpm)")
    rr: float | None = Field(None, gt=0, le=80, description="Frequência respiratória (irpm)")
    bt: float | None = Field(None, ge=25, le=45, description="Temperatura (°C)")
    saturation: float | None = Field(None, gt=0, le=100, description="Saturação de O2 (%)")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Serve a versão ONNX do modelo (Etapa 4: menor latência que o pipeline scikit-learn).
    path = Path(settings.model_path).with_suffix(".onnx")
    app.state.model = load_onnx(str(path)) if path.exists() else None
    yield


app = FastAPI(title="Triagem de urgência em pronto-socorro", version="0.1.0", lifespan=lifespan)
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
    features = {name: [value] for name, value in req.model_dump(exclude={"text"}).items()}
    features["shock_index"] = [shock_index(req.hr, req.sbp)]
    labels, proba = predict_onnx(model, [req.text], features)
    label = str(labels[0])
    PREDICTIONS.labels(label).inc()
    return PredictResponse(
        label=label,
        confidence=round(float(proba[0].max()), 4),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
