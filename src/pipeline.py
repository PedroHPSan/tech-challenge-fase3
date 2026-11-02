"""Etapas do pipeline de treino, independentes do Airflow (facilita testar)."""

import json
from pathlib import Path

from src.config import settings
from src.data import load_dataset
from src.train import train_and_save


def ingest(data_path: str | None = None, staging_path: str = "models/staging.csv") -> str:
    """Lê e valida o CSV de origem e grava a cópia limpa usada no treino."""
    df = load_dataset(data_path or settings.data_path)
    Path(staging_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(staging_path, index=False)
    return staging_path


def train(
    staging_path: str = "models/staging.csv",
    model_path: str | None = None,
    metrics_path: str = "models/metrics.json",
) -> dict:
    """Treina com os dados preparados, salva o modelo e as métricas."""
    metrics = train_and_save(staging_path, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    return metrics
