"""Treino e avaliação do classificador de texto (TF-IDF + modelo linear)."""

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import settings
from src.data import load_dataset, split_by_text


def build_pipeline(model: str = "logreg", seed: int = 42) -> Pipeline:
    """Monta o pipeline TF-IDF + classificador."""
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)), ("clf", clf)])


def evaluate(pipe: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    pred = pipe.predict(test["text"])
    return {
        "accuracy": float(accuracy_score(test["label"], pred)),
        "f1_macro": float(f1_score(test["label"], pred, average="macro")),
    }


def export_onnx(pipe: Pipeline, path: str) -> None:
    """Converte o pipeline treinado para ONNX (inferência mais rápida no ONNX Runtime)."""
    options = {
        # locale "C": a imagem slim não tem en_US.UTF-8, padrão do StringNormalizer.
        id(pipe.named_steps["tfidf"]): {"locale": "C"},
        # Probabilidades como tensor (sem ZipMap), mais rápido de ler.
        id(pipe.named_steps["clf"]): {"zipmap": False},
    }
    onx = to_onnx(pipe, initial_types=[("text", StringTensorType([None, 1]))], options=options)
    Path(path).write_bytes(onx.SerializeToString())


def load_onnx(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict_onnx(session: ort.InferenceSession, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (rótulos, probabilidades por classe) para uma lista de textos."""
    labels, proba = session.run(None, {"text": np.array(texts, dtype=object).reshape(-1, 1)})
    return labels, proba


def train_and_save(
    data_path: str | None = None,
    model_path: str | None = None,
    model: str = "logreg",
    seed: int | None = None,
) -> dict:
    """Treina, avalia em frases inéditas, salva o modelo e devolve as métricas."""
    data_path = data_path or settings.data_path
    model_path = model_path or settings.model_path
    seed = settings.seed if seed is None else seed

    df = load_dataset(data_path)
    train, test = split_by_text(df, seed=seed)

    pipe = build_pipeline(model, seed)
    pipe.fit(train["text"], train["label"])
    metrics = evaluate(pipe, test)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        build_pipeline(model, seed),
        df["text"],
        df["label"],
        groups=df["text"],
        cv=cv,
    )
    metrics.update(
        cv_accuracy_mean=float(scores.mean()),
        n_samples=len(df),
        n_unique_texts=int(df["text"].nunique()),
        n_test_unique_texts=int(test["text"].nunique()),
    )

    # Modelo final: treina com todos os dados antes de salvar.
    pipe.fit(df["text"], df["label"])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    export_onnx(pipe, str(Path(model_path).with_suffix(".onnx")))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
