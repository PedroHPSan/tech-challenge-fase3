"""Treino e avaliação do classificador (TF-IDF no texto + sinais clínicos)."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.data import load_dataset, split_by_text
from src.features import usable_features


def build_pipeline(model: str = "logreg", seed: int = 42, features: Sequence[str] = ()) -> Pipeline:
    """Monta o pipeline: TF-IDF no texto + sinais clínicos (mediana e padronização) + modelo.

    Valores ausentes dos sinais são preenchidos com a mediana do treino, então a API
    funciona mesmo quando só o texto é informado.
    """
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")
    blocks = [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True), "text")
    ]
    if features:
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        blocks.append(("num", numeric, list(features)))
    return Pipeline([("pre", ColumnTransformer(blocks)), ("clf", clf)])


def evaluate(pipe: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    pred = pipe.predict(test)
    return {
        "accuracy": float(accuracy_score(test["label"], pred)),
        "f1_macro": float(f1_score(test["label"], pred, average="macro")),
    }


def export_onnx(pipe: Pipeline, path: str, features: Sequence[str] = ()) -> None:
    """Converte o pipeline treinado para ONNX (inferência mais rápida no ONNX Runtime).

    Entradas do grafo: `text` (string) e uma entrada float por feature clínica.
    """
    pre = pipe.named_steps["pre"]
    options = {
        # locale "C": a imagem slim não tem en_US.UTF-8, padrão do StringNormalizer.
        id(pre.named_transformers_["tfidf"]): {"locale": "C"},
        # Probabilidades como tensor (sem ZipMap), mais rápido de ler.
        id(pipe.named_steps["clf"]): {"zipmap": False},
    }
    initial_types = [("text", StringTensorType([None, 1]))]
    initial_types += [(name, FloatTensorType([None, 1])) for name in features]
    onx = to_onnx(pipe, initial_types=initial_types, options=options)
    Path(path).write_bytes(onx.SerializeToString())


def load_onnx(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict_onnx(
    session: ort.InferenceSession,
    texts: Sequence[str],
    features: Mapping[str, Sequence[float | None]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (rótulos, probabilidades por classe) para uma lista de textos.

    `features` traz uma lista de valores por sinal clínico; o que faltar (ou for None)
    entra como NaN e é preenchido pela mediana do treino dentro do próprio grafo.
    """
    features = features or {}
    n = len(texts)
    feed = {"text": np.array(texts, dtype=object).reshape(-1, 1)}
    for inp in session.get_inputs():
        if inp.name == "text":
            continue
        values = features.get(inp.name) or [None] * n
        column = [np.nan if v is None else v for v in values]
        feed[inp.name] = np.array(column, dtype=np.float32).reshape(-1, 1)
    labels, proba = session.run(None, feed)
    return labels, proba


def train_and_save(
    data_path: str | None = None,
    model_path: str | None = None,
    model: str = "logreg",
    seed: int | None = None,
    use_features: bool = True,
) -> dict:
    """Treina, avalia em queixas inéditas, salva o modelo e devolve as métricas.

    Com `use_features=False` o modelo usa só o texto (referência para comparação).
    """
    data_path = data_path or settings.data_path
    model_path = model_path or settings.model_path
    seed = settings.seed if seed is None else seed

    df = load_dataset(data_path)
    features = usable_features(df) if use_features else []
    train, test = split_by_text(df, seed=seed)

    pipe = build_pipeline(model, seed, features)
    pipe.fit(train, train["label"])
    metrics = evaluate(pipe, test)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        build_pipeline(model, seed, features), df, df["label"], groups=df["text"], cv=cv
    )
    metrics.update(
        cv_accuracy_mean=float(scores.mean()),
        n_samples=len(df),
        n_unique_texts=int(df["text"].nunique()),
        n_test_unique_texts=int(test["text"].nunique()),
        features=features,
    )

    # Modelo final: treina com todos os dados antes de salvar.
    pipe.fit(df, df["label"])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    export_onnx(pipe, str(Path(model_path).with_suffix(".onnx")), features)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
