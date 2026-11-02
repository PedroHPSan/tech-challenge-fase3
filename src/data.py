"""Carga, validação e divisão do dataset de textos médicos."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("text", "label")


def load_dataset(path: str) -> pd.DataFrame:
    """Lê o CSV, valida as colunas e remove linhas vazias."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def split_by_text(
    df: pd.DataFrame, texts_per_label: int = 1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste sem vazamento.

    O dataset repete poucas frases muitas vezes. Um split aleatório colocaria a
    mesma frase no treino e no teste e inflaria a métrica. Aqui, frases inteiras
    (`texts_per_label` por classe) vão para o teste e nunca aparecem no treino.
    """
    rng = np.random.default_rng(seed)
    test_texts: list[str] = []
    for _, group in df.groupby("label"):
        unique = sorted(group["text"].unique())
        if len(unique) <= texts_per_label:
            raise ValueError("Poucas frases únicas por classe para separar um teste.")
        test_texts.extend(rng.choice(unique, size=texts_per_label, replace=False))
    is_test = df["text"].isin(test_texts)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
