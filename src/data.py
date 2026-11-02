"""Carga, validação e divisão do dataset de textos médicos."""

import numpy as np
import pandas as pd

from src.features import add_derived

REQUIRED_COLUMNS = ("text", "label")

# Dataset KTAS (pronto-socorro coreano): a queixa principal vira o texto e o nível
# de triagem definido pelos especialistas (1-5) vira a classe de urgência. Os sinais
# medidos na chegada do paciente viram features. Ficam de fora KTAS_RN (nível dado pela
# enfermagem), diagnóstico, destino e erro de triagem, porque revelam o resultado.
KTAS_TEXT_COLUMN = "Chief_complain"
KTAS_LABEL_COLUMN = "KTAS_expert"
KTAS_TO_URGENCY = {1: "urgente", 2: "urgente", 3: "atencao", 4: "normal", 5: "normal"}
KTAS_FEATURES = {
    "Age": "age",
    "Sex": "sex",
    "Arrival mode": "arrival_mode",
    "Injury": "injury",
    "Mental": "mental",
    "Pain": "pain",
    "NRS_pain": "nrs_pain",
    "SBP": "sbp",
    "DBP": "dbp",
    "HR": "hr",
    "RR": "rr",
    "BT": "bt",
    "Saturation": "saturation",
}


def _read_ktas(path: str) -> pd.DataFrame:
    wanted = {KTAS_TEXT_COLUMN, KTAS_LABEL_COLUMN, *KTAS_FEATURES}
    raw = pd.read_csv(path, sep=";", encoding="latin-1", usecols=lambda c: c in wanted)
    level = pd.to_numeric(raw[KTAS_LABEL_COLUMN], errors="coerce")
    df = pd.DataFrame(
        {"text": raw[KTAS_TEXT_COLUMN].str.lower(), "label": level.map(KTAS_TO_URGENCY)}
    )
    for source, name in KTAS_FEATURES.items():
        if source in raw.columns:
            # Células com lixo ("??", "#BOÞ!") viram NaN.
            df[name] = pd.to_numeric(raw[source].astype(str).str.replace(",", "."), errors="coerce")
    return add_derived(df)


def load_dataset(path: str) -> pd.DataFrame:
    """Lê o CSV (formato `text,label` ou o bruto KTAS), valida e remove linhas vazias."""
    with open(path, encoding="latin-1") as f:
        is_ktas = KTAS_TEXT_COLUMN in f.readline()
    df = _read_ktas(path) if is_ktas else pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def split_by_text(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste sem vazamento.

    A mesma queixa aparece em várias linhas (e às vezes com classes diferentes). Um
    split aleatório colocaria a mesma queixa no treino e no teste e inflaria a
    métrica. Aqui, queixas inteiras (`test_size` das únicas de cada classe, pela classe
    mais frequente) vão para o teste e nunca aparecem no treino.
    """
    rng = np.random.default_rng(seed)
    majority = df.groupby("text")["label"].agg(lambda s: s.value_counts().index[0])
    test_texts: list[str] = []
    for _, texts in majority.groupby(majority):
        unique = sorted(texts.index)
        if len(unique) < 2:
            raise ValueError("Poucas frases únicas por classe para separar um teste.")
        n_test = min(len(unique) - 1, max(1, round(len(unique) * test_size)))
        test_texts.extend(rng.choice(unique, size=n_test, replace=False))
    is_test = df["text"].isin(test_texts)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
