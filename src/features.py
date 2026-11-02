"""Sinais clínicos usados junto com o texto da queixa."""

import numpy as np
import pandas as pd

# Colunas numéricas aceitas como entrada (todas opcionais na API).
BASE_FEATURES = (
    "age",
    "sex",
    "arrival_mode",
    "injury",
    "mental",
    "pain",
    "nrs_pain",
    "sbp",
    "dbp",
    "hr",
    "rr",
    "bt",
    "saturation",
)
# Calculada a partir das outras: frequência cardíaca / pressão sistólica.
DERIVED_FEATURES = ("shock_index",)
FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES


def shock_index(hr: float | None, sbp: float | None) -> float:
    """Índice de choque (FC / PAS). NaN se faltar algum valor ou se a PAS for inválida."""
    if hr is None or sbp is None or np.isnan(hr) or np.isnan(sbp) or sbp <= 0:
        return float("nan")
    return float(hr) / float(sbp)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as colunas derivadas quando as colunas de origem existem."""
    if {"hr", "sbp"} <= set(df.columns):
        sbp = df["sbp"].where(df["sbp"] > 0)
        df["shock_index"] = df["hr"] / sbp
    return df


def usable_features(df: pd.DataFrame) -> list[str]:
    """Features presentes no dataset e com pelo menos um valor preenchido."""
    return [c for c in FEATURE_COLUMNS if c in df.columns and df[c].notna().any()]
