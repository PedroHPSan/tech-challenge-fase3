import pandas as pd
import pytest

SAMPLES = {
    "Cancer": [
        "Chemotherapy and radiation therapy treat cancer.",
        "Tumor screening saves lives.",
    ],
    "Diabetes": [
        "Insulin therapy controls diabetes.",
        "Glucose monitoring helps diabetic patients.",
    ],
    "Cardiology": [
        "Angioplasty restores blood flow in arteries.",
        "Echocardiography images the heart.",
    ],
    "Hypertension": [
        "ACE inhibitors lower blood pressure.",
        "Salt reduction controls hypertension.",
    ],
}


@pytest.fixture
def csv_path(tmp_path):
    rows = [(t, label) for label, texts in SAMPLES.items() for t in texts for _ in range(5)]
    path = tmp_path / "data.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def features_csv_path(tmp_path):
    """Mesmo dataset, com sinais clínicos (hr e sbp) que variam conforme a classe."""
    base = {"Cancer": 70, "Diabetes": 85, "Cardiology": 100, "Hypertension": 115}
    rows = [
        (t, label, base[label] + i, 120 + i)
        for label, texts in SAMPLES.items()
        for t in texts
        for i in range(5)
    ]
    path = tmp_path / "features.csv"
    pd.DataFrame(rows, columns=["text", "label", "hr", "sbp"]).to_csv(path, index=False)
    return str(path)
