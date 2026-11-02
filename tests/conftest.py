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
