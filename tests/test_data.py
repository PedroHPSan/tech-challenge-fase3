import pandas as pd
import pytest

from src.data import load_dataset, split_by_text


def test_load_dataset_valida_colunas(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_dataset(str(bad))


def test_split_nao_vaza_frases(csv_path):
    df = load_dataset(csv_path)
    train, test = split_by_text(df)
    assert set(train["text"]).isdisjoint(set(test["text"]))
    assert set(test["label"]) == set(df["label"])


def test_load_dataset_ktas_mapeia_urgencia(tmp_path):
    path = tmp_path / "ktas.csv"
    linhas = [
        "Chief_complain;KTAS_expert;HR;SBP;Outra",
        "Chest Pain;2;100;80;x",
        "fever;3;??;120;x",
        "rash;5;70;140;x",
    ]
    path.write_bytes("\n".join(linhas).encode("latin-1"))
    df = load_dataset(str(path))
    assert list(df["text"]) == ["chest pain", "fever", "rash"]
    assert list(df["label"]) == ["urgente", "atencao", "normal"]
    assert df["hr"].isna().tolist() == [False, True, False]  # "??" vira NaN
    assert round(df["shock_index"].iloc[0], 3) == 1.25
    assert "Outra" not in df.columns
