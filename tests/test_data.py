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
    path.write_bytes(
        "Chief_complain;KTAS_expert;Outra\nChest Pain;2;x\nfever;3;x\nrash;5;x\n".encode("latin-1")
    )
    df = load_dataset(str(path))
    assert list(df["text"]) == ["chest pain", "fever", "rash"]
    assert list(df["label"]) == ["urgente", "atencao", "normal"]
