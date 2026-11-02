import joblib
import pandas as pd

from src.train import load_onnx, predict_onnx, train_and_save
from tests.conftest import SAMPLES


def test_train_salva_modelo_e_metricas(csv_path, tmp_path):
    model_path = str(tmp_path / "m.joblib")
    metrics = train_and_save(csv_path, model_path)
    assert {"accuracy", "f1_macro", "cv_accuracy_mean"} <= set(metrics)
    model = joblib.load(model_path)
    assert set(model.classes_) == {"Cancer", "Diabetes", "Cardiology", "Hypertension"}


def test_train_exporta_onnx_equivalente(csv_path, tmp_path):
    model_path = tmp_path / "m.joblib"
    train_and_save(csv_path, str(model_path))
    pipe = joblib.load(model_path)
    session = load_onnx(str(model_path.with_suffix(".onnx")))
    texts = [t for group in SAMPLES.values() for t in group] + ["Unseen text about the heart"]
    labels, proba = predict_onnx(session, texts)
    assert list(labels) == list(pipe.predict(pd.DataFrame({"text": texts})))
    assert proba.shape == (len(texts), 4)


def test_train_com_sinais_clinicos_usa_features_no_onnx(features_csv_path, tmp_path):
    model_path = tmp_path / "m.joblib"
    metrics = train_and_save(features_csv_path, str(model_path))
    assert set(metrics["features"]) == {"hr", "sbp"}
    session = load_onnx(str(model_path.with_suffix(".onnx")))
    assert {i.name for i in session.get_inputs()} == {"text", "hr", "sbp"}
    # Sem os sinais, a predição ainda funciona (mediana do treino).
    labels, proba = predict_onnx(session, ["Insulin therapy controls diabetes."])
    assert proba.shape == (1, 4) and labels[0] in set(SAMPLES)
    # O sinal muda a probabilidade: FC alta empurra para Hypertension.
    _, baixa = predict_onnx(session, ["x"], {"hr": [70], "sbp": [120]})
    _, alta = predict_onnx(session, ["x"], {"hr": [120], "sbp": [120]})
    assert not (baixa == alta).all()
