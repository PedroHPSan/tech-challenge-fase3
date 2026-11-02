import joblib

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
    assert list(labels) == list(pipe.predict(texts))
    assert proba.shape == (len(texts), 4)
