import joblib

from src.train import train_and_save


def test_train_salva_modelo_e_metricas(csv_path, tmp_path):
    model_path = str(tmp_path / "m.joblib")
    metrics = train_and_save(csv_path, model_path)
    assert {"accuracy", "f1_macro", "cv_accuracy_mean"} <= set(metrics)
    model = joblib.load(model_path)
    assert set(model.classes_) == {"Cancer", "Diabetes", "Cardiology", "Hypertension"}
