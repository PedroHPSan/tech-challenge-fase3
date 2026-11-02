import pytest
from fastapi.testclient import TestClient

from src import api
from src.train import train_and_save


@pytest.fixture
def client(csv_path, tmp_path, monkeypatch):
    model_path = str(tmp_path / "m.joblib")
    train_and_save(csv_path, model_path)
    monkeypatch.setattr(api.settings, "model_path", model_path)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json={"text": "Insulin therapy for diabetes"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Diabetes"
    assert 0 <= body["confidence"] <= 1


def test_predict_texto_vazio_retorna_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_sem_modelo_retorna_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "model_path", str(tmp_path / "nao_existe.joblib"))
    with TestClient(api.app) as c:
        assert c.post("/predict", json={"text": "x"}).status_code == 503
