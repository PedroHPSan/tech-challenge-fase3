import json

from src.pipeline import ingest, train


def test_pipeline_ingest_e_train(csv_path, tmp_path):
    staging = ingest(csv_path, str(tmp_path / "staging.csv"))
    metrics = train(staging, str(tmp_path / "m.joblib"), str(tmp_path / "metrics.json"))
    assert (tmp_path / "m.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
