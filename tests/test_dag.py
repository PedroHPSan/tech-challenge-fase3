import pytest

pytest.importorskip("airflow")


def test_dag_carrega_e_tem_ordem_correta():
    from airflow.models import DagBag

    bag = DagBag(dag_folder="dags", include_examples=False)
    assert not bag.import_errors
    dag = bag.get_dag("retrain_text_classifier")
    assert [t.task_id for t in dag.topological_sort()] == ["ingest_data", "train_model"]
