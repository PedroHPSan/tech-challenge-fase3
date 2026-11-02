"""DAG de treino/retreino: ingestão do CSV -> treino -> salvamento do modelo."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline import ingest, train

with DAG(
    dag_id="retrain_text_classifier",
    description="Carrega o CSV, treina o classificador TF-IDF e salva o modelo",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["techchallenge", "nlp"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest_data", python_callable=ingest)
    train_task = PythonOperator(task_id="train_model", python_callable=train)

    ingest_task >> train_task
