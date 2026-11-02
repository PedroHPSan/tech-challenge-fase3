# Tech Challenge Fase 3 — Triagem automática de laudos médicos

Serviço de classificação de textos médicos servido por API REST em container
Docker, com pipeline de CI (GitHub Actions) e DAG de retreino (Airflow).

## Dataset

`data/raw/medical_text_classification_fake_dataset.csv`: colunas `text` e `label`.

| Característica | Valor |
|---|---|
| Linhas | 1.000 |
| Classes | Cancer, Diabetes, Cardiology, Hypertension (250 cada) |
| Frases únicas | **20** (5 por classe, repetidas várias vezes) |

**Limitações em relação ao enunciado:** o enunciado sugere ≥ 2.000 amostras e uma
classe de urgência (normal / atenção / urgente). Este dataset tem 1.000 linhas e o
alvo é a **especialidade** do texto. Não criamos um mapeamento especialidade → urgência
porque ele não teria base clínica. Para trocar o alvo por urgência basta fornecer um CSV
com as mesmas colunas (`text`, `label`) e apontar `DATA_PATH` para ele.

### Avaliação sem vazamento

Como só existem 20 frases únicas, um split aleatório coloca a mesma frase no treino e no
teste. Nesse caso a acurácia chega a **100%**, sem medir generalização. Em `src/data.py`
separamos frases inteiras (1 por classe) para o teste, e a validação cruzada agrupa por
frase. Resultado atual (`python -m src.train`):

| Métrica | Valor |
|---|---|
| Acurácia em frases inéditas (4 frases) | 0,27 |
| F1 macro em frases inéditas | 0,17 |
| Acurácia média na validação cruzada agrupada (5 folds) | 0,39 |
| Referência (acaso, 4 classes) | 0,25 |

O modelo só reconhece as frases que viu no treino. Com 5 frases por classe, não há
vocabulário suficiente para generalizar. Um dataset maior e mais variado é o principal
ponto de melhoria do projeto.

## Estrutura do projeto

```
├── data/raw/            # dataset (CSV versionado, 80 KB)
├── src/
│   ├── config.py        # configuração via .env
│   ├── data.py          # carga, validação e split sem vazamento
│   ├── train.py         # TF-IDF + modelo, avaliação e salvamento
│   ├── pipeline.py      # etapas ingestão/treino usadas pela DAG
│   └── api.py           # API FastAPI (/health e /predict)
├── dags/retrain_dag.py  # DAG do Airflow
├── scripts/benchmark_latency.py
├── tests/               # pytest
├── .github/workflows/ci.yml
└── Dockerfile
```

## Como executar

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) e Docker.

```bash
poetry install
cp .env.example .env
poetry run python -m src.train          # gera models/model.joblib
poetry run uvicorn src.api:app --reload # API em http://localhost:8000/docs
```

Exemplo de chamada:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "Insulin therapy remains the cornerstone for type 1 diabetes treatment."}'
# {"label":"Diabetes","confidence":0.9482,"latency_ms":6.939}
```

### Docker

```bash
docker build -t triagem-api:0.1 .
docker run --rm -p 8000:8000 triagem-api:0.1
```

O modelo é treinado durante o build, então a imagem já sobe pronta.

## Baseline de latência (local, container Docker)

Medido com `poetry run python -m scripts.benchmark_latency` (200 requisições
sequenciais a `localhost:8000`, após aquecimento), em um MacBook com Docker Desktop:

| Métrica | Tempo |
|---|---|
| Média | 1,07 ms |
| p50 | 1,03 ms |
| p95 | 1,37 ms |
| Máximo | 3,18 ms |

É o ponto de comparação para a otimização da Etapa 4.

## CI (GitHub Actions)

`.github/workflows/ci.yml` roda a cada `push` e `pull_request`, em três jobs
encadeados: **lint** (`ruff check` e `ruff format --check`) → **test** (`pytest`) →
**build** (`docker build`). Para rodar o mesmo localmente:

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest
```

## Orquestração (Airflow)

A DAG `retrain_text_classifier` (`dags/retrain_dag.py`) tem duas tarefas encadeadas:

1. `ingest_data`: lê e valida o CSV e grava `models/staging.csv`;
2. `train_model`: treina o classificador, salva `models/model.joblib` e
   `models/metrics.json`.

A lógica fica em `src/pipeline.py` (testável sem Airflow). O agendamento é semanal.
O Airflow é uma dependência opcional por ser pesada:

```bash
poetry install --with airflow
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__LOAD_EXAMPLES=False PYTHONPATH=.
poetry run airflow db migrate
poetry run airflow dags test retrain_text_classifier 2026-11-02 -S dags
poetry run pytest tests/test_dag.py   # integridade da DAG (pulado sem Airflow)
```

> Se o `poetry install --with airflow` apresentar conflito de versões, instale o Airflow
> em um ambiente separado usando o arquivo de *constraints* oficial da versão 2.10.

## Decisão arquitetural de deploy

**Contexto:** triagem de urgência de laudos. Uma resposta rápida ao profissional de saúde
é o objetivo, e o modelo é leve (TF-IDF + modelo linear, poucos MB).

**Inferência: tempo real (online).** O laudo chega, é classificado e a resposta volta na
mesma requisição. Um processamento em lote atrasaria justamente os casos urgentes. O
container FastAPI é *stateless*, então escala horizontalmente por réplicas.

**Treino/retreino: lote (batch).** Roda em agenda (Airflow) ou sob demanda, fora do
caminho da requisição, e publica um novo artefato de modelo.

**Nuvem escolhida: AWS** (Azure e GCP são equivalentes para este caso):

| Necessidade | Serviço AWS | Equivalentes |
|---|---|---|
| Registro da imagem | ECR | ACR / Artifact Registry |
| Execução da API | ECS Fargate atrás de um ALB, com auto scaling por CPU/latência | Container Apps / Cloud Run |
| Artefatos do modelo | S3 versionado | Blob Storage / GCS |
| Orquestração do retreino | MWAA (Airflow gerenciado) ou a DAG deste repositório em container | Cloud Composer |
| Métricas e logs | CloudWatch + Prometheus/Grafana | Azure Monitor / Cloud Monitoring |

**Por que não GPU nem serverless por função:** o modelo roda em CPU em ~1 ms, então GPU
não compensa o custo. Funções sob demanda (Lambda) sofreriam com *cold start* num
fluxo clínico.

**Cuidados para produção real:** laudos são dados pessoais de saúde (LGPD). Seria
necessário TLS, autenticação na API, rede privada, criptografia em repouso e logs sem o
texto do laudo. A saída do modelo deve apoiar a priorização, e a decisão clínica continua
com o profissional.
