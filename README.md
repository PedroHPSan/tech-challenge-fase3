# Tech Challenge Fase 3 — Triagem de urgência em pronto-socorro

Serviço que classifica a urgência de um paciente (**urgente**, **atenção** ou **normal**)
a partir da queixa principal e dos sinais clínicos medidos na chegada. O modelo é servido
por uma API REST (FastAPI) em Docker, com CI no GitHub Actions, retreino orquestrado por
uma DAG do Airflow, monitoramento com Prometheus + Grafana e inferência otimizada com
ONNX Runtime.

- **Vídeo (STAR):** _link a preencher após a gravação_
- **Detalhes do dataset e da escolha do modelo:** [docs/metodo.md](docs/metodo.md)

> **Sobre o enunciado:** ele fala em "laudos médicos". Usamos a **queixa principal** de
> pacientes de pronto-socorro (texto curto, uma a poucas palavras) porque é um dado real,
> público e com rótulo de urgência definido por especialistas. O fluxo é o mesmo: texto
> entra, classe de urgência sai.

## Resultados em resumo

| | Resultado |
|---|---|
| Modelo | TF-IDF + sinais clínicos + regressão logística (~32 KB em ONNX) |
| Acurácia em queixas inéditas | 0,75 (F1 macro 0,74); referência da classe mais frequente: 0,42 |
| Latência do modelo (scikit-learn → ONNX) | 1,115 ms → **0,033 ms** (~34x) |
| Latência da API em Docker (scikit-learn → ONNX) | 3,23 ms → **1,13 ms** (média) |
| CI | lint → test → build |
| Dashboard Grafana | 5 painéis (mínimo do enunciado: 3) |

## Como executar

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) e Docker.

### 1. API em Docker

```bash
docker build -t triagem-api:0.1 .
docker run --rm -p 8000:8000 triagem-api:0.1
```

O modelo é treinado durante o build, então a imagem já sobe pronta. A documentação
interativa fica em http://localhost:8000/docs.

### 2. Stack completa (API + Prometheus + Grafana)

```bash
docker compose up -d --build
poetry install
poetry run python -m scripts.benchmark_latency --n 1000   # gera tráfego
docker compose down                                        # para encerrar
```

| Serviço | Endereço |
|---|---|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin; leitura anônima liberada) |

### 3. Sem Docker (desenvolvimento)

```bash
poetry install
cp .env.example .env
poetry run python -m src.train           # gera models/model.joblib e models/model.onnx
poetry run uvicorn src.api:app --reload
```

### Exemplo de chamada

Só `text` é obrigatório; os sinais clínicos são opcionais.

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "chest pain", "age": 58, "hr": 110, "sbp": 95, "saturation": 92}'
# {"label":"urgente","confidence":0.986,"latency_ms":0.962}
```

Campos aceitos além de `text`: `age`, `sex`, `arrival_mode`, `injury`, `mental`, `pain`,
`nrs_pain`, `sbp`, `dbp`, `hr`, `rr`, `bt` e `saturation` (faixas e códigos em `/docs`).
O que faltar é preenchido pela mediana do treino, e o índice de choque (FC / PAS) é
calculado pela própria API.

## Estrutura do projeto

```
├── data/raw/            # dataset KTAS (CSV versionado, 126 KB)
├── src/
│   ├── config.py        # configuração via .env
│   ├── data.py          # carga, validação e split sem vazamento
│   ├── features.py      # sinais clínicos e índice de choque
│   ├── train.py         # treino, avaliação, salvamento e export ONNX
│   ├── pipeline.py      # etapas ingestão/treino usadas pela DAG
│   └── api.py           # API FastAPI (/health, /predict e /metrics)
├── dags/retrain_dag.py  # DAG do Airflow
├── monitoring/          # Prometheus e Grafana (datasource + dashboard provisionados)
├── docker-compose.yml   # API + Prometheus + Grafana
├── scripts/             # benchmarks de latência
├── tests/               # pytest
├── .github/workflows/ci.yml
└── Dockerfile
```

## Decisão arquitetural de deploy

**Contexto:** triagem de urgência em pronto-socorro. O objetivo é devolver a resposta ao
profissional de saúde rapidamente, e o modelo é leve (TF-IDF + modelo linear, poucos KB).

**Inferência: tempo real (online).** O paciente chega, é classificado e a resposta volta na
mesma requisição. Um processamento em lote atrasaria justamente os casos urgentes. O
container FastAPI é *stateless*, então escala horizontalmente por réplicas.

**Treino/retreino: lote (batch).** Roda em agenda (Airflow) ou sob demanda, fora do caminho
da requisição, e publica um novo artefato de modelo.

**Nuvem escolhida: AWS** (Azure e GCP são equivalentes para este caso):

| Necessidade | Serviço AWS | Equivalentes |
|---|---|---|
| Registro da imagem | ECR | ACR / Artifact Registry |
| Execução da API | ECS Fargate atrás de um ALB, com auto scaling por CPU/latência | Container Apps / Cloud Run |
| Artefatos do modelo | S3 versionado | Blob Storage / GCS |
| Orquestração do retreino | MWAA (Airflow gerenciado) ou a DAG deste repositório em container | Cloud Composer |
| Métricas e logs | CloudWatch + Prometheus/Grafana | Azure Monitor / Cloud Monitoring |

**Por que não GPU nem serverless por função:** o modelo roda em CPU em ~0,03 ms (ONNX), então
GPU não compensa o custo. Funções sob demanda (Lambda) sofreriam com *cold start* num fluxo
clínico.

**Cuidados para produção real:** os dados são pessoais de saúde (LGPD). Seria necessário TLS,
autenticação na API, rede privada, criptografia em repouso e logs sem o texto da queixa. A
saída do modelo deve apoiar a priorização, e a decisão clínica continua com o profissional.

## Otimização de latência (ONNX Runtime)

Ao fim do treino, `src/train.py` converte o pipeline (TF-IDF + sinais + regressão logística)
para ONNX com `skl2onnx` e grava `models/model.onnx` ao lado do `model.joblib`. A API carrega
o `.onnx` e faz a inferência com o **ONNX Runtime**. O pipeline scikit-learn continua salvo
para comparação.

Por que ONNX: o scikit-learn passa por várias camadas Python a cada chamada (validação,
vetorização, matriz esparsa, imputação). O ONNX Runtime executa o mesmo grafo em C++ numa
única chamada. Quantização e pruning dariam pouco ganho aqui, porque o modelo já é pequeno.

Detalhes da conversão:

- `zipmap=False`: as probabilidades saem como tensor, e não como lista de dicionários;
- `locale="C"` no TF-IDF: o operador que converte para minúsculas usa `en_US.UTF-8` por
  padrão, e esse locale não existe na imagem `python:3.11-slim`;
- o grafo tem uma entrada `text` e uma entrada float por sinal clínico. Sinal ausente entra
  como NaN e é preenchido pela mediana do treino dentro do grafo.

### Baseline e resultados

Medições em um MacBook com Docker Desktop (2026-11-02), com o modelo de texto + sinais. O
baseline (Etapa 1) é a API servindo o pipeline scikit-learn; o comparativo troca só o
backend do modelo, na mesma máquina e no mesmo container.

**API ponta a ponta em Docker** (`poetry run python -m scripts.benchmark_latency --n 1000`;
as requisições alternam pacientes com e sem sinais clínicos, após aquecimento):

| API | Média | p50 | p95 |
|---|---|---|---|
| scikit-learn (baseline) | 3,23 ms | 3,08 ms | 4,15 ms |
| ONNX Runtime | **1,13 ms** | **1,04 ms** | **1,70 ms** |

Na API o ganho é de ~65% (cerca de 2,9x). Há variação entre rodadas: em quatro rodadas o
scikit-learn ficou entre 3,06 e 3,82 ms e o ONNX entre 1,13 e 1,35 ms.

**Só o modelo** (`poetry run python -m scripts.compare_latency --n 3000`, 1 paciente por
chamada, entradas já montadas):

| Modelo | Média | p50 | p95 | Tamanho |
|---|---|---|---|---|
| scikit-learn (joblib) | 1,115 ms | 1,056 ms | 1,391 ms | 37 KB |
| ONNX Runtime | **0,033 ms** | **0,033 ms** | **0,036 ms** | 32 KB |

Ganho de **~34x** na média, com 100% de concordância de rótulos nos pacientes do dataset. O
modelo passou de ~1,1 ms para ~0,03 ms, e o tempo que sobra na API vem de HTTP, validação do
FastAPI e métricas. As probabilidades diferem em até ~0,01 porque a tokenização do ONNX não é
idêntica à do scikit-learn; `tests/test_train.py` garante os mesmos rótulos.

## CI (GitHub Actions)

`.github/workflows/ci.yml` roda a cada `push` e `pull_request`, em três jobs encadeados:
**lint** (`ruff check` e `ruff format --check`) → **test** (`pytest`) → **build**
(`docker build`). Para rodar o mesmo localmente:

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest
```

## Orquestração (Airflow)

A DAG `retrain_text_classifier` (`dags/retrain_dag.py`) tem duas tarefas encadeadas:

1. `ingest_data`: lê e valida o CSV e grava `models/staging.csv`;
2. `train_model`: treina o classificador e salva `models/model.joblib`,
   `models/model.onnx` (o que a API carrega) e `models/metrics.json`.

A lógica fica em `src/pipeline.py` (testável sem Airflow). O agendamento é semanal. O
Airflow é uma dependência opcional por ser pesada:

```bash
poetry install --with airflow
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__LOAD_EXAMPLES=False PYTHONPATH=.
poetry run airflow db migrate
poetry run airflow dags test retrain_text_classifier 2026-11-02 -S dags
poetry run pytest tests/test_dag.py   # integridade da DAG (pulado sem Airflow)
```

Execução verificada com Airflow 2.10.5 (`airflow dags test`, 2026-11-03): as duas tarefas
terminam em `SUCCESS` e o `DagRun` fecha como concluído.

```
Marking task as SUCCESS. dag_id=retrain_text_classifier, task_id=ingest_data
Marking task as SUCCESS. dag_id=retrain_text_classifier, task_id=train_model
DagRun Finished: dag_id=retrain_text_classifier, ... state=success
```

Na primeira execução, o import da DAG (que carrega scikit-learn, skl2onnx e onnxruntime)
pode passar do limite padrão de 30 s do Airflow e falhar com `DagBag import timeout`. Se
acontecer, aumente com `export AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=120`.

## Monitoramento (Prometheus + Grafana)

A API usa `prometheus_client` e expõe as métricas em `/metrics`:

| Métrica | Tipo | Rótulos |
|---|---|---|
| `http_requests_total` | Counter | `method`, `path`, `status` |
| `http_request_duration_seconds` | Histogram | `method`, `path` |
| `predictions_total` | Counter | `label` (classe prevista) |

O rótulo `path` usa o template da rota (ex.: `/predict`). URLs desconhecidas viram `other`,
para não criar uma série por URL. O Prometheus coleta a API a cada 5 s.

O Grafana já sobe com o datasource e o dashboard **Triagem de urgência — API**
provisionados ([JSON](monitoring/grafana/dashboards/triagem-api.json)). Painéis:

1. Total de requisições em `/predict`;
2. Taxa de erro em `/predict` (respostas 4xx/5xx, em %);
3. Requisições por segundo, por status;
4. Latência de `/predict` (p50, p95 e p99 do histograma);
5. Predições por classe.

![Dashboard Grafana](docs/grafana-dashboard.jpg)

## Limitações

- O dataset tem 1.267 linhas, abaixo das 2.000 sugeridas no enunciado. Conseguir mais
  amostras reais exigiria credenciamento no PhysioNet (MIMIC-IV-ED). Contexto em
  [docs/metodo.md](docs/metodo.md).
- O agrupamento dos 5 níveis KTAS em 3 classes é decisão do grupo.
- O KTAS é definido a partir dos sinais vitais e da queixa; parte do ganho do modelo vem de
  reaprender essa regra. A métrica não prova ganho clínico.
