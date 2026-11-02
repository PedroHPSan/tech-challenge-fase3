# Tech Challenge Fase 3 — Triagem de urgência em pronto-socorro

Serviço que classifica a urgência de um paciente (urgente, atenção ou normal) a partir da
queixa principal e dos sinais clínicos, servido por API REST em container
Docker, com pipeline de CI (GitHub Actions) e DAG de retreino (Airflow).

## Dataset

`data/raw/ktas_triagem_ps.csv`: triagem de dois pronto-socorros coreanos (Kaggle,
"Emergency Service - Triage Application"). O CSV é o bruto (separador `;`, latin-1) e
`src/data.py` o converte para o formato do projeto:

- `text`: coluna `Chief_complain` (queixa principal, em minúsculas).
- `label`: coluna `KTAS_expert` (nível 1-5 definido por especialistas), agrupada em
  **urgente** (1-2), **atenção** (3) e **normal** (4-5).
- Sinais medidos na chegada (`src/features.py`): idade, sexo, meio de chegada, trauma,
  nível de consciência, dor (sim/não e 0-10), pressão sistólica e diastólica, frequência
  cardíaca e respiratória, temperatura, saturação e o índice de choque (FC / PAS).

| Característica | Valor |
|---|---|
| Linhas | 1.267 |
| Classes | normal 534, atenção 487, urgente 246 |
| Queixas únicas | 412 |
| Valores ausentes | saturação 55%, dor 0-10 44%, demais sinais ≤ 2% |

**Limitações em relação ao enunciado:** o enunciado sugere ≥ 2.000 amostras; este
dataset tem 1.267. O agrupamento dos 5 níveis KTAS em 3 classes é decisão do grupo.

**Fonte e licença:** os dados vêm do estudo de Moon et al., publicado na PLOS ONE sob
CC BY 4.0, e foram obtidos pelo upload no Kaggle de İlker Yıldız (que não declara
licença própria e remete ao artigo). Tratamos o CSV sob os termos do artigo, que exigem
citar os autores. Os prontuários foram anonimizados pelos autores e o estudo foi aprovado
pelo comitê de ética do hospital.

> Moon SH, Shim JL, Park KS, Park CS. Triage accuracy and causes of mistriage using the
> Korean Triage and Acuity Scale. PLoS ONE 2019;14(9):e0216972.
> [doi:10.1371/journal.pone.0216972](https://doi.org/10.1371/journal.pone.0216972)

**Vazamento evitado:** `KTAS_RN` (nível dado pela enfermagem), `Diagnosis in ED`,
`Disposition` e `mistriage` ficam de fora porque revelam o resultado da triagem. Os
sinais vitais são medidos antes da classificação, então podem ser usados.

**Cuidado ao interpretar a métrica:** o KTAS usa sinais vitais e queixa para definir o
nível. Parte do ganho do modelo com sinais vem de reaprender essa regra, o que é coerente
com o uso real (a triagem acontece com esses dados), mas não prova ganho clínico.

### Por que trocamos o dataset

O primeiro dataset (`medical_text_classification_fake_dataset.csv`, 1.000 linhas) não
servia ao enunciado: o alvo era a **especialidade** (Cancer, Diabetes, etc.) e não a
urgência, e só tinha 20 frases únicas repetidas. Em frases inéditas o modelo ficou em
0,27 de acurácia, abaixo do acaso (0,25). Procuramos alternativas no Hugging Face, no
Kaggle e no PhysioNet e comparamos as duas que cumpriam o alvo de urgência (nesta
comparação, só com o texto, para ficarem em pé de igualdade):

| | KTAS (adotado) | KurMed-Triage (inglês) |
|---|---|---|
| Origem | pacientes reais de dois pronto-socorros | gerado por LLM (Gemini), sem revisão médica |
| Rótulo | nível definido por especialistas | urgência atribuída na geração |
| Linhas | 1.267 (412 queixas únicas) | 2.000 (todas únicas) |
| Acurácia só com texto, em texto inédito | 0,60 (CV agrupada: 0,55) | 0,71 (CV agrupada: 0,73) |
| Sinais clínicos | sim (idade, vitais, etc.) | não |
| Licença | CC BY 4.0 (artigo de origem) | CC BY 4.0 |

O KurMed tem mais linhas e métrica maior só com texto, mas a métrica mede a capacidade
de reproduzir o padrão de um gerador, não uma triagem clínica. Ficamos com o KTAS por ser
dado real com rótulo de especialistas e por trazer os sinais vitais, e deixamos o KurMed
como comparação. Ter ≥ 2.000 amostras reais exigiria credenciamento no PhysioNet
(MIMIC-IV-ED), que fica como evolução.

### Avaliação sem vazamento

A mesma queixa se repete em várias linhas (e 106 das 412 aparecem com mais de uma
classe). Em `src/data.py` separamos queixas inteiras (20% das únicas de cada classe) para
o teste, e a validação cruzada agrupa por queixa. Resultado atual
(`python -m src.train`):

| Métrica | Só texto | Texto + sinais clínicos (atual) |
|---|---|---|
| Acurácia em queixas inéditas (82 queixas) | 0,60 | **0,75** |
| F1 macro em queixas inéditas | 0,52 | **0,74** |
| Acurácia média na validação cruzada agrupada (5 folds) | 0,55 | **0,68** |
| Referência (classe mais frequente) | 0,42 | 0,42 |

### Como escolhemos o método

Testamos alternativas com validação cruzada agrupada por queixa (5 folds, repetida 3
vezes com sementes diferentes). Valores são acurácia / F1 macro:

| Abordagem | Acurácia | F1 macro |
|---|---|---|
| TF-IDF de palavras + regressão logística (versão anterior) | 0,575 | 0,546 |
| TF-IDF de palavras e caracteres + regressão logística | 0,593 | 0,576 |
| TF-IDF de palavras e caracteres + LinearSVC | 0,594 | 0,572 |
| TF-IDF de palavras + Naive Bayes | 0,586 | 0,583 |
| Só sinais clínicos, Random Forest | 0,626 | 0,606 |
| Só sinais clínicos, gradient boosting | 0,607 | 0,589 |
| Empilhar texto (regressão) + sinais (gradient boosting) | 0,633 | 0,617 |
| **Texto + sinais em um único modelo linear (adotado)** | **0,68** | **0,67** |

Melhorar só o texto rende cerca de 2 pontos, porque a queixa é curta (poucas palavras).
Os sinais clínicos trazem informação que o texto não tem, e o modelo único com texto e
sinais superou até os modelos mais complexos. Ficamos com a regressão logística porque:
(1) converte para ONNX sem perda (100% de concordância de rótulos), o que mantém a
otimização de latência da Etapa 4; (2) é pequena (~37 KB); (3) tolera sinais ausentes,
preenchidos pela mediana do treino dentro do próprio modelo.

## Estrutura do projeto

```
├── data/raw/            # dataset KTAS (CSV versionado, 126 KB)
├── src/
│   ├── config.py        # configuração via .env
│   ├── data.py          # carga, validação e split sem vazamento
│   ├── features.py      # sinais clínicos e índice de choque
│   ├── train.py         # TF-IDF + sinais clínicos, avaliação, salvamento e export ONNX
│   ├── pipeline.py      # etapas ingestão/treino usadas pela DAG
│   └── api.py           # API FastAPI (/health, /predict e /metrics)
├── dags/retrain_dag.py  # DAG do Airflow
├── monitoring/          # Prometheus e Grafana (datasource + dashboard provisionados)
├── docker-compose.yml   # API + Prometheus + Grafana
├── scripts/
│   ├── benchmark_latency.py  # latência da API via HTTP
│   └── compare_latency.py    # scikit-learn x ONNX Runtime (só o modelo)
├── tests/               # pytest
├── .github/workflows/ci.yml
└── Dockerfile
```

## Como executar

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) e Docker.

```bash
poetry install
cp .env.example .env
poetry run python -m src.train          # gera models/model.joblib e models/model.onnx
poetry run uvicorn src.api:app --reload # API em http://localhost:8000/docs
```

Exemplo de chamada (só `text` é obrigatório; os sinais clínicos são opcionais):

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "chest pain", "age": 58, "hr": 110, "sbp": 95, "saturation": 92}'
# {"label":"urgente","confidence":0.986,"latency_ms":0.962}
```

Campos aceitos além de `text`: `age`, `sex`, `arrival_mode`, `injury`, `mental`, `pain`,
`nrs_pain`, `sbp`, `dbp`, `hr`, `rr`, `bt` e `saturation` (faixas e códigos em
`/docs`). O que faltar é preenchido pela mediana do treino, e o índice de choque
(FC / PAS) é calculado pela própria API.

### Docker

```bash
docker build -t triagem-api:0.1 .
docker run --rm -p 8000:8000 triagem-api:0.1
```

O modelo é treinado durante o build, então a imagem já sobe pronta.

## Baseline de latência (local, container Docker)

O ponto de comparação da Etapa 4 é a API servindo o pipeline scikit-learn (joblib).
Medido com `poetry run python -m scripts.benchmark_latency --n 1000` contra o container,
em um MacBook com Docker Desktop. As requisições alternam pacientes com e sem sinais
clínicos, e a medição vem após aquecimento:

| Métrica | Tempo |
|---|---|
| Média | 3,23 ms |
| p50 | 3,08 ms |
| p95 | 4,15 ms |

## Otimização de latência (ONNX Runtime)

Ao fim do treino, `src/train.py` converte o pipeline (TF-IDF + sinais + regressão
logística) para ONNX com `skl2onnx` e grava `models/model.onnx` ao lado do
`model.joblib`. A API carrega o `.onnx` e faz a inferência com o **ONNX Runtime**. O
pipeline scikit-learn continua salvo para comparação.

Por que ONNX: o scikit-learn passa por várias camadas Python a cada chamada (validação,
vetorização, matriz esparsa, imputação). O ONNX Runtime executa o mesmo grafo em C++ numa
única chamada. Quantização e pruning dariam pouco ganho aqui, porque o modelo já é
pequeno (~32 KB).

Detalhes da conversão:

- `zipmap=False`: as probabilidades saem como tensor, e não como lista de dicionários;
- `locale="C"` no TF-IDF: o operador que converte para minúsculas usa `en_US.UTF-8` por
  padrão, e esse locale não existe na imagem `python:3.11-slim`;
- o grafo tem uma entrada `text` e uma entrada float por sinal clínico. Sinal ausente
  entra como NaN e é preenchido pela mediana do treino dentro do grafo.

### Resultados (MacBook, 2026-11-02, modelo com texto + sinais clínicos)

**Só o modelo** (`poetry run python -m scripts.compare_latency --n 3000`, 1 paciente por
chamada, entradas já montadas):

| Modelo | Média | p50 | p95 | Tamanho |
|---|---|---|---|---|
| scikit-learn (joblib) | 1,115 ms | 1,056 ms | 1,391 ms | 37 KB |
| ONNX Runtime | **0,033 ms** | **0,033 ms** | **0,036 ms** | 32 KB |

Ganho de **~34x** na média, com 100% de concordância de rótulos nos pacientes do
dataset. As probabilidades diferem em até ~0,01 porque a tokenização do ONNX não é
idêntica à do scikit-learn. Testes em `tests/test_train.py` garantem os mesmos rótulos.

**API ponta a ponta em Docker** (`scripts.benchmark_latency --n 1000`, mesma máquina e
mesmo container, trocando só o backend do modelo):

| API | Média | p50 | p95 |
|---|---|---|---|
| scikit-learn | 3,23 ms | 3,08 ms | 4,15 ms |
| ONNX Runtime | **1,13 ms** | **1,04 ms** | **1,70 ms** |

Na API o ganho é de ~65% (cerca de 2,9x). O modelo passou de ~1,1 ms para ~0,03 ms, e o
tempo que sobra vem de HTTP, validação do FastAPI e métricas. Há variação entre rodadas:
em quatro rodadas o scikit-learn ficou entre 3,06 e 3,82 ms e o ONNX entre 1,13 e 1,35 ms.
Os números anteriores a esta versão (modelo só com texto, outra máquina) não são
comparáveis com estes.

## CI (GitHub Actions)

`.github/workflows/ci.yml` roda a cada `push` e `pull_request`, em três jobs
encadeados: **lint** (`ruff check` e `ruff format --check`) → **test** (`pytest`) →
**build** (`docker build`). Para rodar o mesmo localmente:

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest
```

## Monitoramento (Prometheus + Grafana)

A API usa `prometheus_client` e expõe as métricas em `/metrics`:

| Métrica | Tipo | Rótulos |
|---|---|---|
| `http_requests_total` | Counter | `method`, `path`, `status` |
| `http_request_duration_seconds` | Histogram | `method`, `path` |
| `predictions_total` | Counter | `label` (classe prevista) |

O rótulo `path` usa o template da rota (ex.: `/predict`). URLs desconhecidas viram
`other`, para não criar uma série por URL.

Para subir a stack completa:

```bash
docker compose up -d --build
poetry run python -m scripts.benchmark_latency --n 1000   # gera tráfego
```

| Serviço | Endereço |
|---|---|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 (coleta a API a cada 5 s) |
| Grafana | http://localhost:3000 (admin/admin; leitura anônima liberada) |

O Grafana já sobe com o datasource e o dashboard **Triagem de laudos — API**
provisionados (`monitoring/grafana/dashboards/triagem-api.json`). Painéis:

1. Total de requisições em `/predict`;
2. Taxa de erro em `/predict` (respostas 4xx/5xx, em %);
3. Requisições por segundo, por status;
4. Latência de `/predict` (p50, p95 e p99 do histograma);
5. Predições por classe.

![Dashboard Grafana](docs/grafana-dashboard.jpg)

Para encerrar: `docker compose down`.

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

**Por que não GPU nem serverless por função:** o modelo roda em CPU em ~0,03 ms (ONNX), então GPU
não compensa o custo. Funções sob demanda (Lambda) sofreriam com *cold start* num
fluxo clínico.

**Cuidados para produção real:** laudos são dados pessoais de saúde (LGPD). Seria
necessário TLS, autenticação na API, rede privada, criptografia em repouso e logs sem o
texto do laudo. A saída do modelo deve apoiar a priorização, e a decisão clínica continua
com o profissional.
