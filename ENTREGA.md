# Entrega — Tech Challenge Fase 3

**Repositório:** https://github.com/PedroHPSan/tech-challenge-fase3

**Vídeo de apresentação (método STAR):** _a preencher após a gravação_

## Status

| Etapa | Situação |
|---|---|
| 1. Arquitetura e API em Docker | Concluída |
| 2. CI/CD e DAG Airflow | Concluída |
| 3. Monitoramento (Prometheus + Grafana) | Concluída |
| 4. Otimização de latência (ONNX) | Concluída |
| 4. Vídeo STAR | Pendente |

## Dataset e método

Triagem KTAS de dois pronto-socorros coreanos (Kaggle), com 1.267 pacientes reais. A queixa
principal é o texto, o nível definido por especialistas vira urgente, atenção ou normal, e
os sinais medidos na chegada (idade, vitais, dor, consciência) entram como features.

| | Só texto | Texto + sinais clínicos (atual) |
|---|---|---|
| Acurácia em queixas inéditas | 0,60 | **0,75** |
| F1 macro em queixas inéditas | 0,52 | **0,74** |
| Validação cruzada agrupada | 0,55 | **0,68** |

Comparamos 8 abordagens (README) e a regressão logística com texto + sinais foi a melhor.
A API continua exigindo só `text`; os sinais são opcionais e ausentes são imputados.

## Latência (MacBook, Docker, 2026-11-02)

| | scikit-learn | ONNX Runtime |
|---|---|---|
| Só o modelo (média) | 1,115 ms | 0,033 ms |
| API ponta a ponta (média / p95) | 3,23 / 4,15 ms | 1,13 / 1,70 ms |

## Pendências

- Confirmar a licença do dataset KTAS no Kaggle antes de redistribuir o CSV.
- Ficamos abaixo das 2.000 amostras sugeridas no enunciado (1.267 linhas).
- Gravar o vídeo STAR.
