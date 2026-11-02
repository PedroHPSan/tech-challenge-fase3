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

## Dataset

Triagem KTAS de dois pronto-socorros coreanos (Kaggle), com 1.267 pacientes reais. A queixa
principal é o texto e o nível definido por especialistas vira urgente, atenção ou normal.
Em queixas inéditas o modelo chega a 0,60 de acurácia (classe mais frequente: 0,42). O
README explica por que trocamos o dataset anterior e como comparamos com o KurMed.

## Pendências

- Confirmar a licença do dataset KTAS no Kaggle antes de redistribuir o CSV.
- Ficamos abaixo das 2.000 amostras sugeridas no enunciado (1.267 linhas).
- Atualizar as tabelas de latência do README com o dataset novo.
- Gravar o vídeo STAR.
