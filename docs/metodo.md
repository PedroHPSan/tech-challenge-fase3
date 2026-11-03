# Dataset e método — detalhes

Complemento do [README](../README.md): como o dataset foi escolhido, como a avaliação evita
vazamento e como chegamos ao modelo atual.

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

**Fonte e licença:** estudo de Moon et al. (PLOS ONE, CC BY 4.0), obtido pelo upload no
Kaggle de İlker Yıldız, que não declara licença própria e remete ao artigo. Tratamos o CSV
sob os termos do artigo, que exigem citar os autores. Os prontuários foram anonimizados e o
estudo foi aprovado pelo comitê de ética do hospital.

> Moon SH, Shim JL, Park KS, Park CS. Triage accuracy and causes of mistriage using the
> Korean Triage and Acuity Scale. PLoS ONE 2019;14(9):e0216972.
> [doi:10.1371/journal.pone.0216972](https://doi.org/10.1371/journal.pone.0216972)

**Vazamento evitado:** `KTAS_RN` (nível dado pela enfermagem), `Diagnosis in ED`,
`Disposition` e `mistriage` ficam de fora porque revelam o resultado da triagem. Os sinais
vitais são medidos antes da classificação, então podem ser usados.

**Cuidado ao interpretar a métrica:** o KTAS usa sinais vitais e queixa para definir o
nível. Parte do ganho do modelo com sinais vem de reaprender essa regra. Isso é coerente com
o uso real (a triagem acontece com esses dados), mas não prova ganho clínico.

## Por que trocamos o dataset

O primeiro dataset (`medical_text_classification_fake_dataset.csv`, 1.000 linhas) não servia
ao enunciado: o alvo era a **especialidade** (Cancer, Diabetes, etc.) e não a urgência, e só
tinha 20 frases únicas repetidas. Em frases inéditas o modelo ficou em 0,27 de acurácia,
abaixo do acaso (0,25). Procuramos alternativas no Hugging Face, no Kaggle e no PhysioNet e
comparamos as duas que cumpriam o alvo de urgência (só com o texto, para ficarem em pé de
igualdade):

| | KTAS (adotado) | KurMed-Triage (inglês) |
|---|---|---|
| Origem | pacientes reais de dois pronto-socorros | gerado por LLM (Gemini), sem revisão médica |
| Rótulo | nível definido por especialistas | urgência atribuída na geração |
| Linhas | 1.267 (412 queixas únicas) | 2.000 (todas únicas) |
| Acurácia só com texto, em texto inédito | 0,60 (CV agrupada: 0,55) | 0,71 (CV agrupada: 0,73) |
| Sinais clínicos | sim (idade, vitais, etc.) | não |
| Licença | CC BY 4.0 (artigo de origem) | CC BY 4.0 |

O KurMed tem mais linhas e métrica maior só com texto, mas a métrica mede a capacidade de
reproduzir o padrão de um gerador, não uma triagem clínica. Ficamos com o KTAS por ser dado
real com rótulo de especialistas e por trazer os sinais vitais. Ter ≥ 2.000 amostras reais
exigiria credenciamento no PhysioNet (MIMIC-IV-ED), que fica como evolução.

## Avaliação sem vazamento

A mesma queixa se repete em várias linhas (e 106 das 412 aparecem com mais de uma classe).
Em `src/data.py` separamos queixas inteiras (20% das únicas de cada classe) para o teste, e a
validação cruzada agrupa por queixa. Resultado atual (`python -m src.train`):

| Métrica | Só texto | Texto + sinais clínicos (atual) |
|---|---|---|
| Acurácia em queixas inéditas (82 queixas) | 0,60 | **0,75** |
| F1 macro em queixas inéditas | 0,52 | **0,74** |
| Acurácia média na validação cruzada agrupada (5 folds) | 0,55 | **0,68** |
| Referência (classe mais frequente) | 0,42 | 0,42 |

## Como escolhemos o método

Validação cruzada agrupada por queixa (5 folds, repetida 3 vezes com sementes diferentes).
Valores são acurácia / F1 macro:

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

Melhorar só o texto rende cerca de 2 pontos, porque a queixa é curta. Os sinais clínicos
trazem informação que o texto não tem, e o modelo único com texto e sinais superou até os
modelos mais complexos. Ficamos com a regressão logística porque (1) converte para ONNX sem
perda (100% de concordância de rótulos), (2) é pequena (~37 KB em joblib, ~32 KB em ONNX) e
(3) tolera sinais ausentes, preenchidos pela mediana do treino dentro do próprio modelo.
