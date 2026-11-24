# Bases brutas

Coloque os arquivos originais em `data/raw/<chave>/` com estes nomes e rode
`python -m data.prepare`. O script grava `data/processed/<chave>.csv`,
cria `data/splits/<chave>.json` (10 folds estratificados, semente 2023) e
confere casos, classes e número de variáveis contra a Tabela 1 do artigo.

| chave | arquivos esperados | fonte |
|---|---|---|
| parkinson | `parkinsons.data` | UCI "Parkinsons" |
| qadc | `green.csv`, `hinselmann.csv`, `schiller.csv` | UCI "Quality Assessment of Digital Colposcopies" |
| spectf | `SPECTF.train`, `SPECTF.test` | UCI "SPECTF Heart" |
| wdbc | `wdbc.data` | UCI "Breast Cancer Wisconsin (Diagnostic)" |
| wpbc | `wpbc.data` | UCI "Breast Cancer Wisconsin (Prognostic)" |
| alzheimer | `alzheimer.csv` + `alzheimer.json` (`{"target": "...", "positive": ...}`) | site da UBU citado no artigo |
| cervical | `risk_factors_cervical_cancer.csv` | UCI "Cervical cancer (Risk Factors)" |
| glioma | `TCGA_InfoWithGrade.csv` | UCI "Glioma Grading Clinical and Mutation Features" |
| mesothelioma | `Mesothelioma data set.xlsx` (ou `mesothelioma.csv`) | UCI "Mesothelioma's disease" |

`y = 1` é sempre a classe doente ou de interesse (Parkinson, maligno,
recorrência, anormal, consenso positivo, Alzheimer, biópsia positiva, GBM,
mesotelioma). A checagem compara o par de contagens sem ordem, porque a
Tabela 1 nem sempre lista a classe doente primeiro.

Divergências conhecidas antes de baixar (anotar o que encontrar em `NOTES.md`):

- Parkinson: a UCI tem 147/48; o artigo diz 46/146 (soma 192, não 195).
- QADC: a UCI tem 62 variáveis + 6 colunas de especialistas; o artigo diz 67.
  `read_qadc(include_experts=..., modality_dummies=...)` permite testar alternativas.
- WPBC: padrão = 30 variáveis + `tumor_size` (31). `time` sai porque vaza o rótulo.
- Câncer cervical: alvo padrão `Biopsy`. Conferir se o artigo usou outro alvo.
- Glioma: a UCI tem 839 casos; o artigo diz 838.
- Alzheimer: formato desconhecido; criar o JSON com o nome do alvo.

O ambiente onde o código foi escrito não tinha acesso à UCI. Os leitores foram
testados com arquivos falsos no mesmo formato (`tests/fakes.py`). Os nomes de
colunas reais podem variar entre versões dos arquivos: se um leitor falhar,
ajuste-o e rode `pytest tests/test_datasets.py`.
