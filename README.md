# MultiStartTabu para seleção de variáveis (Grupo 5)



## Instalação

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate    Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                      # ~670 testes, cerca de 20 s
```

## Fluxo completo

```bash
# 1. Bases: coloque os arquivos em data/raw/<base>/ (ver data/README.md)
python -m data.prepare                 # processa, cria folds e confere a Tabela 1

# 2. Ensaio rápido sem bases reais (base sintética)
python -m experiments.run --config experiments/configs/smoke.yaml

# 3. Experimentos da SPEC (§6), na ordem
python -m experiments.run --config experiments/configs/E0_sensibilidade.yaml
python -m experiments.run --config experiments/configs/E1_reproducao.yaml
python -m experiments.run --config experiments/configs/E2_P1.yaml
python -m experiments.run --config experiments/configs/E2a_ajuste.yaml   # escolhe λ e gatilho
python -m experiments.run --config experiments/configs/E2_P2.yaml        # depois de ajustar λ/gatilho
python -m experiments.run --config experiments/configs/E5_fuga.yaml
# opcionais
python -m experiments.run --config experiments/configs/E3_ablacoes.yaml
python -m experiments.run --config experiments/configs/E4_baselines.yaml
python -m experiments.run --config experiments/configs/E4_outros_classificadores.yaml

# 4. Tabelas e figuras
python -m reports.tabelas results/E1-reproducao-DA/runs.jsonl --saida results/E1-reproducao-DA/tabelas.md
python -m reports.tabelas results/E2-P2/runs.jsonl --comparar MST_FREQ MST --comparar MST_FREQ GULOSO
python -m reports.tabelas results/E2a-ajuste/runs.jsonl --grade
python -m reports.tabela4 results/E2-P1/runs.jsonl --base wpbc --fold 0 --seed 0 --saida results/tabela4
python -m reports.figuras results/E1-reproducao-DA/runs.jsonl --base wpbc --saida results/fig_wpbc.png
python -m reports.fugas results/E5-fuga/runs.jsonl --exemplo
```

Opções úteis de `experiments.run`:

- `--n-jobs 7` ou `--n-jobs auto` (núcleos físicos − 1). As configs já usam 7
  processos, pensando num Ryzen 7 de 8 núcleos, e BLAS com 1 thread por processo
  (`blas_threads: 1`). O paralelismo é entre execuções, então o tempo de cada
  execução continua comparável. A GPU não é usada: os classificadores são
  pequenos e rodam na CPU (ver "Desempenho" abaixo).
- `--listar` mostra as execuções sem rodar.
- `--set chave=valor` sobrescreve a config, por exemplo `--set "sementes=[0]"`.
- As campanhas retomam de onde pararam: execuções já gravadas em `runs.jsonl`
  são puladas. Falhas vão para `erros.jsonl` sem derrubar a campanha.

## Estrutura

```
src/mst/
  objective.py     f(S) = β·Rat(S) + (1−β)(1−|S|/m), cache, orçamento de avaliações
  classifiers.py   DA (LDA rápida em numpy, igual ao scikit-learn), DA_SK, LR, SVM
  neighborhood.py  adição, remoção, troca; status tabu (eqs. 1 e 2 do artigo)
  constructive.py  Pseudocódigo 2 e construção diversificada (MST+FREQ)
  tabu.py          Pseudocódigo 3, trace, detecção de máximo local e fugas
  multistart.py    Pseudocódigo 1, MST+FREQ, paradas, resolve_params (§3.5)
  greedy.py        baseline guloso (α = 1 + subida de encosta)
  baselines/       GA, PSO, GWO, WOA, FPA binários (opcionais)
  datasets.py      leitores das 9 bases + checagem da Tabela 1 + base sintética
  splits.py        10 folds estratificados e min-max ajustado no treino
  metrics.py       ACC, AUC do artigo, Gmean, F1, ROC-AUC
  analysis.py      diversidade, estabilidade, Wilcoxon, t pareado, Holm
  runner.py        campanhas, protocolos P1/P2, paralelismo, retomada
  runlog.py        registro JSONL (§5.5)
  reporting.py     resumos, comparação com o artigo, Tabela 4, testes pareados
data/prepare.py    CLI do preparo das bases
experiments/       run.py e configs/*.yaml (E0 a E5, smoke)
reports/           tabela4.py, tabelas.py, figuras.py, fugas.py
tests/             testes (pytest)
```

## Desempenho

Com `DA` e `DA_PASCAL`, o objetivo usa `LDASubsetEvaluator`
(`classifiers.py`): médias, desvios e Σ_z = ZᵀZ são calculados uma vez por fold,
e a vizinhança inteira da Busca Tabu é avaliada em lote (`evaluate_many`), com
uma matriz k × k por subconjunto em vez de uma SVD n × k. As decisões são as
mesmas da `FastLDA`: casos perto do corte do posto ou da fronteira de decisão
voltam para o ajuste original. Conferido contra 42 execuções gravadas
(E0, E1, E1b, E2-P1, nas nove bases): `S*`, `f`, `n_evals`, acertos de cache,
iterações e métricas de validação idênticos. Ganho medido na mesma máquina:
2× na Parkinson, 8 a 10× na WPBC e WDBC, 13× na SPECTF, 18× na Alzheimer e 5 a 8× na QADC (menos nas fases com muitas matrizes de posto incompleto).

- `MST_LDA_RAPIDA=0` desliga o avaliador e volta ao ajuste por subconjunto.
- Os `tempo_s` mudam com a versão do código. Para comparar tempos (ou usar
  `tempo_de` nas baselines), use execuções da mesma versão.
- GPU: cada avaliação é uma matriz de ~258 × 28. Mesmo em lote, a vizinhança
  tem ~1 100 matrizes 28 × 28, pequenas demais para compensar a transferência e
  o lançamento de kernels; o ganho real vem de não refazer o ajuste.

## Decisões de implementação

- **Representação:** `frozenset` de índices; cache por subconjunto dentro de
  cada (base, classificador, fold).
- **`n` em `tenure` e `maxiterTS`:** `scale_base: casos` (leitura literal do
  artigo) ou `variaveis` (convenção do código Pascal). Num ensaio sintético do
  tamanho da WPBC (178 casos de treino, 31 variáveis), a leitura literal deu
  `tenure = 89` e 66% das iterações locais sem movimento admissível. A E0
  mede isso nas bases reais.
- **Parada externa:** `outer_stop_strict: false` segue o Pseudocódigo 1;
  `true` imita a versão pública (um reinício a mais).
- **Protocolo P2:** orçamento = mediana de `n_evals` do MST em P1 no mesmo
  fold. Se o espaço de busca se esgotar no cache antes do orçamento, a
  execução para com `stop_reason = "sem_avaliacoes_novas"`.
- **Normalização:** min-max ajustado só no treino de cada fold (o artigo
  normaliza antes de dividir; desvio declarado).
- **AUC:** a fórmula do artigo é de um único ponto de corte, igual à
  acurácia balanceada. A ROC-AUC usual fica em `validacao.roc_auc`.
- **Baselines:** versões binárias simplificadas, não cópias dos artigos de
  origem. Conferir antes de usar no relatório.

