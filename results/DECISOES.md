# Decisões de parâmetros e de execução

Registro das decisões tomadas ao rodar as campanhas, com o motivo. Decisões de
pré-processamento das bases estão em `data/NOTES.md`.

## Ambiente

- Máquina usada: WSL2 (Linux 6.18) com 32 threads lógicas e 62 GB de RAM (o
  pedido previa um Ryzen 7 de 8 núcleos com 128 GB). Python 3.12.16, numpy 2.5.5,
  scikit-learn 1.9.1, pandas 3.1.1, scipy 1.18.1.
- Todas as campanhas com `--n-jobs 7` (configs) e BLAS com 1 thread por processo,
  como pedido. Os tempos (`tempo_s`) são de uma execução sozinha num núcleo, com
  outras 6 rodando ao lado.

## D1. Cache de f com chave em máscara de bits e um processo por execução

- **Problema:** no primeiro E0, os 7 processos chegaram a 60 GB de RAM e a swap
  encheu (14 de 16 GB) nas execuções da QADC. Cada chave do cache era um
  `frozenset` de até 65 índices (≈ 2 KB), e execuções da QADC passam de 2 milhões
  de avaliações. Além disso, o processo trabalhador não devolvia a memória entre
  execuções. Os tempos medidos sob swap não eram confiáveis.
- **Decisão:** (1) a chave do cache passou a ser a máscara de bits `int` do
  subconjunto (a SPEC §2 permite frozenset ou bitmask); (2) cada execução roda num
  processo novo (`max_tasks_per_child=1`, contexto `spawn`).
- **Efeito no algoritmo:** nenhum. Conferido em 16 execuções do E0 (Parkinson e
  WPBC, folds 0 e 1, duas sementes, as duas leituras de n): `S*`, `f`, `n_evals`,
  `n_cache_hits`, iterações, reinícios e métricas de validação idênticos. O tempo
  ficou de 0 a 20% maior nas execuções curtas (custo de montar a máscara).
- O E0 foi refeito do zero com o código novo, para todos os tempos virem da mesma
  versão. O registro interrompido está em `results/E0-sensibilidade-v0/` (91 de
  120 execuções, código antigo).

## D2. Leitura de `n` em `tenure` e `maxiterTS` (E0): `scale_base = variaveis`

Critério combinado: trocar para `variaveis` se, com `casos`, mais de 50% das
iterações locais ficarem sem movimento admissível **e** o f não for melhor.

E0 (MST, DA, Parkinson, WPBC e QADC, 10 folds × 2 sementes, 120 execuções;
tabela completa em `results/relatorio/E0.md`):

| base | leitura | tenure | maxiterTS | f treino (%) | tempo (s) | iter. locais | sem admissível |
|---|---|---|---|---|---|---|---|
| Parkinson | casos | 87 | 1755 | 92,79 ± 0,60 | 5,2 | 20 185 | 76,0% |
| Parkinson | variaveis | 11 | 220 | 93,08 ± 0,47 | 7,6 | 3 320 | 0% |
| WPBC | casos | 89 | 1780 | 84,69 ± 0,83 | 25,3 | 25 577 | 66,2% |
| WPBC | variaveis | 15 | 310 | 85,37 ± 0,62 | 82,4 | 6 871 | 0% |
| QADC | casos | 129 | 2580 | 89,03 ± 0,67 | 657,8 | 40 838 | 50,1% |
| QADC | variaveis | 32 | 650 | 89,56 ± 0,64 | 2 666,1 | 15 549 | 0% |

- No agregado, com `casos`, **60,9%** das iterações ficam sem movimento
  admissível (> 50%), e o f médio é **88,84%** contra **89,34%** com
  `variaveis` (não é melhor).
- Pareando por (fold, semente): `variaveis` tem f maior em 44 dos 60 pares,
  `casos` em 3, e 13 empatam. Diferença média: Parkinson −0,30 p.p., WPBC −0,68 p.p.,
  QADC −0,53 p.p. (casos − variaveis).
- Sonda nas outras seis bases (fold 0, semente 0): com `casos`, de 64% (SPECTF)
  a 96% (Alzheimer) das iterações ficam sem movimento admissível; f igual em
  Alzheimer, Câncer cervical e Glioma, e menor em SPECTF (−0,27 p.p.), WDBC
  (−0,22 p.p.) e Mesotelioma (−1,40 p.p.).
- **Decisão:** `scale_base = variaveis` em todas as configs a partir do E1
  (E1, E1b, E2-P1, E2a, E2-P2, E3, E4-LR-SVM, E5).
- **Custo:** com `variaveis` a busca explora muito mais subconjuntos novos (QADC:
  7,9 milhões de avaliações por execução contra 1,6 milhão) e fica de 1,5 a 4
  vezes mais lenta. O f no treino só melhora.
- **O que isso diz sobre `tenure = n/2`:** com `n` = casos, o tenure (87 a 696)
  passa do número de variáveis (22 a 65) em todas as bases. Depois de cerca de m
  movimentos, toda variável fica tabu, e a busca só se move por aspiração (quando
  o vizinho bate f*). O resto das iterações é gasto sem mover. A leitura literal
  do artigo transforma a Busca Tabu, na prática, numa subida de encosta com
  longas esperas. O código Pascal público dimensiona as listas tabu pelo número
  de variáveis (`SetLength(TabuListaIn, n)`, com `n` = variáveis), o que reforça
  que o `n` efetivo do artigo é o de variáveis. O valor de `tenure` em si vem do
  programa principal, que não foi publicado.

Observação de ordem: o E1 foi lançado com `variaveis` quando o E0 estava em
101/120 (faltavam execuções da QADC). Os dados das nove bases já apontavam a
mesma decisão, e o E0 completo a confirmou. Nada precisou ser refeito.

## D3. Duas campanhas ao mesmo tempo

A CPU real é um AMD Ryzen 9 7950X (16 núcleos físicos, 32 threads), e não um
Ryzen 7 de 8 núcleos. Cada campanha continua com `--n-jobs 7` e BLAS em 1 thread,
mas, para caber no tempo, até duas campanhas independentes rodam juntas (no
máximo 14 processos em 16 núcleos). Os tempos (`tempo_s`) podem ficar um pouco
maiores do que com uma campanha sozinha, por causa de cache L3 e banda de memória
compartilhados. Dentro de uma campanha, as variantes rodam intercaladas nas
mesmas condições, então as comparações de tempo entre variantes continuam justas.
Comparações de tempo com o artigo (Pascal, outra máquina) já eram só qualitativas.

## D4. Pausa das campanhas (24/11)

E1 e E2-P1 foram interrompidos às 08:34 de 24/11 para desligar a máquina e
retomados às 21:07 do mesmo dia pela chave de execução (`run_key`). Os registros
estavam íntegros (91 e 525 linhas válidas). As 14 execuções da QADC que estavam em
andamento foram perdidas e rodaram de novo do zero; como o MST é determinístico
dada a semente, o resultado não muda. Os tempos dessas execuções vêm da segunda
tentativa.

## D5. E2a adiantado como terceira campanha (24/11, 22:42)

- O E2a (protocolo P2) precisa só da mediana de `n_evals` do MST em P1 por base e
  fold. O E1 já tinha as 5 sementes completas da Parkinson e da QADC nos folds 0–4,
  então o E2a dessas duas bases foi lançado lendo os orçamentos direto de
  `results/E1-reproducao-DA/runs.jsonl`, sem esperar o E1 terminar. A WPBC entra
  quando o E1 a concluir. A chave de execução não depende do arquivo de origem do
  orçamento, e os registros reaproveitados no E2-P1 têm o mesmo `n_evals`, então o
  resultado é o mesmo de lançar depois.
- Corte do desenho: semente [0] e folds 0–4 (150 + 75 execuções em vez de 900). A
  grade completa levaria cerca de 33 h, quase toda na QADC.
- Com três campanhas, rodam 21 processos em 16 núcleos físicos (32 threads). `S*`,
  `f`, `n_evals` e iterações não mudam (a busca é determinística dada a semente e
  não depende da carga). O `tempo_s` das execuções que rodaram nesse período fica
  maior (núcleos divididos por SMT; estimativa de 20 a 40%). Isso afeta as colunas
  de tempo do E1 e do E2-P1 e o orçamento de tempo do E4 (baselines), que vem do
  `tempo_s` médio do MST no E1. Se o E4 rodar, isso entra nas limitações.

## D6. E2-P2 (MST e GULOSO) e E5 adiantados; 3 sementes no E2-P2 e no E3 (24/11, 23:00)

- **Sementes:** a estimativa do E2-P2 e do E3 com 5 sementes passava de 12 h cada
  (≈ 15 h), então os dois rodam com `sementes = [0, 1, 2]`, como previsto no pedido.
- **E2-P2 adiantado:** as variantes MST e GULOSO do E2-P2 não dependem de λ nem do
  gatilho, só do orçamento (mediana de `n_evals` do MST em P1). Elas foram lançadas
  para as bases e folds que já tinham as 5 sementes do MST no E1: Parkinson
  (todos os folds) e QADC (todos menos o fold 6, ainda incompleto na hora). Só
  entra (base, fold) com as 5 sementes, para a mediana ser a mesma da campanha
  completa; o resto entra depois pela retomada. O MST_FREQ do E2-P2 espera o E2a.
- **E5 adiantado:** não depende de nenhuma outra campanha.
- Com isso rodam até 28 processos nas 32 threads. Os resultados não mudam; o
  `tempo_s` fica maior nesse período (ver D5).

## D7. Código rápido da LDA, resultados do ianfefe e fila única (24/11, 23:53)

- **Merge do `origin/master`:** o commit 88ed288 (ianfefe) trouxe a
  `LDASubsetEvaluator`. Ela pré-calcula Σ_z = ZᵀZ por fold, avalia a vizinhança em
  lote e volta para a `FastLDA` original quando algum autovalor fica perto do corte
  do posto (10⁻¹⁰ a 10⁻⁶) ou algum caso de treino fica a menos de 10⁻⁷ (relativo)
  da fronteira. Como f depende só do número de acertos e de |S|, decisões iguais
  dão f igual bit a bit. Revisado: o erro esperado do caminho rápido (≈ κ·ε, com
  κ ≤ ~10⁸ no caminho de posto completo) fica cerca de duas ordens abaixo da margem.
- **Conferência:** o ianfefe rodou de novo, com o código novo, execuções que já
  tínhamos. Nas 820 execuções em comum (E1, E2-P1, E2a), `S*`, `f_treino`,
  `n_evals`, `n_cache_hits`, iterações e reinícios são idênticos. A suíte (722
  testes) passa. Ganho medido: QADC ≈ 3×, SPECTF ≈ 5×, demais bases 2 a 5×.
- **Resultados adotados:** E1 (450) e E2-P1 (1 350) ficaram completos com as
  execuções do ianfefe; o E2a ganhou 610 execuções. Os registros mantêm o campo
  `ambiente`: as execuções vêm de três origens (este Linux com o código antigo,
  commit 160e357/323792d; o Windows 11 do ianfefe, CPU AMD família 23 modelo 113
  (Zen 2), commit 39d5953; e este Linux com o código novo). **O `tempo_s` só é
  comparável dentro da mesma origem.** Nas tabelas, tempos são indicativos.
- **Fila única:** `experiments/fila.py` junta várias configs num só pool de 31
  processos (32 threads), com a retomada de cada campanha e as execuções mais
  longas primeiro. A ordem só muda quando cada execução começa. Com 31 processos
  em 16 núcleos físicos, o `tempo_s` fica mais alto que com 7 (SMT).
- **Desenho revisto com o código rápido:**
  - E2a volta ao desenho completo da config (2 sementes, 10 folds); faltavam 215
    execuções da QADC. Substitui o corte de D5.
  - E2-P2 e E3 voltam a **5 sementes**: a estimativa nova de cada um fica bem
    abaixo de 12 h. Substitui a redução de D6. As 90 execuções do E2-P2 já feitas
    (orçamento lido do E1) têm as mesmas chaves e foram mantidas.
  - E4 baselines: 3 sementes (estimativa com 5 ≈ 15 h de máquina só para ele) e
    orçamento de tempo lido do MST do **E2-P1** (`tempo_de`), cujas 450 execuções
    vêm todas da mesma origem (código novo, Windows do ianfefe). O E1 mistura
    código antigo e novo, e o orçamento ficaria 3× maior na QADC em 40 dos 50 folds.

## D8. O que o texto completo do artigo confirma (PDF recebido em 25/11)

As tabelas do artigo foram transcritas em `reports/artigo.py` (Tabelas 1 a 11),
com testes de conferência em `tests/test_artigo.py`.

- **n = casos.** A seção "Notation" define n como o número de casos de X (treino)
  e m como o de variáveis; `tenure = n/2` e `maxiterTS = 10·n` usam esse n. A
  leitura literal é, portanto, `casos`. A escolha de `variaveis` (D2) continua
  valendo pelo critério do E0, mas é um **desvio declarado** do texto, não uma
  interpretação dele. O código Pascal público dimensiona as listas pelo número de
  variáveis, o que sugere que o programa usado pode diferir do texto.
- **Normalização antes da divisão em folds** ("the data of the features were
  normalized, and the database was divided into 10 folds"). Aqui ela é feita
  dentro do fold, só com o treino (SPEC). Divergência metodológica a registrar nas
  limitações; ela deixa o artigo com um leve vazamento de informação do teste.
- **Tempo:** Tabela 2 é por fold, em Object-Pascal (Delphi 10.3), num Intel
  i9-10920X com 128 GB. Com DA, de 0,5 s (Parkinson) a 72 s (QADC).
- **AUC** do artigo é (1 + TPR − FPR)/2, a mesma `auc` do projeto.
- **Parâmetros:** α e tenure foram ajustados com `maxiterMS = 20`; os valores
  finais são α = 0,99, tenure = n/2, maxiterTS = 10·n, maxiterMS = 10.

## D9. Fila 2: prioridade aos números do artigo, prazo 25/11 12:00 (00:53)

Prazo para resultados e relatório: 25/11 às 12:00. A fila 1 foi parada (as
execuções em andamento rodam de novo) e trocada pela `experiments/filas/fila2.yaml`,
que o `experiments.fila` relê a cada minuto. Ordem de prioridade:

1. E2a (resto da QADC): decide λ e gatilho do MST_FREQ.
2. **E1-casos** (novo): MST com DA na leitura literal do artigo (`scale_base =
   casos`), 9 bases × 10 folds × 5 sementes. Dá a comparação direta com a Tabela 3
   sem o desvio de D2.
3. **E4 baselines com DA, 1 semente**: o artigo roda uma vez por fold (média ± dp
   nos 10 folds), então 1 semente é o mesmo desenho. Substitui as 3 sementes de D7.
4. **MST com LR (8 bases) e SVM (7 bases), 1 semente**, para comparar com as
   Tabelas 9 e 11 (ACC). Medido aqui: LR ≈ 1–2 ms e SVM ≈ 1–17 ms por avaliação.
   Com as ~8 milhões de avaliações do MST na QADC, uma execução levaria ~4 h (LR) e
   ~17 h (SVM); a QADC fica de fora com LR e SVM, e a SPECTF com SVM (~3 h por
   execução) fica no fim da fila, só se der tempo. **As baselines com LR e SVM não
   rodam**: custariam 5× o tempo do MST com o mesmo classificador.
5. E2-P2 (GULOSO e MST) e E3 voltam a **3 sementes** por causa do prazo (D7 tinha
   voltado a 5).

O MST_FREQ do E2-P2 entra na fila quando o E2a terminar.

## D10. λ e gatilho do MST_FREQ (E2a concluído, 25/11 02:06)

- Grade completa (5 λ × 3 gatilhos, 3 bases de ajuste, 10 folds, 2 sementes, P2) em
  `results/relatorio/E2a.md`. Critério da spec §6.2: maior `f` médio no treino em P2,
  desempate pela diversidade.
- Escolha: **λ = 0,05, gatilho = 5** (f médio 0,89364). O padrão da spec (λ = 0,01,
  gatilho = 3) dá 0,89347.
- A grade é quase plana: em relação a λ = 0 (que reproduz o MST bit a bit), a melhor
  combinação ganha só **+0,014 p.p.** em média, e em 53 de 60 execuções o resultado é
  idêntico ao do MST (4 melhores, 3 piores). Todas as combinações ficam dentro de
  ±0,014 p.p. A escolha segue o critério, mas a diferença entre elas não é relevante;
  isso já é um resultado sobre a extensão: a penalidade por frequência quase nunca
  muda a trajetória de modo que altere o melhor `f`.
- O MST_FREQ antigo do E2-P1 (λ = 0,01, gatilho = 3; 450 execuções) saiu de
  `results/E2-P1/runs.jsonl` para `results/E2-P1/runs_MST_FREQ_lam0.01_trigger3.jsonl`,
  para não misturar parâmetros nos relatórios. Fica guardado para referência.
- O MST_FREQ de E2-P1 e de E2-P2 roda com os novos valores
  (`E2_P1_freq.yaml`, `E2_P2_freq.yaml`). GULOSO e MST continuam com λ = 0,01 e
  gatilho = 3 nos params: esses valores não têm efeito neles, mas entram na `run_key`,
  então mantê-los evita refazer execuções já prontas.
- Folga de tempo: a estimativa da fila com 3 sementes é de ~4 h (até ~06:30). Por isso
  a fila2 ganhou, com prioridade menor, as sementes 3 e 4 de E2-P1/MST_FREQ, E2-P2 e
  E3, voltando às 5 sementes da spec se der tempo. O relatório usa só sementes
  completas em todas as bases de cada comparação.
