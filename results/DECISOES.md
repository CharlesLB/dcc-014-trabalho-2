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
