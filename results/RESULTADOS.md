# Resultados: MultiStartTabu (MST) e extensão MST_FREQ

Grupo 5, Inteligência Artificial. Reprodução de Pacheco et al. (2023, *Scientific
Reports* 13:17140) com a extensão do grupo (memória de longo prazo por frequência).
Todas as execuções, decisões e tabelas completas estão em `results/` (veja o índice no
fim). Valores de `f`, ACC, AUC, Gmean e F1 em %. Cada linha "média ± desvio" é
calculada entre os 10 folds, como no artigo; com várias sementes, cada fold entra com
a média das suas sementes.

<!-- SECAO_RESUMO -->

---

## Seção 3.3: leitura de `n` em `tenure = n/2` e `maxiterTS = 10·n` (E0)

O texto do artigo define `n` como o número de **casos** de treino. O código Pascal
público dos autores dimensiona as listas tabu pelo número de **variáveis**. O E0
comparou as duas leituras (MST, DA, Parkinson, WPBC e QADC, 10 folds × 2 sementes,
120 execuções).

| base | leitura | tenure | maxiterTS | f treino (%) | tempo (s) | iter. locais | iterações sem movimento admissível |
|---|---|---|---|---|---|---|---|
| Parkinson | casos | 87 | 1755 | 92,79 ± 0,60 | 5,2 | 20 185 | 76,0% |
| Parkinson | variáveis | 11 | 220 | 93,08 ± 0,47 | 7,6 | 3 320 | 0% |
| WPBC | casos | 89 | 1780 | 84,69 ± 0,83 | 25,3 | 25 577 | 66,2% |
| WPBC | variáveis | 15 | 310 | 85,37 ± 0,62 | 82,4 | 6 871 | 0% |
| QADC | casos | 129 | 2580 | 89,03 ± 0,67 | 657,8 | 40 838 | 50,1% |
| QADC | variáveis | 32 | 650 | 89,56 ± 0,64 | 2 666,1 | 15 549 | 0% |

**Critério** (definido antes de rodar): trocar para `variaveis` se, com `casos`, mais
de 50% das iterações ficarem sem movimento admissível **e** o `f` não for melhor.

**Resultado:** com `casos`, 60,9% das iterações ficam sem movimento admissível, e o
`f` médio é 88,84% contra 89,34% com `variaveis`. Pareando por (fold, semente),
`variaveis` vence em 44 dos 60 pares, `casos` em 3, e 13 empatam. Uma sonda nas
outras seis bases dá de 64% (SPECTF) a 96% (Alzheimer) de iterações paradas com
`casos`. **Decisão: `scale_base = variaveis`** em todos os experimentos seguintes. É um
desvio declarado do texto do artigo. Para medir o efeito dele, o E1 também foi rodado
com a leitura literal (E1-casos; veja a comparação com o artigo).

**O que isso diz sobre `tenure = n/2`:** com `n` = casos, o tenure (87 a 696) passa
do número de variáveis (22 a 65) em todas as bases. Depois de cerca de `m`
movimentos, toda variável está tabu, e a busca só se move por aspiração, quando o
vizinho supera `f*`. As demais iterações passam sem movimento. Lida ao pé da letra,
a regra transforma a Busca Tabu numa subida de encosta com longas esperas. Com
`n` = variáveis, o tenure fica entre 11 e 32: a lista tabu bloqueia cerca de metade
das variáveis e sempre sobra movimento admissível.

---

## Comparação com o artigo

Gerado por `python -m reports.comparacao`. Os gráficos e as tabelas completas estão
em `results/relatorio/comparacao/` (`comparacao.md`). As tabelas do artigo foram
transcritas em `reports/artigo.py`, e os testes em `tests/test_artigo.py` conferem a
transcrição.

<!-- SECAO_ARTIGO -->

---

## Seção 6.3: avaliação da extensão (E2)

<!-- SECAO_E2 -->

---

## Ablações (E3)

<!-- SECAO_E3 -->

---

## Seção 7: fuga de máximo local (E5)

### Exemplo real

E5: MST e GULOSO com trajetória completa, DA, Parkinson e WPBC, 10 folds, semente 0
(`results/relatorio/E5.md`). Um **máximo local** é uma solução cujos vizinhos são
todos piores. Uma **fuga confirmada** acontece quando a busca sai desse ponto
aceitando um vizinho pior e depois encontra uma solução melhor que ele, no mesmo
reinício.

| base | variante | máximos locais (10 folds) | fugas confirmadas | % | f treino (%) |
|---|---|---|---|---|---|
| Parkinson | GULOSO | 10 | 0 | 0% | 88,57 |
| Parkinson | MST | 2 064 | 570 | 27,6% | 93,08 |
| WPBC | GULOSO | 10 | 0 | 0% | 78,38 |
| WPBC | MST | 4 236 | 1 732 | 40,9% | 85,37 |

O guloso (construção determinística, α = 1, seguida de subida de encosta) para no primeiro máximo local
que encontra, um por fold. O MST sai de 28% (Parkinson) a 41% (WPBC) dos máximos
locais por onde passa e termina 4,5 p.p. (Parkinson) e 7,0 p.p. (WPBC) acima. Parte
dessa vantagem vem dos 10 reinícios, e não só das fugas; a ablação do E3 separa as
duas contribuições.

Trecho da trajetória (Parkinson, fold 2, semente 0, reinício 1;
`python -m reports.fugas results/E5-fuga/runs.jsonl --exemplo`):

| it. | movimento | f antes → depois | melhor f até aqui (f*) | situação |
|---|---|---|---|---|
| 1 | entra x8 | 0,8859 → 0,8855 | 0,8859 | o ponto de partida era máximo local: todo vizinho é pior |
| 2 | entra x3 | 0,8855 → 0,8850 | 0,8859 | aceita outro vizinho pior |
| 3 | entra x20 | 0,8850 → 0,8959 | 0,8959 | supera o máximo local em 1,0 p.p. |

Na iteração 1, nenhum vizinho melhora. O guloso pararia aqui com f = 0,8859. A Busca
Tabu aceita o melhor vizinho mesmo sendo pior, desce por duas iterações e, na
terceira, chega a f = 0,8959. Na mesma execução há outra fuga (de 0,8959 a 0,8997,
entre as iterações 4 e 9).

### Correção do exemplo sintético {A, B, C}

No exemplo do relatório, com `f(A) = 0,88`, `f(A,C) = 0,86` e `f(B,C) = 0,93`, a
explicação atual atribui a fuga à lista tabu. Isso não está certo. **A fuga vem de
aceitar um vizinho pior, não da lista tabu.**

- O que tira a busca de {A} é aceitar o melhor vizinho, {A, C} com 0,86, mesmo sendo
  pior que 0,88. O guloso não aceita e para em {A}.
- Em {A, C}, o melhor vizinho já é {B, C} (troca A → B, 0,93), com ou sem lista tabu.
  A lista tabu impede "remover C", que voltaria para {A} (0,88), mas esse movimento
  não seria escolhido de qualquer forma, porque 0,88 < 0,93.
- Com `tenure = 0` (sem lista tabu), a trajetória é a mesma: {A} → {A, C} → {B, C}.
  O teste `test_report_example_escape_comes_from_accepting_worse_move`
  (`tests/test_tabu.py`) verifica isso.

**Texto sugerido para a seção 7:** "No exemplo, a busca sai do máximo local {A}
porque aceita o melhor vizinho mesmo quando ele é pior: vai para {A, C} (0,86). De
lá, a troca de A por B leva a {B, C} (0,93), o ótimo global. A lista tabu proíbe
remover C logo depois de inseri-lo, o que impediria o retorno imediato a {A}, mas
neste exemplo esse retorno não seria escolhido de qualquer forma, já que {B, C} é
melhor. A lista tabu passa a ser decisiva quando o melhor vizinho depois de uma
piora é desfazer o próprio movimento. Sem ela, a busca oscilaria entre as duas
soluções."

---

## Seção 8: implementação e resultados do grupo (Tabela 4)

<!-- SECAO_T4 -->

---

## Limitações

<!-- SECAO_LIMITACOES -->

---

## TODOs do relatório

<!-- SECAO_TODOS -->
