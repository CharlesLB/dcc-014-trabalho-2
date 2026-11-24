# Prompt para o Claude Code (copie tudo abaixo da linha)

---

Você está numa pasta que contém `busca-tabu.tar.gz`, o projeto do trabalho de Busca Tabu do Grupo 5 (Inteligência Artificial). O projeto implementa o MultiStartTabu (MST) de Pacheco et al. (2023, Scientific Reports 13:17140) para seleção de variáveis em 9 bases médicas, mais a extensão do grupo com memória de longo prazo por frequência (MST_FREQ). Quero que você faça tudo, de ponta a ponta, sem me perguntar a cada passo: montar o ambiente, baixar as bases, rodar todos os experimentos e entregar os resultados prontos para o relatório. Escreva tudo em português.

## Regras

- Siga `spec/SPEC-busca-tabu.md`. Ela manda em caso de dúvida.
- Não mude o algoritmo para melhorar números. Diferenças em relação ao artigo são resultado, não erro: registre e explique.
- Toda decisão de pré-processamento ou de parâmetro vai para `data/NOTES.md` ou `results/DECISOES.md`, com o motivo.
- Se mudar código, escreva ou ajuste testes e rode `pytest` de novo. A suíte tem que terminar verde.
- Rode as campanhas longas em segundo plano e acompanhe o progresso. Elas retomam de onde pararam se forem interrompidas.
- Git: rode `git init` e faça commits pequenos ao fim de cada etapa, **sem** linhas de atribuição de IA (nada de "Co-Authored-By" ou "Generated with"). `spec/`, `.claude/`, `data/raw/` e `results/` ficam fora do git (já estão no `.gitignore`). Não faça push.
- Só me pergunte se estiver realmente bloqueado (por exemplo, uma base que não dá para baixar de jeito nenhum). Nesse caso, siga com as outras bases e deixe a pendência anotada.

## Máquina

Ryzen 7 com 8 núcleos e 16 threads, 128 GB de RAM. Use `--n-jobs 7` (as configs já vêm assim, com BLAS em 1 thread por processo). A GPU não é usada.

## Passo a passo

### 1. Montar o projeto
1. Extraia: `tar -xzf busca-tabu.tar.gz` e entre em `busca-tabu/`.
2. Leia `README.md`, `spec/SPEC-busca-tabu.md` e `data/README.md`.
3. Crie um venv com Python ≥ 3.10 e instale: `pip install -r requirements.txt` e `pip install -e .`
   (Windows: `.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`).
4. Rode `pytest`. Devem passar cerca de 668 testes. Se algo falhar no meu sistema operacional, corrija e anote o motivo.
5. Rode o ensaio sintético: `python -m experiments.run --config experiments/configs/smoke.yaml`.

### 2. Baixar e preparar as bases
1. Baixe as 9 bases para `data/raw/<chave>/` com os nomes de arquivo da tabela em `data/README.md`. Oito vêm do UCI Machine Learning Repository (archive.ics.uci.edu): procure pelo nome de cada base. A base de Alzheimer vem da página da Universidade de Burgos citada no artigo: https://www.ubu.es/metaheuristicos-grinubumet/ejemplos-y-datos-de-problemas. Para ela, crie `alzheimer.json` com o nome da coluna alvo e o valor positivo.
2. Rode `python -m data.prepare`.
3. Se um leitor falhar por causa do nome real das colunas, ajuste o leitor em `src/mst/datasets.py`, ajuste o arquivo falso correspondente em `tests/fakes.py` para o formato real, e rode `pytest tests/test_datasets.py`.
4. Para cada base que divergir da Tabela 1, tente as alternativas documentadas em `data/README.md` (QADC com especialistas ou dummies de modalidade, WPBC com ou sem `lymph_node_status`, outro alvo no câncer cervical). Fique com a opção mais próxima da Tabela 1 e anote tudo em `data/NOTES.md`. Se nenhuma bater, siga com a mais próxima.
5. Commit: "Bases preparadas".

### 3. Experimentos (nesta ordem)
Antes de cada campanha grande, rode `--listar` para ver o número de execuções. Meça o tempo das primeiras e estime o total. Se a estimativa passar de 12 horas, reduza as sementes de 5 para 3 com `--set "sementes=[0, 1, 2]"` e anote em `results/DECISOES.md`.

1. **E0 (sensibilidade de `n`):** `python -m experiments.run --config experiments/configs/E0_sensibilidade.yaml`.
   Compare `scale_base = casos` e `variaveis` (f no treino, tempo, iterações e `iter_sem_admissivel / iter_locais_total`).
   Critério: se com `casos` mais de 50% das iterações ficarem sem movimento admissível **e** o f não for melhor, troque `scale_base` para `variaveis` em todas as configs seguintes. Caso contrário, mantenha `casos`. Anote a decisão com os números.
2. **E1 (reprodução, 450 execuções):** `E1_reproducao.yaml`. Gere `python -m reports.tabelas results/E1-reproducao-DA/runs.jsonl --saida results/relatorio/E1.md`. Toda base com diferença maior que 1 p.p. em relação ao artigo precisa de uma explicação provável.
3. **E2 P1:** `E2_P1.yaml`.
4. **E2a (ajuste de λ e gatilho):** `E2a_ajuste.yaml`, depois `python -m reports.tabelas results/E2a-ajuste/runs.jsonl --grade`. Coloque o melhor λ e o melhor gatilho em `E2_P2.yaml` (e em `E2_P1.yaml`, se mudarem, e rode a P1 do MST_FREQ de novo com `--set`).
5. **E2 P2 (comparação principal):** `E2_P2.yaml`, depois
   `python -m reports.tabelas results/E2-P2/runs.jsonl --comparar MST_FREQ MST --comparar MST_FREQ GULOSO --saida results/relatorio/E2-P2.md`.
6. **E5 (fuga de máximo local):** `E5_fuga.yaml`, depois `python -m reports.fugas results/E5-fuga/runs.jsonl --exemplo > results/relatorio/E5.md`.
7. **E3 (ablações):** `E3_ablacoes.yaml` e as tabelas com `--comparar MST MST_SEM_TABU --comparar MST MST_SEM_REINICIO` (junte com os runs do MST em E2-P2).
8. **E4 (opcional, faça se o tempo deixar):** `E4_baselines.yaml` (DA) e `E4_outros_classificadores.yaml` (LR e SVM nas bases menores).

### 4. Saídas para o relatório
1. Tabela 4: `python -m reports.tabela4 results/E2-P1/runs.jsonl --base wpbc --fold 0 --seed 0 --saida results/relatorio/tabela4`.
2. Figura: `python -m reports.figuras results/E1-reproducao-DA/runs.jsonl --base wpbc --saida results/relatorio/fig_wpbc.png`.
3. Escreva `results/RESULTADOS.md`, pronto para colar no relatório, com:
   - **Seção 8 (Implementação e resultados do grupo):** a Tabela 4 preenchida (solução encontrada, função objetivo, tempo e iterações) para MST e MST_FREQ, com a execução representativa e o agregado.
   - **Seção 6.3 (avaliação da extensão):** o resultado do E2-P2 com p-valores corrigidos por Holm, diversidade e estabilidade, interpretado pela regra do relatório ("queda de f com mais diversidade = mais exploração sem ganho de qualidade").
   - **Seção 7 (fuga de máximo local):** o exemplo real do E5. Inclua também esta observação: no exemplo sintético de {A, B, C}, a fuga não depende da lista tabu. Mesmo com `tenure = 0`, o melhor vizinho de {A, C} já é {B, C}. O que tira a busca de {A} é aceitar um vizinho pior (o teste `test_report_example_escape_comes_from_accepting_worse_move` mostra isso). O texto da seção 7 deve ser corrigido.
   - **Seção 3.3 e pendências:** a decisão do E0 com os números e o que ela diz sobre `tenure = n/2`.
   - **Comparação com o artigo:** a tabela do E1 contra a Tabela 3 e as divergências das bases.
   - **Limitações:** normalização dentro do fold (o artigo normaliza antes), baselines simplificados, divergências de pré-processamento, e qualquer redução de sementes.
   - Uma lista das TODOs do relatório que agora podem ser fechadas e das que continuam abertas (por exemplo, a confirmação no enunciado de que a memória de longo prazo é obrigatória).
4. Último commit: "Resultados".

## Ao terminar
Me mande um resumo curto: o que rodou, quanto tempo levou, as decisões tomadas (`scale_base`, λ, gatilho, pré-processamento), os três achados principais e onde está cada arquivo em `results/relatorio/`.
