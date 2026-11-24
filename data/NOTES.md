# Notas de pré-processamento

Decisões e divergências encontradas ao rodar `python -m data.prepare` com as
bases reais (SPEC §5.1). Download feito em 2026-11-24. Folds: 10 estratificados,
`SPLIT_SEED = 2023` (`data/splits/<base>.json`).

## Resumo

| Base | Obtido (casos, classes, m) | Tabela 1 | Situação |
|---|---|---|---|
| Parkinson | 195, 147/48, 22 | 195, 46/146, 22 | classes divergem em 1–2 casos (ver abaixo) |
| QADC | 287, 216/71, 65 | 287, 72/215, 67 | m = 65 (mais próximo sem vazamento); classes a 1 caso |
| SPECTF | 267, 212/55, 44 | 267, 56/211, 44 | classes a 1 caso |
| WDBC | 569, 212/357, 30 | igual | OK |
| WPBC | 198, 47/151, 31 | igual | OK |
| Alzheimer | 1548, 630/918, 28 | igual | OK |
| Câncer cervical | 668, 63/605, 28 | igual | OK com alvo `Schiller` |
| Glioma | 838, 352/486, 23 | igual | OK após remover 1 linha repetida |
| Mesotelioma | 324, 96/228, 33 | igual | OK |

Seis das nove bases batem exatamente. Nas outras três, a comparação com os
números do artigo continua válida, porque a diferença é de no máximo 1 a 2 casos
de rótulo, ou de 2 variáveis na QADC.

## parkinson
- Arquivo: `parkinsons.data` (UCI id 174, `parkinsons.zip`).
- Obtido: 195 casos, 147 Parkinson (`status = 1`) / 48 saudáveis, m = 22 (sai `name`).
- Divergência: a Tabela 1 diz 46/146, que soma 192, e não 195. O total de casos
  (195) e o m batem. É provável que seja um erro de digitação no artigo; não há
  alternativa de pré-processamento que leve a 46/146 com 195 casos.
- Decisão: usar o arquivo da UCI sem mudança.

## qadc
- Arquivos: `green.csv`, `hinselmann.csv`, `schiller.csv` (UCI id 384). 98 + 97 + 92 = 287 casos.
- Cada arquivo tem 62 variáveis, 6 colunas `experts::0..5` e `consensus`.
- Alternativas testadas (`read_qadc`):

  | experts | dummies de modalidade | m |
  |---|---|---|
  | não | não | 62 |
  | não | **sim** | **65** |
  | sim | não | 68 |
  | sim | sim | 71 |

- `consensus` é exatamente a maioria dos 6 especialistas (0 divergências em 287
  casos). Incluir os experts vaza o rótulo: a acurácia ficaria perto de 100%, e
  o artigo reporta 83,95% com DA. Por isso as opções com experts foram descartadas,
  apesar de 68 ser numericamente a mais próxima de 67.
- **Decisão:** 62 variáveis + 3 dummies da modalidade (`mod_green`,
  `mod_hinselmann`, `mod_schiller`) = 65. É a opção sem vazamento mais próxima
  de 67. As três dummies somam 1 (colineares com o intercepto); a LDA por SVD
  trata o posto incompleto sem problema. Não descobrimos de onde vêm as 2
  variáveis restantes do artigo.
- Classes: 216 `consensus = 1` / 71 `consensus = 0`; o artigo diz 72/215. Diferença
  de 1 caso, sem explicação pelo arquivo. `y = 1` = `consensus = 1`.

## spectf
- Arquivos: `SPECTF.train` (80) + `SPECTF.test` (187) da UCI id 96. Usamos o
  `SPECTF.test` corrigido (a UCI renomeou a versão errada para `SPECTFincorrect.test`).
- Obtido: 267 casos, 212 anormais (`y = 1`) / 55 normais, m = 44.
- Divergência: o artigo diz 56/211. Diferença de 1 caso. Pode vir de uma versão
  antiga do arquivo de teste. Mantido como está.

## wdbc
- Arquivo: `wdbc.data` (UCI id 17). 569 casos, 212 malignos / 357 benignos, m = 30. OK.

## wpbc
- Arquivo: `wpbc.data` (UCI id 16).
- Padrão: 30 variáveis + `tumor_size` = 31; saem `id`, `outcome` (alvo) e `time`
  (tempo até recorrência ou último acompanhamento: vaza o rótulo).
  `lymph_node_status` (4 faltantes) fica de fora.
- Alternativa testada: com `lymph_node_status` ficam 194 casos (46/148) e m = 32,
  mais longe da Tabela 1. **Decisão:** padrão (198 casos, 47/151, m = 31). OK.

## alzheimer
- Fonte: a página citada no artigo
  (https://www.ubu.es/metaheuristicos-grinubumet/ejemplos-y-datos-de-problemas)
  hoje não mostra mais a lista de arquivos, e o repositório institucional
  (riubu.ubu.es, handle 10259/9823) recusou conexão. Uma cópia da página de
  12/04/2024 no Internet Archive tinha o link
  `https://www.ubu.es/sites/default/files/portal_page/files/new_database_for_the_paper_a_multi-start_tabu_search-based_method_for_feature_selection_in_medical_applications.zip`,
  que ainda funciona no servidor da UBU. O zip traz `Alzheimer data.txt` e
  `Read_me_Leeme.txt` (ambos guardados em `data/raw/alzheimer/`).
- Formato: texto separado por espaço, sem cabeçalho, 29 colunas. O leia-me diz
  que a classe é a última coluna (0 = sem Alzheimer, 1 = com Alzheimer) e que as
  primeiras variáveis são dados pessoais (sexo, idade, altura, peso...) e as
  seguintes indicam presença de sintomas.
- Conversão: `alzheimer.csv` = o mesmo arquivo com cabeçalho `v1..v28,alzheimer`
  (sem nenhuma outra mudança). `alzheimer.json` = `{"target": "alzheimer", "positive": 1}`.
- Obtido: 1548 casos, 630/918, m = 28, sem faltantes. OK.

## cervical
- Arquivo: `risk_factors_cervical_cancer.csv` (UCI id 383), 858 casos, 36 colunas.
- Pré-processamento (artigo + SPEC): saem as 4 colunas-alvo; saem as 2 variáveis com
  mais faltantes (`STDs: Time since first diagnosis`, `STDs: Time since last diagnosis`,
  787 faltantes cada); saem os casos com qualquer faltante (sobram 668); saem as
  colunas constantes (`STDs:cervical condylomatosis`, `STDs:AIDS`). m = 28.
- Alvos testados nos 668 casos:

  | alvo | positivos / negativos |
  |---|---|
  | Hinselmann | 30 / 638 |
  | **Schiller** | **63 / 605** |
  | Citology | 39 / 629 |
  | Biopsy (padrão antigo) | 45 / 623 |

- **Decisão:** alvo `Schiller`, o único que reproduz 63/605 da Tabela 1.
  `read_cervical` passou a usar `Schiller` como padrão.

## glioma
- Arquivo: `TCGA_InfoWithGrade.csv` (UCI id 759), 839 linhas, 352 GBM / 487 LGG.
- As linhas 409 e 424 são idênticas em todas as colunas (LGG). Sem a repetição ficam
  838 casos, 352/486, exatamente a Tabela 1.
- **Decisão:** `read_glioma` remove linhas repetidas (`drop_duplicates=True`).

## mesothelioma
- Arquivo: `Mesothelioma data set.xlsx` (UCI id 351), primeira planilha.
- Alvo `class of diagnosis` (2 = mesotelioma, `y = 1`); `diagnosis method` é
  igual ao alvo e sai (vazamento). 324 casos, 96/228, m = 33. OK.

## Integridade (sha256 dos arquivos brutos)

```
9115621364c36e92081c7fcd7eed9b37be8a35c2f58cca9ec33c5ef7a874ba20  alzheimer/Alzheimer data.txt
951f2492be14ce5a2119654eb4da465c556e73f15e27884591a89309e0a517c8  alzheimer/alzheimer.csv
8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063  cervical/risk_factors_cervical_cancer.csv
1b25614ff94c8cf95f80f2c91c33e69843044c62283332690c9ff88f195144a1  glioma/TCGA_InfoWithGrade.csv
cb3a7b5b9352ff30224f5080769db031339dd78c5a4ffb6860cb2a2a54cba73f  mesothelioma/Mesothelioma data set.xlsx
a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5  parkinson/parkinsons.data
5008416001d5d9547fcfe48d0940de871016c8d97623d95ce3feb36332915b67  qadc/green.csv
b0a30014067bd720c4262919143f0ab797a184473b7122b3520359836df38357  qadc/hinselmann.csv
8eb7e87415e776b1b04b746320561ae41ccaa455b306e96b8eecbc046fd53eb7  qadc/schiller.csv
467a65c81e14a71c029fa58bf9350ecdf28ff072303b99a832e8606e193a3882  spectf/SPECTF.test
9c837e4fd4b7931d9be5304989e39f284f35f767518ffd4dfd42a6a519a9f30c  spectf/SPECTF.train
d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a  wdbc/wdbc.data
f2bd5ccdb6086fa832370dc043319907430cf6bfcfd3b96a3961a357be345c8e  wpbc/wpbc.data
```
