"""Baselines populacionais binários (GA, PSO, GWO, WOA, FPA).

São versões binárias simplificadas, com a mesma função objetivo f e o mesmo
critério de parada do artigo (tempo do MST no mesmo fold) ou um orçamento de
avaliações. Não reproduzem linha a linha os artigos de origem (refs. 26, 39,
71, 36 e 31 de Pacheco et al.); conferir parâmetros antes de usar no relatório.
"""

from .population import BASELINES, run_baseline

__all__ = ["BASELINES", "run_baseline"]
