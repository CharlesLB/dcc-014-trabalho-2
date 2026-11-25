"""Baseline guloso: construção determinística (α = 1) + subida de encosta.

A subida usa a mesma vizinhança da Busca Tabu (adição, remoção, troca), move
para o melhor vizinho só se ele melhora estritamente, e para no primeiro
máximo local.
"""

from __future__ import annotations

import time

import numpy as np

from .constructive import construct
from .multistart import MSTResult
from .neighborhood import moves
from .objective import BaseObjective, BudgetExhausted


def hill_climb(objective: BaseObjective, S0: frozenset, eps: float = 1e-12,
               min_size: int = 1) -> tuple[frozenset, float, int, bool]:
    """Retorna (S, f, iterações, budget_hit)."""
    S = frozenset(S0)
    fS = objective(S)
    it = 0
    while True:
        best, fbest = None, fS
        cand = [mv.apply(S) for mv in moves(S, objective.m, min_size=min_size)]
        try:
            fvals = objective.evaluate_many(cand)
        except BudgetExhausted:
            return S, fS, it, True
        for Sp, fp in zip(cand, fvals):
            if fp > fbest + eps:
                best, fbest = Sp, fp
        if best is None:
            return S, fS, it, False
        S, fS = best, fbest
        it += 1


def run_greedy(objective: BaseObjective, eps: float = 1e-12, local_search: bool = True,
               seed: int = 0) -> MSTResult:
    """Executa o baseline guloso. Determinístico (seed só para manter a assinatura)."""
    ev0, hit0 = objective.n_evals, objective.n_cache_hits
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    cons = construct(objective, 1.0, rng, eps=eps)
    S, fS, iters, hit = cons.S, cons.f, 0, cons.budget_hit
    if local_search and not hit:
        S, fS, iters, hit = hill_climb(objective, cons.S, eps=eps)
    return MSTResult(
        S_best=S, f_best=fS, n_local_searches=1, total_local_iters=iters,
        n_evals=objective.n_evals - ev0, n_cache_hits=objective.n_cache_hits - hit0,
        time_s=time.perf_counter() - t0,
        stop_reason="orcamento" if hit else "max_local",
        no_admissible=0, local_maxima=0 if hit else 1, n_escapes=0,
        restarts=[{"r": 1, "f_construido": cons.f, "tamanho_construido": len(cons.S),
                   "f_local": fS, "S_local": sorted(S), "iters": iters,
                   "sem_admissivel": 0, "diversificado": False,
                   "melhorou_global": True, "f_global": fS}],
    )
