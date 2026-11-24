"""Busca Tabu (Pseudocódigo 3 de Pacheco et al., 2023)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neighborhood import is_tabu, moves, update_memory
from .objective import BaseObjective, BudgetExhausted, as_subset


@dataclass
class TabuResult:
    S_best: frozenset
    f_best: float
    iterations: int
    no_admissible: int = 0
    local_maxima: int = 0
    escapes: list = field(default_factory=list)
    trace: list | None = None
    budget_hit: bool = False
    f_start: float = float("nan")

    @property
    def n_escapes(self) -> int:
        return len(self.escapes)


def tabu_search(objective: BaseObjective, S0, tenure: int, maxiter: int,
                eps: float = 1e-12, trace: bool = False,
                detect_local_max: bool = True, min_size: int = 1,
                hard_limit: int | None = None) -> TabuResult:
    """Executa a Busca Tabu a partir de S0.

    Parâmetros
    ----------
    tenure : duração tabu (iterações).
    maxiter : maxiterTS, iterações sem melhora de f* até parar.
              Para quando it > itbest + maxiter.
    detect_local_max : marca iterações em que todos os vizinhos válidos
              (tabu ou não) têm f ≤ f(S), e conta fugas confirmadas.
    hard_limit : teto de iterações (segurança em testes); None = sem teto.

    Memória (VectorIn/VectorOut) é criada aqui, ou seja, reinicializada a cada
    chamada, como no artigo.
    """
    if tenure < 0:
        raise ValueError("tenure não pode ser negativo")
    if maxiter < 0:
        raise ValueError("maxiter não pode ser negativo")
    S = as_subset(S0)
    if len(S) < min_size:
        raise ValueError("solução inicial menor que o tamanho mínimo")
    m = objective.m
    vin = np.full(m, -tenure, dtype=np.int64)
    vout = np.full(m, -tenure, dtype=np.int64)

    try:
        fS = objective(S)
    except BudgetExhausted:
        return TabuResult(S, float("nan"), 0, budget_hit=True)

    f0 = fS
    S_best, f_best = S, fS
    it = 0
    itbest = 0
    no_adm = 0
    n_lm = 0
    pending_lm: list[tuple[int, float]] = []  # (it, f) de máximos locais deixados por piora
    escapes: list[dict] = []
    tr: list | None = [] if trace else None
    budget_hit = False

    while it <= itbest + maxiter:
        if hard_limit is not None and it >= hard_limit:
            break
        it += 1
        fb = float("-inf")
        Sb = None
        mv_b = None
        b_tabu = False
        b_asp = False
        all_le = True
        # A vizinhança inteira é avaliada de uma vez (em lote quando o objetivo
        # permite); a escolha abaixo percorre os vizinhos na mesma ordem de antes.
        nbrs = list(moves(S, m, min_size=min_size))
        cand = [mv.apply(S) for mv in nbrs]
        try:
            fvals = objective.evaluate_many(cand)
        except BudgetExhausted:
            budget_hit = True
            it -= 1
            break
        for mv, Sp, fp in zip(nbrs, cand, fvals):
            if fp > fS + eps:
                all_le = False
            tabu = is_tabu(mv, it, vin, vout, tenure)
            asp = fp > f_best + eps
            if (not tabu or asp) and fp > fb:
                fb, Sb, mv_b, b_tabu, b_asp = fp, Sp, mv, tabu, asp

        is_lm = detect_local_max and all_le
        f_before = fS
        if Sb is None:
            no_adm += 1
            accepted = None
        else:
            update_memory(mv_b, it, vin, vout)
            S, fS = Sb, fb
            accepted = mv_b
        if is_lm:
            n_lm += 1
            if accepted is not None and fS < f_before - eps:
                pending_lm.append((it, f_before))

        improved = False
        if fS > f_best + eps:
            S_best, f_best, itbest = S, fS, it
            improved = True
            still = []
            for it_lm, f_lm in pending_lm:
                if f_best > f_lm + eps:
                    escapes.append({"it_max_local": it_lm, "f_max_local": f_lm,
                                    "it_melhora": it, "f_nova": f_best})
                else:
                    still.append((it_lm, f_lm))
            pending_lm = still

        if tr is not None:
            tr.append({
                "it": it,
                "mov": accepted.as_dict() if accepted is not None else None,
                "f_antes": f_before,
                "f_depois": fS,
                "f_star": f_best,
                "tabu": b_tabu if accepted is not None else None,
                "aspiracao": (b_tabu and b_asp) if accepted is not None else None,
                "sem_admissivel": accepted is None,
                "max_local": is_lm,
                "melhorou": improved,
                "tamanho": len(S),
            })

    return TabuResult(S_best=S_best, f_best=f_best, iterations=it,
                      no_admissible=no_adm, local_maxima=n_lm, escapes=escapes,
                      trace=tr, budget_hit=budget_hit, f_start=f0)
