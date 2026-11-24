"""Construção aleatorizada (Pseudocódigo 2) e construção diversificada (MST+FREQ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import BaseObjective, BudgetExhausted


@dataclass
class ConstructResult:
    S: frozenset
    f: float
    steps: int
    budget_hit: bool = False
    diversified: bool = False


def restricted_candidates(eligible: list[int], scores: dict[int, float], alpha: float,
                          eps: float = 1e-12) -> list[int]:
    """L = { j ∈ L' : score(j) ≥ α·max + (1 − α)·min − eps }, em ordem crescente."""
    if not eligible:
        return []
    vals = [scores[j] for j in eligible]
    gmax, gmin = max(vals), min(vals)
    thr = alpha * gmax + (1.0 - alpha) * gmin
    return sorted(j for j in eligible if scores[j] >= thr - eps)


def construct(objective: BaseObjective, alpha: float, rng: np.random.Generator,
              eps: float = 1e-12, penalty: np.ndarray | None = None,
              max_size: int | None = None) -> ConstructResult:
    """Constrói S a partir de ∅.

    penalty : vetor (m,) subtraído de g(j) apenas para ranquear a lista restrita
              (construção diversificada da extensão). A elegibilidade continua
              sendo f(S ∪ {j}) ≥ f(S) − eps.
    max_size : teto opcional para |S| (None = sem teto).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha deve estar em [0, 1]")
    m = objective.m
    if penalty is not None:
        penalty = np.asarray(penalty, dtype=float)
        if penalty.shape != (m,):
            raise ValueError("penalty deve ter forma (m,)")
    S: frozenset = frozenset()
    fS = 0.0
    steps = 0
    try:
        g = dict(zip(range(m), objective.evaluate_many([frozenset([j]) for j in range(m)])))
        eligible = list(range(m))  # L' inicial = V (f(∅) = 0)
        while eligible:
            rank = g if penalty is None else {j: g[j] - penalty[j] for j in eligible}
            L = restricted_candidates(eligible, rank, alpha, eps)
            j_star = L[int(rng.integers(len(L)))]
            S = S | {j_star}
            fS = g[j_star]
            steps += 1
            if max_size is not None and len(S) >= max_size:
                break
            rest = [j for j in range(m) if j not in S]
            g = dict(zip(rest, objective.evaluate_many([S | {j} for j in rest])))
            eligible = [j for j in rest if g[j] >= fS - eps]
    except BudgetExhausted:
        return ConstructResult(S, fS, steps, budget_hit=True, diversified=penalty is not None)
    return ConstructResult(S, fS, steps, diversified=penalty is not None)
