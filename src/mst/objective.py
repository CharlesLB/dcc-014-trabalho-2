"""Função objetivo f(S) = β·Rat(S) + (1 − β)·(1 − |S|/m) com cache e orçamento.

Subconjuntos são representados por ``frozenset`` de índices de colunas (0..m−1).
No cache, a chave é a máscara de bits ``int`` do subconjunto (SPEC §2 permite
frozenset ou bitmask). Com m = 65 e |S| ≈ 28, um frozenset ocupa cerca de 2 KB e
a máscara cerca de 36 bytes: execuções com milhões de avaliações cabem na memória.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np

from .classifiers import fit_predict

Subset = frozenset


class BudgetExhausted(Exception):
    """Levantada quando o orçamento de avaliações (max_evals) acaba."""


def as_subset(S: Iterable[int]) -> frozenset:
    return S if isinstance(S, frozenset) else frozenset(int(j) for j in S)


def subset_mask(S: Iterable[int]) -> int:
    """Máscara de bits do subconjunto: bit j ligado se j ∈ S."""
    mask = 0
    for j in S:
        if j < 0:
            raise IndexError(f"variável {j} negativa")
        mask |= 1 << j
    return mask


class BaseObjective:
    """Base com cache, contadores e orçamento.

    Subclasses implementam ``_score(S)`` (valor de f sem cache).
    """

    def __init__(self, m: int, beta: float = 0.99, cache: bool = True,
                 max_evals: int | None = None):
        if m <= 0:
            raise ValueError("m deve ser positivo")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta deve estar em [0, 1]")
        if max_evals is not None and max_evals < 0:
            raise ValueError("max_evals não pode ser negativo")
        self.m = int(m)
        self.beta = float(beta)
        self.use_cache = cache
        self.max_evals = max_evals
        self._cache: dict[int, float] = {}  # chave = subset_mask(S)
        self.n_evals = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ API
    def __call__(self, S: Iterable[int]) -> float:
        S = as_subset(S)
        if not S:
            return 0.0  # f(∅) = 0 por convenção (Pseudocódigo 2)
        if self.use_cache:
            key = subset_mask(S)
            hit = self._cache.get(key)
            if hit is not None:
                self.n_cache_hits += 1
                return hit
        if self.max_evals is not None and self.n_evals >= self.max_evals:
            raise BudgetExhausted(f"orçamento de {self.max_evals} avaliações esgotado")
        for j in S:
            if j < 0 or j >= self.m:
                raise IndexError(f"variável {j} fora de 0..{self.m - 1}")
        value = float(self._score(S))
        self.n_evals += 1
        if self.use_cache:
            self._cache[key] = value
        return value

    @property
    def budget_left(self) -> int | None:
        if self.max_evals is None:
            return None
        return max(self.max_evals - self.n_evals, 0)

    def size_term(self, S: frozenset) -> float:
        return (1.0 - self.beta) * (1.0 - len(S) / self.m)

    def reset_counters(self) -> None:
        self.n_evals = 0
        self.n_cache_hits = 0

    def clear_cache(self) -> None:
        self._cache.clear()

    def _score(self, S: frozenset) -> float:  # pragma: no cover - abstrato
        raise NotImplementedError


class ClassifierObjective(BaseObjective):
    """f(S) com Rat(S) = acurácia de ressubstituição no treino."""

    def __init__(self, X: np.ndarray, y: np.ndarray, classifier: str = "DA",
                 beta: float = 0.99, seed: int = 0, cache: bool = True,
                 max_evals: int | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X deve ser 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y com número de linhas diferente")
        if np.unique(y).size != 2:
            raise ValueError("y deve ter exatamente duas classes")
        super().__init__(X.shape[1], beta=beta, cache=cache, max_evals=max_evals)
        self.X = X
        self.y = y
        self.classifier = classifier
        self.seed = seed

    def rat(self, S: Iterable[int]) -> float:
        S = sorted(as_subset(S))
        if not S:
            return 0.0
        Xs = self.X[:, S]
        pred = fit_predict(self.classifier, Xs, self.y, Xs, seed=self.seed)
        return float(np.mean(pred == self.y))

    def _score(self, S: frozenset) -> float:
        return self.beta * self.rat(S) + self.size_term(S)


class TableObjective(BaseObjective):
    """Objetivo dado por tabela explícita (testes e exemplos sintéticos).

    ``table`` mapeia subconjuntos (iteráveis de índices) para f. Subconjuntos
    ausentes usam ``default`` (que pode ser um callable S -> float).
    """

    def __init__(self, m: int, table: Mapping[Iterable[int], float],
                 default: float | Callable[[frozenset], float] = float("-inf"),
                 cache: bool = True, max_evals: int | None = None):
        super().__init__(m, beta=0.99, cache=cache, max_evals=max_evals)
        self.table = {as_subset(k): float(v) for k, v in table.items()}
        self.default = default

    def _score(self, S: frozenset) -> float:
        if S in self.table:
            return self.table[S]
        if callable(self.default):
            return float(self.default(S))
        return float(self.default)


class FunctionObjective(BaseObjective):
    """Objetivo dado por função arbitrária S -> float."""

    def __init__(self, m: int, fn: Callable[[frozenset], float], cache: bool = True,
                 max_evals: int | None = None):
        super().__init__(m, cache=cache, max_evals=max_evals)
        self.fn = fn

    def _score(self, S: frozenset) -> float:
        return float(self.fn(S))
