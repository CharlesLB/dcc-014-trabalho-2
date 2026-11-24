from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

from mst.objective import FunctionObjective, TableObjective

# Exemplo sintético da seção 7 do relatório (A=0, B=1, C=2).
TOY_TABLE = {
    (0,): 0.88, (1,): 0.81, (2,): 0.80,
    (0, 1): 0.85, (0, 2): 0.86, (1, 2): 0.93,
    (0, 1, 2): 0.84,
}


@pytest.fixture
def toy():
    return TableObjective(3, TOY_TABLE)


def all_subsets(m: int, min_size: int = 1):
    for k in range(min_size, m + 1):
        for c in itertools.combinations(range(m), k):
            yield frozenset(c)


def random_objective(m: int, seed: int, cache: bool = True, max_evals=None) -> FunctionObjective:
    """Objetivo pseudoaleatório determinístico: f(S) ∈ [0, 1) dependente só de (S, seed)."""

    def fn(S):
        key = f"{seed}:{sorted(S)}".encode()
        return (zlib.crc32(key) % 100000) / 100000.0

    return FunctionObjective(m, fn, cache=cache, max_evals=max_evals)


def additive_objective(weights, penalty: float = 0.0, cache: bool = True, max_evals=None):
    """f(S) = Σ w_j − penalty·|S| (ótimo = {j: w_j > penalty})."""
    w = np.asarray(weights, dtype=float)

    def fn(S):
        return float(sum(w[j] for j in S) - penalty * len(S))

    return FunctionObjective(len(w), fn, cache=cache, max_evals=max_evals)


def brute_force_best(obj, m: int):
    best, fb = None, -np.inf
    for S in all_subsets(m):
        v = obj(S)
        if v > fb:
            best, fb = S, v
    return best, fb


@pytest.fixture
def synth_xy():
    from mst.datasets import synthetic

    df = synthetic(n=120, m=10, informative=4, seed=0)
    X = df.drop(columns="y").to_numpy(float)
    y = df["y"].to_numpy(int)
    return X, y
