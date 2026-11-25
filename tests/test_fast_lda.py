"""Avaliador em lote da LDA e ``evaluate_many``: mesmas decisões e contadores."""

import numpy as np
import pytest

from mst.classifiers import FastLDA, LDASubsetEvaluator
from mst.objective import BudgetExhausted, ClassifierObjective, FunctionObjective


def _data(n=150, m=12, seed=0, collinear=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    if collinear:  # colunas dependentes, como na QADC
        X[:, 3] = X[:, 1] + X[:, 2]
        X[:, 5] = 2.0 * X[:, 4]
        X[:, 7] = 1.0  # constante
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("equal_priors", [False, True])
@pytest.mark.parametrize("collinear", [False, True])
def test_evaluator_matches_fastlda(equal_priors, collinear):
    X, y = _data(collinear=collinear)
    ev = LDASubsetEvaluator(X, y, equal_priors=equal_priors)
    rng = np.random.default_rng(1)
    subsets = [sorted(rng.choice(X.shape[1], size=k, replace=False).tolist())
               for k in rng.integers(1, X.shape[1] + 1, size=300)]
    got = ev.accuracy_many(subsets)
    for S, a in zip(subsets, got):
        pred = FastLDA(equal_priors=equal_priors).fit(X[:, S], y).predict(X[:, S])
        assert a == np.mean(pred == y), S


def test_evaluator_alternates_paths_consistently():
    X, y = _data()
    ev = LDASubsetEvaluator(X, y)
    subsets = [[1, 2, 3], [0, 1], [4, 5, 6], [0, 2, 8]] * 40
    a1 = ev.accuracy_many(subsets)
    ev._deficient_rate = 1.0  # força o caminho só com eigh
    a2 = ev.accuracy_many(subsets)
    assert np.array_equal(a1, a2)


@pytest.mark.parametrize("clf", ["DA", "DA_PASCAL"])
def test_objective_fast_equals_slow(clf):
    X, y = _data()
    fast = ClassifierObjective(X, y, clf, fast=True)
    slow = ClassifierObjective(X, y, clf, fast=False)
    rng = np.random.default_rng(2)
    subsets = [frozenset(rng.choice(12, size=int(k), replace=False).tolist())
               for k in rng.integers(1, 13, size=200)]
    assert fast.evaluate_many(subsets) == [slow(S) for S in subsets]
    assert (fast.n_evals, fast.n_cache_hits) == (slow.n_evals, slow.n_cache_hits)


def test_evaluate_many_same_counters_and_budget_as_loop():
    fn = lambda S: sum(S) / 10
    seq = [{0}, {1}, {0}, {2, 3}, frozenset(), {1}, {4}, {5}, {4, 1}]
    a = FunctionObjective(6, fn, max_evals=4)
    b = FunctionObjective(6, fn, max_evals=4)
    out_a = []
    with pytest.raises(BudgetExhausted):
        for S in seq:
            out_a.append(a(S))
    with pytest.raises(BudgetExhausted):
        b.evaluate_many(seq)
    assert (a.n_evals, a.n_cache_hits) == (b.n_evals, b.n_cache_hits)
    assert a._cache == b._cache
    # sem orçamento: valores iguais, repetidos dentro do lote contam como acerto
    c = FunctionObjective(6, fn)
    d = FunctionObjective(6, fn)
    assert c.evaluate_many(seq) == [d(S) for S in seq]
    assert (c.n_evals, c.n_cache_hits) == (d.n_evals, d.n_cache_hits)


def test_evaluate_many_index_error_after_prefix():
    a = FunctionObjective(3, lambda S: 1.0)
    with pytest.raises(IndexError):
        a.evaluate_many([{0}, {1}, {7}, {2}])
    assert a.n_evals == 2
