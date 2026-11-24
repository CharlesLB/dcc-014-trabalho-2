import numpy as np
import pytest

from mst.objective import (BaseObjective, BudgetExhausted, ClassifierObjective, FunctionObjective,
                           TableObjective, as_subset)


def test_empty_set_is_zero(toy):
    assert toy(frozenset()) == 0.0
    assert toy.n_evals == 0


def test_table_lookup(toy):
    assert toy({0}) == 0.88
    assert toy([1, 2]) == 0.93


def test_as_subset_accepts_iterables():
    assert as_subset([2, 1, 2]) == frozenset({1, 2})
    s = frozenset({3})
    assert as_subset(s) is s


def test_cache_counts_hits(toy):
    toy({0})
    toy({0})
    toy(frozenset({0}))
    assert toy.n_evals == 1
    assert toy.n_cache_hits == 2


def test_cache_disabled_counts_every_eval():
    o = TableObjective(3, {(0,): 1.0}, cache=False)
    o({0}); o({0})
    assert o.n_evals == 2 and o.n_cache_hits == 0


def test_default_value_and_callable_default():
    o = TableObjective(3, {}, default=0.5)
    assert o({1}) == 0.5
    o2 = TableObjective(3, {}, default=lambda S: len(S) / 10)
    assert o2({0, 1}) == pytest.approx(0.2)


def test_missing_subset_defaults_to_minus_inf():
    o = TableObjective(2, {(0,): 1.0})
    assert o({1}) == float("-inf")


def test_index_out_of_range():
    o = TableObjective(2, {})
    with pytest.raises(IndexError):
        o({5})
    with pytest.raises(IndexError):
        o({-1})


def test_budget_exhausted_after_max_evals():
    o = FunctionObjective(4, lambda S: len(S), max_evals=2)
    o({0}); o({1})
    with pytest.raises(BudgetExhausted):
        o({2})
    # cache continua respondendo depois do orçamento
    assert o({0}) == 1
    assert o.budget_left == 0


def test_budget_zero():
    o = FunctionObjective(4, lambda S: 1.0, max_evals=0)
    with pytest.raises(BudgetExhausted):
        o({0})


def test_budget_left_none_without_budget(toy):
    assert toy.budget_left is None


@pytest.mark.parametrize("bad", [dict(m=0), dict(m=3, beta=1.5), dict(m=3, max_evals=-1)])
def test_invalid_constructor_args(bad):
    with pytest.raises(ValueError):
        BaseObjective(**bad)


def test_reset_counters_and_clear_cache(toy):
    toy({0}); toy({0})
    toy.reset_counters()
    assert toy.n_evals == 0 and toy.n_cache_hits == 0
    toy.clear_cache()
    toy({0})
    assert toy.n_evals == 1


def test_size_term():
    o = FunctionObjective(10, lambda S: 0)
    assert o.size_term(frozenset({1, 2})) == pytest.approx(0.01 * 0.8)


def test_classifier_objective_formula(synth_xy):
    X, y = synth_xy
    o = ClassifierObjective(X, y, "DA", beta=0.99)
    S = frozenset({0, 3, 5})
    expected = 0.99 * o.rat(S) + 0.01 * (1 - 3 / X.shape[1])
    assert o(S) == pytest.approx(expected)


def test_rat_is_resubstitution_accuracy(synth_xy):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = synth_xy
    o = ClassifierObjective(X, y, "DA")
    S = [1, 2, 4]
    lda = LinearDiscriminantAnalysis().fit(X[:, S], y)
    assert o.rat(S) == pytest.approx(np.mean(lda.predict(X[:, S]) == y))


def test_rat_empty_is_zero(synth_xy):
    X, y = synth_xy
    assert ClassifierObjective(X, y).rat([]) == 0.0


def test_beta_one_ignores_size(synth_xy):
    X, y = synth_xy
    o = ClassifierObjective(X, y, beta=1.0)
    assert o({0}) == pytest.approx(o.rat({0}))


def test_beta_zero_only_size(synth_xy):
    X, y = synth_xy
    o = ClassifierObjective(X, y, beta=0.0)
    assert o({0, 1}) == pytest.approx(1 - 2 / X.shape[1])


def test_smaller_subset_wins_on_tie():
    # mesma acurácia, menos variáveis -> f maior
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    o = ClassifierObjective(X, y, beta=0.99)
    assert o({0}) > o({0, 1})


def test_classifier_objective_validation():
    X = np.zeros((4, 2))
    with pytest.raises(ValueError):
        ClassifierObjective(X, np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError):
        ClassifierObjective(X, np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        ClassifierObjective(np.zeros(4), np.array([0, 1, 0, 1]))


@pytest.mark.parametrize("clf", ["DA", "DA_SK", "LR", "SVM"])
def test_all_classifiers_give_values_in_range(synth_xy, clf):
    X, y = synth_xy
    o = ClassifierObjective(X, y, clf)
    v = o({0, 1, 2})
    assert 0.0 <= v <= 1.0


def test_cache_does_not_change_values(synth_xy):
    X, y = synth_xy
    a = ClassifierObjective(X, y, cache=True)
    b = ClassifierObjective(X, y, cache=False)
    for S in [{0}, {1, 2}, {0, 1, 2, 3}, {0}]:
        assert a(S) == b(S)
