import numpy as np
import pytest

from mst.constructive import construct, restricted_candidates
from mst.objective import FunctionObjective, TableObjective

from conftest import TOY_TABLE, additive_objective, random_objective


def test_rcl_alpha_one_only_max():
    assert restricted_candidates([0, 1, 2], {0: 0.5, 1: 0.9, 2: 0.9}, 1.0) == [1, 2]


def test_rcl_alpha_zero_everything():
    assert restricted_candidates([2, 0, 1], {0: 0.1, 1: 0.5, 2: 0.9}, 0.0) == [0, 1, 2]


def test_rcl_threshold():
    # limiar = 0.5*0.9 + 0.5*0.1 = 0.5
    assert restricted_candidates([0, 1, 2], {0: 0.1, 1: 0.5, 2: 0.9}, 0.5) == [1, 2]


def test_rcl_empty():
    assert restricted_candidates([], {}, 0.5) == []


def test_rcl_eps_tolerance():
    assert restricted_candidates([0, 1], {0: 1.0, 1: 1.0 - 1e-15}, 1.0) == [0, 1]


def test_alpha_one_is_deterministic_greedy():
    w = [0.3, 0.1, 0.5, -0.2]
    o = additive_objective(w)
    rs = {construct(o, 1.0, np.random.default_rng(s)).S for s in range(10)}
    assert rs == {frozenset({0, 1, 2})}


def test_toy_construction_alpha_one(toy):
    r = construct(toy, 1.0, np.random.default_rng(0))
    # g({A})=0.88 é o maior; depois nenhuma adição mantém f -> para em {A}
    assert r.S == {0} and r.f == pytest.approx(0.88) and r.steps == 1


def test_stops_when_no_addition_keeps_f():
    o = additive_objective([0.5, -0.1, -0.2])
    r = construct(o, 0.99, np.random.default_rng(0))
    assert r.S == {0}


def test_ties_are_eligible():
    # adicionar variável com peso 0 mantém f: é elegível (g ≥ f(S))
    o = additive_objective([0.5, 0.0, 0.0])
    r = construct(o, 1.0, np.random.default_rng(0))
    assert r.S == {0, 1, 2}


def test_constant_objective_fills_set():
    o = FunctionObjective(5, lambda S: 0.5)
    r = construct(o, 0.5, np.random.default_rng(0))
    assert r.S == frozenset(range(5))


def test_result_f_matches_objective():
    for s in range(20):
        o = random_objective(6, s)
        r = construct(o, 0.9, np.random.default_rng(s))
        assert r.f == pytest.approx(o(r.S))
        assert len(r.S) >= 1


@pytest.mark.parametrize("seed", range(15))
def test_alpha_one_final_f_at_least_best_single(seed):
    # α = 1: a 1ª escolha é o melhor singleton e cada adição mantém ou eleva f
    o = random_objective(7, seed)
    r = construct(o, 1.0, np.random.default_rng(seed))
    assert r.f + 1e-12 >= max(o({j}) for j in range(7))


@pytest.mark.parametrize("seed", range(15))
def test_final_solution_has_no_improving_addition(seed):
    o = random_objective(7, seed)
    r = construct(o, 0.8, np.random.default_rng(seed))
    assert all(o(r.S | {j}) < r.f - 1e-12 for j in range(7) if j not in r.S)


def test_same_seed_same_solution():
    o1, o2 = random_objective(8, 4), random_objective(8, 4)
    a = construct(o1, 0.5, np.random.default_rng(11))
    b = construct(o2, 0.5, np.random.default_rng(11))
    assert a.S == b.S


def test_alpha_zero_explores_more_than_alpha_one():
    o = random_objective(8, 9)
    s0 = {construct(o, 0.0, np.random.default_rng(s)).S for s in range(30)}
    s1 = {construct(o, 1.0, np.random.default_rng(s)).S for s in range(30)}
    assert len(s0) > len(s1) == 1


def test_invalid_alpha(toy):
    with pytest.raises(ValueError):
        construct(toy, 1.5, np.random.default_rng(0))


def test_penalty_shape_checked(toy):
    with pytest.raises(ValueError):
        construct(toy, 0.9, np.random.default_rng(0), penalty=np.zeros(5))


def test_zero_penalty_equals_plain():
    o = random_objective(7, 2)
    a = construct(o, 0.9, np.random.default_rng(3))
    b = construct(o, 0.9, np.random.default_rng(3), penalty=np.zeros(7))
    assert a.S == b.S and b.diversified and not a.diversified


def test_penalty_changes_ranking_not_eligibility():
    # Exemplo 6.2 do relatório: B (f=0.845, freq 0.8) vs C (f=0.843, freq 0.2), λ = 0.01
    table = {(0,): 0.845, (1,): 0.843}
    o = TableObjective(2, table, default=0.0)
    pen = 0.01 * np.array([0.8, 0.2])
    r = construct(o, 0.99, np.random.default_rng(0), penalty=pen)
    assert r.S == {1}
    assert r.f == pytest.approx(0.843)  # valor real, sem penalidade
    r0 = construct(o, 0.99, np.random.default_rng(0))
    assert r0.S == {0}


def test_penalty_cannot_make_ineligible_addition_eligible():
    # adicionar 1 piora f; penalidade enorme no 0 não muda isso depois da 1ª escolha
    o = additive_objective([0.5, -0.3, 0.2])
    pen = np.array([0.0, 0.0, 10.0])
    r = construct(o, 1.0, np.random.default_rng(0), penalty=pen)
    assert 1 not in r.S


def test_max_size():
    o = FunctionObjective(6, lambda S: 0.5)
    r = construct(o, 0.5, np.random.default_rng(0), max_size=2)
    assert len(r.S) == 2


def test_budget_during_construction():
    o = random_objective(8, 1, max_evals=5)
    r = construct(o, 0.9, np.random.default_rng(0))
    assert r.budget_hit and r.S == frozenset()


def test_budget_after_first_step():
    o = random_objective(4, 1, max_evals=6)
    r = construct(o, 0.9, np.random.default_rng(0))
    assert r.budget_hit and len(r.S) == 1


def test_initial_candidates_all_variables_even_if_negative():
    # f(∅) = 0 mas o artigo começa com L' = V: escolhe a melhor mesmo se todas forem negativas
    o = additive_objective([-0.5, -0.1, -0.9])
    r = construct(o, 1.0, np.random.default_rng(0))
    assert r.S == {1}
