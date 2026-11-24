import pytest

from mst.greedy import hill_climb, run_greedy
from mst.multistart import MSTConfig, run_mst
from mst.objective import FunctionObjective

from conftest import additive_objective, random_objective


def test_greedy_toy_stops_at_local_max(toy):
    r = run_greedy(toy)
    assert r.S_best == {0} and r.f_best == pytest.approx(0.88)
    assert r.stop_reason == "max_local"


def test_greedy_deterministic():
    a = run_greedy(random_objective(7, 1), seed=0)
    b = run_greedy(random_objective(7, 1), seed=99)
    assert a.S_best == b.S_best


def test_greedy_without_local_search():
    r = run_greedy(random_objective(7, 1), local_search=False)
    assert r.total_local_iters == 0


@pytest.mark.parametrize("seed", range(15))
def test_hill_climb_result_is_local_max(seed):
    from mst.neighborhood import moves

    o = random_objective(6, seed)
    S, f, _, _ = hill_climb(o, frozenset({seed % 6}))
    assert all(o(mv.apply(S)) <= f + 1e-12 for mv in moves(S, 6))


@pytest.mark.parametrize("seed", range(15))
def test_hill_climb_never_worse_than_start(seed):
    o = random_objective(6, seed)
    S0 = frozenset({0, 1})
    _, f, _, _ = hill_climb(o, S0)
    assert f >= o(S0)


def test_hill_climb_budget():
    o = random_objective(10, 0, max_evals=15)
    S, f, it, hit = hill_climb(o, frozenset({0}))
    assert hit


def test_greedy_additive_optimum():
    r = run_greedy(additive_objective([0.2, -0.1, 0.3]))
    assert r.S_best == {0, 2}


@pytest.mark.parametrize("seed", range(10))
def test_mst_at_least_as_good_as_greedy_when_starting_same(seed):
    # MST com α = 1 parte da mesma construção do guloso; a Busca Tabu nunca devolve pior
    o1, o2 = random_objective(7, seed), random_objective(7, seed)
    g = run_greedy(o1)
    m = run_mst(o2, MSTConfig(tenure=2, maxiter_ts=30, alpha=1.0, maxiter_ms=1), seed=seed)
    assert m.f_best >= g.f_best - 1e-12


def test_greedy_counters():
    o = random_objective(6, 2)
    r = run_greedy(o)
    assert r.n_evals == o.n_evals and r.n_local_searches == 1
