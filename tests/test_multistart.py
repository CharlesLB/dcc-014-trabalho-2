import numpy as np
import pytest

from mst.multistart import MSTConfig, resolve_params, run_mst
from mst.objective import ClassifierObjective, FunctionObjective

from conftest import additive_objective, brute_force_best, random_objective


def cfg(**kw):
    base = dict(tenure=2, maxiter_ts=5, alpha=0.9, maxiter_ms=3)
    base.update(kw)
    return MSTConfig(**base)


# ---------------------------------------------------------------- resolve_params
def test_resolve_params_casos():
    p = resolve_params(175, 22, "casos")
    assert p == {"tenure": 87, "maxiter_ts": 1750, "scale_base": "casos", "n_base": 175}


def test_resolve_params_variaveis():
    p = resolve_params(175, 22, "variaveis")
    assert p["tenure"] == 11 and p["maxiter_ts"] == 220


def test_resolve_params_min_tenure_one():
    assert resolve_params(1, 1, "casos")["tenure"] == 1


def test_resolve_params_odd_floor():
    assert resolve_params(31, 5, "casos")["tenure"] == 15


def test_resolve_params_invalid():
    with pytest.raises(ValueError):
        resolve_params(10, 5, "outra")


# --------------------------------------------------------------- validação
@pytest.mark.parametrize("bad", [dict(tenure=-1), dict(maxiter_ts=-1), dict(maxiter_ms=0),
                                 dict(alpha=2.0), dict(lam=-1.0),
                                 dict(use_freq=True, trigger=0)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        run_mst(random_objective(4, 0), cfg(**bad))


def test_no_stagnation_requires_budget():
    with pytest.raises(ValueError):
        run_mst(random_objective(4, 0), cfg(stop_on_stagnation=False))


def test_config_to_dict():
    d = cfg().to_dict()
    assert d["tenure"] == 2 and "lam" in d


# ------------------------------------------------------------- parada externa
@pytest.mark.parametrize("maxiter_ms", [1, 2, 5, 10])
def test_outer_stop_pseudocode(maxiter_ms):
    o = FunctionObjective(4, lambda S: 0.5)  # nunca melhora depois do 1º reinício
    r = run_mst(o, cfg(maxiter_ms=maxiter_ms))
    assert r.n_local_searches == maxiter_ms + 1  # 1 com melhora + maxiter_ms sem melhora
    assert r.stop_reason == "estagnacao"


@pytest.mark.parametrize("maxiter_ms", [1, 2, 5, 10])
def test_outer_stop_strict_one_more(maxiter_ms):
    o = FunctionObjective(4, lambda S: 0.5)
    r = run_mst(o, cfg(maxiter_ms=maxiter_ms, outer_stop_strict=True))
    assert r.n_local_searches == maxiter_ms + 2


def test_max_restarts():
    r = run_mst(random_objective(6, 0), cfg(maxiter_ms=100, max_restarts=4))
    assert r.n_local_searches == 4 and r.stop_reason == "max_reinicios"


def test_budget_stop_and_respected():
    o = random_objective(14, 3, max_evals=300)
    r = run_mst(o, cfg(stop_on_stagnation=False))
    assert r.stop_reason == "orcamento"
    assert r.n_evals == 300
    assert r.f_best == pytest.approx(o(r.S_best))


def test_budget_mode_stops_when_space_exhausted():
    # m = 4 tem só 15 subconjuntos: o orçamento nunca acaba, mas a busca precisa parar
    o = random_objective(4, 3, max_evals=10_000)
    r = run_mst(o, cfg(stop_on_stagnation=False, budget_patience=3))
    assert r.stop_reason == "sem_avaliacoes_novas"
    assert o.n_evals <= 15


def test_budget_hit_in_construction_keeps_best():
    o = random_objective(10, 3, max_evals=4)
    r = run_mst(o, cfg(stop_on_stagnation=False))
    assert r.stop_reason == "orcamento"
    assert r.n_local_searches == 0


# ------------------------------------------------------------- consistência
@pytest.mark.parametrize("seed", range(20))
def test_best_is_max_of_local_results(seed):
    o = random_objective(6, seed)
    r = run_mst(o, cfg(), seed=seed)
    assert r.f_best == pytest.approx(max(x["f_local"] for x in r.restarts))
    assert r.f_best == pytest.approx(o(r.S_best))


@pytest.mark.parametrize("seed", range(10))
def test_global_record_monotone(seed):
    r = run_mst(random_objective(6, seed), cfg(), seed=seed)
    g = [x["f_global"] for x in r.restarts]
    assert all(b >= a for a, b in zip(g, g[1:]))


@pytest.mark.parametrize("seed", range(10))
def test_improvement_flags_match_record(seed):
    r = run_mst(random_objective(6, seed), cfg(), seed=seed)
    prev = float("-inf")
    for x in r.restarts:
        assert x["melhorou_global"] == (x["f_global"] > prev)
        prev = x["f_global"]


@pytest.mark.parametrize("seed", range(10))
def test_reproducible_with_seed(seed):
    a = run_mst(random_objective(7, seed), cfg(), seed=seed)
    b = run_mst(random_objective(7, seed), cfg(), seed=seed)
    sa, sb = a.summary(), b.summary()
    sa.pop("time_s"); sb.pop("time_s")
    assert sa == sb


def test_different_seeds_can_differ():
    sols = {tuple(sorted(run_mst(random_objective(8, 1), cfg(alpha=0.0), seed=s).restarts[0]["S_local"]))
            for s in range(10)}
    assert len(sols) > 1


def test_external_rng_used():
    a = run_mst(random_objective(6, 2), cfg(), rng=np.random.default_rng(5))
    b = run_mst(random_objective(6, 2), cfg(), seed=5)
    assert a.S_best == b.S_best


def test_cache_equivalence():
    a = run_mst(random_objective(6, 4, cache=True), cfg(), seed=1)
    b = run_mst(random_objective(6, 4, cache=False), cfg(), seed=1)
    assert a.S_best == b.S_best and a.f_best == b.f_best
    assert a.total_local_iters == b.total_local_iters
    assert a.n_evals < b.n_evals


def test_counters_consistent():
    r = run_mst(random_objective(6, 5), cfg())
    assert r.total_local_iters == sum(x["iters"] for x in r.restarts)
    assert r.no_admissible == sum(x["sem_admissivel"] for x in r.restarts)
    assert r.n_local_searches == len(r.restarts)


@pytest.mark.parametrize("seed", range(10))
def test_finds_brute_force_optimum_small(seed):
    o = random_objective(5, seed)
    _, fopt = brute_force_best(random_objective(5, seed), 5)
    r = run_mst(o, cfg(tenure=1, maxiter_ts=40, maxiter_ms=10, alpha=0.5), seed=seed)
    assert r.f_best == pytest.approx(fopt)


def test_additive_optimum():
    w = [0.3, -0.2, 0.1, 0.4, -0.5, 0.05]
    r = run_mst(additive_objective(w), cfg())
    assert r.S_best == {0, 2, 3, 5}


def test_summary_serializable():
    import json

    r = run_mst(random_objective(5, 0), cfg(trace=True))
    s = r.summary()
    json.dumps(s)
    assert "traces" not in s and isinstance(s["S_best"], list)


def test_trace_per_restart():
    r = run_mst(random_objective(5, 0), cfg(trace=True))
    assert len(r.traces) == r.n_local_searches
    assert all(t["passos"] for t in r.traces)


def test_time_measured():
    r = run_mst(random_objective(5, 0), cfg())
    assert r.time_s >= 0


def test_with_classifier_objective(synth_xy):
    X, y = synth_xy
    o = ClassifierObjective(X, y, "DA")
    p = resolve_params(len(y), X.shape[1], "variaveis")
    r = run_mst(o, MSTConfig(tenure=p["tenure"], maxiter_ts=20, maxiter_ms=3), seed=0)
    assert 0 < r.f_best <= 1
    assert r.f_best >= max(o({j}) for j in range(X.shape[1]))


def test_freq_counts_none_without_extension():
    assert run_mst(random_objective(5, 0), cfg()).freq_counts is None
