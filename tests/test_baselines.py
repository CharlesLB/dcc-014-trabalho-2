import time

import numpy as np
import pytest

from mst.baselines import BASELINES, run_baseline
from mst.objective import ClassifierObjective

from conftest import additive_objective, random_objective

NAMES = sorted(BASELINES)


@pytest.mark.parametrize("name", NAMES)
def test_eval_budget_respected(name):
    o = random_objective(10, 0, max_evals=120)
    r = run_baseline(name, o, seed=0)
    # GWO e WOA podem convergir (população colapsa) e parar por ociosidade antes do orçamento
    assert r.n_evals <= 120
    assert r.stop_reason == "orcamento" or (r.stop_reason == "sem_avaliacoes_novas"
                                            and name in ("GWO", "WOA"))


@pytest.mark.parametrize("name", NAMES)
def test_time_budget_respected(name):
    o = random_objective(10, 0)
    t0 = time.perf_counter()
    r = run_baseline(name, o, seed=0, time_budget=0.15)
    assert time.perf_counter() - t0 < 1.0
    assert r.stop_reason in ("tempo", "sem_avaliacoes_novas")


@pytest.mark.parametrize("name", NAMES)
def test_max_iter(name):
    r = run_baseline(name, random_objective(8, 1), seed=0, max_iter=5)
    assert r.total_local_iters == 5 and r.stop_reason == "max_iter"


@pytest.mark.parametrize("name", NAMES)
def test_best_consistent(name):
    o = random_objective(8, 2, max_evals=200)
    r = run_baseline(name, o, seed=1)
    assert len(r.S_best) >= 1
    assert r.f_best == pytest.approx(o(r.S_best))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic_under_eval_budget(name, seed):
    a = run_baseline(name, random_objective(9, 3, max_evals=150), seed=seed)
    b = run_baseline(name, random_objective(9, 3, max_evals=150), seed=seed)
    assert a.S_best == b.S_best and a.f_best == b.f_best


@pytest.mark.parametrize("name", NAMES)
def test_reasonable_on_additive(name):
    rng = np.random.default_rng(0)
    w = rng.uniform(-1, 1, 10)
    o = additive_objective(w, max_evals=1500)
    r = run_baseline(name, o, seed=0)
    opt = w[w > 0].sum()
    assert r.f_best >= 0.6 * opt


@pytest.mark.parametrize("name", NAMES)
def test_with_classifier(name, synth_xy):
    X, y = synth_xy
    o = ClassifierObjective(X, y, max_evals=60)
    r = run_baseline(name, o, seed=0)
    assert 0 < r.f_best <= 1


@pytest.mark.parametrize("name", NAMES)
def test_stops_when_everything_cached(name):
    # m = 3 tem 7 subconjuntos: o orçamento nunca acaba, a parada por ociosidade evita laço infinito
    o = random_objective(3, 0, max_evals=10_000)
    r = run_baseline(name, o, seed=0, patience=20)
    assert r.stop_reason == "sem_avaliacoes_novas" and o.n_evals <= 7


def test_unknown_baseline():
    with pytest.raises(ValueError):
        run_baseline("ABC", random_objective(4, 0), max_iter=1)


def test_requires_some_stop():
    with pytest.raises(ValueError):
        run_baseline("GA", random_objective(4, 0))


def test_ga_custom_params():
    r = run_baseline("GA", random_objective(8, 0), seed=0, max_iter=3, pop=6, pc=1.0, pm=0.2)
    assert r.total_local_iters == 3
