"""Testes da extensão MST+FREQ (SPEC §4)."""

import numpy as np
import pytest

import mst.multistart as msmod
from mst.multistart import MSTConfig, run_mst
from mst.objective import FunctionObjective

from conftest import random_objective


def cfg(**kw):
    base = dict(tenure=2, maxiter_ts=5, alpha=0.9, maxiter_ms=10)
    base.update(kw)
    return MSTConfig(**base)


@pytest.mark.parametrize("seed", range(15))
def test_lambda_zero_equals_mst(seed):
    a = run_mst(random_objective(7, seed), cfg(), seed=seed)
    b = run_mst(random_objective(7, seed), cfg(use_freq=True, lam=0.0), seed=seed)
    assert a.S_best == b.S_best and a.f_best == b.f_best
    assert [x["S_local"] for x in a.restarts] == [x["S_local"] for x in b.restarts]
    assert a.total_local_iters == b.total_local_iters
    assert a.n_evals == b.n_evals


@pytest.mark.parametrize("seed", range(5))
def test_lambda_zero_equals_mst_with_trace(seed):
    a = run_mst(random_objective(6, seed), cfg(trace=True), seed=seed)
    b = run_mst(random_objective(6, seed), cfg(trace=True, use_freq=True, lam=0.0), seed=seed)
    assert a.traces == b.traces


def test_counts_equal_sum_of_local_bests():
    r = run_mst(random_objective(7, 1), cfg(use_freq=True), seed=1)
    expected = np.zeros(7, dtype=int)
    for x in r.restarts:
        for j in x["S_local"]:
            expected[j] += 1
    assert r.freq_counts == expected.tolist()


def test_no_diversification_in_first_restart():
    r = run_mst(random_objective(7, 1), cfg(use_freq=True), seed=1)
    assert r.restarts[0]["diversificado"] is False


def _flags(restarts):
    return [x["diversificado"] for x in restarts]


def _stalls(restarts):
    s, out = 0, []
    for x in restarts:
        s = 0 if x["melhorou_global"] else s + 1
        out.append(s)
    return out


@pytest.mark.parametrize("trigger", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(6))
def test_diversification_follows_trigger_rule(trigger, seed):
    r = run_mst(random_objective(7, seed), cfg(use_freq=True, trigger=trigger, maxiter_ms=12),
                seed=seed)
    stalls = _stalls(r.restarts)
    flags = _flags(r.restarts)
    for i in range(1, len(flags)):
        prev = stalls[i - 1]
        assert flags[i] == (prev > 0 and prev % trigger == 0), (i, stalls, flags)


def test_constant_objective_diversifies_on_multiples_of_trigger():
    o = FunctionObjective(4, lambda S: 0.5)
    r = run_mst(o, cfg(use_freq=True, trigger=3, maxiter_ms=10))
    # stalls: 0,1,2,3,4,5,6,7,8,9,10 -> diversifica nos reinícios seguintes a stall 3,6,9
    idx = [i for i, f in enumerate(_flags(r.restarts)) if f]
    assert idx == [4, 7, 10]


def test_extension_does_not_change_stopping():
    o1 = FunctionObjective(4, lambda S: 0.5)
    o2 = FunctionObjective(4, lambda S: 0.5)
    a = run_mst(o1, cfg(maxiter_ms=7))
    b = run_mst(o2, cfg(use_freq=True, maxiter_ms=7, lam=0.5))
    assert a.n_local_searches == b.n_local_searches


def test_penalty_passed_is_lambda_times_frequency(monkeypatch):
    seen = []
    real = msmod.construct

    def spy(objective, alpha, rng, eps=1e-12, penalty=None, **kw):
        seen.append(None if penalty is None else np.array(penalty))
        return real(objective, alpha, rng, eps=eps, penalty=penalty, **kw)

    monkeypatch.setattr(msmod, "construct", spy)
    o = FunctionObjective(4, lambda S: 0.5)
    r = run_mst(o, cfg(use_freq=True, trigger=3, lam=0.02, maxiter_ms=5))
    # no 5º reinício (índice 4) houve 4 buscas locais concluídas, todas com S = {0,1,2,3}
    assert seen[4] is not None
    np.testing.assert_allclose(seen[4], 0.02 * np.ones(4))
    assert all(p is None for i, p in enumerate(seen) if i != 4)
    assert r.restarts[4]["diversificado"]


def test_freq_uses_f_for_global_record():
    r = run_mst(random_objective(7, 3), cfg(use_freq=True, lam=1.0, trigger=1), seed=3)
    o = random_objective(7, 3)
    assert r.f_best == pytest.approx(o(r.S_best))
    assert r.f_best == pytest.approx(max(x["f_local"] for x in r.restarts))


def test_large_lambda_increases_diversity_on_average():
    from mst.analysis import restart_diversity

    d0, d1 = [], []
    for s in range(12):
        a = run_mst(random_objective(9, s), cfg(use_freq=True, lam=0.0, trigger=1, alpha=0.99), seed=s)
        b = run_mst(random_objective(9, s), cfg(use_freq=True, lam=1.0, trigger=1, alpha=0.99), seed=s)
        d0.append(restart_diversity([x["S_local"] for x in a.restarts]))
        d1.append(restart_diversity([x["S_local"] for x in b.restarts]))
    assert np.nanmean(d1) >= np.nanmean(d0) - 0.05
