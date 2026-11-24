import numpy as np
import pytest

from mst.greedy import hill_climb
from mst.objective import FunctionObjective, TableObjective
from mst.tabu import tabu_search

from conftest import TOY_TABLE, additive_objective, random_objective


# ------------------------------------------------- exemplo da seção 7 do relatório
def test_report_example_escapes_local_max(toy):
    r = tabu_search(toy, {0}, tenure=1, maxiter=3, trace=True)
    assert r.S_best == frozenset({1, 2})
    assert r.f_best == pytest.approx(0.93)


def test_report_example_trajectory(toy):
    r = tabu_search(toy, {0}, tenure=1, maxiter=3, trace=True)
    t = r.trace
    assert t[0]["mov"] == {"tipo": "add", "sai": None, "entra": 2}
    assert t[0]["f_depois"] == pytest.approx(0.86)
    assert t[0]["max_local"] is True
    assert t[1]["mov"] == {"tipo": "swap", "sai": 0, "entra": 1}
    assert t[1]["f_depois"] == pytest.approx(0.93)
    assert t[1]["melhorou"] is True


def test_report_example_return_to_A_blocked(toy):
    # na iteração 2, remover C (que entrou na 1) é tabu; sem tabu a busca voltaria para {A}
    r = tabu_search(toy, {0}, tenure=1, maxiter=5, trace=True, hard_limit=2)
    assert len(r.trace) == 2
    assert r.trace[1]["mov"] != {"tipo": "remove", "sai": 2, "entra": None}
    assert r.trace[1]["f_depois"] == pytest.approx(0.93)


def test_report_example_escape_counted(toy):
    r = tabu_search(toy, {0}, tenure=1, maxiter=3)
    assert r.n_escapes >= 1
    e = r.escapes[0]
    assert e["f_max_local"] == pytest.approx(0.88)
    assert e["f_nova"] == pytest.approx(0.93)
    assert e["it_max_local"] == 1 and e["it_melhora"] == 2


def test_report_example_greedy_stops_at_A(toy):
    S, f, it, hit = hill_climb(toy, frozenset({0}))
    assert S == {0} and f == pytest.approx(0.88) and it == 0 and not hit


def test_report_example_optimum_is_global(toy):
    from conftest import brute_force_best

    S, f = brute_force_best(TableObjective(3, TOY_TABLE), 3)
    assert S == {1, 2} and f == pytest.approx(0.93)


# --------------------------------------------------------------------- parada
@pytest.mark.parametrize("maxiter", [0, 1, 2, 5, 10])
def test_stops_after_maxiter_plus_one_without_improvement(maxiter):
    # f decresce com o tamanho e {0} é o melhor: nunca melhora
    o = FunctionObjective(4, lambda S: 1.0 - 0.1 * len(S) - 0.01 * sum(S))
    r = tabu_search(o, {0}, tenure=1, maxiter=maxiter)
    assert r.iterations == maxiter + 1
    assert r.S_best == {0}


def test_improvement_extends_search():
    w = [0.1, 0.2, 0.3, 0.4, 0.5]
    o = additive_objective(w)
    r = tabu_search(o, {0}, tenure=1, maxiter=2)
    assert r.S_best == frozenset(range(5))
    # última melhora em it = 4 (4 adições), então para em 4 + 2 + 1
    assert r.iterations == 4 + 2 + 1


def test_hard_limit():
    o = random_objective(6, 1)
    r = tabu_search(o, {0}, tenure=2, maxiter=1000, hard_limit=7)
    assert r.iterations == 7


def test_negative_params_rejected(toy):
    with pytest.raises(ValueError):
        tabu_search(toy, {0}, tenure=-1, maxiter=1)
    with pytest.raises(ValueError):
        tabu_search(toy, {0}, tenure=1, maxiter=-1)


def test_empty_start_rejected(toy):
    with pytest.raises(ValueError):
        tabu_search(toy, set(), tenure=1, maxiter=1)


# ------------------------------------------------------------- aspiração e tabu
def test_aspiration_overrides_tabu():
    # Começa em {0}. Adicionar 1 piora (0.5). Remover 1 seria tabu, mas a troca 0->2 a partir de
    # {0,1}... Construímos: f({0})=0.6, f({0,1})=0.5, f({1})=0.9 (remover 0 = saída do 0, que entrou
    # antes da busca e não é tabu). Para forçar aspiração: f({0}) só é atingível removendo 1 (tabu).
    table = {(0,): 0.6, (1,): 0.1, (0, 1): 0.5, (2,): 0.0, (0, 2): 0.0, (1, 2): 0.0, (0, 1, 2): 0.0}
    o = TableObjective(3, table)
    r = tabu_search(o, {0}, tenure=10, maxiter=3, trace=True)
    # nada supera 0.6 estritamente -> nunca há aspiração; f* fica em 0.6
    assert r.f_best == pytest.approx(0.6)
    assert all(not s["aspiracao"] for s in r.trace if s["mov"] is not None)


def test_aspiration_accepts_tabu_move_that_beats_best():
    # {0,1} -> troca 1->3 ({0,3}); depois add 2 ({0,2,3}); depois a reentrada de 1 é tabu,
    # mas {0,1,2,3} = 0.99 > f* e entra por aspiração. Estados não listados valem −inf.
    table = {(0, 1): 0.5, (0, 3): 0.45, (0, 2, 3): 0.4, (0, 1, 2, 3): 0.99}
    o = TableObjective(4, table)
    r = tabu_search(o, {0, 1}, tenure=5, maxiter=2, trace=True)
    assert r.f_best == pytest.approx(0.99)
    step = r.trace[2]
    assert step["mov"] == {"tipo": "add", "sai": None, "entra": 1}
    assert step["tabu"] is True and step["aspiracao"] is True


def test_without_aspiration_the_tabu_move_would_be_blocked():
    table = {(0, 1): 0.5, (0, 3): 0.45, (0, 2, 3): 0.4, (0, 1, 2, 3): 0.49}
    o = TableObjective(4, table)
    r = tabu_search(o, {0, 1}, tenure=5, maxiter=2, trace=True)
    # 0.49 não supera f* = 0.5: o movimento continua tabu e não é aceito
    assert r.trace[2]["mov"] != {"tipo": "add", "sai": None, "entra": 1}
    assert r.f_best == pytest.approx(0.5)


def test_equal_to_best_does_not_aspire():
    table = {(0,): 0.9, (1,): 0.1, (0, 1): 0.2}
    o = TableObjective(2, table)
    r = tabu_search(o, {0}, tenure=50, maxiter=4, trace=True)
    # depois de add 1 (0.2), voltar para {0} (0.9 = f*) é tabu e não aspira
    assert r.trace[1]["mov"] != {"tipo": "remove", "sai": 1, "entra": None}


def test_no_admissible_move_keeps_solution_and_counts():
    # m=2, tenure enorme: depois de dois movimentos tudo fica tabu
    table = {(0,): 0.9, (1,): 0.2, (0, 1): 0.3}
    o = TableObjective(2, table)
    r = tabu_search(o, {0}, tenure=100, maxiter=5, trace=True)
    blocked = [s for s in r.trace if s["sem_admissivel"]]
    assert r.no_admissible == len(blocked) > 0
    for s in blocked:
        assert s["f_antes"] == s["f_depois"]


def test_report_example_escape_comes_from_accepting_worse_move(toy):
    # Observação para o relatório (seção 7): mesmo sem lista tabu (tenure = 0) a busca chega a
    # {B,C}, porque a troca A->B (0,93) já é o melhor vizinho de {A,C}. O que tira a busca de {A}
    # é aceitar o melhor vizinho mesmo quando ele piora; o bloqueio de "remover C" não muda a
    # escolha neste exemplo.
    r = tabu_search(toy, {0}, tenure=0, maxiter=4, trace=True)
    assert [s["f_depois"] for s in r.trace[:2]] == [pytest.approx(0.86), pytest.approx(0.93)]
    assert r.f_best == pytest.approx(0.93)


def test_tenure_zero_can_cycle():
    # f({0}) = 0.9 é máximo local; o melhor vizinho de {0,1} é voltar para {0}
    table = {(0,): 0.9, (1,): 0.1, (0, 1): 0.5}
    o = TableObjective(2, table)
    r = tabu_search(o, {0}, tenure=0, maxiter=4, trace=True)
    seq = [s["f_depois"] for s in r.trace]
    assert seq[:4] == [0.5, 0.9, 0.5, 0.9]


# --------------------------------------------------------- propriedades gerais
@pytest.mark.parametrize("seed", range(30))
def test_f_best_consistent_with_objective(seed):
    o = random_objective(7, seed)
    r = tabu_search(o, {seed % 7}, tenure=2, maxiter=15, trace=True)
    assert r.f_best == pytest.approx(o(r.S_best))
    assert r.f_best >= o({seed % 7})
    assert r.f_best == pytest.approx(max([o({seed % 7})] + [s["f_depois"] for s in r.trace]))
    assert len(r.S_best) >= 1


@pytest.mark.parametrize("seed", range(10))
def test_deterministic(seed):
    a = tabu_search(random_objective(6, seed), {0, 1}, tenure=2, maxiter=10, trace=True)
    b = tabu_search(random_objective(6, seed), {0, 1}, tenure=2, maxiter=10, trace=True)
    assert a.S_best == b.S_best and a.trace == b.trace


@pytest.mark.parametrize("seed", range(10))
def test_never_worse_than_hill_climbing(seed):
    o = random_objective(6, seed)
    S0 = frozenset({seed % 6})
    hc = hill_climb(o, S0)
    ts = tabu_search(o, S0, tenure=2, maxiter=30)
    assert ts.f_best >= hc[1] - 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_finds_additive_optimum(seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, 8)
    o = additive_objective(w)
    r = tabu_search(o, {int(np.argmax(w))}, tenure=2, maxiter=20)
    assert r.S_best == frozenset(np.flatnonzero(w > 0).tolist()) or (w <= 0).all()


def test_trace_fields(toy):
    r = tabu_search(toy, {0}, tenure=1, maxiter=2, trace=True)
    keys = {"it", "mov", "f_antes", "f_depois", "f_star", "tabu", "aspiracao",
            "sem_admissivel", "max_local", "melhorou", "tamanho"}
    assert all(keys <= set(s) for s in r.trace)
    assert [s["it"] for s in r.trace] == list(range(1, r.iterations + 1))


def test_trace_off_by_default(toy):
    assert tabu_search(toy, {0}, tenure=1, maxiter=2).trace is None


def test_f_star_monotone_in_trace():
    o = random_objective(6, 3)
    r = tabu_search(o, {0}, tenure=2, maxiter=20, trace=True)
    fs = [s["f_star"] for s in r.trace]
    assert all(b >= a for a, b in zip(fs, fs[1:]))


def test_local_max_detection_disabled(toy):
    r = tabu_search(toy, {0}, tenure=1, maxiter=3, detect_local_max=False)
    assert r.local_maxima == 0 and r.n_escapes == 0


def test_budget_hit_during_search():
    o = random_objective(8, 0, max_evals=20)
    r = tabu_search(o, {0}, tenure=2, maxiter=100)
    assert r.budget_hit
    assert o.n_evals == 20
    assert r.f_best == pytest.approx(o(r.S_best))


def test_budget_hit_before_start():
    o = random_objective(4, 0, max_evals=0)
    r = tabu_search(o, {0}, tenure=1, maxiter=1)
    assert r.budget_hit and r.iterations == 0


def test_cache_reduces_evaluations():
    a = random_objective(6, 2, cache=True)
    b = random_objective(6, 2, cache=False)
    ra = tabu_search(a, {0}, tenure=1, maxiter=20)
    rb = tabu_search(b, {0}, tenure=1, maxiter=20)
    assert ra.S_best == rb.S_best and ra.f_best == rb.f_best
    assert a.n_evals < b.n_evals
