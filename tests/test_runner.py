import json

import numpy as np
import pytest

from mst import runlog
from mst import runner as R

BASE = "sintetica-80-6-3-0"
FAST = {"scale_base": "variaveis", "maxiter_ms": 2}


def spec(tmp_path, variante="MST", **kw):
    d = dict(campanha="t", variante=variante, base=BASE, classificador="DA", fold=0, seed=0,
             params=dict(FAST), saida=str(tmp_path), n_splits=4)
    d.update(kw)
    return R.RunSpec(**d)


def config(tmp_path, **kw):
    c = {"campanha": "c", "saida": str(tmp_path), "bases": [BASE], "classificadores": ["DA"],
         "variantes": ["MST"], "folds": [0, 1], "sementes": [0], "params": dict(FAST),
         "n_splits": 4}
    c.update(kw)
    return c


# ------------------------------------------------------------------ load_base
def test_load_synthetic_variants():
    X, y, names = R.load_base("sintetica-50-4", "x")
    assert X.shape == (50, 4) and len(names) == 4
    X2, _, _ = R.load_base("sintetica", "x")
    assert X2.shape == (200, 20)


def test_load_processed_from_dir(tmp_path):
    from mst.datasets import save_processed, synthetic

    save_processed(synthetic(n=40, m=3), tmp_path / "abc.csv")
    X, y, _ = R.load_base("abc", str(tmp_path))
    assert X.shape == (40, 3)


def test_folds_written_for_real_bases(tmp_path):
    from mst.datasets import save_processed, synthetic

    save_processed(synthetic(n=60, m=3), tmp_path / "p" / "b.csv")
    s = spec(tmp_path, base="b", dados=str(tmp_path / "p"), splits=str(tmp_path / "s"))
    R.execute(s)
    assert (tmp_path / "s" / "b.json").exists()


# -------------------------------------------------------------------- execute
@pytest.mark.parametrize("variante", ["MST", "MST_FREQ", "MST_SEM_TABU", "MST_SEM_REINICIO",
                                      "GULOSO"])
def test_execute_record_complete(tmp_path, variante):
    rec = R.execute(spec(tmp_path, variante))
    assert runlog.validate_record(rec) == []
    assert rec["variante"] == variante
    assert rec["tamanho_S_star"] == len(rec["S_star"]) >= 1
    assert len(rec["S_star_nomes"]) == rec["tamanho_S_star"]
    assert 0 < rec["f_treino"] <= 1
    assert set(rec["validacao"]) >= {"acc", "auc", "gmean", "f1"}
    json.loads(runlog.dumps(rec))


def test_f_treino_consistent_with_rat(tmp_path):
    rec = R.execute(spec(tmp_path))
    m = rec["m"]
    expected = 0.99 * rec["rat_treino"] + 0.01 * (1 - rec["tamanho_S_star"] / m)
    assert rec["f_treino"] == pytest.approx(expected)


def test_execute_reproducible(tmp_path):
    a = R.execute(spec(tmp_path))
    b = R.execute(spec(tmp_path))
    for k in ("S_star", "f_treino", "validacao", "iter_locais_total", "n_evals", "reinicios"):
        assert a[k] == b[k]


def test_variant_params(tmp_path):
    a = R.execute(spec(tmp_path, "MST_SEM_TABU"))
    assert a["params"]["tenure"] == 0
    b = R.execute(spec(tmp_path, "MST_SEM_REINICIO"))
    assert b["n_buscas_locais"] == 1
    c = R.execute(spec(tmp_path, "MST_FREQ"))
    assert c["params"]["use_freq"] is True and c["freq_contadores"] is not None


def test_params_resolved_from_scale_base(tmp_path):
    rec = R.execute(spec(tmp_path, params={"scale_base": "casos", "maxiter_ms": 1,
                                           "maxiter_ts": 5}))
    assert rec["params"]["tenure"] == rec["n_treino"] // 2
    assert rec["params"]["maxiter_ts"] == 5


def test_tenure_override(tmp_path):
    rec = R.execute(spec(tmp_path, params={**FAST, "tenure": 3}))
    assert rec["params"]["tenure"] == 3


def test_budget_mode(tmp_path):
    rec = R.execute(spec(tmp_path, max_evals=150))
    assert rec["stop_reason"] in ("orcamento", "sem_avaliacoes_novas")
    assert rec["n_evals"] <= 150


@pytest.mark.parametrize("bl", ["GA", "PSO", "GWO", "WOA", "FPA"])
def test_baselines_with_time_budget(tmp_path, bl):
    rec = R.execute(spec(tmp_path, bl, time_budget=0.2))
    assert rec["stop_reason"] in ("tempo", "sem_avaliacoes_novas")
    assert rec["tamanho_S_star"] >= 1


def test_baseline_needs_budget(tmp_path):
    with pytest.raises(ValueError):
        R.execute(spec(tmp_path, "GA"))


def test_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        R.execute(spec(tmp_path, "XYZ"))


def test_trace_written(tmp_path):
    rec = R.execute(spec(tmp_path, trace=True))
    rows = runlog.read(rec["trace_path"])
    assert rows and {"reinicio", "it", "max_local"} <= set(rows[0])
    assert rec["fugas_confirmadas"] is not None


@pytest.mark.parametrize("clf", ["LR", "SVM", "DA_SK"])
def test_other_classifiers(tmp_path, clf):
    rec = R.execute(spec(tmp_path, classificador=clf, params={**FAST, "maxiter_ts": 3,
                                                              "maxiter_ms": 1}))
    assert rec["classificador"] == clf and rec["f_treino"] > 0


def test_no_normalization_option(tmp_path):
    rec = R.execute(spec(tmp_path, params={**FAST, "normalizar": False}))
    assert rec["params"]["normalizar"] is False


# ---------------------------------------------------------------------- expand
def test_expand_cartesian(tmp_path):
    c = config(tmp_path, variantes=["MST", "GULOSO"], sementes=[0, 1], folds=[0, 1, 2])
    assert len(R.expand(c)) == 2 * 2 * 3


def test_expand_all_folds(tmp_path):
    assert len(R.expand(config(tmp_path, folds="todos"))) == 4


def test_expand_grid(tmp_path):
    c = config(tmp_path, variantes=["MST_FREQ"], grade={"lam": [0, 0.01], "trigger": [2, 3, 5]})
    specs = R.expand(c)
    assert len(specs) == 2 * 6
    assert {(s.params["lam"], s.params["trigger"]) for s in specs} == {
        (l, t) for l in (0, 0.01) for t in (2, 3, 5)}


def test_expand_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        R.expand(config(tmp_path, variantes=["NOPE"]))


def test_keys_unique(tmp_path):
    specs = R.expand(config(tmp_path, variantes=["MST", "MST_FREQ"], sementes=[0, 1]))
    assert len({s.key for s in specs}) == len(specs)


def _p1(tmp_path):
    out = R.run_campaign(config(tmp_path, campanha="p1"), progress=False)
    return out


def test_budgets_from(tmp_path):
    out = _p1(tmp_path)
    b = R.budgets_from(out)
    assert set(b) == {(BASE, "DA", 0), (BASE, "DA", 1)}
    recs = runlog.read(out)
    assert b[(BASE, "DA", 0)] == [r["n_evals"] for r in recs if r["fold"] == 0][0]


def test_expand_p2_sets_budget(tmp_path):
    out = _p1(tmp_path)
    c = config(tmp_path, campanha="p2", variantes=["MST", "MST_FREQ", "GULOSO"],
               protocolo={"tipo": "P2", "orcamento_de": str(out)})
    specs = R.expand(c)
    b = R.budgets_from(out)
    for s in specs:
        if s.variante == "GULOSO":
            assert s.max_evals is None
        else:
            assert s.max_evals == int(b[(s.base, s.classificador, s.fold)])


def test_expand_p2_missing_budget(tmp_path):
    out = _p1(tmp_path)
    c = config(tmp_path, folds=[3], protocolo={"tipo": "P2", "orcamento_de": str(out)})
    with pytest.raises(KeyError):
        R.expand(c)


def test_expand_baseline_time_from(tmp_path):
    out = _p1(tmp_path)
    c = config(tmp_path, variantes=["GA"], tempo_de=str(out))
    specs = R.expand(c)
    assert all(s.time_budget is not None and s.time_budget > 0 for s in specs)


def test_expand_baseline_fixed_time(tmp_path):
    specs = R.expand(config(tmp_path, variantes=["PSO"], tempo_s=0.1))
    assert all(s.time_budget == 0.1 for s in specs)


def test_expand_baseline_without_time_fails(tmp_path):
    with pytest.raises(KeyError):
        R.expand(config(tmp_path, variantes=["PSO"]))


# ---------------------------------------------------------------- run_campaign
def test_run_campaign_writes_and_resumes(tmp_path):
    c = config(tmp_path)
    out = R.run_campaign(c, progress=False)
    assert len(runlog.read(out)) == 2
    R.run_campaign(c, progress=False)  # retoma: nada novo
    assert len(runlog.read(out)) == 2
    R.run_campaign(c, resume=False, progress=False)
    assert len(runlog.read(out)) == 4


def test_run_campaign_errors_logged(tmp_path):
    c = config(tmp_path, bases=["inexistente"], dados=str(tmp_path / "vazio"))
    out = R.run_campaign(c, progress=False)
    errs = runlog.read(out.with_name("erros.jsonl"))
    assert len(errs) == 2 and "erro" in errs[0]
    assert runlog.read(out) == []


def test_run_campaign_parallel_same_results(tmp_path):
    a = R.run_campaign(config(tmp_path / "a", sementes=[0, 1]), n_jobs=1, progress=False)
    b = R.run_campaign(config(tmp_path / "b", sementes=[0, 1]), n_jobs=2, progress=False)
    ka = {(r["fold"], r["seed"]): (r["S_star"], r["f_treino"]) for r in runlog.read(a)}
    kb = {(r["fold"], r["seed"]): (r["S_star"], r["f_treino"]) for r in runlog.read(b)}
    assert ka == kb


def test_run_campaign_progress_output(tmp_path, capsys):
    R.run_campaign(config(tmp_path, folds=[0]), progress=True)
    out = capsys.readouterr().out
    assert "processo" in out and "concluído" in out


def test_n_jobs_auto(tmp_path):
    assert R.default_n_jobs() >= 1
    out = R.run_campaign(config(tmp_path, folds=[0], n_jobs="auto"), progress=False)
    assert len(runlog.read(out)) == 1


def test_limit_blas_threads_sets_env(monkeypatch):
    import os

    R.limit_blas_threads(1)
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_pool_one_process_per_run():
    import sys

    from mst.runner import pool_kwargs

    kw = pool_kwargs(3, 1)
    assert kw["max_workers"] == 3 and kw["initargs"] == (1,)
    if sys.version_info >= (3, 11):
        assert kw["max_tasks_per_child"] == 1
        assert kw["mp_context"].get_start_method() == "spawn"
