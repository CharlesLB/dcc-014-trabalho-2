"""Testes de experiments/fila.py."""

import yaml

from experiments import fila
from mst import runlog
from mst.runner import run_campaign


def _cfg(tmp_path, campanha, bases=("sintetica-80-8-3-0",), variantes=("MST",)):
    return {"campanha": campanha, "saida": str(tmp_path), "bases": list(bases),
            "classificadores": ["DA"], "variantes": list(variantes), "folds": [0, 1],
            "sementes": [0], "params": {"scale_base": "variaveis", "maxiter_ms": 2}}


def _write(tmp_path, name, cfg):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_queue_equals_separate_campaigns(tmp_path):
    a = _write(tmp_path, "a.yaml", _cfg(tmp_path, "A"))
    b = _write(tmp_path, "b.yaml", _cfg(tmp_path, "B", variantes=("GULOSO", "MST_FREQ")))
    n = fila.run_queue({"n_jobs": 2, "entradas": [{"config": a}, {"config": b}]}, progress=False)
    assert n == 6
    ref = run_campaign(_cfg(tmp_path / "ref", "A"), n_jobs=1, progress=False)
    got = {(r["fold"], r["seed"]): r for r in runlog.read(tmp_path / "A" / "runs.jsonl")}
    exp = {(r["fold"], r["seed"]): r for r in runlog.read(ref)}
    for k in exp:
        for c in ("S_star", "f_treino", "n_evals", "run_key"):
            assert got[k][c] == exp[k][c]
    assert len(runlog.read(tmp_path / "B" / "runs.jsonl")) == 4


def test_queue_resumes_and_skips_done(tmp_path):
    a = _write(tmp_path, "a.yaml", _cfg(tmp_path, "A"))
    run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    assert fila.plan({"entradas": [{"config": a}]}) == []
    assert fila.run_queue({"entradas": [{"config": a}]}, progress=False) == 0


def test_plan_order_priority_then_cost(tmp_path):
    a = _write(tmp_path, "a.yaml", _cfg(tmp_path, "A", bases=("sintetica-80-8-3-0", "sintetica-60-6-2-0")))
    b = _write(tmp_path, "b.yaml", _cfg(tmp_path, "B"))
    specs = fila.plan({"custo_base": {"sintetica-60-6-2-0": 50},
                       "entradas": [{"config": a, "prioridade": 1}, {"config": b, "prioridade": 0}]})
    assert [s.campanha for s in specs] == ["B", "B", "A", "A", "A", "A"]
    assert [s.base for s in specs[2:]] == ["sintetica-60-6-2-0"] * 2 + ["sintetica-80-8-3-0"] * 2


def test_set_overrides_and_dedupe(tmp_path):
    a = _write(tmp_path, "a.yaml", _cfg(tmp_path, "A"))
    specs = fila.plan({"entradas": [{"config": a, "set": ["folds=[0]"]}, {"config": a}]})
    assert len(specs) == 2  # fold 0 não se repete


def test_cli_listar(tmp_path, capsys):
    a = _write(tmp_path, "a.yaml", _cfg(tmp_path, "A"))
    q = tmp_path / "fila.yaml"
    q.write_text(yaml.safe_dump({"entradas": [{"config": a}]}))
    assert fila.main([str(q), "--listar"]) == 0
    assert "2 execuções" in capsys.readouterr().out
