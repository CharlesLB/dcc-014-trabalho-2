"""Testes de experiments/reutilizar.py."""

from experiments import reutilizar
from mst import runlog
from mst.runner import expand, run_campaign


def _cfg(tmp_path, campanha, variantes=("MST",), **params):
    p = {"scale_base": "variaveis", "maxiter_ms": 2, **params}
    return {"campanha": campanha, "saida": str(tmp_path), "bases": ["sintetica-80-8-3-0"],
            "classificadores": ["DA"], "variantes": list(variantes), "folds": [0, 1],
            "sementes": [0], "params": p}


def test_reuse_copies_identical_runs(tmp_path):
    src = run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    # campanha B pede o mesmo MST, só com lam/trigger explícitos (iguais aos padrões)
    cfg_b = _cfg(tmp_path, "B", variantes=("GULOSO", "MST", "MST_FREQ"), lam=0.01, trigger=3)
    c, f = reutilizar.reuse(cfg_b, src, variantes=["MST"])
    assert (c, f) == (2, 0)
    out = tmp_path / "B" / "runs.jsonl"
    recs = runlog.read(out)
    assert all(r["reaproveitado_de"]["campanha"] == "A" for r in recs)
    # a retomada de B pula os MST copiados e roda só o resto
    keys_b = {s.key for s in expand(cfg_b) if s.variante == "MST"}
    assert keys_b <= runlog.done_keys(out)
    run_campaign(cfg_b, n_jobs=1, progress=False)
    recs = runlog.read(out)
    assert sorted(r["variante"] for r in recs) == ["GULOSO"] * 2 + ["MST"] * 2 + ["MST_FREQ"] * 2


def test_reused_equals_fresh_run(tmp_path):
    src = run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    reutilizar.reuse(_cfg(tmp_path, "B", lam=0.01), src)
    fresh = run_campaign(_cfg(tmp_path, "C", lam=0.01), n_jobs=1, progress=False)
    a = {(r["fold"], r["seed"]): r for r in runlog.read(tmp_path / "B" / "runs.jsonl")}
    b = {(r["fold"], r["seed"]): r for r in runlog.read(fresh)}
    for k in a:
        for campo in ("S_star", "f_treino", "n_evals", "iter_locais_total", "reinicios"):
            assert a[k][campo] == b[k][campo]


def test_reuse_refuses_different_params(tmp_path):
    src = run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    c, f = reutilizar.reuse(_cfg(tmp_path, "B", maxiter_ms=3), src)
    assert (c, f) == (0, 2)
    c, f = reutilizar.reuse(_cfg(tmp_path, "D", scale_base="casos"), src)
    assert (c, f) == (0, 2)


def test_reuse_is_idempotent(tmp_path):
    src = run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    cfg = _cfg(tmp_path, "B")
    assert reutilizar.reuse(cfg, src) == (2, 0)
    assert reutilizar.reuse(cfg, src) == (0, 0)
    assert len(runlog.read(tmp_path / "B" / "runs.jsonl")) == 2


def test_cli(tmp_path, capsys):
    import yaml

    src = run_campaign(_cfg(tmp_path, "A"), n_jobs=1, progress=False)
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(_cfg(tmp_path, "B")))
    assert reutilizar.main(["--de", str(src), "--config", str(p)]) == 0
    assert "2 execuções reaproveitadas" in capsys.readouterr().out
