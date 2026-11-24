"""Testes das linhas de comando (data.prepare, experiments.run, reports.*)."""

from pathlib import Path

import pytest
import yaml

import fakes
from data import prepare
from experiments import run as runcli
from mst import runlog
from reports import figuras, fugas, tabela4, tabelas

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "experiments" / "configs").glob("*.yaml"))


# ------------------------------------------------------------------- prepare
def test_prepare_all_fakes(tmp_path, capsys):
    raw = tmp_path / "raw"
    fakes.make_all(raw)
    rc = prepare.main(["--raw", str(raw), "--processed", str(tmp_path / "p"),
                       "--splits", str(tmp_path / "s")])
    assert rc == 0
    out = capsys.readouterr().out
    for k in ("parkinson", "wdbc", "mesothelioma"):
        assert f"[{k}]" in out
    assert len(list((tmp_path / "p").glob("*.csv"))) == 9
    assert len(list((tmp_path / "s").glob("*.json"))) == 9


def test_prepare_strict_fails_on_divergence(tmp_path):
    raw = tmp_path / "raw"
    fakes.qadc(raw / "qadc")  # 62 variáveis != 67
    rc = prepare.main(["--bases", "qadc", "--raw", str(raw), "--processed", str(tmp_path / "p"),
                       "--splits", str(tmp_path / "s"), "--estrito"])
    assert rc == 1


def test_prepare_strict_ok(tmp_path):
    raw = tmp_path / "raw"
    fakes.wdbc(raw / "wdbc")
    rc = prepare.main(["--bases", "wdbc", "--raw", str(raw), "--processed", str(tmp_path / "p"),
                       "--splits", str(tmp_path / "s"), "--estrito"])
    assert rc == 0


def test_prepare_missing_files(tmp_path, capsys):
    rc = prepare.main(["--bases", "wdbc", "--raw", str(tmp_path / "nada"),
                       "--processed", str(tmp_path / "p"), "--splits", str(tmp_path / "s")])
    assert rc == 0
    assert "ausentes" in capsys.readouterr().out


def test_prepare_returns_status(tmp_path):
    raw = tmp_path / "raw"
    fakes.wdbc(raw / "wdbc")
    res = prepare.prepare(["wdbc", "glioma"], raw, tmp_path / "p", tmp_path / "s", verbose=False)
    assert res["wdbc"][0] == "ok" and res["glioma"][0] == "faltando"


# ----------------------------------------------------------------- configs
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_configs_are_valid_yaml_with_required_keys(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"campanha", "bases", "variantes"} <= set(cfg)


@pytest.mark.parametrize("path", [p for p in CONFIGS if "P2" not in p.stem and "ajuste" not in p.stem
                                  and "ablacoes" not in p.stem and "baselines" not in p.stem],
                         ids=lambda p: p.stem)
def test_configs_expand(path):
    from mst.runner import expand

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(expand(cfg)) > 0


def test_e1_has_450_runs():
    from mst.runner import expand

    cfg = yaml.safe_load((ROOT / "experiments/configs/E1_reproducao.yaml").read_text())
    assert len(expand(cfg)) == 450


def test_parallel_settings_in_configs():
    cfg = yaml.safe_load((ROOT / "experiments/configs/E1_reproducao.yaml").read_text())
    assert cfg["n_jobs"] == 7 and cfg["blas_threads"] == 1


# --------------------------------------------------------------- experiments
def _smoke_cfg(tmp_path):
    cfg = yaml.safe_load((ROOT / "experiments/configs/smoke.yaml").read_text())
    cfg["saida"] = str(tmp_path)
    cfg["trace"] = True
    p = tmp_path / "smoke.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_cli_smoke(tmp_path):
    p = _smoke_cfg(tmp_path)
    assert runcli.main(["--config", str(p)]) == 0
    recs = runlog.read(tmp_path / "smoke" / "runs.jsonl")
    assert len(recs) == 6


def test_run_cli_list(tmp_path, capsys):
    p = _smoke_cfg(tmp_path)
    assert runcli.main(["--config", str(p), "--listar"]) == 0
    assert "6 execuções" in capsys.readouterr().out


def test_run_cli_set_override(tmp_path, capsys):
    p = _smoke_cfg(tmp_path)
    runcli.main(["--config", str(p), "--listar", "--set", "sementes=[0, 1, 2]"])
    assert "18 execuções" in capsys.readouterr().out


def test_load_config_override(tmp_path):
    p = _smoke_cfg(tmp_path)
    cfg = runcli.load_config(p, ["n_jobs=3", "folds=todos"])
    assert cfg["n_jobs"] == 3 and cfg["folds"] == "todos"


@pytest.fixture
def smoke_runs(tmp_path):
    p = _smoke_cfg(tmp_path)
    runcli.main(["--config", str(p)])
    return tmp_path / "smoke" / "runs.jsonl"


def test_tabela4_cli(smoke_runs, tmp_path, capsys):
    pref = str(tmp_path / "t4")
    rc = tabela4.main([str(smoke_runs), "--base", "sintetica-120-12-4-0", "--saida", pref])
    assert rc == 0
    assert Path(pref + ".md").exists() and Path(pref + ".tex").exists()
    assert "MST_FREQ" in capsys.readouterr().out


def test_tabelas_cli(smoke_runs, tmp_path):
    out = tmp_path / "t.md"
    rc = tabelas.main([str(smoke_runs), "--comparar", "MST_FREQ", "MST", "--grade",
                       "--saida", str(out)])
    assert rc == 0
    txt = out.read_text()
    assert "Resumo" in txt and "Wilcoxon" in txt and "Grade" in txt


def test_tabelas_empty(tmp_path):
    assert "nenhuma" in tabelas.build([str(tmp_path / "x.jsonl")])


def test_figuras_cli(smoke_runs, tmp_path):
    out = tmp_path / "fig.png"
    rc = figuras.main([str(smoke_runs), "--base", "sintetica-120-12-4-0", "--saida", str(out)])
    assert rc == 0
    assert out.with_suffix(".csv").exists()


def test_figuras_not_found(smoke_runs, tmp_path):
    assert figuras.main([str(smoke_runs), "--base", "nao", "--saida", str(tmp_path / "f.png")]) == 1


def test_fugas_cli(smoke_runs, capsys):
    assert fugas.main([str(smoke_runs), "--exemplo"]) == 0
    out = capsys.readouterr().out
    assert "máximos locais" in out


def test_fugas_example_from_toy_trace(tmp_path):
    rec = {"variante": "MST", "base": "toy", "fold": 0, "maximos_locais": 1,
           "fugas_confirmadas": 1,
           "fugas": [{"reinicio": 1, "it_max_local": 1, "f_max_local": 0.88, "it_melhora": 2,
                      "f_nova": 0.93}],
           "trace_path": str(tmp_path / "t.jsonl")}
    runlog.write_trace(tmp_path / "t.jsonl", [{"reinicio": 1, "passos": [
        {"it": 1, "mov": {"tipo": "add"}, "f_antes": 0.88, "f_depois": 0.86, "f_star": 0.88,
         "tabu": False, "aspiracao": False, "max_local": True},
        {"it": 2, "mov": {"tipo": "swap"}, "f_antes": 0.86, "f_depois": 0.93, "f_star": 0.93,
         "tabu": False, "aspiracao": False, "max_local": False}]}])
    txt = fugas.exemplo([rec])
    assert "0.8800" in txt and "it 2" in txt


def test_fugas_none():
    assert "Nenhuma" in fugas.exemplo([{"fugas": []}])
