import numpy as np
import pandas as pd
import pytest

from mst import reporting as P
from mst import runner as R
from mst import runlog

BASE = "sintetica-80-6-3-0"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rep")
    cfg = {"campanha": "r", "saida": str(tmp), "bases": [BASE], "classificadores": ["DA"],
           "variantes": ["MST", "MST_FREQ", "GULOSO"], "folds": [0, 1, 2], "sementes": [0, 1],
           "params": {"scale_base": "variaveis", "maxiter_ms": 3, "trigger": 1}, "n_splits": 4}
    out = R.run_campaign(cfg, progress=False)
    return runlog.read(out)


@pytest.fixture(scope="module")
def df(runs):
    return P.to_frame(runs)


def test_to_frame_columns(df):
    for c in ("variante", "base", "f_treino", "acc", "auc", "gmean", "f1", "diversidade",
              "S_star", "tempo_s"):
        assert c in df.columns
    assert len(df) == 3 * 3 * 2


def test_to_frame_skips_errors(runs):
    assert len(P.to_frame(runs + [{"erro": "x"}])) == len(runs)


def test_summary_table(df):
    t = P.summary_table(df)
    assert set(t.variante) == {"MST", "MST_FREQ", "GULOSO"}
    assert all("±" in v for v in t["f_treino (%)"])


def test_compare_with_paper_no_ref_for_synthetic(df):
    t = P.compare_with_paper(df)
    assert t["f artigo (%)"].isna().all()


def test_compare_with_paper_alert():
    d = pd.DataFrame({"variante": ["MST"] * 2, "classificador": ["DA"] * 2,
                      "base": ["wdbc"] * 2, "f_treino": [0.93, 0.93]})
    t = P.compare_with_paper(d)
    assert t["alerta (>1 p.p.)"].iloc[0]
    assert t["diferença (p.p.)"].iloc[0] == pytest.approx(93 - 95.58, abs=0.01)


def test_paper_reference_tables_complete():
    from mst.datasets import PAPER_ORDER

    assert set(P.PAPER_F_DA) == set(PAPER_ORDER) == set(P.PAPER_TIME)


def test_table4(df):
    t = P.table4(df, base=BASE, fold=0, seed=0)
    assert list(t.variante) == ["MST", "MST_FREQ"]
    for c in ("solução encontrada", "função objetivo", "tempo (s)", "iterações"):
        assert c in t.columns
    assert "fold 0, semente 0" in t["execução representativa"].iloc[0]


def test_table4_missing_rep_falls_back(df):
    t = P.table4(df, base=BASE, fold=99, seed=99)
    assert len(t) == 2


def test_stability_table(df):
    t = P.stability_table(df)
    assert t["estabilidade"].between(0, 1).all()


def test_paired_tests(df):
    t = P.paired_tests(df, "MST_FREQ", "MST")
    assert {"p", "p_holm", "medida", "n"} <= set(t.columns)
    assert (t["n"] == 6).all()
    assert (t["p_holm"] >= t["p"] - 1e-12).all()


def test_choose_grid():
    d = pd.DataFrame({"variante": ["MST_FREQ"] * 4, "lam": [0, 0, 0.01, 0.01],
                      "trigger": [3, 3, 3, 3], "f_treino": [0.8, 0.8, 0.9, 0.9],
                      "diversidade": [0.1, 0.1, 0.2, 0.2]})
    g = P.choose_grid(d)
    assert g.iloc[0]["lam"] == 0.01


def test_restart_curve(runs):
    c = P.restart_curve(runs[0])
    assert list(c.columns) == ["reinicio", "f_local", "f_global"]
    assert (np.diff(c.f_global) >= 0).all()


def test_markdown_escapes_pipes():
    md = P.to_markdown(pd.DataFrame({"a": ["x|y"]}))
    assert "x\\|y" in md
    assert md.count("\n") == 3


def test_markdown_and_latex_empty():
    assert "sem dados" in P.to_markdown(pd.DataFrame())
    assert "sem dados" in P.to_latex(pd.DataFrame())


def test_latex_escapes():
    tex = P.to_latex(pd.DataFrame({"f_treino (%)": ["1 ± 2"]}))
    assert r"f\_treino (\%)" in tex and r"$\pm$" in tex and r"\begin{tabular}" in tex
