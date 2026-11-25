"""Teste de fumaça do relatório comparativo (reports/comparacao.py)."""

from mst import runlog
from reports import artigo as A
from reports import comparacao as C


def _rec(camp, var, base, fold, seed, f, clf="DA"):
    return {"campanha": camp, "variante": var, "base": base, "classificador": clf, "fold": fold,
            "seed": seed, "f_treino": f, "tempo_s": 1.0, "tamanho_S_star": 3, "m": 10,
            "validacao": {"acc": 0.8, "auc": 0.7, "gmean": 0.6, "f1": 0.5}}


def test_build_gera_tabelas_e_figuras(tmp_path):
    res = tmp_path / "results"
    for b in A.BASES[:2]:
        for fold in range(3):
            for seed in range(2):
                runlog.append(res / "E1-reproducao-DA" / "runs.jsonl",
                              _rec("E1-reproducao-DA", "MST", b, fold, seed, 0.9 + 0.01 * seed))
                runlog.append(res / "E4-baselines-DA" / "runs.jsonl",
                              _rec("E4-baselines-DA", "GA", b, fold, seed, 0.85))
    out = tmp_path / "comp"
    md = C.build(res, out)
    assert (out / "fig_f_mst.png").exists() and (out / "fig_baselines_f_treino.png").exists()
    # média das sementes no fold (90,5) e dp entre folds (0)
    assert "90.50 ± 0.00" in md
    assert "Parkinson" in md


def test_por_fold_tira_media_das_sementes():
    import pandas as pd

    df = pd.DataFrame([{"campanha": "X", "variante": "MST", "clf": "DA", "base": "b", "fold": 0,
                        "seed": s, "f_treino": v, "tempo_s": 1, "tam": 2, "acc": 1}
                       for s, v in ((0, 80.0), (1, 90.0))])
    pf = C.por_fold(df, "X", "MST", "DA")
    assert pf.f_treino.tolist() == [85.0]
