"""Dados do painel de resultados (página publicada): um CSV por bloco do report.

Lê as mesmas campanhas de ``reports.comparacao`` e ``reports.tabelas`` e grava, em
formato longo, os números que a página desenha.

Uso:
    python -m reports.painel [--resultados results] [--saida results/relatorio/painel]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from mst.reporting import choose_grid, paired_tests, stability_table, to_frame  # noqa: E402
from reports import artigo as A  # noqa: E402
from reports.comparacao import FONTES, METRICAS, carregar, por_fold, resumo, wtl  # noqa: E402

NOME_FONTE = {"Grupo (n = variáveis)": "Grupo, n = variáveis",
              "Grupo (n = casos)": "Grupo, n = casos",
              "Grupo DA_PASCAL": "Grupo, prioris iguais"}
NOME_MED = {"f_treino": "f (treino)", "acc": "ACC", "auc": "AUC", "gmean": "Gmean", "f1": "F1"}


def _long(series: dict, extra: dict | None = None) -> list[dict]:
    out = []
    for fonte, d in series.items():
        for b in A.BASES:
            if b in d and d[b] is not None:
                v = d[b]
                out.append({"base": b, "nome": A.NOMES[b], **(extra or {}), "fonte": fonte,
                            "media": round(float(v[0]), 3), "dp": round(float(v[1]), 3)})
    return out


def build(resultados: Path, saida: Path) -> list[Path]:
    saida.mkdir(parents=True, exist_ok=True)
    df = carregar(resultados)
    pfs = {NOME_FONTE[n]: por_fold(df, *src) for n, src in FONTES.items()}
    files = {}

    # 1. f do MST (Tabela 3)
    s = {"Artigo": {b: A.F_DA[b]["MST"] for b in A.BASES}}
    s.update({n: resumo(pf, "f_treino") for n, pf in pfs.items()})
    files["f_mst"] = pd.DataFrame(_long(s))

    # 2. métricas de validação (Tabelas 4 a 7)
    rows = []
    for met, tab in zip(METRICAS, (A.ACC_DA, A.AUC_DA, A.GMEAN_DA, A.F1_DA)):
        s = {"Artigo": {b: tab[b]["MST"] for b in A.BASES}}
        s.update({n: resumo(pf, met) for n, pf in pfs.items()})
        rows += _long(s, {"metrica": NOME_MED[met]})
    files["metricas"] = pd.DataFrame(rows)

    # 3. tempo (Tabela 2): MST e MST_FREQ rodados juntos, mesma máquina e carga
    s = {"Artigo (Pascal, i9-10920X)": {b: A.TEMPO["DA"][b] for b in A.BASES}}
    for v in ("MST", "MST_FREQ"):
        pf = por_fold(df, "E2-P1-tempo", v, "DA")
        s[f"Grupo {v} (Python, Ryzen 9 7950X)"] = resumo(pf, "tempo_s")
    files["tempo"] = pd.DataFrame(_long(s))

    # 4. baselines (Tabelas 3 a 8)
    rows, difs = [], []
    mst = por_fold(df, "E1-reproducao-DA", "MST", "DA")
    for med, tab in (("f_treino", A.F_DA), ("acc", A.ACC_DA), ("auc", A.AUC_DA)):
        grupo = {m: resumo(mst if m == "MST" else por_fold(df, "E4-baselines-DA", m, "DA"), med)
                 for m in A.METODOS}
        for b in A.BASES:
            for m in A.METODOS:
                g = grupo[m].get(b)
                rows.append({"chave": f"{b}|{med}|{m}", "base": b, "nome": A.NOMES[b],
                             "medida": NOME_MED[med], "metodo": m,
                             "artigo": tab[b][m], "grupo": round(g[0], 3) if g else None})
            art = tab[b]["MST"][0] - max(tab[b][m][0] for m in A.METODOS[1:])
            gr = grupo["MST"][b][0] - max(grupo[m][b][0] for m in A.METODOS[1:])
            for fonte, v in (("Artigo", art), ("Grupo, n = variáveis", gr)):
                difs.append({"base": b, "nome": A.NOMES[b], "medida": NOME_MED[med],
                             "fonte": fonte, "media": round(v, 3), "dp": 0.0})
    bl = pd.DataFrame(rows)
    bl["artigo"] = [v[0] for v in bl.artigo]
    files["baselines"] = bl
    files["baselines_dif"] = pd.DataFrame(difs)

    rows = []
    for med in ("f_treino", "acc", "auc", "gmean", "f1"):
        g = wtl(df, "E1-reproducao-DA", "E4-baselines-DA", med)
        for m in A.METODOS[1:]:
            rows.append({"chave": f"{med}|{m}", "medida": NOME_MED[med], "metodo": m,
                         "artigo": A.TABELA8[med][m],
                         "grupo": "/".join(map(str, g[m]))})
    files["wtl"] = pd.DataFrame(rows)

    # 5. LR e SVM (Tabelas 9 e 11)
    rows = []
    for clf in ("LR", "SVM"):
        s = {"Artigo": {b: A.ACC[clf][b]["MST"] for b in A.BASES},
             "Grupo, n = variáveis": resumo(por_fold(df, "E4-LR-SVM", "MST", clf), "acc")}
        rows += _long(s, {"clf": clf})
    files["lr_svm"] = pd.DataFrame(rows)

    # 6. E2-P2 e 7. E3 (testes pareados de reporting)
    p2 = to_frame(runlog.read(resultados / "E2-P2" / "runs.jsonl"))
    e3 = to_frame(r for c in ("E3-ablacoes", "E2-P2") for r in runlog.read(resultados / c / "runs.jsonl"))
    t = paired_tests(p2, "MST_FREQ", "MST")
    est = stability_table(p2).set_index(["variante", "base"])["estabilidade"]
    raw = {(r["variante"], r["base"], r["fold"], r["seed"]): frozenset(r["S_star"])
           for r in runlog.read(resultados / "E2-P2" / "runs.jsonl")}
    rows = []
    for b in A.BASES:
        tb = t[t.base == b].set_index("medida")
        fm = p2[(p2.base == b) & (p2.variante == "MST")]
        ff = p2[(p2.base == b) & (p2.variante == "MST_FREQ")]
        iguais = sum(raw[("MST", b, f, s)] == raw[("MST_FREQ", b, f, s)]
                     for (v, bb, f, s) in raw if v == "MST" and bb == b)
        rows.append({"base": b, "nome": A.NOMES[b],
                     "f_mst": round(100 * fm.f_treino.mean(), 2), "f_freq": round(100 * ff.f_treino.mean(), 2),
                     "dif_f": round(100 * tb.loc["f_treino", "media_dif"], 3),
                     "p_holm_f": round(tb.loc["f_treino", "p_holm"], 3),
                     "acc_mst": round(100 * fm.acc.mean(), 2), "acc_freq": round(100 * ff.acc.mean(), 2),
                     "dif_div": round(tb.loc["diversidade", "media_dif"], 4),
                     "p_holm_div": round(tb.loc["diversidade", "p_holm"], 3),
                     "est_mst": round(est[("MST", b)], 3), "est_freq": round(est[("MST_FREQ", b)], 3),
                     "iguais": int(iguais), "n": int(len(fm))})
    files["e2"] = pd.DataFrame(rows)

    rows = []
    for v, nome in (("MST", "MST"), ("MST_SEM_TABU", "Sem lista tabu"), ("MST_SEM_REINICIO", "Sem reinícios")):
        d = e3[e3.variante == v]
        for b in A.BASES:
            g = d[d.base == b].groupby("fold").f_treino.mean() * 100
            rows.append({"base": b, "nome": A.NOMES[b], "fonte": nome,
                         "media": round(g.mean(), 3), "dp": round(g.std(ddof=1), 3)})
    files["e3"] = pd.DataFrame(rows)

    # 8. E5
    rows = []
    for r in runlog.read(resultados / "E5-fuga" / "runs.jsonl"):
        rows.append({"base": r["base"], "variante": r["variante"], "max": r["maximos_locais"],
                     "fugas": r["fugas_confirmadas"], "f": 100 * r["f_treino"]})
    e5 = pd.DataFrame(rows).groupby(["base", "variante"], as_index=False).agg(
        maximos=("max", "sum"), fugas=("fugas", "sum"), f=("f", "mean"))
    e5["pct"] = (100 * e5.fugas / e5.maximos).round(1)
    e5["f"] = e5.f.round(2)
    e5["chave"] = e5.base + "|" + e5.variante
    e5["nome"] = e5.base.map(A.NOMES)
    files["e5"] = e5[["chave", "nome", "variante", "maximos", "fugas", "pct", "f"]]

    # 9. E2a
    g = choose_grid(to_frame(runlog.read(resultados / "E2a-ajuste" / "runs.jsonl"))).reset_index(drop=True)
    g["f_medio"] = (100 * g.f_medio).round(3)
    g["chave"] = g.lam.astype(str) + "|" + g.trigger.astype(int).astype(str)
    g["trigger"] = g.trigger.astype(int)
    files["e2a"] = g[["chave", "lam", "trigger", "f_medio", "diversidade"]].round(4)

    out = []
    for k, d in files.items():
        p = saida / f"{k}.csv"
        d.to_csv(p, index=False)
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resultados", default="results")
    ap.add_argument("--saida", default="results/relatorio/painel")
    a = ap.parse_args(argv)
    for p in build(Path(a.resultados), Path(a.saida)):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
