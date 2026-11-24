"""Tabelas a partir dos runs.jsonl (Tabela 4 do relatório, comparações, estatística)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis import holm, paired_wilcoxon, restart_diversity, stability

# Tabela 3 do artigo: f (%) no treino com DA, MST (média ± dp).
PAPER_F_DA = {
    "parkinson": (83.76, 0.36), "qadc": (83.95, 0.51), "spectf": (81.49, 0.52),
    "wdbc": (95.58, 0.16), "wpbc": (79.44, 0.48), "alzheimer": (82.38, 0.45),
    "cervical": (91.53, 0.19), "glioma": (89.56, 0.70), "mesothelioma": (75.03, 0.64),
}
# Tabela 2 do artigo: tempo por fold (s) do MST com DA, LR e SVM.
PAPER_TIME = {
    "parkinson": (0.529, 17.761, 10.412), "qadc": (71.748, 1906.915, 1513.337),
    "spectf": (9.091, 642.467, 383.022), "wdbc": (1.187, 171.111, 176.336),
    "wpbc": (1.019, 86.006, 37.534), "alzheimer": (1.630, 260.755, 274.872),
    "cervical": (1.386, 392.119, 227.206), "glioma": (1.463, 162.417, 156.452),
    "mesothelioma": (3.125, 763.716, 396.045),
}

METRICS = ("acc", "auc", "gmean", "f1")


def to_frame(records: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if "erro" in r:
            continue
        row = {k: r.get(k) for k in ("variante", "base", "classificador", "fold", "seed",
                                      "f_treino", "tamanho_S_star", "tempo_s",
                                      "iter_locais_total", "n_buscas_locais", "n_evals",
                                      "stop_reason", "fugas_confirmadas", "maximos_locais")}
        for mk in METRICS:
            row[mk] = (r.get("validacao") or {}).get(mk, np.nan)
        locs = [x["S_local"] for x in r.get("reinicios") or []]
        row["diversidade"] = restart_diversity(locs) if len(locs) > 1 else np.nan
        row["S_star"] = tuple(r.get("S_star") or [])
        row["S_star_nomes"] = tuple(r.get("S_star_nomes") or [])
        p = r.get("params") or {}
        row["lam"] = p.get("lam")
        row["trigger"] = p.get("trigger")
        row["scale_base"] = p.get("scale_base")
        rows.append(row)
    return pd.DataFrame(rows)


def _ms(x: pd.Series, scale: float = 1.0, nd: int = 2) -> str:
    x = x.dropna() * scale
    if x.empty:
        return "—"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{nd}f} ± {sd:.{nd}f}"


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Média ± dp por (variante, base, classificador)."""
    out = []
    for (v, b, c), g in df.groupby(["variante", "base", "classificador"], sort=True):
        out.append({
            "variante": v, "base": b, "classificador": c, "n": len(g),
            "f_treino (%)": _ms(g["f_treino"], 100),
            **{f"{mk} (%)": _ms(g[mk], 100) for mk in METRICS},
            "tam. S*": _ms(g["tamanho_S_star"], 1, 1),
            "tempo (s)": _ms(g["tempo_s"], 1, 3),
            "iter. locais": _ms(g["iter_locais_total"], 1, 0),
            "buscas locais": _ms(g["n_buscas_locais"], 1, 1),
            "avaliações": _ms(g["n_evals"], 1, 0),
        })
    return pd.DataFrame(out)


def compare_with_paper(df: pd.DataFrame, variante: str = "MST", classificador: str = "DA") -> pd.DataFrame:
    g = df[(df.variante == variante) & (df.classificador == classificador)]
    rows = []
    for b, gb in g.groupby("base"):
        ref = PAPER_F_DA.get(b) if classificador == "DA" else None
        ours = gb["f_treino"].mean() * 100
        rows.append({"base": b, "f grupo (%)": round(ours, 2),
                     "f artigo (%)": ref[0] if ref else None,
                     "diferença (p.p.)": round(ours - ref[0], 2) if ref else None,
                     "alerta (>1 p.p.)": bool(ref and abs(ours - ref[0]) > 1.0)})
    return pd.DataFrame(rows)


def table4(df: pd.DataFrame, variantes=("MST", "MST_FREQ"), base: str = "wpbc",
           classificador: str = "DA", fold: int = 0, seed: int = 0) -> pd.DataFrame:
    """Tabela 4 do relatório: execução representativa + agregado."""
    rows = []
    for v in variantes:
        g = df[(df.variante == v) & (df.base == base) & (df.classificador == classificador)]
        if g.empty:
            continue
        rep = g[(g.fold == fold) & (g.seed == seed)]
        rep = rep.iloc[0] if not rep.empty else g.iloc[0]
        rows.append({
            "variante": v,
            "solução encontrada": f"{', '.join(rep['S_star_nomes'])} (tamanho {rep['tamanho_S_star']})",
            "função objetivo": f"{rep['f_treino']:.4f}",
            "tempo (s)": f"{rep['tempo_s']:.3f}",
            "iterações": f"{int(rep['iter_locais_total'])} locais em {int(rep['n_buscas_locais'])} buscas",
            "agregado f (%)": _ms(g["f_treino"], 100),
            "agregado tamanho": _ms(g["tamanho_S_star"], 1, 1),
            "agregado tempo (s)": _ms(g["tempo_s"], 1, 3),
            "agregado iterações": _ms(g["iter_locais_total"], 1, 0),
            "execução representativa": f"fold {int(rep['fold'])}, semente {int(rep['seed'])}",
        })
    return pd.DataFrame(rows)


def stability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, b, c, s), g in df.groupby(["variante", "base", "classificador", "seed"]):
        rows.append({"variante": v, "base": b, "classificador": c, "seed": s,
                     "estabilidade": stability(list(g["S_star"]))})
    st = pd.DataFrame(rows)
    if st.empty:
        return st
    return st.groupby(["variante", "base", "classificador"], as_index=False)["estabilidade"].mean()


def paired_tests(df: pd.DataFrame, a: str, b: str,
                 medidas=("f_treino", "acc", "auc", "gmean", "f1", "tamanho_S_star", "diversidade"),
                 classificador: str = "DA") -> pd.DataFrame:
    """Wilcoxon pareado a vs b por base e medida, pareando por (fold, seed), com Holm."""
    rows = []
    d = df[df.classificador == classificador]
    for base, g in d.groupby("base"):
        ga = g[g.variante == a].set_index(["fold", "seed"])
        gb = g[g.variante == b].set_index(["fold", "seed"])
        idx = ga.index.intersection(gb.index)
        for med in medidas:
            xa = ga.loc[idx, med].astype(float)
            xb = gb.loc[idx, med].astype(float)
            ok = ~(xa.isna() | xb.isna())
            r = paired_wilcoxon(xa[ok].to_numpy(), xb[ok].to_numpy())
            rows.append({"base": base, "medida": med, "a": a, "b": b, **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].tolist())
    return out


def choose_grid(df: pd.DataFrame, variante: str = "MST_FREQ") -> pd.DataFrame:
    """Ranqueia (lam, trigger): maior f médio, desempate por diversidade."""
    g = df[df.variante == variante]
    if g.empty:
        return g
    agg = g.groupby(["lam", "trigger"], as_index=False).agg(
        f_medio=("f_treino", "mean"), diversidade=("diversidade", "mean"), n=("f_treino", "size"))
    return agg.sort_values(["f_medio", "diversidade"], ascending=[False, False]).reset_index(drop=True)


def restart_curve(record: dict) -> pd.DataFrame:
    """Evolução de f por reinício (como a Figura 1 do artigo)."""
    rs = record.get("reinicios") or []
    return pd.DataFrame({"reinicio": [x["r"] for x in rs],
                         "f_local": [x["f_local"] for x in rs],
                         "f_global": [x["f_global"] for x in rs]})


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sem dados)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return "% sem dados\n"
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
             " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
