"""Comparação com o artigo: tabelas com todos os números e gráficos.

Lê as campanhas do projeto, resume cada (fonte, base, classificador, método) como
no artigo (média ± dp entre os 10 folds; com várias sementes, primeiro a média
das sementes em cada fold) e põe ao lado das Tabelas 2 a 11 de Pacheco et al.

Uso:
    python -m reports.comparacao [--resultados results] [--saida results/relatorio/comparacao]
Gera ``comparacao.md`` e ``fig_*.png`` na pasta de saída.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from reports import artigo as A  # noqa: E402

# paleta categórica de referência (skill dataviz), em ordem fixa
COR = {"Artigo": "#2a78d6", "Grupo (n = variáveis)": "#eb6834", "Grupo (n = casos)": "#1baf7a",
       "Grupo DA_PASCAL": "#eda100"}
COR_METODO = dict(zip(A.METODOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                  "#6250d6"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICAS = ["acc", "auc", "gmean", "f1"]
NOME_MET = {"f_treino": "f (treino)", "acc": "ACC", "auc": "AUC", "gmean": "Gmean", "f1": "F1"}

# fonte -> (campanha, variante, classificador, filtro de scale_base)
FONTES = {
    "Grupo (n = variáveis)": ("E1-reproducao-DA", "MST", "DA"),
    "Grupo (n = casos)": ("E1-casos-DA", "MST", "DA"),
    "Grupo DA_PASCAL": ("E1b-diagnostico-DA_PASCAL", "MST", "DA_PASCAL"),
}


# ------------------------------------------------------------------ dados
def carregar(resultados: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(resultados.glob("*/runs.jsonl")):
        for r in runlog.iter_records(p):
            if "erro" in r:
                continue
            v = r.get("validacao") or {}
            rows.append({"campanha": r["campanha"], "variante": r["variante"],
                         "base": r["base"], "clf": r["classificador"], "fold": int(r["fold"]),
                         "seed": int(r["seed"]), "f_treino": 100 * r["f_treino"],
                         "tempo_s": r["tempo_s"], "tam": r["tamanho_S_star"], "m": r["m"],
                         "origem": (r.get("ambiente") or {}).get("git_commit"),
                         **{k: 100 * v[k] for k in METRICAS if k in v}})
    return pd.DataFrame(rows)


def por_fold(df: pd.DataFrame, campanha: str, variante: str, clf: str) -> pd.DataFrame:
    """Média das sementes em cada (base, fold)."""
    d = df[(df.campanha == campanha) & (df.variante == variante) & (df.clf == clf)]
    if d.empty:
        return d
    cols = ["f_treino", "tempo_s", "tam", *[m for m in METRICAS if m in d]]
    return d.groupby(["base", "fold"], as_index=False)[cols].mean()


def resumo(pf: pd.DataFrame, col: str) -> dict:
    """{base: (média, dp entre folds, nº de folds)}."""
    out = {}
    if pf.empty or col not in pf:
        return out
    for b, g in pf.groupby("base"):
        x = g[col].dropna()
        out[b] = (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, len(x))
    return out


def fmt(v) -> str:
    return "—" if v is None else f"{v[0]:.2f} ± {v[1]:.2f}"


# ------------------------------------------------------------------ gráficos
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 150,
                         "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _dot(ax, series: dict, bases, xlabel=None, offset=0.16):
    """Dot plot horizontal: uma linha por base, um ponto (± dp) por série."""
    k = len(series)
    for i, (nome, vals) in enumerate(series.items()):
        dy = (i - (k - 1) / 2) * offset
        ys, xs, es = [], [], []
        for j, b in enumerate(bases):
            if b in vals:
                ys.append(j + dy)
                xs.append(vals[b][0])
                es.append(vals[b][1])
        ax.errorbar(xs, ys, xerr=es, fmt="o", ms=5, lw=1.2, capsize=0, color=COR.get(nome, INK),
                    mec="white", mew=1, label=nome)
    ax.set_yticks(range(len(bases)), [A.NOMES[b] for b in bases])
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)


def fig_f(series, saida, plt):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _dot(ax, series, A.BASES, "f no treino (%)")
    ax.set_title("f do MST com DA: artigo × grupo (média ± dp entre folds)", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=8)
    fig.savefig(saida / "fig_f_mst.png")
    plt.close(fig)


def fig_dif(series, saida, plt):
    art = series["Artigo"]
    outras = {k: v for k, v in series.items() if k != "Artigo"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    k = len(outras)
    h = 0.8 / max(k, 1)
    for i, (nome, vals) in enumerate(outras.items()):
        ys = [j + (i - (k - 1) / 2) * h for j, b in enumerate(A.BASES) if b in vals]
        ds = [vals[b][0] - art[b][0] for b in A.BASES if b in vals]
        ax.barh(ys, ds, height=h - 0.04, color=COR[nome], label=nome)
    ax.axvline(0, color=INK2, lw=1)
    for x in (-1, 1):
        ax.axvline(x, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(range(len(A.BASES)), [A.NOMES[b] for b in A.BASES])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("diferença de f: grupo − artigo (p.p.); pontilhado = ±1 p.p.")
    ax.set_title("Quanto o f do grupo se afasta da Tabela 3", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=8)
    fig.savefig(saida / "fig_f_diferenca.png")
    plt.close(fig)


def fig_metricas(por_metrica, saida, plt):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for ax, met in zip(axs.flat, METRICAS):
        _dot(ax, por_metrica[met], A.BASES, f"{NOME_MET[met]} na validação (%)")
        ax.set_title(NOME_MET[met], loc="left")
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=4, bbox_to_anchor=(0.02, 1.02), fontsize=8)
    fig.suptitle("Métricas na validação, MST com DA (Tabelas 4 a 7)", x=0.02, y=1.06,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(saida / "fig_metricas_validacao.png")
    plt.close(fig)


def fig_tempo(series, saida, plt):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _dot(ax, series, A.BASES, "tempo por fold (s, escala log)")
    ax.set_xscale("log")
    ax.set_title("Tempo do MST com DA por fold (máquinas e linguagens diferentes)", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=8)
    fig.savefig(saida / "fig_tempo.png")
    plt.close(fig)


def fig_baselines(art, grupo, saida, plt, met="f_treino"):
    fig, axs = plt.subplots(3, 3, figsize=(11, 8.5))
    for ax, b in zip(axs.flat, A.BASES):
        for i, (nome, tab) in enumerate((("Artigo", art), ("Grupo (n = variáveis)", grupo))):
            dy = (i - 0.5) * 0.2
            ys, xs, es = [], [], []
            for j, mt in enumerate(A.METODOS):
                v = tab.get(b, {}).get(mt)
                if v is not None:
                    ys.append(j + dy)
                    xs.append(v[0])
                    es.append(v[1])
            ax.errorbar(xs, ys, xerr=es, fmt="o", ms=4.5, lw=1, capsize=0, color=COR[nome],
                        mec="white", mew=0.8, label=nome)
        ax.set_yticks(range(len(A.METODOS)), A.METODOS)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(A.NOMES[b], loc="left")
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=2, bbox_to_anchor=(0.02, 1.03), fontsize=8)
    fig.suptitle(f"{NOME_MET[met]} (%) do MST e das baselines com DA (eixos por base)",
                 x=0.02, y=1.07, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(saida / f"fig_baselines_{met}.png")
    plt.close(fig)


def fig_lr_svm(art, grupo, saida, plt):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, clf in zip(axs, ("LR", "SVM")):
        _dot(ax, {"Artigo": art[clf], "Grupo (n = variáveis)": grupo.get(clf, {})}, A.BASES,
             "ACC na validação (%)")
        ax.set_title(f"MST com {clf} (Tabela {9 if clf == 'LR' else 11})", loc="left")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=2, bbox_to_anchor=(0.02, 1.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(saida / "fig_acc_lr_svm.png")
    plt.close(fig)


def fig_tamanho(series, saida, plt):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _dot(ax, series, A.BASES, "|S*| (variáveis escolhidas)")
    ax.set_title("Tamanho do subconjunto escolhido pelo MST (o artigo não publica)", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=8)
    fig.savefig(saida / "fig_tamanho_S.png")
    plt.close(fig)


# ------------------------------------------------------------------ testes t
def wtl(df, campanha_mst, campanha_base, met, clf="DA"):
    """Vitórias/empates/derrotas do MST (t pareado bicaudal, 5%) contra cada baseline."""
    mst = por_fold(df, campanha_mst, "MST", clf)
    out = {}
    for bl in A.METODOS[1:]:
        o = por_fold(df, campanha_base, bl, clf)
        w = t = l = 0
        for b in A.BASES:
            x = mst[mst.base == b].set_index("fold")[met] if met in mst else pd.Series(dtype=float)
            y = o[o.base == b].set_index("fold")[met] if met in o else pd.Series(dtype=float)
            j = x.index.intersection(y.index)
            if len(j) < 3:
                continue
            d = x[j] - y[j]
            if np.allclose(d, 0):
                t += 1
                continue
            p = stats.ttest_rel(x[j], y[j]).pvalue
            if p < 0.05:
                w, l = (w + 1, l) if d.mean() > 0 else (w, l + 1)
            else:
                t += 1
        out[bl] = (w, t, l)
    return out


# ------------------------------------------------------------------ relatório
def build(resultados: Path, saida: Path) -> str:
    saida.mkdir(parents=True, exist_ok=True)
    df = carregar(resultados)
    plt = _plt()
    md = ["# Comparação com o artigo (Pacheco et al., 2023)\n",
          "Média ± desvio-padrão entre os 10 folds, como no artigo. Com várias sementes, "
          "cada fold entra com a média das suas sementes. Valores em %; tempo em segundos.\n"]

    pfs = {nome: por_fold(df, *src) for nome, src in FONTES.items()}

    # f e diferença
    sf = {"Artigo": {b: A.F_DA[b]["MST"] for b in A.BASES}}
    sf.update({n: resumo(pf, "f_treino") for n, pf in pfs.items() if not pf.empty})
    fig_f(sf, saida, plt)
    fig_dif(sf, saida, plt)
    md.append("## 1. f do MST com DA (Tabela 3)\n")
    md.append("![f](fig_f_mst.png)\n\n![diferença](fig_f_diferenca.png)\n")
    cab = list(sf)
    md.append("| base | " + " | ".join(cab) + " | " +
              " | ".join(f"dif. {c.split('(')[-1].rstrip(')') if '(' in c else c} (p.p.)"
                         for c in cab[1:]) + " |")
    md.append("|---" * (1 + 2 * len(cab) - 1) + "|")
    for b in A.BASES:
        vals = [sf[c].get(b) for c in cab]
        difs = [f"{v[0] - vals[0][0]:+.2f}" if v else "—" for v in vals[1:]]
        md.append(f"| {A.NOMES[b]} | " + " | ".join(fmt(v) for v in vals) + " | " +
                  " | ".join(difs) + " |")

    # métricas de validação
    por_met = {}
    for met, tab in zip(METRICAS, (A.ACC_DA, A.AUC_DA, A.GMEAN_DA, A.F1_DA)):
        s = {"Artigo": {b: tab[b]["MST"] for b in A.BASES}}
        for n in ("Grupo (n = variáveis)", "Grupo (n = casos)", "Grupo DA_PASCAL"):
            if not pfs[n].empty:
                s[n] = resumo(pfs[n], met)
        por_met[met] = s
    fig_metricas(por_met, saida, plt)
    md.append("\n## 2. Métricas na validação, MST com DA (Tabelas 4 a 7)\n")
    md.append("![métricas](fig_metricas_validacao.png)\n")
    for met in METRICAS:
        s = por_met[met]
        cab = list(s)
        md.append(f"\n**{NOME_MET[met]}**\n")
        md.append("| base | " + " | ".join(cab) + " |")
        md.append("|---" * (len(cab) + 1) + "|")
        for b in A.BASES:
            md.append(f"| {A.NOMES[b]} | " + " | ".join(fmt(s[c].get(b)) for c in cab) + " |")

    # tempo
    st = {"Artigo": {b: A.TEMPO["DA"][b] for b in A.BASES}}
    p1 = por_fold(df, "E2-P1", "MST", "DA")
    if not p1.empty:
        st["Grupo (n = variáveis)"] = resumo(p1, "tempo_s")
    if not pfs["Grupo (n = casos)"].empty:
        st["Grupo (n = casos)"] = resumo(pfs["Grupo (n = casos)"], "tempo_s")
    fig_tempo(st, saida, plt)
    md.append("\n## 3. Tempo do MST com DA por fold (Tabela 2)\n")
    md.append("![tempo](fig_tempo.png)\n")
    md.append("Artigo: Object-Pascal (Delphi 10.3), Intel i9-10920X. Grupo, n = variáveis: "
              "MST do E2-P1 (Python, código rápido da LDA, Windows/AMD Zen 2, 7 processos). "
              "Grupo, n = casos: E1-casos (Python, código rápido, Ryzen 9 7950X com 31 "
              "processos em 16 núcleos). Comparação só qualitativa.\n")
    cab = list(st)
    md.append("| base | " + " | ".join(cab) + " |")
    md.append("|---" * (len(cab) + 1) + "|")
    for b in A.BASES:
        md.append(f"| {A.NOMES[b]} | " + " | ".join(fmt(st[c].get(b)) for c in cab) + " |")

    # baselines
    bl = df[df.campanha == "E4-baselines-DA"]
    md.append("\n## 4. MST × baselines com DA (Tabelas 3 a 8)\n")
    if bl.empty:
        md.append("_(E4-baselines-DA ainda sem execuções)_\n")
    else:
        mst_pf = pfs["Grupo (n = variáveis)"]
        for met, tab in (("f_treino", A.F_DA), ("acc", A.ACC_DA), ("auc", A.AUC_DA),
                         ("gmean", A.GMEAN_DA), ("f1", A.F1_DA)):
            grupo = {}
            for b in A.BASES:
                g = {}
                r = resumo(mst_pf[mst_pf.base == b], met).get(b)
                if r:
                    g["MST"] = r
                for mt in A.METODOS[1:]:
                    r = resumo(por_fold(df, "E4-baselines-DA", mt, "DA"), met).get(b)
                    if r:
                        g[mt] = r
                grupo[b] = g
            if met in ("f_treino", "acc"):
                fig_baselines(tab, grupo, saida, plt, met)
                md.append(f"![baselines {met}](fig_baselines_{met}.png)\n")
            md.append(f"\n**{NOME_MET[met]}** (artigo / grupo)\n")
            md.append("| base | " + " | ".join(A.METODOS) + " |")
            md.append("|---" * (len(A.METODOS) + 1) + "|")
            for b in A.BASES:
                cel = []
                for mt in A.METODOS:
                    a = tab[b][mt]
                    g = grupo[b].get(mt)
                    cel.append(f"{a[0]:.2f} / {g[0]:.2f}" if g else f"{a[0]:.2f} / —")
                md.append(f"| {A.NOMES[b]} | " + " | ".join(cel) + " |")
        md.append("\n**Vitórias/empates/derrotas do MST** (t pareado bicaudal, 5%, nos folds; "
                  "artigo → grupo)\n")
        art_t8 = {"f_treino": ["9/0/0"] * 5, "acc": ["7/2/0", "4/5/0", "7/2/0", "9/0/0", "8/1/0"],
                  "auc": ["7/2/0", "4/5/0", "8/1/0", "8/1/0", "8/1/0"],
                  "gmean": ["6/3/0", "3/6/0", "7/2/0", "7/2/0", "7/2/0"],
                  "f1": ["7/2/0", "3/6/0", "7/2/0", "9/0/0", "8/1/0"]}
        md.append("| medida | " + " | ".join(A.METODOS[1:]) + " |")
        md.append("|---" * 6 + "|")
        for met in ("f_treino", *METRICAS):
            r = wtl(df, "E1-reproducao-DA", "E4-baselines-DA", met)
            cel = [f"{a} → {'/'.join(map(str, r[mt]))}" for a, mt in zip(art_t8[met],
                                                                         A.METODOS[1:])]
            md.append(f"| {NOME_MET[met]} | " + " | ".join(cel) + " |")

    # LR e SVM
    md.append("\n## 5. MST com LR e SVM: ACC na validação (Tabelas 9 e 11)\n")
    lrsvm = {}
    for clf in ("LR", "SVM"):
        lrsvm[clf] = resumo(por_fold(df, "E4-LR-SVM", "MST", clf), "acc")
    art = {c: {b: A.ACC[c][b]["MST"] for b in A.BASES} for c in ("LR", "SVM")}
    fig_lr_svm(art, lrsvm, saida, plt)
    md.append("![LR e SVM](fig_acc_lr_svm.png)\n")
    md.append("| base | LR artigo | LR grupo | SVM artigo | SVM grupo |")
    md.append("|---|---|---|---|---|")
    for b in A.BASES:
        md.append(f"| {A.NOMES[b]} | {fmt(art['LR'][b])} | {fmt(lrsvm['LR'].get(b))} | "
                  f"{fmt(art['SVM'][b])} | {fmt(lrsvm['SVM'].get(b))} |")

    # tamanho de S*
    stam = {n: resumo(pf, "tam") for n, pf in pfs.items() if not pf.empty}
    fig_tamanho(stam, saida, plt)
    md.append("\n## 6. Tamanho de S* (não publicado no artigo)\n")
    md.append("![tamanho](fig_tamanho_S.png)\n")
    cab = list(stam)
    md.append("| base | m | " + " | ".join(cab) + " |")
    md.append("|---" * (len(cab) + 2) + "|")
    for b in A.BASES:
        md.append(f"| {A.NOMES[b]} | {A.TABELA1[b][3]} | " +
                  " | ".join(fmt(stam[c].get(b)) for c in cab) + " |")

    texto = "\n".join(md) + "\n"
    (saida / "comparacao.md").write_text(texto, encoding="utf-8")
    return texto


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resultados", default="results")
    ap.add_argument("--saida", default="results/relatorio/comparacao")
    a = ap.parse_args(argv)
    build(Path(a.resultados), Path(a.saida))
    print(f"comparação gravada em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
