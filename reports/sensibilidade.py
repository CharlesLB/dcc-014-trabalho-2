"""Resumo do E0 (SPEC §3.5): compara scale_base = casos e variaveis.

Para cada base e leitura de n: f no treino, tempo, iterações locais, fração de
iterações sem movimento admissível e quantas execuções pareadas (fold, semente)
cada leitura vence em f. Aplica o critério de decisão do grupo:
troca para "variaveis" se, com "casos", mais de 50% das iterações ficam sem
movimento admissível e o f não é melhor.

Uso:
    python -m reports.sensibilidade results/E0-sensibilidade/runs.jsonl [--saida E0.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from mst.reporting import to_markdown  # noqa: E402

EPS = 1e-12


def frame(recs) -> pd.DataFrame:
    rows = []
    for r in recs:
        if "erro" in r:
            continue
        p = r.get("params") or {}
        rows.append({"base": r["base"], "fold": r["fold"], "seed": r["seed"],
                     "scale_base": p.get("scale_base"), "tenure": p.get("tenure"),
                     "maxiter_ts": p.get("maxiter_ts"), "f": r["f_treino"],
                     "tempo_s": r["tempo_s"], "iters": r["iter_locais_total"],
                     "sem_adm": r["iter_sem_admissivel"], "buscas": r["n_buscas_locais"],
                     "n_evals": r["n_evals"], "tam": r["tamanho_S_star"]})
    return pd.DataFrame(rows)


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (b, sb), g in df.groupby(["base", "scale_base"], sort=True):
        out.append({"base": b, "scale_base": sb, "n": len(g),
                    "tenure": int(g["tenure"].median()), "maxiterTS": int(g["maxiter_ts"].median()),
                    "f (%)": f"{100 * g.f.mean():.2f} ± {100 * g.f.std(ddof=1):.2f}" if len(g) > 1
                    else f"{100 * g.f.mean():.2f}",
                    "tempo (s)": round(g.tempo_s.mean(), 1),
                    "iter. locais": round(g.iters.mean()),
                    "buscas locais": round(g.buscas.mean(), 1),
                    "avaliações": round(g.n_evals.mean()),
                    "|S*|": round(g.tam.mean(), 1),
                    "sem admissível (%)": round(100 * g.sem_adm.sum() / g.iters.sum(), 1)})
    return pd.DataFrame(out)


def pareado(df: pd.DataFrame) -> pd.DataFrame:
    """Vitórias em f por par (fold, semente): casos vs variaveis."""
    out = []
    for b, g in df.groupby("base", sort=True):
        w = g.pivot_table(index=["fold", "seed"], columns="scale_base", values="f")
        if not {"casos", "variaveis"} <= set(w.columns):
            continue
        w = w.dropna()
        d = w["casos"] - w["variaveis"]
        out.append({"base": b, "pares": len(w),
                    "casos melhor": int((d > EPS).sum()), "empate": int((d.abs() <= EPS).sum()),
                    "variaveis melhor": int((d < -EPS).sum()),
                    "dif. média f (p.p.)": round(100 * d.mean(), 3)})
    return pd.DataFrame(out)


def decisao(df: pd.DataFrame) -> tuple[str, str]:
    """Critério: >50% sem admissível com 'casos' E f de 'casos' não melhor -> 'variaveis'."""
    c = df[df.scale_base == "casos"]
    v = df[df.scale_base == "variaveis"]
    if c.empty or v.empty:
        return "casos", "faltam execuções de uma das leituras; mantém casos"
    frac = c.sem_adm.sum() / c.iters.sum()
    fc, fv = c.f.mean(), v.f.mean()
    trava = frac > 0.5
    nao_melhor = fc <= fv + EPS
    escolha = "variaveis" if (trava and nao_melhor) else "casos"
    motivo = (f"com casos, {100 * frac:.1f}% das iterações sem movimento admissível "
              f"({'>' if trava else '≤'} 50%); f médio casos = {100 * fc:.3f}% vs "
              f"variaveis = {100 * fv:.3f}% ({'não melhor' if nao_melhor else 'melhor'})")
    return escolha, motivo


def build(recs) -> str:
    df = frame(recs)
    if df.empty:
        return "_(nenhuma execução)_\n"
    esc, mot = decisao(df)
    parts = ["## E0: resumo por base e leitura de n\n", to_markdown(resumo(df)),
             "\n## E0: comparação pareada de f (fold, semente)\n", to_markdown(pareado(df)),
             f"\n## Decisão\n\n`scale_base = {esc}`: {mot}.\n"]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    md = build(r for p in a.runs for r in runlog.read(p))
    print(md)
    if a.saida:
        Path(a.saida).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
