"""Tabelas de resumo, comparação com o artigo, estabilidade e testes pareados.

Uso:
    python -m reports.tabelas results/E1-reproducao-DA/runs.jsonl --saida results/E1-reproducao-DA/tabelas.md
    python -m reports.tabelas results/E2-P2/runs.jsonl --comparar MST_FREQ MST --comparar MST_FREQ GULOSO
    python -m reports.tabelas results/E2a-ajuste/runs.jsonl --grade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from mst.reporting import (choose_grid, compare_with_paper, paired_tests,  # noqa: E402
                           stability_table, summary_table, to_frame, to_markdown)


def build(paths, comparar=(), grade=False, classificador="DA") -> str:
    df = to_frame(r for p in paths for r in runlog.read(p))
    if df.empty:
        return "_(nenhuma execução)_\n"
    parts = ["## Resumo por variante e base\n", to_markdown(summary_table(df))]
    if "MST" in set(df.variante):
        parts += ["\n## MST vs artigo (Tabela 3, DA)\n", to_markdown(compare_with_paper(df))]
    parts += ["\n## Estabilidade entre folds (Jaccard médio)\n", to_markdown(stability_table(df).round(4))]
    for a, b in comparar:
        t = paired_tests(df, a, b, classificador=classificador)
        parts += [f"\n## Wilcoxon pareado {a} vs {b} (Holm)\n", to_markdown(t.round(4))]
    if grade:
        parts += ["\n## Grade λ × gatilho\n", to_markdown(choose_grid(df).round(5))]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--comparar", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--classificador", default="DA")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    md = build(a.runs, a.comparar, a.grade, a.classificador)
    print(md)
    if a.saida:
        Path(a.saida).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
