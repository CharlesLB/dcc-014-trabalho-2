"""Gera a Tabela 4 do relatório (Markdown e LaTeX) a partir de um runs.jsonl.

Uso:
    python -m reports.tabela4 results/E2-P1/runs.jsonl --base wpbc --fold 0 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from mst.reporting import table4, to_frame, to_latex, to_markdown  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default="wpbc")
    ap.add_argument("--classificador", default="DA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variantes", nargs="*", default=["MST", "MST_FREQ"])
    ap.add_argument("--saida", default=None, help="prefixo dos arquivos .md e .tex")
    a = ap.parse_args(argv)
    recs = [r for p in a.runs for r in runlog.read(p)]
    df = to_frame(recs)
    t = table4(df, a.variantes, a.base, a.classificador, a.fold, a.seed)
    md, tex = to_markdown(t), to_latex(t)
    print(md)
    if a.saida:
        Path(a.saida + ".md").write_text(md, encoding="utf-8")
        Path(a.saida + ".tex").write_text(tex, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
