"""Resume as fugas de máximo local (E5) a partir de runs.jsonl e dos traces.

Uso:
    python -m reports.fugas results/E5-fuga/runs.jsonl [--exemplo]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402


def resumo(recs):
    linhas = ["| variante | base | fold | máximos locais | fugas confirmadas |", "|---|---|---|---|---|"]
    for r in recs:
        if "erro" in r:
            continue
        linhas.append(f"| {r['variante']} | {r['base']} | {r['fold']} | {r.get('maximos_locais')} | "
                      f"{r.get('fugas_confirmadas')} |")
    return "\n".join(linhas) + "\n"


def exemplo(recs, janela: int = 3) -> str:
    """Primeira fuga encontrada, com os passos do trace ao redor."""
    for r in recs:
        for e in r.get("fugas") or []:
            tp = r.get("trace_path")
            out = [f"Base {r['base']}, fold {r['fold']}, reinício {e['reinicio']}: máximo local "
                   f"f = {e['f_max_local']:.4f} na iteração {e['it_max_local']}, superado na "
                   f"iteração {e['it_melhora']} com f = {e['f_nova']:.4f}."]
            if tp and Path(tp).exists():
                passos = [s for s in runlog.read(tp) if s["reinicio"] == e["reinicio"]
                          and e["it_max_local"] - janela <= s["it"] <= e["it_melhora"]]
                for s in passos:
                    out.append(f"  it {s['it']}: mov {s['mov']} f {s['f_antes']:.4f} -> "
                               f"{s['f_depois']:.4f} (f* {s['f_star']:.4f}) tabu={s['tabu']} "
                               f"asp={s['aspiracao']} max_local={s['max_local']}")
            return "\n".join(out) + "\n"
    return "Nenhuma fuga confirmada nas execuções.\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs")
    ap.add_argument("--exemplo", action="store_true")
    a = ap.parse_args(argv)
    recs = runlog.read(a.runs)
    print(resumo(recs))
    if a.exemplo:
        print(exemplo(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
