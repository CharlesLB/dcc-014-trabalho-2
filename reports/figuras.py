"""Figuras: evolução de f por reinício (como a Figura 1 do artigo).

Uso:
    python -m reports.figuras results/E1-reproducao-DA/runs.jsonl --base wpbc --fold 0 --seed 0 --saida fig_wpbc.png
Sem matplotlib, grava só o CSV da curva.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst import runlog  # noqa: E402
from mst.reporting import restart_curve  # noqa: E402


def find(recs, variante, base, fold, seed, classificador):
    for r in recs:
        if (r.get("variante") == variante and r.get("base") == base and r.get("fold") == fold
                and r.get("seed") == seed and r.get("classificador") == classificador):
            return r
    return None


def plot(rec: dict, out: str | Path) -> Path:
    curve = restart_curve(rec)
    out = Path(out)
    curve.to_csv(out.with_suffix(".csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return out.with_suffix(".csv")
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    ax.plot(curve.reinicio, curve.f_local, color="tab:blue", label="f da busca local")
    ax.plot(curve.reinicio, curve.f_global, color="tab:red", label="melhor f")
    ax.set_xlabel("reinício")
    ax.set_ylabel("f no treino")
    ax.set_title(f"{rec['variante']} — {rec['base']} (fold {rec['fold']}, semente {rec['seed']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs")
    ap.add_argument("--variante", default="MST")
    ap.add_argument("--base", default="wpbc")
    ap.add_argument("--classificador", default="DA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default="figura_reinicios.png")
    a = ap.parse_args(argv)
    rec = find(runlog.read(a.runs), a.variante, a.base, a.fold, a.seed, a.classificador)
    if rec is None:
        print("execução não encontrada", file=sys.stderr)
        return 1
    print(plot(rec, a.saida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
