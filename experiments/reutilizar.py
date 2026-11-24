"""Reaproveita execuções de uma campanha anterior em outra campanha.

Copia um registro de ``--de`` para a campanha da ``--config`` quando a execução
pedida é a mesma: mesma variante, base, classificador, fold e semente, mesmo
orçamento (``max_evals``, ``time_budget``), mesmo ``trace`` e os mesmos
parâmetros *resolvidos* (tenure, maxiterTS, etc.). Como o MST é determinístico
dada a semente, rodar de novo daria o mesmo ``S*`` e o mesmo ``f``.

O registro copiado ganha a chave da campanha nova (para a retomada pulá-lo) e o
campo ``reaproveitado_de`` com a campanha e o ``run_id`` de origem. O tempo
(``tempo_s``) é o medido na campanha de origem.

Uso:
    python -m experiments.reutilizar --de results/E1-reproducao-DA/runs.jsonl \\
        --config experiments/configs/E2_P1.yaml [--set chave=valor] [--variantes MST]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.run import load_config  # noqa: E402
from mst import runlog  # noqa: E402
from mst.runner import MST_VARIANTS, build_mst_config, expand  # noqa: E402

IGNORAR = ("max_evals", "time_budget")


def _comparavel(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in IGNORAR}


def reuse(config: dict, source: str | Path, variantes=None) -> tuple[int, int]:
    """Copia execuções compatíveis. Devolve (copiadas, sem correspondência)."""
    specs = [s for s in expand(config) if s.variante in MST_VARIANTS
             and (variantes is None or s.variante in variantes)]
    out = Path(config.get("saida", "results")) / config["campanha"] / "runs.jsonl"
    done = runlog.done_keys(out)
    fonte = {}
    for r in runlog.iter_records(source):
        if "erro" in r:
            continue
        fonte[(r["variante"], r["base"], r["classificador"], int(r["fold"]), int(r["seed"]))] = r
    copiadas = faltando = 0
    for s in specs:
        if s.key in done:
            continue
        r = fonte.get((s.variante, s.base, s.classificador, s.fold, s.seed))
        if r is None:
            faltando += 1
            continue
        _, resolved = build_mst_config(s.variante, s.params, int(r["n_treino"]), int(r["m"]),
                                       s.max_evals is not None, s.trace)
        rp = r.get("params") or {}
        ok = (_comparavel(resolved) == _comparavel(rp)
              and rp.get("max_evals") == s.max_evals and rp.get("time_budget") == s.time_budget
              and bool(r.get("trace_path")) == s.trace and int(r["split_seed"]) == s.split_seed)
        if not ok:
            faltando += 1
            continue
        novo = dict(r)
        novo["run_key"] = s.key
        novo["campanha"] = config["campanha"]
        novo["reaproveitado_de"] = {"campanha": r["campanha"], "run_id": r["run_id"]}
        novo["run_id"] = runlog.new_run_id()
        runlog.append(out, novo)
        done.add(s.key)
        copiadas += 1
    return copiadas, faltando


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--de", required=True, help="runs.jsonl de origem")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR")
    ap.add_argument("--variantes", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    c, f = reuse(cfg, a.de, a.variantes)
    print(f"[{cfg['campanha']}] {c} execuções reaproveitadas de {a.de}; {f} sem correspondência")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
