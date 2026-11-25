"""Roda várias campanhas numa única fila de processos.

Cada entrada da fila é uma config (com ``--set`` opcionais), como em
``experiments.run``. As execuções de todas as entradas vão para um só
``ProcessPoolExecutor``, então nenhum processo fica parado esperando o fim de uma
campanha. Cada registro continua indo para o ``runs.jsonl`` da sua campanha, e a
retomada pula o que já está gravado, como em ``run_campaign``.

Ordem: menor ``prioridade`` primeiro; dentro da mesma prioridade, maior custo
estimado primeiro (``custo_base``, em segundos por execução; base sem custo vale 1).
A ordem só muda quando cada execução começa, não o que ela calcula.

Arquivo da fila (YAML):

    n_jobs: 31
    custo_base: {qadc: 1000, spectf: 100}
    entradas:
      - config: experiments/configs/E2a_ajuste.yaml
        set: ["sementes=[0, 1]"]
        prioridade: 0

Uso:
    python -m experiments.fila fila.yaml [--listar]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.run import load_config  # noqa: E402
from mst import runlog  # noqa: E402
from mst.runner import _exec_safe, expand, limit_blas_threads, pool_kwargs  # noqa: E402


def _out(spec) -> Path:
    return Path(spec.saida) / spec.campanha / "runs.jsonl"


def plan(fila: dict) -> list:
    """Lista ordenada das execuções que faltam em todas as entradas."""
    custo = fila.get("custo_base", {}) or {}
    todo, vistos = [], set()
    for ordem, e in enumerate(fila["entradas"]):
        cfg = load_config(e["config"], e.get("set", []))
        prio = e.get("prioridade", ordem)
        done = {}
        for s in expand(cfg):
            out = _out(s)
            if out not in done:
                done[out] = runlog.done_keys(out)
            ident = (str(out), s.key)
            if s.key in done[out] or ident in vistos:
                continue
            vistos.add(ident)
            todo.append((prio, -float(custo.get(s.base, 1)), len(todo), s))
    todo.sort(key=lambda t: t[:3])
    return [t[3] for t in todo]


def run_queue(fila: dict, progress: bool = True) -> int:
    specs = plan(fila)
    n_jobs = int(fila.get("n_jobs", 1))
    blas = int(fila.get("blas_threads", 1))
    if progress:
        print(f"[fila] {len(specs)} execuções, {n_jobs} processo(s), BLAS com {blas} thread(s)",
              flush=True)
    t0 = time.perf_counter()

    def _store(i, rec, spec):
        out = _out(spec)
        if "erro" in rec:
            runlog.append(out.with_name("erros.jsonl"), rec)
            if progress:
                print(f"  ERRO {spec.campanha} {rec['run_key']}: {rec['erro']}", flush=True)
            return
        runlog.append(out, rec)
        if progress:
            print(f"  {i}/{len(specs)} [{spec.campanha}] {rec['variante']} {rec['base']} "
                  f"f{rec['fold']} s{rec['seed']} f={rec['f_treino']:.4f} "
                  f"|S|={rec['tamanho_S_star']} {rec['tempo_s']:.1f}s", flush=True)

    if n_jobs <= 1:
        limit_blas_threads(blas)
        for i, s in enumerate(specs, 1):
            _store(i, _exec_safe(s), s)
    else:
        with ProcessPoolExecutor(**pool_kwargs(n_jobs, blas)) as ex:
            futs = {ex.submit(_exec_safe, s): s for s in specs}
            for i, fut in enumerate(as_completed(futs), 1):
                _store(i, fut.result(), futs[fut])
    if progress:
        print(f"[fila] concluído em {time.perf_counter() - t0:.1f}s", flush=True)
    return len(specs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fila")
    ap.add_argument("--listar", action="store_true", help="só lista as execuções, na ordem")
    a = ap.parse_args(argv)
    fila = yaml.safe_load(Path(a.fila).read_text(encoding="utf-8"))
    if a.listar:
        specs = plan(fila)
        for s in specs:
            print(s.campanha, s.key, "max_evals=", s.max_evals, "tempo=", s.time_budget)
        print(f"{len(specs)} execuções")
        return 0
    run_queue(fila)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
