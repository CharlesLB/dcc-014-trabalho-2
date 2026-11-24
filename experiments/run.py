"""Roda uma campanha de experimentos.

Uso:
    python -m experiments.run --config experiments/configs/E1_reproducao.yaml [--n-jobs 4]
    python -m experiments.run --config experiments/configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

# BLAS com 1 thread por processo antes de importar numpy (o paralelismo é entre execuções).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst.runner import expand, run_campaign  # noqa: E402


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-jobs", default=None,
                    help="processos em paralelo (número ou 'auto' = núcleos físicos − 1)")
    ap.add_argument("--sem-retomar", action="store_true", help="não pular execuções já gravadas")
    ap.add_argument("--listar", action="store_true", help="só lista as execuções")
    ap.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                    help="sobrescreve uma chave da config (valor em YAML)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.listar:
        specs = expand(cfg)
        for s in specs:
            print(s.key, "max_evals=", s.max_evals, "tempo=", s.time_budget)
        print(f"{len(specs)} execuções")
        return 0
    nj = a.n_jobs if a.n_jobs in (None, "auto") else int(a.n_jobs)
    run_campaign(cfg, n_jobs=nj, resume=not a.sem_retomar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
