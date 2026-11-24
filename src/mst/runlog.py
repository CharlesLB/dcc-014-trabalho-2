"""Registro das execuções em JSONL (SPEC §5.5)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REQUIRED_FIELDS = (
    "run_id", "campanha", "variante", "base", "classificador", "fold", "split_seed",
    "seed", "params", "S_star", "S_star_nomes", "tamanho_S_star", "f_treino",
    "rat_treino", "validacao", "tempo_s", "iter_locais_total", "n_buscas_locais",
    "n_evals", "n_cache_hits", "iter_sem_admissivel", "stop_reason", "reinicios",
    "fugas_confirmadas", "ambiente",
)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"não serializável: {type(o)}")


def dumps(rec: dict) -> str:
    return json.dumps(rec, default=_default, ensure_ascii=False, sort_keys=True)


def new_run_id() -> str:
    return uuid.uuid4().hex


def run_key(variante: str, base: str, classificador: str, fold: int, seed: int,
            params: dict) -> str:
    """Chave estável de uma execução (para retomar campanhas sem repetir)."""
    p = json.dumps(params, sort_keys=True, default=_default)
    h = hashlib.sha1(p.encode()).hexdigest()[:10]
    return f"{variante}|{base}|{classificador}|{fold}|{seed}|{h}"


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def environment() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cpu": platform.processor() or platform.machine(),
        "sistema": platform.platform(),
        "git_commit": git_commit(),
    }


def validate_record(rec: dict) -> list[str]:
    """Lista campos obrigatórios ausentes."""
    return [k for k in REQUIRED_FIELDS if k not in rec]


def append(path: str | Path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(dumps(rec) + "\n")


def append_many(path: str | Path, recs: Iterable[dict]) -> None:
    for r in recs:
        append(path, r)


def read(path: str | Path) -> list[dict]:
    return list(iter_records(path))


def iter_records(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_keys(path: str | Path) -> set[str]:
    return {r["run_key"] for r in iter_records(path) if "run_key" in r}


def write_trace(path: str | Path, traces: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in traces or []:
            for step in t.get("passos") or []:
                fh.write(dumps({"reinicio": t["reinicio"], **step}) + "\n")
