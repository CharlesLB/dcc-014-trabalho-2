"""Prepara as nove bases: lê data/raw/<base>/, padroniza e grava
data/processed/<base>.csv, cria data/splits/<base>.json e confere as contagens
da Tabela 1 do artigo.

Uso:
    python -m data.prepare                 # todas as bases encontradas
    python -m data.prepare --bases wdbc wpbc
    python -m data.prepare --estrito       # sai com erro se alguma contagem divergir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mst.datasets import PAPER_ORDER, REGISTRY, check_counts, load_raw, save_processed  # noqa: E402
from mst.splits import SPLIT_SEED, get_or_create_folds  # noqa: E402

ROOT = Path(__file__).resolve().parent


def prepare(bases, raw: Path, processed: Path, splits: Path, n_splits: int = 10,
            seed: int = SPLIT_SEED, verbose: bool = True):
    resultados = {}
    for key in bases:
        spec = REGISTRY[key]
        d = raw / key
        faltando = [f for f in spec.arquivos if not (d / f).exists()]
        if faltando and key != "mesothelioma":
            resultados[key] = ("faltando", faltando)
            if verbose:
                print(f"[{key}] arquivos ausentes em {d}: {faltando}")
            continue
        try:
            df = load_raw(key, raw)
        except FileNotFoundError as e:
            resultados[key] = ("faltando", [str(e)])
            if verbose:
                print(f"[{key}] {e}")
            continue
        save_processed(df, processed / f"{key}.csv")
        get_or_create_folds(df["y"].to_numpy(), splits / f"{key}.json", n_splits, seed)
        chk = check_counts(key, df)
        resultados[key] = ("ok" if chk.ok else "divergente", chk)
        if verbose:
            status = "OK" if chk.ok else "DIVERGE: " + "; ".join(chk.problemas)
            print(f"[{key}] {chk.n_casos} casos, {chk.contagens[0]}/{chk.contagens[1]}, "
                  f"m={chk.m} -> {status}")
    return resultados


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bases", nargs="*", default=PAPER_ORDER)
    ap.add_argument("--raw", default=str(ROOT / "raw"))
    ap.add_argument("--processed", default=str(ROOT / "processed"))
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--estrito", action="store_true")
    a = ap.parse_args(argv)
    res = prepare(a.bases, Path(a.raw), Path(a.processed), Path(a.splits))
    if a.estrito and any(s != "ok" for s, _ in res.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
