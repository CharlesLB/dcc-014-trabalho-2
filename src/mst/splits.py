"""Divisão em folds estratificados e normalização min-max ajustada no treino."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

SPLIT_SEED = 2023


def make_folds(y, n_splits: int = 10, seed: int = SPLIT_SEED) -> list[dict]:
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        out.append({"fold": k, "train": tr.tolist(), "test": te.tolist()})
    return out


def save_folds(folds: list[dict], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "folds": folds}))


def load_folds(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())["folds"]


def get_or_create_folds(y, path: str | Path, n_splits: int = 10, seed: int = SPLIT_SEED) -> list[dict]:
    """Lê os folds gravados ou cria e grava. Confere tamanho com y."""
    path = Path(path)
    if path.exists():
        folds = load_folds(path)
        n = sum(len(f["test"]) for f in folds)
        if n != len(y):
            raise ValueError(f"{path}: folds cobrem {n} casos, base tem {len(y)}")
        return folds
    folds = make_folds(y, n_splits, seed)
    save_folds(folds, path, {"n_splits": n_splits, "seed": seed, "n": int(len(y))})
    return folds


class MinMax:
    """Min-max ajustado no treino. Colunas constantes viram 0."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        rng = X.max(axis=0) - self.min_
        rng[rng == 0] = 1.0
        self.range_ = rng
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def split_fold(X, y, fold: dict, normalize: bool = True):
    tr = np.asarray(fold["train"], dtype=int)
    te = np.asarray(fold["test"], dtype=int)
    Xtr, Xte = X[tr], X[te]
    if normalize:
        sc = MinMax().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    return Xtr, y[tr], Xte, y[te]
