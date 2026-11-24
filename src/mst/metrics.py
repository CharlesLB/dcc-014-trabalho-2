"""Métricas do artigo (validação). Positivo = rótulo 1.

AUC aqui é a fórmula do artigo, de um único ponto de corte:
    AUC = (1 + TP/(TP+FN) − FP/(TN+FP)) / 2
que é igual à acurácia balanceada. A ROC-AUC usual fica em ``roc_auc``.
"""

from __future__ import annotations

import math

import numpy as np


def confusion(y_true, y_pred, positive=1) -> tuple[int, int, int, int]:
    """Retorna (TP, TN, FP, FN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred com formas diferentes")
    pt = y_true == positive
    pp = y_pred == positive
    tp = int(np.sum(pt & pp))
    tn = int(np.sum(~pt & ~pp))
    fp = int(np.sum(~pt & pp))
    fn = int(np.sum(pt & ~pp))
    return tp, tn, fp, fn


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def accuracy(tp, tn, fp, fn) -> float:
    return _div(tp + tn, tp + tn + fp + fn)


def tpr(tp, tn, fp, fn) -> float:
    return _div(tp, tp + fn)


def tnr(tp, tn, fp, fn) -> float:
    return _div(tn, tn + fp)


def auc_paper(tp, tn, fp, fn) -> float:
    return (1.0 + tpr(tp, tn, fp, fn) - _div(fp, tn + fp)) / 2.0


def gmean(tp, tn, fp, fn) -> float:
    return math.sqrt(tpr(tp, tn, fp, fn) * tnr(tp, tn, fp, fn))


def f1(tp, tn, fp, fn) -> float:
    return _div(2 * tp, 2 * tp + fp + fn)


def all_metrics(y_true, y_pred, positive=1) -> dict[str, float]:
    c = confusion(y_true, y_pred, positive)
    return {
        "acc": accuracy(*c),
        "auc": auc_paper(*c),
        "gmean": gmean(*c),
        "f1": f1(*c),
        "tp": c[0],
        "tn": c[1],
        "fp": c[2],
        "fn": c[3],
    }


def roc_auc(y_true, scores, positive=1) -> float:
    """ROC-AUC usual (Mann-Whitney), com empates valendo 1/2."""
    y_true = np.asarray(y_true) == positive
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true]
    neg = scores[~y_true]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))
