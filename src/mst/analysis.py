"""Diversidade, estabilidade e testes estatísticos (SPEC §6.2)."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_distance(a, b) -> float:
    return 1.0 - jaccard(a, b)


def mean_pairwise(sets: Sequence[Iterable[int]], fn) -> float:
    sets = [set(s) for s in sets]
    if len(sets) < 2:
        return float("nan")
    vals = [fn(a, b) for a, b in combinations(sets, 2)]
    return float(np.mean(vals))


def restart_diversity(local_sets: Sequence[Iterable[int]]) -> float:
    """Média da distância de Jaccard entre os S_local dos reinícios."""
    return mean_pairwise(local_sets, jaccard_distance)


def stability(fold_sets: Sequence[Iterable[int]]) -> float:
    """Média da similaridade de Jaccard entre os S* dos folds."""
    return mean_pairwise(fold_sets, jaccard)


def kuncheva(fold_sets: Sequence[Iterable[int]], m: int) -> float:
    """Índice de consistência de Kuncheva (exige subconjuntos de mesmo tamanho k;
    com tamanhos diferentes usa k = média dos dois)."""
    sets = [set(s) for s in fold_sets]
    if len(sets) < 2:
        return float("nan")
    vals = []
    for a, b in combinations(sets, 2):
        k = (len(a) + len(b)) / 2
        r = len(a & b)
        den = k - k * k / m
        vals.append((r * m - k * k) / (m * den) if den > 0 else float("nan"))
    return float(np.nanmean(vals))


def selection_frequency(sets: Sequence[Iterable[int]], m: int) -> np.ndarray:
    f = np.zeros(m)
    for s in sets:
        for j in s:
            f[j] += 1
    return f / max(len(sets), 1)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Correção de Holm-Bonferroni (p ajustados, monotônicos, ≤ 1). NaN é preservado."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    k = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (k - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon pareado bicaudal. Retorna p, mediana da diferença (a − b) e vitórias."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("amostras pareadas com tamanhos diferentes")
    d = a - b
    res = {"n": int(d.size), "mediana_dif": float(np.median(d)) if d.size else float("nan"),
           "media_dif": float(np.mean(d)) if d.size else float("nan"),
           "vitorias": int(np.sum(d > 0)), "empates": int(np.sum(d == 0)),
           "derrotas": int(np.sum(d < 0))}
    if d.size == 0 or np.all(d == 0):
        res["p"] = 1.0
        return res
    res["p"] = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    return res


def paired_ttest(a, b) -> dict:
    """Teste t pareado bicaudal (o usado no artigo)."""
    from scipy.stats import ttest_rel

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, 0):
        return {"p": 1.0, "media_dif": 0.0, "n": int(d.size)}
    r = ttest_rel(a, b)
    return {"p": float(r.pvalue), "media_dif": float(d.mean()), "n": int(d.size)}


def wtl(a, b, alpha: float = 0.05, test=paired_ttest) -> str:
    """'+' se a é significativamente maior, '−' se menor, '=' sem diferença (notação do artigo)."""
    r = test(a, b)
    if r["p"] >= alpha:
        return "="
    return "+" if r.get("media_dif", r.get("mediana_dif", 0)) > 0 else "-"
