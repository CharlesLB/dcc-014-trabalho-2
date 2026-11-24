"""Implementação dos baselines populacionais binários."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..multistart import MSTResult
from ..objective import BaseObjective, BudgetExhausted


class _Stop(Exception):
    pass


@dataclass
class StopRule:
    time_budget: float | None = None
    max_iter: int | None = None
    # gerações seguidas sem nenhuma avaliação nova (tudo em cache) antes de parar
    patience: int = 200

    def __post_init__(self):
        if self.time_budget is None and self.max_iter is None:
            raise ValueError("defina time_budget, max_iter ou max_evals")


class _Ctx:
    """Avalia indivíduos, guarda o melhor e aplica as paradas."""

    def __init__(self, objective: BaseObjective, rule: StopRule, rng: np.random.Generator):
        self.obj = objective
        self.rule = rule
        self.rng = rng
        self.t0 = time.perf_counter()
        self.best_S: frozenset = frozenset()
        self.best_f = float("-inf")
        self.iters = 0
        self._last_evals = objective.n_evals
        self._idle = 0
        self.idle_stop = False

    def repair(self, bits: np.ndarray) -> np.ndarray:
        if not bits.any():
            bits = bits.copy()
            bits[int(self.rng.integers(bits.size))] = True
        return bits

    def fitness(self, bits: np.ndarray) -> float:
        if self.rule.time_budget is not None and time.perf_counter() - self.t0 >= self.rule.time_budget:
            raise _Stop()
        S = frozenset(np.flatnonzero(bits).tolist())
        try:
            f = self.obj(S)
        except BudgetExhausted:
            raise _Stop()
        if f > self.best_f:
            self.best_f, self.best_S = f, S
        return f

    def next_iter(self) -> None:
        self.iters += 1
        if self.obj.n_evals == self._last_evals:
            self._idle += 1
            if self._idle >= self.rule.patience:
                self.idle_stop = True
                raise _Stop()
        else:
            self._idle = 0
            self._last_evals = self.obj.n_evals
        if self.rule.max_iter is not None and self.iters >= self.rule.max_iter:
            raise _Stop()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _levy(rng, size, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size)
    v = rng.normal(0, 1, size)
    return 0.01 * u / np.abs(v) ** (1 / beta)


# ----------------------------------------------------------------------------- GA
def ga(ctx: _Ctx, m: int, pop: int = 10, pc: float = 0.8, pm: float | None = None,
       tour: int = 2, **_):
    rng = ctx.rng
    pm = 1.0 / m if pm is None else pm
    P = np.array([ctx.repair(rng.random(m) < 0.5) for _ in range(pop)])
    F = np.array([ctx.fitness(x) for x in P])
    while True:
        ctx.next_iter()
        elite = P[np.argmax(F)].copy()
        children = [elite]
        while len(children) < pop:
            par = []
            for _ in range(2):
                idx = rng.integers(pop, size=tour)
                par.append(P[idx[np.argmax(F[idx])]])
            a, b = par[0].copy(), par[1].copy()
            if rng.random() < pc:
                cut = int(rng.integers(1, m)) if m > 1 else 0
                a[cut:], b[cut:] = par[1][cut:], par[0][cut:]
            for c in (a, b):
                flip = rng.random(m) < pm
                c[flip] = ~c[flip]
                children.append(ctx.repair(c))
        P = np.array(children[:pop])
        F = np.array([ctx.fitness(x) for x in P])


# ---------------------------------------------------------------------------- PSO
def pso(ctx: _Ctx, m: int, pop: int = 10, w: float = 0.9, c1: float = 2.0, c2: float = 2.0,
        vmax: float = 6.0, **_):
    rng = ctx.rng
    X = np.array([ctx.repair(rng.random(m) < 0.5) for _ in range(pop)])
    V = rng.uniform(-1, 1, (pop, m))
    F = np.array([ctx.fitness(x) for x in X])
    Pb, Fb = X.copy(), F.copy()
    while True:
        ctx.next_iter()
        g = Pb[np.argmax(Fb)]
        r1, r2 = rng.random((pop, m)), rng.random((pop, m))
        V = w * V + c1 * r1 * (Pb.astype(float) - X) + c2 * r2 * (g.astype(float) - X)
        V = np.clip(V, -vmax, vmax)
        X = rng.random((pop, m)) < _sigmoid(V)
        X = np.array([ctx.repair(x) for x in X])
        for i in range(pop):
            F[i] = ctx.fitness(X[i])
            if F[i] > Fb[i]:
                Pb[i], Fb[i] = X[i].copy(), F[i]


# ---------------------------------------------------------------------- contínuos
def _binarize(ctx, pos):
    return ctx.repair(pos > 0.5)


def gwo(ctx: _Ctx, m: int, pop: int = 10, max_t: int = 100, **_):
    rng = ctx.rng
    X = rng.random((pop, m))
    F = np.array([ctx.fitness(_binarize(ctx, x)) for x in X])
    t = 0
    while True:
        ctx.next_iter()
        order = np.argsort(-F)
        alpha, beta, delta = X[order[0]], X[order[1 % pop]], X[order[2 % pop]]
        a = 2 - 2 * min(t, max_t) / max_t
        for i in range(pop):
            new = np.zeros(m)
            for leader in (alpha, beta, delta):
                A = 2 * a * rng.random(m) - a
                C = 2 * rng.random(m)
                D = np.abs(C * leader - X[i])
                new += leader - A * D
            X[i] = np.clip(new / 3, 0, 1)
            F[i] = ctx.fitness(_binarize(ctx, X[i]))
        t += 1


def woa(ctx: _Ctx, m: int, pop: int = 10, max_t: int = 100, b: float = 1.0, **_):
    rng = ctx.rng
    X = rng.random((pop, m))
    F = np.array([ctx.fitness(_binarize(ctx, x)) for x in X])
    t = 0
    while True:
        ctx.next_iter()
        best = X[np.argmax(F)].copy()
        a = 2 - 2 * min(t, max_t) / max_t
        for i in range(pop):
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            p = rng.random()
            l = rng.uniform(-1, 1)
            if p < 0.5:
                ref = best if abs(A) < 1 else X[int(rng.integers(pop))]
                X[i] = ref - A * np.abs(C * ref - X[i])
            else:
                D = np.abs(best - X[i])
                X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            X[i] = np.clip(X[i], 0, 1)
            F[i] = ctx.fitness(_binarize(ctx, X[i]))
        t += 1


def fpa(ctx: _Ctx, m: int, pop: int = 10, p: float = 0.8, **_):
    rng = ctx.rng
    X = rng.random((pop, m))
    F = np.array([ctx.fitness(_binarize(ctx, x)) for x in X])
    while True:
        ctx.next_iter()
        g = X[np.argmax(F)].copy()
        for i in range(pop):
            if rng.random() < p:
                new = X[i] + _levy(rng, m) * (g - X[i])
            else:
                j, k = rng.integers(pop, size=2)
                new = X[i] + rng.random() * (X[j] - X[k])
            new = np.clip(new, 0, 1)
            fn = ctx.fitness(_binarize(ctx, new))
            if fn > F[i]:
                X[i], F[i] = new, fn


BASELINES = {"GA": ga, "PSO": pso, "GWO": gwo, "WOA": woa, "FPA": fpa}


def run_baseline(name: str, objective: BaseObjective, seed: int = 0,
                 time_budget: float | None = None, max_iter: int | None = None,
                 patience: int = 200, **params) -> MSTResult:
    """Executa um baseline até o tempo, o número de gerações ou o orçamento de avaliações."""
    name = name.upper()
    if name not in BASELINES:
        raise ValueError(f"baseline desconhecido: {name}; use {sorted(BASELINES)}")
    if time_budget is None and max_iter is None and objective.max_evals is None:
        raise ValueError("defina time_budget, max_iter ou max_evals no objetivo")
    rule = StopRule(time_budget=time_budget,
                    max_iter=max_iter if (max_iter is not None or time_budget is not None)
                    else 10 ** 12, patience=patience)
    rng = np.random.default_rng(seed)
    ctx = _Ctx(objective, rule, rng)
    ev0, hit0 = objective.n_evals, objective.n_cache_hits
    try:
        BASELINES[name](ctx, objective.m, **params)
    except _Stop:
        pass
    reason = "orcamento"
    if ctx.idle_stop:
        reason = "sem_avaliacoes_novas"
    elif time_budget is not None and time.perf_counter() - ctx.t0 >= time_budget:
        reason = "tempo"
    elif max_iter is not None and ctx.iters >= max_iter:
        reason = "max_iter"
    return MSTResult(
        S_best=ctx.best_S, f_best=ctx.best_f, n_local_searches=0,
        total_local_iters=ctx.iters, n_evals=objective.n_evals - ev0,
        n_cache_hits=objective.n_cache_hits - hit0,
        time_s=time.perf_counter() - ctx.t0, stop_reason=reason,
        no_admissible=0, local_maxima=0, n_escapes=0, restarts=[],
    )
