"""Laço externo MultiStartTabu (Pseudocódigo 1) e a extensão MST+FREQ."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .constructive import construct
from .objective import BaseObjective
from .tabu import tabu_search

SCALE_BASES = ("casos", "variaveis")


def resolve_params(n_train: int, m: int, scale_base: str = "casos",
                   tenure_factor: float = 0.5, maxiter_factor: int = 10) -> dict:
    """tenure = max(1, floor(fator·n_base)), maxiterTS = 10·n_base.

    scale_base = "casos" usa n = casos de treino (leitura literal do artigo);
    "variaveis" usa m (convenção do código Pascal). Ver SPEC §3.5.
    """
    if scale_base not in SCALE_BASES:
        raise ValueError(f"scale_base deve ser um de {SCALE_BASES}")
    nb = n_train if scale_base == "casos" else m
    return {
        "tenure": max(1, int(math.floor(tenure_factor * nb))),
        "maxiter_ts": int(maxiter_factor * nb),
        "scale_base": scale_base,
        "n_base": int(nb),
    }


@dataclass
class MSTConfig:
    tenure: int
    maxiter_ts: int
    alpha: float = 0.99
    maxiter_ms: int = 10
    outer_stop_strict: bool = False
    eps: float = 1e-12
    # Extensão MST+FREQ
    use_freq: bool = False
    lam: float = 0.01
    trigger: int = 3
    # Orçamento / controle
    stop_on_stagnation: bool = True
    max_restarts: int | None = None
    # No modo orçamento, para se tantos reinícios seguidos não gerarem avaliação nova
    # (espaço pequeno já todo em cache); evita laço infinito.
    budget_patience: int = 10
    trace: bool = False
    detect_local_max: bool = True
    ts_hard_limit: int | None = None

    def validate(self) -> None:
        if self.tenure < 0:
            raise ValueError("tenure < 0")
        if self.maxiter_ts < 0:
            raise ValueError("maxiter_ts < 0")
        if self.maxiter_ms < 1:
            raise ValueError("maxiter_ms deve ser ≥ 1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha fora de [0, 1]")
        if self.use_freq and self.trigger < 1:
            raise ValueError("trigger deve ser ≥ 1")
        if self.lam < 0:
            raise ValueError("lam < 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MSTResult:
    S_best: frozenset
    f_best: float
    n_local_searches: int
    total_local_iters: int
    n_evals: int
    n_cache_hits: int
    time_s: float
    stop_reason: str
    no_admissible: int
    local_maxima: int
    n_escapes: int
    restarts: list = field(default_factory=list)
    freq_counts: list | None = None
    traces: list | None = None
    escapes: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d["S_best"] = sorted(self.S_best)
        d.pop("traces", None)
        return d


def _stagnated(it: int, itbest: int, cfg: MSTConfig) -> bool:
    gap = it - itbest
    return gap > cfg.maxiter_ms if cfg.outer_stop_strict else gap >= cfg.maxiter_ms


def run_mst(objective: BaseObjective, cfg: MSTConfig, seed: int = 0,
            rng: np.random.Generator | None = None) -> MSTResult:
    """Executa o MST (ou MST+FREQ se cfg.use_freq) e devolve o melhor subconjunto.

    Paradas:
      - estagnação: it − itbest ≥ maxiterMS (ou > com outer_stop_strict);
      - orçamento: o objetivo levanta BudgetExhausted (max_evals);
      - max_restarts: teto de reinícios (opcional).
    """
    cfg.validate()
    if not cfg.stop_on_stagnation and objective.max_evals is None and cfg.max_restarts is None:
        raise ValueError("sem estagnação, defina max_evals no objetivo ou max_restarts")
    rng = rng if rng is not None else np.random.default_rng(seed)
    m = objective.m
    ev0, hit0 = objective.n_evals, objective.n_cache_hits
    t0 = time.perf_counter()

    it = 0
    itbest = 0
    f_best = float("-inf")
    S_best: frozenset = frozenset()
    total_iters = 0
    no_adm = 0
    n_lm = 0
    escapes: list = []
    restarts: list = []
    traces: list | None = [] if cfg.trace else None
    counts = np.zeros(m, dtype=np.int64)
    R = 0
    stall = 0
    diversify_next = False
    stop_reason = "estagnacao"
    no_new_evals = 0

    while True:
        it += 1
        ev_before = objective.n_evals
        penalty = None
        diversified = False
        if cfg.use_freq and diversify_next and R > 0:
            penalty = cfg.lam * counts / R
            diversified = True
        diversify_next = False

        cons = construct(objective, cfg.alpha, rng, eps=cfg.eps, penalty=penalty)
        if cons.budget_hit:
            if cons.S and cons.f > f_best:
                S_best, f_best = cons.S, cons.f
            stop_reason = "orcamento"
            break

        ts = tabu_search(objective, cons.S, cfg.tenure, cfg.maxiter_ts, eps=cfg.eps,
                         trace=cfg.trace, detect_local_max=cfg.detect_local_max,
                         hard_limit=cfg.ts_hard_limit)
        total_iters += ts.iterations
        no_adm += ts.no_admissible
        n_lm += ts.local_maxima
        for e in ts.escapes:
            escapes.append({**e, "reinicio": it})
        if traces is not None:
            traces.append({"reinicio": it, "passos": ts.trace})

        improved = False
        if not math.isnan(ts.f_best) and ts.f_best > f_best + cfg.eps:
            S_best, f_best, itbest = ts.S_best, ts.f_best, it
            improved = True

        if not ts.budget_hit:
            R += 1
            for j in ts.S_best:
                counts[j] += 1
        restarts.append({
            "r": it,
            "f_construido": cons.f,
            "tamanho_construido": len(cons.S),
            "f_local": ts.f_best,
            "S_local": sorted(ts.S_best),
            "iters": ts.iterations,
            "sem_admissivel": ts.no_admissible,
            "diversificado": diversified,
            "melhorou_global": improved,
            "f_global": f_best,
        })
        if ts.budget_hit:
            stop_reason = "orcamento"
            break

        stall = 0 if improved else stall + 1
        if cfg.use_freq and stall > 0 and stall % cfg.trigger == 0:
            diversify_next = True

        no_new_evals = no_new_evals + 1 if objective.n_evals == ev_before else 0
        if not cfg.stop_on_stagnation and no_new_evals >= cfg.budget_patience:
            stop_reason = "sem_avaliacoes_novas"
            break
        if cfg.max_restarts is not None and it >= cfg.max_restarts:
            stop_reason = "max_reinicios"
            break
        if cfg.stop_on_stagnation and _stagnated(it, itbest, cfg):
            stop_reason = "estagnacao"
            break

    return MSTResult(
        S_best=S_best,
        f_best=f_best,
        n_local_searches=len(restarts),
        total_local_iters=total_iters,
        n_evals=objective.n_evals - ev0,
        n_cache_hits=objective.n_cache_hits - hit0,
        time_s=time.perf_counter() - t0,
        stop_reason=stop_reason,
        no_admissible=no_adm,
        local_maxima=n_lm,
        n_escapes=len(escapes),
        restarts=restarts,
        freq_counts=counts.tolist() if cfg.use_freq else None,
        traces=traces,
        escapes=escapes,
    )
