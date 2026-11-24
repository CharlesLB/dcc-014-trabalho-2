"""Execução de campanhas de experimentos (SPEC §6).

Uma campanha é descrita por um dicionário (lido de YAML em experiments/configs).
Cada combinação (variante, base, classificador, fold, semente, parâmetros) vira
uma execução registrada em ``<saida>/<campanha>/runs.jsonl``.
"""

from __future__ import annotations

import itertools
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import runlog
from .baselines import BASELINES, run_baseline
from .classifiers import fit_decision, fit_predict
from .datasets import load_processed, synthetic
from .greedy import run_greedy
from .metrics import all_metrics, roc_auc
from .multistart import MSTConfig, resolve_params, run_mst
from .objective import ClassifierObjective
from .splits import SPLIT_SEED, get_or_create_folds, make_folds, split_fold

MST_VARIANTS = ("MST", "MST_FREQ", "MST_SEM_TABU", "MST_SEM_REINICIO")
VARIANTS = MST_VARIANTS + ("GULOSO",) + tuple(BASELINES)

DEFAULT_PARAMS = {
    "beta": 0.99,
    "alpha": 0.99,
    "scale_base": "casos",
    "tenure": None,          # None = resolve_params
    "maxiter_ts": None,      # None = resolve_params
    "maxiter_ms": 10,
    "outer_stop_strict": False,
    "lam": 0.01,
    "trigger": 3,
    "eps": 1e-12,
    "normalizar": True,
}


@dataclass
class RunSpec:
    campanha: str
    variante: str
    base: str
    classificador: str
    fold: int
    seed: int
    params: dict
    dados: str = "data/processed"
    splits: str = "data/splits"
    split_seed: int = SPLIT_SEED
    n_splits: int = 10
    max_evals: int | None = None
    time_budget: float | None = None
    trace: bool = False
    saida: str = "results"
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        p = dict(self.params)
        p["max_evals"] = self.max_evals
        p["time_budget"] = self.time_budget
        return runlog.run_key(self.variante, self.base, self.classificador,
                              self.fold, self.seed, p)


# ----------------------------------------------------------------------- dados
_DATA_CACHE: dict = {}


def load_base(base: str, dados: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega uma base processada, ou uma sintética.

    Bases sintéticas: "sintetica" ou "sintetica-<n>-<m>[-<informativas>[-<seed>]]".
    """
    ck = (base, dados)
    if ck in _DATA_CACHE:
        return _DATA_CACHE[ck]
    if base.startswith("sintetica"):
        parts = base.split("-")[1:]
        nums = [int(p) for p in parts]
        n, m, inf, sd = (nums + [200, 20, 5, 0][len(nums):])[:4]
        df = synthetic(n=n, m=m, informative=min(inf, m), seed=sd)
        names = [c for c in df.columns if c != "y"]
        out = (df[names].to_numpy(float), df["y"].to_numpy(int), names)
    else:
        out = load_processed(Path(dados) / f"{base}.csv")
    _DATA_CACHE[ck] = out
    return out


def folds_for(spec: RunSpec, y: np.ndarray) -> list[dict]:
    if spec.base.startswith("sintetica"):
        return make_folds(y, spec.n_splits, spec.split_seed)
    return get_or_create_folds(y, Path(spec.splits) / f"{spec.base}.json",
                               spec.n_splits, spec.split_seed)


# -------------------------------------------------------------------- execução
def build_mst_config(variante: str, params: dict, n_train: int, m: int,
                     budget_mode: bool, trace: bool) -> tuple[MSTConfig, dict]:
    p = {**DEFAULT_PARAMS, **params}
    res = resolve_params(n_train, m, p["scale_base"])
    tenure = p["tenure"] if p["tenure"] is not None else res["tenure"]
    maxiter_ts = p["maxiter_ts"] if p["maxiter_ts"] is not None else res["maxiter_ts"]
    max_restarts = None
    maxiter_ms = p["maxiter_ms"]
    if variante == "MST_SEM_TABU":
        tenure = 0
    if variante == "MST_SEM_REINICIO":
        maxiter_ms = 1
        max_restarts = 1
    cfg = MSTConfig(
        tenure=int(tenure), maxiter_ts=int(maxiter_ts), alpha=p["alpha"],
        maxiter_ms=int(maxiter_ms), outer_stop_strict=bool(p["outer_stop_strict"]),
        eps=p["eps"], use_freq=variante == "MST_FREQ", lam=p["lam"],
        trigger=int(p["trigger"]), stop_on_stagnation=not budget_mode,
        max_restarts=max_restarts, trace=trace,
    )
    resolved = {**p, "tenure": cfg.tenure, "maxiter_ts": cfg.maxiter_ts,
                "maxiter_ms": cfg.maxiter_ms, "n_base": res["n_base"],
                "use_freq": cfg.use_freq}
    return cfg, resolved


def execute(spec: RunSpec) -> dict:
    """Executa uma RunSpec e devolve o registro (dict). Traço gravado à parte."""
    if spec.variante not in VARIANTS:
        raise ValueError(f"variante desconhecida: {spec.variante}; use {VARIANTS}")
    X, y, names = load_base(spec.base, spec.dados)
    folds = folds_for(spec, y)
    fold = folds[spec.fold]
    p = {**DEFAULT_PARAMS, **spec.params}
    Xtr, ytr, Xte, yte = split_fold(X, y, fold, normalize=p["normalizar"])
    n_train, m = Xtr.shape

    budget_mode = spec.max_evals is not None
    obj = ClassifierObjective(Xtr, ytr, spec.classificador, beta=p["beta"], seed=spec.seed,
                              max_evals=spec.max_evals)
    resolved = dict(p)
    if spec.variante in MST_VARIANTS:
        cfg, resolved = build_mst_config(spec.variante, spec.params, n_train, m,
                                         budget_mode, spec.trace)
        res = run_mst(obj, cfg, seed=spec.seed)
    elif spec.variante == "GULOSO":
        res = run_greedy(obj, eps=p["eps"], seed=spec.seed)
    else:
        if spec.time_budget is None and spec.max_evals is None:
            raise ValueError(f"{spec.variante} precisa de time_budget ou max_evals")
        bl_params = spec.extra.get("baseline_params", {})
        res = run_baseline(spec.variante, obj, seed=spec.seed,
                           time_budget=spec.time_budget, **bl_params)
    resolved["max_evals"] = spec.max_evals
    resolved["time_budget"] = spec.time_budget

    S = sorted(res.S_best)
    val = {}
    rat = float("nan")
    if S:
        pred = fit_predict(spec.classificador, Xtr[:, S], ytr, Xte[:, S], seed=spec.seed)
        val = all_metrics(yte, pred)
        try:
            val["roc_auc"] = roc_auc(yte, fit_decision(spec.classificador, Xtr[:, S], ytr,
                                                       Xte[:, S], seed=spec.seed))
        except Exception:
            val["roc_auc"] = float("nan")
        rat = ClassifierObjective(Xtr, ytr, spec.classificador, beta=p["beta"],
                                  seed=spec.seed, cache=False).rat(S)

    rec = {
        "run_id": runlog.new_run_id(),
        "run_key": spec.key,
        "campanha": spec.campanha,
        "variante": spec.variante,
        "base": spec.base,
        "classificador": spec.classificador,
        "fold": spec.fold,
        "split_seed": spec.split_seed,
        "seed": spec.seed,
        "n_treino": int(n_train),
        "m": int(m),
        "params": resolved,
        "S_star": S,
        "S_star_nomes": [names[j] for j in S],
        "tamanho_S_star": len(S),
        "f_treino": res.f_best,
        "rat_treino": rat,
        "validacao": val,
        "tempo_s": res.time_s,
        "iter_locais_total": res.total_local_iters,
        "n_buscas_locais": res.n_local_searches,
        "n_evals": res.n_evals,
        "n_cache_hits": res.n_cache_hits,
        "iter_sem_admissivel": res.no_admissible,
        "maximos_locais": res.local_maxima,
        "stop_reason": res.stop_reason,
        "reinicios": res.restarts,
        "fugas_confirmadas": res.n_escapes if spec.trace or spec.variante in MST_VARIANTS else None,
        "fugas": res.escapes,
        "freq_contadores": res.freq_counts,
        "ambiente": runlog.environment(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if spec.trace and res.traces is not None:
        tpath = Path(spec.saida) / spec.campanha / "traces" / f"{rec['run_id']}.jsonl"
        runlog.write_trace(tpath, res.traces)
        rec["trace_path"] = str(tpath)
    return rec


# ------------------------------------------------------------------- campanhas
def _grid(params: dict, grade: dict | None) -> list[dict]:
    if not grade:
        return [dict(params)]
    keys = sorted(grade)
    out = []
    for combo in itertools.product(*(grade[k] for k in keys)):
        out.append({**params, **dict(zip(keys, combo))})
    return out


def budgets_from(path: str | Path, variante: str = "MST", campo: str = "n_evals",
                 agg: str = "mediana") -> dict:
    """{(base, classificador, fold): orçamento} a partir de um runs.jsonl anterior."""
    vals = defaultdict(list)
    for r in runlog.iter_records(path):
        if r["variante"] == variante:
            vals[(r["base"], r["classificador"], int(r["fold"]))].append(r[campo])
    fn = statistics.median if agg == "mediana" else statistics.mean
    return {k: fn(v) for k, v in vals.items()}


def expand(config: dict) -> list[RunSpec]:
    """Expande uma configuração em RunSpecs."""
    camp = config["campanha"]
    variantes = config.get("variantes", ["MST"])
    for v in variantes:
        if v not in VARIANTS:
            raise ValueError(f"variante desconhecida: {v}")
    bases = config["bases"]
    clfs = config.get("classificadores", ["DA"])
    seeds = config.get("sementes", [0])
    n_splits = int(config.get("n_splits", 10))
    folds_cfg = config.get("folds", "todos")
    folds = list(range(n_splits)) if folds_cfg in ("todos", "all", None) else list(folds_cfg)
    params_list = _grid(config.get("params", {}), config.get("grade"))
    proto = config.get("protocolo", {"tipo": "P1"}) or {"tipo": "P1"}
    budgets = {}
    if proto.get("tipo") == "P2":
        budgets = budgets_from(proto["orcamento_de"], proto.get("variante_ref", "MST"))
    times = {}
    if config.get("tempo_de"):
        times = budgets_from(config["tempo_de"], config.get("tempo_variante_ref", "MST"),
                             campo="tempo_s", agg="media")
    specs = []
    for v, b, c, f, s, p in itertools.product(variantes, bases, clfs, folds, seeds, params_list):
        max_evals = None
        if proto.get("tipo") == "P2" and v in MST_VARIANTS:
            k = (b, c, f)
            if k not in budgets:
                raise KeyError(f"sem orçamento para {k} em {proto['orcamento_de']}")
            max_evals = int(budgets[k])
        tb = None
        if v in BASELINES:
            k = (b, c, f)
            if k in times:
                tb = float(times[k])
            elif "tempo_s" in config:
                tb = float(config["tempo_s"])
            elif proto.get("tipo") == "P2" and k in budgets:
                max_evals = int(budgets[k])
            else:
                raise KeyError(f"baseline {v} sem orçamento de tempo para {k}")
        specs.append(RunSpec(
            campanha=camp, variante=v, base=b, classificador=c, fold=f, seed=s,
            params=p, dados=config.get("dados", "data/processed"),
            splits=config.get("splits", "data/splits"),
            split_seed=int(config.get("split_seed", SPLIT_SEED)), n_splits=n_splits,
            max_evals=max_evals, time_budget=tb, trace=bool(config.get("trace", False)),
            saida=config.get("saida", "results"),
            extra={"baseline_params": config.get("baseline_params", {})},
        ))
    return specs


def limit_blas_threads(n: int = 1) -> None:
    """Limita BLAS/OpenMP a n threads no processo atual (uma por processo em paralelo)."""
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n)
    except Exception:  # threadpoolctl vem com o scikit-learn; se faltar, fica só o ambiente
        pass


def default_n_jobs() -> int:
    """Núcleos físicos − 1 (ex.: Ryzen 7 de 8 núcleos -> 7). Usa os lógicos/2 se não souber."""
    import os

    logical = os.cpu_count() or 2
    return max(1, logical // 2 - 1) if logical >= 4 else 1


def _exec_safe(spec: RunSpec) -> dict:
    try:
        return execute(spec)
    except Exception as e:  # registra a falha sem derrubar a campanha
        return {"erro": f"{type(e).__name__}: {e}", "run_key": spec.key,
                "variante": spec.variante, "base": spec.base, "fold": spec.fold,
                "seed": spec.seed, "campanha": spec.campanha}


def run_campaign(config: dict, n_jobs: int | None = None, resume: bool = True,
                 progress: bool = True) -> Path:
    specs = expand(config)
    out = Path(config.get("saida", "results")) / config["campanha"] / "runs.jsonl"
    errs = out.with_name("erros.jsonl")
    done = runlog.done_keys(out) if resume else set()
    todo = [s for s in specs if s.key not in done]
    n_jobs = n_jobs or config.get("n_jobs", 1)
    n_jobs = default_n_jobs() if n_jobs == "auto" else int(n_jobs)
    blas = int(config.get("blas_threads", 1))
    if progress:
        print(f"[{config['campanha']}] {len(todo)} execuções ({len(specs) - len(todo)} já feitas), "
              f"{n_jobs} processo(s), BLAS com {blas} thread(s)")
    t0 = time.perf_counter()

    def _store(i, rec):
        if "erro" in rec:
            runlog.append(errs, rec)
            if progress:
                print(f"  ERRO {rec['run_key']}: {rec['erro']}")
        else:
            runlog.append(out, rec)
            if progress:
                print(f"  {i}/{len(todo)} {rec['variante']} {rec['base']} f{rec['fold']} "
                      f"s{rec['seed']} f={rec['f_treino']:.4f} |S|={rec['tamanho_S_star']} "
                      f"{rec['tempo_s']:.1f}s")

    if n_jobs <= 1:
        limit_blas_threads(blas)
        for i, s in enumerate(todo, 1):
            _store(i, _exec_safe(s))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=limit_blas_threads,
                                 initargs=(blas,)) as ex:
            futs = [ex.submit(_exec_safe, s) for s in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                _store(i, fut.result())
    if progress:
        print(f"[{config['campanha']}] concluído em {time.perf_counter() - t0:.1f}s -> {out}")
    return out
