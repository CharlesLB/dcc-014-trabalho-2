"""Leitura e pré-processamento das nove bases do artigo (Tabela 1).

Cada base tem um ``DatasetSpec`` com o leitor do arquivo bruto e as contagens
esperadas. Os arquivos brutos ficam em ``data/raw/<chave>/`` (ver
``data/README.md`` para nomes e fontes). O resultado padronizado é um
DataFrame só com variáveis numéricas e a coluna ``y`` ∈ {0, 1}.

Observação sobre "pos/neg": na Tabela 1 do artigo, a ordem dos números nem
sempre corresponde à classe doente (ex.: WPBC 151/47 = N/R, Mesotelioma
228/96 = saudável/doente). Por isso a checagem compara o *par* de contagens
sem ordem, e ``y = 1`` é sempre a classe doente/de interesse definida aqui.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

Y = "y"


@dataclass
class DatasetSpec:
    key: str
    nome: str
    n_casos: int
    contagens: tuple[int, int]
    m: int
    leitor: Callable[[Path], pd.DataFrame]
    arquivos: tuple[str, ...]
    fonte: str
    notas: str = ""
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------- utilidades
def _finalize(X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    df = X.copy()
    df[Y] = np.asarray(y).astype(int)
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"valores faltantes após o pré-processamento: {bad}")
    return df.reset_index(drop=True)


def drop_constant(X: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    return X[keep]


# ------------------------------------------------------------------------ leitores
def read_parkinson(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "parkinsons.data")
    y = df["status"].astype(int)  # 1 = Parkinson
    X = df.drop(columns=["name", "status"])
    return _finalize(X, y)


WDBC_FEATS = [f"{s}_{k}" for k in ("mean", "se", "worst") for s in (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension")]


def read_wdbc(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "wdbc.data", header=None)
    df.columns = ["id", "diagnosis"] + WDBC_FEATS
    y = (df["diagnosis"] == "M").astype(int)  # 1 = maligno
    return _finalize(df[WDBC_FEATS], y)


def read_wpbc(d: Path, keep_lymph: bool = False) -> pd.DataFrame:
    """WPBC: remove id, outcome e time (time vaza o rótulo).

    Padrão: 30 variáveis + tumor_size = 31 (Tabela 1). ``lymph_node_status``
    tem 4 faltantes e fica de fora; com keep_lymph=True entra e os 4 casos saem.
    """
    df = pd.read_csv(d / "wpbc.data", header=None, na_values="?")
    df.columns = ["id", "outcome", "time"] + WDBC_FEATS + ["tumor_size", "lymph_node_status"]
    y = (df["outcome"] == "R").astype(int)  # 1 = recorrência
    cols = WDBC_FEATS + ["tumor_size"] + (["lymph_node_status"] if keep_lymph else [])
    X = df[cols]
    if keep_lymph:
        ok = X.notna().all(axis=1)
        X, y = X[ok], y[ok]
    return _finalize(X, y)


def read_spectf(d: Path) -> pd.DataFrame:
    parts = [pd.read_csv(d / f, header=None) for f in ("SPECTF.train", "SPECTF.test")]
    df = pd.concat(parts, ignore_index=True)
    y = df[0].astype(int)
    X = df.drop(columns=[0])
    X.columns = [f"F{i}" for i in range(1, X.shape[1] + 1)]
    # No SPECTF, 1 = anormal. A classe minoritária (55-56) é 0.
    return _finalize(X, y)


def read_qadc(d: Path, include_experts: bool = False, modality_dummies: bool = False) -> pd.DataFrame:
    """QADC: junta green, hinselmann e schiller. Alvo = consensus."""
    frames = []
    for mod in ("green", "hinselmann", "schiller"):
        f = d / f"{mod}.csv"
        part = pd.read_csv(f)
        part["_mod"] = mod
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    y = df["consensus"].astype(int)
    expert_cols = [c for c in df.columns if c.startswith("experts::")]
    drop = ["consensus", "_mod"] + ([] if include_experts else expert_cols)
    X = df.drop(columns=drop)
    if modality_dummies:
        X = pd.concat([X, pd.get_dummies(df["_mod"], prefix="mod").astype(int)], axis=1)
    return _finalize(X, y)


CERVICAL_TARGETS = ("Hinselmann", "Schiller", "Citology", "Biopsy")


def read_cervical(d: Path, target: str = "Biopsy") -> pd.DataFrame:
    """Remove as 2 variáveis com mais faltantes, depois os casos com faltantes,
    depois colunas constantes (artigo + SPEC §5.1)."""
    df = pd.read_csv(d / "risk_factors_cervical_cancer.csv", na_values="?")
    y_all = df[target].astype(int)
    X = df.drop(columns=list(CERVICAL_TARGETS))
    worst = X.isna().sum().sort_values(ascending=False).index[:2].tolist()
    X = X.drop(columns=worst)
    ok = X.notna().all(axis=1)
    X, y = X[ok], y_all[ok]
    X = drop_constant(X)
    return _finalize(X, y)


def read_glioma(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "TCGA_InfoWithGrade.csv")
    y = df["Grade"].astype(int)  # 1 = GBM
    X = df.drop(columns=["Grade"])
    return _finalize(X, y)


def read_mesothelioma(d: Path) -> pd.DataFrame:
    """Alvo 'class of diagnosis' (2 = mesotelioma). 'diagnosis method' é
    idêntico ao alvo e sai (vazamento)."""
    xlsx = d / "Mesothelioma data set.xlsx"
    csv = d / "mesothelioma.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name=0)
    else:
        df = pd.read_csv(csv)
    df.columns = [str(c).strip() for c in df.columns]
    target = "class of diagnosis"
    y = (df[target].astype(int) == 2).astype(int)
    drop = [target] + [c for c in df.columns if c.lower() == "diagnosis method"]
    X = df.drop(columns=drop)
    return _finalize(X, y)


def read_alzheimer(d: Path) -> pd.DataFrame:
    """Base própria dos autores (UBU). Formato desconhecido: espera um CSV
    ``alzheimer.csv`` e um ``alzheimer.json`` com {"target": nome, "positive": valor}.
    Sem o JSON, usa a última coluna como alvo e o maior rótulo como positivo."""
    df = pd.read_csv(d / "alzheimer.csv", sep=None, engine="python")
    meta_f = d / "alzheimer.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
        target, positive = meta["target"], meta["positive"]
    else:
        target = df.columns[-1]
        positive = sorted(df[target].unique())[-1]
    y = (df[target] == positive).astype(int)
    X = df.drop(columns=[target])
    return _finalize(X, y)


REGISTRY: dict[str, DatasetSpec] = {
    s.key: s for s in [
        DatasetSpec("parkinson", "Parkinson", 195, (46, 146), 22, read_parkinson,
                    ("parkinsons.data",), "UCI: Parkinsons",
                    "A UCI tem 147/48; o artigo diz 46/146 (soma 192)."),
        DatasetSpec("qadc", "QADC", 287, (72, 215), 67, read_qadc,
                    ("green.csv", "hinselmann.csv", "schiller.csv"),
                    "UCI: Quality Assessment of Digital Colposcopies",
                    "A UCI tem 62 variáveis + experts; conferir como o artigo chegou a 67."),
        DatasetSpec("spectf", "SPECTF Heart", 267, (56, 211), 44, read_spectf,
                    ("SPECTF.train", "SPECTF.test"), "UCI: SPECTF Heart"),
        DatasetSpec("wdbc", "WDBC", 569, (212, 357), 30, read_wdbc,
                    ("wdbc.data",), "UCI: Breast Cancer Wisconsin (Diagnostic)"),
        DatasetSpec("wpbc", "WPBC", 198, (151, 47), 31, read_wpbc,
                    ("wpbc.data",), "UCI: Breast Cancer Wisconsin (Prognostic)"),
        DatasetSpec("alzheimer", "Alzheimer", 1548, (630, 918), 28, read_alzheimer,
                    ("alzheimer.csv",), "UBU (base própria dos autores)",
                    "Formato a confirmar; ver read_alzheimer."),
        DatasetSpec("cervical", "Câncer cervical", 668, (605, 63), 28, read_cervical,
                    ("risk_factors_cervical_cancer.csv",),
                    "UCI: Cervical cancer (Risk Factors)",
                    "Alvo padrão Biopsy; conferir com a Tabela 1."),
        DatasetSpec("glioma", "Glioma", 838, (486, 352), 23, read_glioma,
                    ("TCGA_InfoWithGrade.csv",),
                    "UCI: Glioma Grading Clinical and Mutation Features",
                    "A UCI tem 839 casos; o artigo diz 838."),
        DatasetSpec("mesothelioma", "Mesotelioma", 324, (228, 96), 33, read_mesothelioma,
                    ("Mesothelioma data set.xlsx",), "UCI: Mesothelioma's disease"),
    ]
}

PAPER_ORDER = ["parkinson", "qadc", "spectf", "wdbc", "wpbc", "alzheimer",
               "cervical", "glioma", "mesothelioma"]


@dataclass
class CheckResult:
    key: str
    ok: bool
    n_casos: int
    contagens: tuple[int, int]
    m: int
    problemas: list[str]


def check_counts(key: str, df: pd.DataFrame) -> CheckResult:
    spec = REGISTRY[key]
    n = len(df)
    pos = int(df[Y].sum())
    neg = n - pos
    m = df.shape[1] - 1
    probs = []
    if n != spec.n_casos:
        probs.append(f"casos {n} ≠ {spec.n_casos}")
    if sorted((pos, neg)) != sorted(spec.contagens):
        probs.append(f"contagens {pos}/{neg} ≠ {spec.contagens} (sem ordem)")
    if m != spec.m:
        probs.append(f"variáveis {m} ≠ {spec.m}")
    return CheckResult(key, not probs, n, (pos, neg), m, probs)


def load_raw(key: str, raw_dir: str | Path, **kw) -> pd.DataFrame:
    spec = REGISTRY[key]
    d = Path(raw_dir) / key
    return spec.leitor(d, **kw)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if Y not in df.columns:
        raise ValueError(f"{path} sem coluna '{Y}'")
    names = [c for c in df.columns if c != Y]
    return df[names].to_numpy(dtype=float), df[Y].to_numpy(dtype=int), names


# ---------------------------------------------------------------- dados sintéticos
def synthetic(n: int = 200, m: int = 20, informative: int = 5, seed: int = 0,
              weights: tuple[float, float] | None = None) -> pd.DataFrame:
    """Base sintética para testes e ensaios (make_classification)."""
    from sklearn.datasets import make_classification

    informative = max(1, min(informative, m))
    X, y = make_classification(n_samples=n, n_features=m, n_informative=informative,
                               n_redundant=max(0, min(2, m - informative)), random_state=seed,
                               n_clusters_per_class=1 if informative < 2 else 2,
                               weights=list(weights) if weights else None)
    X = pd.DataFrame(X, columns=[f"x{j}" for j in range(m)])
    return _finalize(X, y)
