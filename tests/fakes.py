"""Gera arquivos brutos falsos no formato de cada base da UCI/UBU.

Os valores são aleatórios; só o formato (colunas, separador, faltantes, rótulos)
imita os arquivos reais. ``counts`` permite gerar exatamente as contagens da
Tabela 1 para testar a checagem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mst.datasets import WDBC_FEATS


def _labels(rng, pos, neg, a, b):
    lab = np.array([a] * pos + [b] * neg, dtype=object)
    rng.shuffle(lab)
    return lab


def parkinson(d: Path, pos=147, neg=48, seed=0):
    rng = np.random.default_rng(seed)
    n = pos + neg
    cols = {"name": [f"phon_R01_S{i:02d}" for i in range(n)]}
    feats = [f"MDVP:F{i}" for i in range(22)]
    for i, c in enumerate(feats):
        cols[c] = rng.normal(size=n)
        if i == 16:
            cols["status"] = _labels(rng, pos, neg, 1, 0).astype(int)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(d / "parkinsons.data", index=False)


def wdbc(d: Path, pos=212, neg=357, seed=0):
    rng = np.random.default_rng(seed)
    n = pos + neg
    df = pd.DataFrame(rng.normal(size=(n, 30)), columns=WDBC_FEATS)
    df.insert(0, "diagnosis", _labels(rng, pos, neg, "M", "B"))
    df.insert(0, "id", np.arange(n) + 842302)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "wdbc.data", index=False, header=False)


def wpbc(d: Path, recur=47, nonrecur=151, n_missing_lymph=4, seed=0):
    rng = np.random.default_rng(seed)
    n = recur + nonrecur
    df = pd.DataFrame(rng.normal(size=(n, 30)), columns=WDBC_FEATS)
    df.insert(0, "time", rng.integers(1, 100, n))
    df.insert(0, "outcome", _labels(rng, recur, nonrecur, "R", "N"))
    df.insert(0, "id", np.arange(n) + 119513)
    df["tumor_size"] = rng.uniform(0.5, 5, n)
    lymph = rng.integers(0, 20, n).astype(object)
    lymph[:n_missing_lymph] = "?"
    df["lymph"] = lymph
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "wpbc.data", index=False, header=False)


def spectf(d: Path, n1_train=40, n0_train=40, n1_test=172, n0_test=15, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    for name, n1, n0 in (("SPECTF.train", n1_train, n0_train), ("SPECTF.test", n1_test, n0_test)):
        n = n1 + n0
        df = pd.DataFrame(rng.integers(0, 80, (n, 44)))
        df.insert(0, "lab", _labels(rng, n1, n0, 1, 0).astype(int))
        df.to_csv(d / name, index=False, header=False)


def qadc(d: Path, sizes=(98, 97, 92), pos_total=215, m=62, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    n = sum(sizes)
    y = _labels(rng, pos_total, n - pos_total, 1, 0).astype(int)
    start = 0
    for mod, sz in zip(("green", "hinselmann", "schiller"), sizes):
        df = pd.DataFrame(rng.normal(size=(sz, m)), columns=[f"feat_{i}" for i in range(m)])
        for e in range(6):
            df[f"experts::{e}"] = rng.integers(0, 2, sz)
        df["consensus"] = y[start:start + sz]
        start += sz
        df.to_csv(d / f"{mod}.csv", index=False)


CERVICAL_COLS = [
    "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
    "Smokes", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)", "IUD", "IUD (years)", "STDs", "STDs (number)",
    "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease",
    "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",
    "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis", "STDs: Time since last diagnosis",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx",
    "Hinselmann", "Schiller", "Citology", "Biopsy",
]


def cervical(d: Path, n=858, n_incomplete=190, seed=0):
    """32 variáveis + 4 alvos; 2 colunas quase todas faltantes; 2 constantes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(0, 5, (n, len(CERVICAL_COLS))).astype(object),
                      columns=CERVICAL_COLS)
    for c in ("Hinselmann", "Schiller", "Citology", "Biopsy"):
        df[c] = rng.integers(0, 2, n)
    df["STDs:cervical condylomatosis"] = 0
    df["STDs:AIDS"] = 0
    many = rng.random(n) < 0.9
    df.loc[many, "STDs: Time since first diagnosis"] = "?"
    df.loc[many, "STDs: Time since last diagnosis"] = "?"
    rows = rng.choice(n, n_incomplete, replace=False)
    df.loc[rows, "Smokes"] = "?"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "risk_factors_cervical_cancer.csv", index=False)


def glioma(d: Path, gbm=352, lgg=487, seed=0, duplicate=True):
    """Como na UCI: com ``duplicate`` uma linha LGG aparece duas vezes."""
    rng = np.random.default_rng(seed)
    n = gbm + lgg
    cols = {"Grade": _labels(rng, gbm, lgg, 1, 0).astype(int), "Gender": rng.integers(0, 2, n),
            "Age_at_diagnosis": rng.uniform(14, 89, n), "Race": rng.integers(0, 4, n)}
    for g in ("IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1",
              "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4",
              "PDGFRA"):
        cols[g] = rng.integers(0, 2, n)
    df = pd.DataFrame(cols)
    if duplicate:
        lgg_row = df.index[df["Grade"] == 0][0]
        df = pd.concat([df, df.loc[[lgg_row]]], ignore_index=True)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "TCGA_InfoWithGrade.csv", index=False)


def mesothelioma(d: Path, healthy=228, meso=96, fmt="xlsx", seed=0):
    rng = np.random.default_rng(seed)
    n = healthy + meso
    cls = _labels(rng, meso, healthy, 2, 1).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, 33)), columns=[f"attr{i}" for i in range(33)])
    df.insert(5, "diagnosis method", cls)
    df["class of diagnosis"] = cls
    d.mkdir(parents=True, exist_ok=True)
    if fmt == "xlsx":
        df.to_excel(d / "Mesothelioma data set.xlsx", index=False)
    else:
        df.to_csv(d / "mesothelioma.csv", index=False)


def alzheimer(d: Path, pos=630, neg=918, m=28, meta=True, sep=";", seed=0):
    rng = np.random.default_rng(seed)
    n = pos + neg
    df = pd.DataFrame(rng.normal(size=(n, m)), columns=[f"v{i}" for i in range(m)])
    df["diag"] = _labels(rng, pos, neg, "AD", "CN")
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "alzheimer.csv", index=False, sep=sep)
    if meta:
        (d / "alzheimer.json").write_text('{"target": "diag", "positive": "AD"}')


ALL = {
    "parkinson": parkinson, "wdbc": wdbc, "wpbc": wpbc, "spectf": spectf, "qadc": qadc,
    "cervical": cervical, "glioma": glioma, "mesothelioma": mesothelioma, "alzheimer": alzheimer,
}


def make_all(raw: Path):
    for key, fn in ALL.items():
        fn(raw / key)
