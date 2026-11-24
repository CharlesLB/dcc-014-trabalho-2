import numpy as np
import pandas as pd
import pytest

import fakes
from mst import datasets as D


@pytest.fixture
def raw(tmp_path):
    return tmp_path / "raw"


def test_registry_has_nine_bases_in_paper_order():
    assert D.PAPER_ORDER == list(D.REGISTRY)
    assert len(D.REGISTRY) == 9


def test_registry_matches_table1():
    t1 = {"parkinson": (195, 22), "qadc": (287, 67), "spectf": (267, 44), "wdbc": (569, 30),
          "wpbc": (198, 31), "alzheimer": (1548, 28), "cervical": (668, 28),
          "glioma": (838, 23), "mesothelioma": (324, 33)}
    for k, (n, m) in t1.items():
        assert (D.REGISTRY[k].n_casos, D.REGISTRY[k].m) == (n, m)
        assert sum(D.REGISTRY[k].contagens) == n or k == "parkinson"


def _check_basic(df, m=None):
    assert "y" in df.columns
    assert set(df["y"].unique()) <= {0, 1}
    assert not df.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in df.dtypes)
    if m is not None:
        assert df.shape[1] - 1 == m


def test_parkinson(raw):
    fakes.parkinson(raw / "parkinson")
    df = D.load_raw("parkinson", raw)
    _check_basic(df, 22)
    assert len(df) == 195 and df.y.sum() == 147
    assert "status" not in df.columns and "name" not in df.columns


def test_wdbc_ok_counts(raw):
    fakes.wdbc(raw / "wdbc")
    df = D.load_raw("wdbc", raw)
    _check_basic(df, 30)
    assert df.y.sum() == 212
    assert D.check_counts("wdbc", df).ok


def test_wdbc_divergent_counts(raw):
    fakes.wdbc(raw / "wdbc", pos=200, neg=300)
    chk = D.check_counts("wdbc", D.load_raw("wdbc", raw))
    assert not chk.ok and any("casos" in p for p in chk.problemas)


def test_wpbc_default_31_features(raw):
    fakes.wpbc(raw / "wpbc")
    df = D.load_raw("wpbc", raw)
    _check_basic(df, 31)
    assert len(df) == 198 and df.y.sum() == 47
    assert "time" not in df.columns
    assert D.check_counts("wpbc", df).ok


def test_wpbc_keep_lymph_drops_missing(raw):
    fakes.wpbc(raw / "wpbc")
    df = D.read_wpbc(raw / "wpbc", keep_lymph=True)
    _check_basic(df, 32)
    assert len(df) == 194


def test_spectf_combines_train_and_test(raw):
    fakes.spectf(raw / "spectf")
    df = D.load_raw("spectf", raw)
    _check_basic(df, 44)
    assert len(df) == 267


def test_qadc_three_modalities(raw):
    fakes.qadc(raw / "qadc")
    df = D.load_raw("qadc", raw)
    _check_basic(df, 62 + 3)  # padrão: dummies de modalidade, sem experts
    assert len(df) == 287
    assert not any(c.startswith("experts::") for c in df.columns)
    assert {"mod_green", "mod_hinselmann", "mod_schiller"} <= set(df.columns)
    assert (df[["mod_green", "mod_hinselmann", "mod_schiller"]].sum(axis=1) == 1).all()


def test_qadc_without_dummies(raw):
    fakes.qadc(raw / "qadc")
    df = D.read_qadc(raw / "qadc", modality_dummies=False)
    _check_basic(df, 62)


def test_qadc_options(raw):
    fakes.qadc(raw / "qadc")
    df = D.read_qadc(raw / "qadc", include_experts=True, modality_dummies=True)
    _check_basic(df, 62 + 6 + 3)


def test_cervical_pipeline(raw):
    fakes.cervical(raw / "cervical")
    df = D.load_raw("cervical", raw)
    _check_basic(df)
    assert "STDs: Time since first diagnosis" not in df.columns
    assert "STDs: Time since last diagnosis" not in df.columns
    assert "STDs:AIDS" not in df.columns  # constante
    assert len(df) == 858 - 190
    assert df.shape[1] - 1 == 32 - 2 - 2
    for t in D.CERVICAL_TARGETS:
        assert t not in df.columns


def test_cervical_other_target(raw):
    fakes.cervical(raw / "cervical")
    a = D.read_cervical(raw / "cervical", target="Schiller")
    b = D.read_cervical(raw / "cervical", target="Biopsy")
    assert len(a) == len(b)


def test_cervical_default_target_is_schiller(raw):
    fakes.cervical(raw / "cervical")
    df = D.load_raw("cervical", raw)
    raw_df = pd.read_csv(raw / "cervical" / "risk_factors_cervical_cancer.csv", na_values="?")
    ok = raw_df.drop(columns=["STDs: Time since first diagnosis",
                              "STDs: Time since last diagnosis"]).notna().all(axis=1)
    assert (df["y"].to_numpy() == raw_df.loc[ok, "Schiller"].astype(int).to_numpy()).all()


def test_glioma(raw):
    fakes.glioma(raw / "glioma")
    df = D.load_raw("glioma", raw)
    _check_basic(df, 23)
    assert len(df) == 839  # 840 linhas no arquivo, 1 repetida removida
    assert (df.y == 1).sum() == 352 and (df.y == 0).sum() == 487
    assert not df.duplicated().any()


def test_glioma_keep_duplicates(raw):
    fakes.glioma(raw / "glioma")
    df = D.read_glioma(raw / "glioma", drop_duplicates=False)
    assert len(df) == 840


def test_glioma_counts_match_table1(raw):
    fakes.glioma(raw / "glioma", gbm=352, lgg=486)
    chk = D.check_counts("glioma", D.load_raw("glioma", raw))
    assert chk.ok, chk.problemas


@pytest.mark.parametrize("fmt", ["xlsx", "csv"])
def test_mesothelioma_drops_leaky_column(raw, fmt):
    fakes.mesothelioma(raw / "mesothelioma", fmt=fmt)
    df = D.load_raw("mesothelioma", raw)
    _check_basic(df, 33)
    assert "diagnosis method" not in df.columns
    assert df.y.sum() == 96
    assert D.check_counts("mesothelioma", df).ok


def test_alzheimer_with_meta(raw):
    fakes.alzheimer(raw / "alzheimer")
    df = D.load_raw("alzheimer", raw)
    _check_basic(df, 28)
    assert df.y.sum() == 630
    assert D.check_counts("alzheimer", df).ok


def test_alzheimer_without_meta_uses_last_column(raw):
    fakes.alzheimer(raw / "alzheimer", meta=False, sep=",")
    df = D.load_raw("alzheimer", raw)
    _check_basic(df, 28)
    # sem JSON, positivo = maior rótulo em ordem: "CN" (918 casos). Por isso o JSON é recomendado.
    assert df.y.sum() == 918


def test_check_counts_order_insensitive(raw):
    fakes.mesothelioma(raw / "mesothelioma")
    df = D.load_raw("mesothelioma", raw)
    chk = D.check_counts("mesothelioma", df)
    assert chk.contagens == (96, 228) and chk.ok


def test_check_counts_reports_m(raw):
    fakes.qadc(raw / "qadc")
    chk = D.check_counts("qadc", D.load_raw("qadc", raw))
    assert not chk.ok and any("variáveis" in p for p in chk.problemas)


def test_finalize_rejects_missing():
    with pytest.raises(ValueError):
        D._finalize(pd.DataFrame({"a": [1.0, None]}), [0, 1])


def test_finalize_coerces_numeric():
    df = D._finalize(pd.DataFrame({"a": ["1", "2"]}), [0, 1])
    assert df["a"].dtype.kind in "if"


def test_drop_constant():
    df = D.drop_constant(pd.DataFrame({"a": [1, 1], "b": [1, 2]}))
    assert list(df.columns) == ["b"]


def test_save_and_load_processed(tmp_path):
    df = D.synthetic(n=50, m=5)
    D.save_processed(df, tmp_path / "x" / "s.csv")
    X, y, names = D.load_processed(tmp_path / "x" / "s.csv")
    assert X.shape == (50, 5) and y.shape == (50,) and names == [f"x{j}" for j in range(5)]


def test_load_processed_requires_y(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        D.load_processed(p)


@pytest.mark.parametrize("n,m", [(60, 4), (200, 20), (100, 50)])
def test_synthetic_shapes(n, m):
    df = D.synthetic(n=n, m=m, informative=min(3, m))
    _check_basic(df, m)
    assert len(df) == n


def test_synthetic_deterministic():
    assert D.synthetic(seed=3).equals(D.synthetic(seed=3))


def test_synthetic_imbalanced():
    df = D.synthetic(n=200, m=6, informative=3, weights=(0.9, 0.1))
    assert df.y.mean() < 0.25


def test_all_fakes_load(raw):
    fakes.make_all(raw)
    for k in D.PAPER_ORDER:
        _check_basic(D.load_raw(k, raw))
