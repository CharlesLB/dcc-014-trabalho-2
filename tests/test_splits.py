import numpy as np
import pytest

from mst.splits import (MinMax, SPLIT_SEED, get_or_create_folds, load_folds, make_folds,
                        save_folds, split_fold)


@pytest.fixture
def y():
    return np.array([0] * 70 + [1] * 30)


def test_ten_folds_partition(y):
    folds = make_folds(y)
    assert len(folds) == 10
    tests = np.concatenate([f["test"] for f in folds])
    assert sorted(tests.tolist()) == list(range(len(y)))


def test_train_test_disjoint(y):
    for f in make_folds(y):
        assert not set(f["train"]) & set(f["test"])
        assert len(f["train"]) + len(f["test"]) == len(y)


def test_stratified(y):
    for f in make_folds(y):
        assert y[f["test"]].sum() == 3


def test_deterministic(y):
    assert make_folds(y, seed=1) == make_folds(y, seed=1)
    assert make_folds(y, seed=1) != make_folds(y, seed=2)


def test_default_seed():
    assert SPLIT_SEED == 2023


def test_save_load_roundtrip(y, tmp_path):
    f = make_folds(y)
    save_folds(f, tmp_path / "s.json", {"n": 100})
    assert load_folds(tmp_path / "s.json") == f


def test_get_or_create_reuses(y, tmp_path):
    p = tmp_path / "s.json"
    a = get_or_create_folds(y, p)
    b = get_or_create_folds(y, p, seed=999)  # lê do arquivo, ignora a nova semente
    assert a == b


def test_get_or_create_size_mismatch(y, tmp_path):
    p = tmp_path / "s.json"
    get_or_create_folds(y, p)
    with pytest.raises(ValueError):
        get_or_create_folds(y[:50], p)


def test_minmax_train_range():
    X = np.array([[0.0, 5], [10, 5], [5, 5]])
    Z = MinMax().fit_transform(X)
    assert Z[:, 0].min() == 0 and Z[:, 0].max() == 1
    assert (Z[:, 1] == 0).all()  # constante


def test_minmax_uses_train_only():
    tr = np.array([[0.0], [10.0]])
    te = np.array([[20.0]])
    sc = MinMax().fit(tr)
    assert sc.transform(te)[0, 0] == 2.0


def test_split_fold_normalizes_with_train(y):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)) * 10
    f = make_folds(y)[0]
    Xtr, ytr, Xte, yte = split_fold(X, y, f)
    assert np.isclose(Xtr.min(), 0) and np.isclose(Xtr.max(), 1)
    assert len(ytr) == 90 and len(yte) == 10


def test_split_fold_without_normalization(y):
    X = np.arange(300, dtype=float).reshape(100, 3)
    f = make_folds(y)[0]
    Xtr, _, _, _ = split_fold(X, y, f, normalize=False)
    assert Xtr.max() > 1
