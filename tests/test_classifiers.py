import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mst.classifiers import CLASSIFIERS, FastLDA, fit_decision, fit_predict, make_classifier


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 250))
    m = int(rng.integers(1, 12))
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(int)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    return X, y


@pytest.mark.parametrize("seed", range(25))
def test_fast_lda_matches_sklearn(seed):
    X, y = _random_problem(seed)
    a = FastLDA().fit(X, y)
    b = LinearDiscriminantAnalysis().fit(X, y)
    np.testing.assert_allclose(a.decision_function(X), b.decision_function(X), atol=1e-8)
    assert np.array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("seed", range(8))
def test_fast_lda_matches_sklearn_collinear(seed):
    X, y = _random_problem(seed + 100)
    if X.shape[1] < 2:
        X = np.hstack([X, X])
    X = np.hstack([X, X[:, :1] * 2.0, np.ones((X.shape[0], 1))])  # colinear + constante
    a = FastLDA().fit(X, y)
    b = LinearDiscriminantAnalysis().fit(X, y)
    assert np.mean(a.predict(X) == b.predict(X)) == 1.0


def test_fast_lda_requires_two_classes():
    with pytest.raises(ValueError):
        FastLDA().fit(np.zeros((3, 1)), np.array([1, 1, 1]))


def test_fast_lda_string_labels():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array(["a", "a", "b", "b"])
    assert list(FastLDA().fit(X, y).predict(X)) == list(y)


def test_fast_lda_all_constant_features():
    X = np.ones((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    p = FastLDA().fit(X, y).predict(X)
    assert set(p) <= {0, 1}


def test_fast_lda_prior_breaks_ties():
    X = np.zeros((5, 1))
    y = np.array([0, 0, 0, 1, 1])
    assert (FastLDA().fit(X, y).predict(X) == 0).all()


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_make_classifier(name):
    clf = make_classifier(name, seed=1)
    assert hasattr(clf, "fit") and hasattr(clf, "predict")


def test_make_classifier_case_insensitive():
    assert isinstance(make_classifier("da"), FastLDA)


def test_make_classifier_unknown():
    with pytest.raises(ValueError):
        make_classifier("knn")


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_fit_predict_separable(name):
    X = np.array([[0.0, 1], [0.1, 0], [0.2, 1], [1.0, 0], [1.1, 1], [1.2, 0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    p = fit_predict(name, X, y, X)
    assert np.mean(p == y) >= 5 / 6


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_fit_decision_shape(name):
    X, y = _random_problem(3)
    d = fit_decision(name, X, y, X[:7])
    assert d.shape == (7,)


def test_lr_svm_deterministic_with_seed():
    X, y = _random_problem(5)
    for name in ("LR", "SVM"):
        a = fit_predict(name, X, y, X, seed=3)
        b = fit_predict(name, X, y, X, seed=3)
        assert np.array_equal(a, b)
