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


# ------------------------------------------- DA_PASCAL (regra do código público)
def _pascal_reference(X, y):
    """Reimplementação literal de FuncionesClas.pas: covariância total (1/n),
    coeficientes Σ_T⁻¹(μ1 − μ0), corte no ponto médio das médias, classe 1 se > 0."""
    mu = X.mean(axis=0)
    cov = X.T @ X / len(X) - np.outer(mu, mu)
    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    w = np.linalg.solve(cov, mu1 - mu0)
    b = -w @ ((mu1 + mu0) / 2)
    return (X @ w + b > 0).astype(int)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.8, 0.2), (0.9, 0.1)])
def test_da_pascal_matches_pascal_rule(seed, weights):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=0,
                               weights=list(weights), random_state=seed)
    ours = fit_predict("DA_PASCAL", X, y, X)
    ref = _pascal_reference(X, y)
    assert (ours == ref).mean() >= 0.995  # empates numéricos no limiar podem trocar 1 caso


def test_da_pascal_differs_from_da_only_in_threshold():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=400, n_features=5, n_informative=3, n_redundant=0,
                               weights=[0.85, 0.15], random_state=0)
    a = FastLDA().fit(X, y)
    b = FastLDA(equal_priors=True).fit(X, y)
    assert np.allclose(a.coef_, b.coef_)
    p1 = y.mean()
    assert a.intercept_ - b.intercept_ == pytest.approx(np.log(p1 / (1 - p1)))


def test_da_pascal_equals_da_when_balanced():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.repeat([0, 1], 100)
    X[y == 1] += 0.7
    assert (fit_predict("DA", X, y, X) == fit_predict("DA_PASCAL", X, y, X)).all()
