"""Classificadores usados para pontuar subconjuntos.

DA      análise discriminante linear em numpy (rápida, mesma regra de decisão
        do LinearDiscriminantAnalysis(solver="svd") do scikit-learn para 2 classes).
DA_SK   LinearDiscriminantAnalysis do scikit-learn (referência).
LR      LogisticRegression(solver="liblinear"), Lin et al. (trust region Newton).
SVM     LinearSVC(dual=True), Hsieh et al. (dual coordinate descent).
"""

from __future__ import annotations

import warnings

import numpy as np

CLASSIFIERS = ("DA", "DA_SK", "LR", "SVM")


class FastLDA:
    """LDA binária com covariância agrupada e pseudo-inversa.

    Regra: classe 1 se x·w + b > 0, com
        w = Σ⁺ (μ1 − μ0)
        b = −½ (μ1 + μ0)·w + log(π1/π0)
    e Σ = Σ_k Σ_{i∈k} (x_i − μ_k)(x_i − μ_k)ᵀ / n (covariância agrupada
    com o mesmo fator 1/n do scikit-learn 1.9, para as decisões coincidirem;
    o teste tests/test_classifiers.py confere a concordância).
    """

    def __init__(self, tol: float = 1e-4):
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastLDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("FastLDA exige exatamente duas classes no treino")
        self.classes_ = classes
        mask1 = y == classes[1]
        X0, X1 = X[~mask1], X[mask1]
        mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
        n = X.shape[0]
        Xc = np.vstack([X0 - mu0, X1 - mu1])
        # Escala por desvio-padrão como no scikit-learn para estabilidade numérica.
        std = Xc.std(axis=0)
        std[std == 0] = 1.0
        Z = Xc / std / np.sqrt(n)
        # Σ_z = Zᵀ Z ; pseudo-inversa via SVD com tolerância absoluta (como o scikit-learn).
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        rank = int(np.sum(s > self.tol)) if s.size else 0
        diff = (mu1 - mu0) / std
        if rank == 0:
            w_z = np.zeros_like(diff)
        else:
            V = Vt[:rank].T
            w_z = V @ ((V.T @ diff) / (s[:rank] ** 2))
        w = w_z / std
        p1 = mask1.mean()
        p0 = 1.0 - p1
        self.coef_ = w
        self.intercept_ = float(-0.5 * (mu1 + mu0) @ w + np.log(p1 / p0))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        d = self.decision_function(X)
        return np.where(d > 0, self.classes_[1], self.classes_[0])


def make_classifier(name: str, seed: int = 0):
    """Cria um classificador novo pelo nome (DA, DA_SK, LR, SVM)."""
    name = name.upper()
    if name == "DA":
        return FastLDA()
    if name == "DA_SK":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(solver="svd")
    if name == "LR":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(solver="liblinear", random_state=seed)
    if name == "SVM":
        from sklearn.svm import LinearSVC

        return LinearSVC(dual=True, random_state=seed, max_iter=5000)
    raise ValueError(f"classificador desconhecido: {name!r}; use um de {CLASSIFIERS}")


def fit_predict(name: str, X_train, y_train, X_eval, seed: int = 0) -> np.ndarray:
    """Ajusta no treino e prediz X_eval. Silencia avisos de colinearidade/convergência."""
    clf = make_classifier(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        return clf.predict(X_eval)


def fit_decision(name: str, X_train, y_train, X_eval, seed: int = 0) -> np.ndarray:
    """Escore contínuo (decision_function) para ROC-AUC opcional."""
    clf = make_classifier(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        return clf.decision_function(X_eval)
