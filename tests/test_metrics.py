import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from mst import metrics as M

Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
# TP=3 FN=1 TN=4 FP=2


def test_confusion_by_hand():
    assert M.confusion(Y_TRUE, Y_PRED) == (3, 4, 2, 1)


def test_accuracy_by_hand():
    assert M.accuracy(3, 4, 2, 1) == pytest.approx(0.7)


def test_auc_paper_by_hand():
    assert M.auc_paper(3, 4, 2, 1) == pytest.approx((1 + 0.75 - 2 / 6) / 2)


def test_gmean_by_hand():
    assert M.gmean(3, 4, 2, 1) == pytest.approx(math.sqrt(0.75 * 4 / 6))


def test_f1_by_hand():
    assert M.f1(3, 4, 2, 1) == pytest.approx(6 / 9)


def test_auc_paper_equals_balanced_accuracy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        yt = rng.integers(0, 2, 50)
        yp = rng.integers(0, 2, 50)
        c = M.confusion(yt, yp)
        assert M.auc_paper(*c) == pytest.approx(balanced_accuracy_score(yt, yp))


def test_f1_matches_sklearn():
    assert M.f1(*M.confusion(Y_TRUE, Y_PRED)) == pytest.approx(f1_score(Y_TRUE, Y_PRED))


def test_all_metrics_keys():
    d = M.all_metrics(Y_TRUE, Y_PRED)
    assert {"acc", "auc", "gmean", "f1", "tp", "tn", "fp", "fn"} <= set(d)


def test_perfect_prediction():
    d = M.all_metrics(Y_TRUE, Y_TRUE)
    assert d["acc"] == d["auc"] == d["gmean"] == d["f1"] == 1.0


def test_all_negative_prediction_zero_division():
    d = M.all_metrics(Y_TRUE, np.zeros_like(Y_TRUE))
    assert d["f1"] == 0.0 and d["gmean"] == 0.0
    assert d["auc"] == pytest.approx(0.5)


def test_no_positives_in_truth():
    yt = np.zeros(5, dtype=int)
    d = M.all_metrics(yt, yt)
    assert d["acc"] == 1.0 and d["f1"] == 0.0


def test_positive_label_switch():
    c1 = M.confusion(Y_TRUE, Y_PRED, positive=1)
    c0 = M.confusion(Y_TRUE, Y_PRED, positive=0)
    assert c0 == (c1[1], c1[0], c1[3], c1[2])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        M.confusion([0, 1], [0])


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 2, 80)
    s = rng.normal(size=80) + yt
    assert M.roc_auc(yt, s) == pytest.approx(roc_auc_score(yt, s))


def test_roc_auc_with_ties():
    yt = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert M.roc_auc(yt, s) == pytest.approx(0.5)


def test_roc_auc_single_class_nan():
    assert math.isnan(M.roc_auc([1, 1], [0.1, 0.2]))
