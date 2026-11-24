import math

import numpy as np
import pytest

from mst import analysis as A


def test_jaccard_basic():
    assert A.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert A.jaccard({1}, {1}) == 1.0
    assert A.jaccard(set(), set()) == 1.0
    assert A.jaccard({1}, {2}) == 0.0


def test_jaccard_distance():
    assert A.jaccard_distance({1, 2}, {2, 3}) == pytest.approx(2 / 3)


def test_restart_diversity_identical_zero():
    assert A.restart_diversity([[1, 2], [1, 2], [2, 1]]) == 0.0


def test_restart_diversity_disjoint_one():
    assert A.restart_diversity([[1], [2], [3]]) == 1.0


def test_restart_diversity_single_nan():
    assert math.isnan(A.restart_diversity([[1, 2]]))


def test_stability_value():
    assert A.stability([[1, 2], [1, 2], [1, 3]]) == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)


def test_kuncheva_identical_is_one():
    assert A.kuncheva([[0, 1, 2], [0, 1, 2]], m=10) == pytest.approx(1.0)


def test_kuncheva_random_near_zero_or_negative():
    assert A.kuncheva([[0, 1], [2, 3]], m=10) < 0


def test_selection_frequency():
    f = A.selection_frequency([[0, 1], [1], [1, 2]], m=4)
    np.testing.assert_allclose(f, [1 / 3, 1, 1 / 3, 0])


def test_holm_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = A.holm(p)
    # ordenados: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotônico 0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_keeps_nan():
    adj = A.holm([0.6, float("nan"), 0.9])
    assert adj[0] == 1.0 and math.isnan(adj[1]) and adj[2] == 1.0


def test_holm_matches_statsmodels_style_monotone():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 15)
    adj = np.array(A.holm(p))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-15)
    assert np.all(adj >= p)


def test_wilcoxon_identical():
    r = A.paired_wilcoxon([1, 2, 3], [1, 2, 3])
    assert r["p"] == 1.0 and r["empates"] == 3


def test_wilcoxon_clear_difference():
    a = np.arange(20) + 1.0
    b = np.arange(20) * 1.0
    r = A.paired_wilcoxon(a, b)
    assert r["p"] < 0.001 and r["vitorias"] == 20 and r["mediana_dif"] == 1.0


def test_wilcoxon_shape_mismatch():
    with pytest.raises(ValueError):
        A.paired_wilcoxon([1, 2], [1])


def test_ttest_and_wtl():
    rng = np.random.default_rng(1)
    b = rng.normal(size=10)
    a = b + 1 + rng.normal(scale=0.01, size=10)
    assert A.wtl(a, b) == "+"
    assert A.wtl(b, a) == "-"
    assert A.wtl(b, b) == "="
