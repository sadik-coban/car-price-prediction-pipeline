"""
test_market.py
notebooks/lib/market.py: Cliff's delta bütün çiftlerin kaba sayımıyla aynı (eşitlikler dahil), ayrık gruplarda ±1,
aynı gruplarda 0. Gerçek veri okunmaz.
"""
import numpy as np
import pytest

from lib import market


def brute_force(a, b):
    """Bütün (a, b) çiftleri tek tek: P(a > b) - P(a < b)."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return ((a > b).sum() - (a < b).sum()) / (a.size * b.size)


def test_cliffs_delta_equals_counting_every_pair():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 20, 300), rng.integers(3, 25, 200)       # çok sayıda eşit değer
    assert market.cliffs_delta(a, b) == pytest.approx(brute_force(a, b), abs=1e-12)


def test_cliffs_delta_extremes():
    assert market.cliffs_delta([5, 6, 7], [1, 2]) == 1
    assert market.cliffs_delta([1, 2], [5, 6, 7]) == -1
    assert market.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0
