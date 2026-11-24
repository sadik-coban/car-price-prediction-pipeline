"""
test_association.py
notebooks/lib/association.py: entropi ve Theil's U küçük, elle kurulmuş kolonlarla: kendisiyle 1, bağımsızla 0, yönlü,
elle hesaplanmış bir değerle aynı, tek değerli kolonda 0. Gerçek veri okunmaz.
"""
import math

import pandas as pd
import pytest

from lib import association


def test_entropy():
    assert association.entropy(pd.Series(["a", "b"])) == pytest.approx(math.log(2))
    assert association.entropy(pd.Series(["a", "a", "a"])) == 0


def test_theils_u_bounds():
    a = pd.Series(["x", "y", "x", "y"])
    assert association.theils_u(a, a) == pytest.approx(1)
    independent = pd.Series(["p", "p", "q", "q"])                  # a'nın her değeri b'nin her değeriyle bir kez
    assert association.theils_u(a, independent) == pytest.approx(0)
    assert association.theils_u(pd.Series(["x"] * 4), a) == 0       # tek değerli kolon


def test_theils_u_is_directional():
    # araç modeli seriyi belirler, seri araç modelini belirlemez
    model = pd.Series(["320i", "330i", "A4 2.0", "A4 1.4"])
    series = pd.Series(["3 Serisi", "3 Serisi", "A4", "A4"])
    assert association.theils_u(series, model) == pytest.approx(1)
    # U(model | seri) = (H(model) - H(model | seri)) / H(model) = (log 4 - log 2) / log 4 = 0.5
    assert association.theils_u(model, series) == pytest.approx(0.5)
