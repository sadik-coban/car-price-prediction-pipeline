"""
test_errors.py
notebooks/lib/errors.py elle yazılmış sayılarla: artık gerçek fiyatın yüzdesi, eksi işaret fazla tahmin; tahmin 0 ile
15 milyon ₺ arasına kırpılır; büyük hata iki haneye yuvarlanmış |artık| > %20, iki yönde de. Gerçek veri okunmaz.
"""
import numpy as np

from lib import errors


def test_residual_pct():
    price = np.array([100.0, 100.0, 100.0, 2e7])
    pred = np.array([90.0, 110.0, -5.0, 3e7])
    assert np.allclose(errors.residual_pct(price, pred), [10, -10, 100, 25])   # kırpma: -5 -> 0, 3e7 -> 1.5e7


def test_large_error_is_two_sided_after_rounding():
    resid = np.array([20.01, -20.01, 19.99, 20.0, 20.004, -20.004, 0.0])
    assert list(errors.large_error(resid)) == [True, True, False, False, False, False, False]
