"""
test_errors_lira.py
notebooks/lib/errors.py'nin ₺ hatası yardımcıları, elle yazılmış sayılarla: en büyük ₺ hataları büyükten küçüğe ve
eşitlikte satır sırasıyla; simetrik karşılaştırmada en büyük hatalar aynı ilanlarsa eksik tahmin sayısı yazı tura
gibi yarı yarıya, log hatası her ilanda simetrik dağılınca en büyük ₺ hatalarının yarıdan fazlası eksik tahmin, aynı
tohum aynı sonucu verir. Gerçek veri okunmaz.
"""
import numpy as np

from lib import errors


def test_top_lira_orders_by_lira_gap_and_keeps_row_order_on_ties():
    price = np.array([100.0, 100.0, 100.0, 100.0])
    pred = np.array([110.0, 70.0, 130.0, 90.0])                 # farklar 10, 30, 30, 10
    assert errors.top_lira(price, pred, 3).tolist() == [1, 2, 0]


def test_symmetric_under_counts():
    rng = np.random.default_rng(0)
    pred = np.full(3000, 1e6)
    price = pred * np.exp(np.where(np.arange(3000) < 100, 0.2, 0.01))             # 100 ilanın log hatası büyük
    counts = errors.symmetric_under_counts(price, pred, 100, 200, 1)
    assert 0 < counts.min() and counts.max() < 100 and abs(counts.mean() - 50) < 5     # işaret yazı tura
    pred = rng.uniform(5e5, 5e6, 3000)
    price = pred * np.exp(rng.normal(0, 0.1, 3000))
    counts = errors.symmetric_under_counts(price, pred, 100, 40, 1)
    assert len(counts) == 40 and counts.mean() > 50 and counts.max() <= 100        # aynı log hata ₺ olarak üstte büyük
    assert np.array_equal(counts, errors.symmetric_under_counts(price, pred, 100, 40, 1))
