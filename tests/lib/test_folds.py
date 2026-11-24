"""
test_folds.py
notebooks/lib/folds.py: ortak 5 kat satırları tam bir kez doğrulamaya koyar, her koşumda aynı çıkar ve karıştırılmış
KFold (tohum 42) ile aynıdır. Gerçek veri okunmaz.
"""
import numpy as np
from sklearn.model_selection import KFold

from lib import folds


def test_folds_partition_the_rows():
    pairs = folds.make_folds(23)
    assert len(pairs) == folds.N_FOLDS
    valid = np.concatenate([v for _, v in pairs])
    assert sorted(valid) == list(range(23))                        # her satır tam bir kez doğrulamada
    for train, v in pairs:
        assert sorted(np.concatenate([train, v])) == list(range(23)) and not set(train) & set(v)


def test_folds_are_fixed_and_shuffled():
    a, b = folds.make_folds(100), folds.make_folds(100)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))   # her koşumda aynı
    assert not np.array_equal(a[0][1], np.arange(20))              # karıştırılmış, ardışık değil
    expected = KFold(5, shuffle=True, random_state=42).split(np.arange(100))
    assert all(np.array_equal(x[1], e[1]) for x, e in zip(a, expected))
