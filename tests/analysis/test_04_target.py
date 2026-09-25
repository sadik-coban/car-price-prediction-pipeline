"""
test_04_target.py
EN: Invariants of metrics/04_target.json (technical report §4 target): the quantiles are ordered, the histogram's
    bins rise and cover every listing, and the report's claim holds — log price is far less skewed than raw price.
TR: metrics/04_target.json'un değişmezleri (teknik rapor §4 hedef): yüzdelikler sıralı, histogram kovaları artan ve
    her ilanı kapsıyor, raporun iddiası tutuyor — log fiyat ham fiyattan çok daha az çarpık.
"""
import pytest


@pytest.fixture(scope="module")
def dom(metrics):
    """EN: The domain section. / TR: Domain bölümü."""
    return metrics("04_target")["domain"]


def test_quantiles_ordered(dom):
    """EN: 0 < P10 ≤ median ≤ P90. / TR: 0 < P10 ≤ medyan ≤ P90."""
    pd_ = dom["price_dist"]
    assert 0 < pd_["p10"] <= pd_["median"] <= pd_["p90"]


def test_histogram_covers_listings(dom, metrics):
    """EN: Bin edges rise, counts are ≥ 0 and sum to the listings. / TR: Kova sınırları artan, sayılar ilanlara toplanır."""
    hist = dom["price_histogram"]
    assert all(a[0] < b[0] for a, b in zip(hist, hist[1:]))
    assert all(n >= 0 for _, n in hist)
    assert sum(n for _, n in hist) == metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_log_is_less_skewed(dom):
    """EN: Claim: log price is less skewed and nearly symmetric. / TR: İddia: log fiyat daha az çarpık, neredeyse simetrik."""
    pd_ = dom["price_dist"]
    assert abs(pd_["skew_log"]) < 1 < pd_["skew_raw"] and pd_["skew_log"] < pd_["skew_raw"]
