"""
test_07_text_flag.py
EN: Invariants of metrics/07_text_flag.json (technical report §7 text flag, decision note): the odds ratio sits in
    its interval and the p-value is a probability; the flagged share is the ratio of the counts; the large-error
    rates of flagged and other listings average, weighted by size, to 08_large_errors' overall rate (same OOF,
    same threshold); the shares are percentages.
TR: metrics/07_text_flag.json'un değişmezleri (teknik rapor §7 metin bayrağı, karar notu): olasılık oranı
    aralığının içinde, p bir olasılık; bayraklı pay sayımların oranı; bayraklı ve öteki ilanların büyük hata
    oranlarının boyla ağırlıklı ortalaması 08_large_errors'ın genel oranı (aynı OOF, aynı eşik); paylar yüzde.
"""
import pytest


@pytest.fixture(scope="module")
def tf(metrics):
    """EN: The published block. / TR: Yayımlanan blok."""
    return metrics("07_text_flag")["error_drivers"]["text_flag"]


def test_odds_interval(tf):
    """EN: 0 < ci_lo ≤ OR ≤ ci_hi; p ∈ [0, 1]. / TR: 0 < ci_lo ≤ OR ≤ ci_hi; p ∈ [0, 1]."""
    c = tf["controlled"]
    assert 0 < c["ci_lo"] <= c["or"] <= c["ci_hi"] and 0 <= c["p"] <= 1
    assert 0 <= c["perf_share_pct"] <= 100


def test_share(tf, metrics):
    """EN: pct = flagged / listings. / TR: pay = bayraklı / ilan."""
    n = metrics("01_dedup_leakage")["meta"]["n_dedup"]
    assert tf["pct"] == pytest.approx(round(100 * tf["n"] / n, 2))


def test_rates_add_up_to_overall(tf, metrics):
    """
    EN: Size-weighted large-error rates of flagged and other listings = 08's overall rate (±0.1 pt, rounding).
    TR: Bayraklı ve öteki ilanların boyla ağırlıklı büyük hata oranı = 08'in genel oranı (±0,1 puan, yuvarlama).
    """
    overall = metrics("08_large_errors")["error_drivers"]["overall"]
    n = overall["n"]
    weighted = (tf["n"] * tf["big_pct"] + (n - tf["n"]) * tf["other_big_pct"]) / n
    assert weighted == pytest.approx(overall["big_pct"], abs=0.1)
    assert tf["median_error_pct"] > 0 and tf["other_median_error_pct"] > 0
