"""
test_08_residuals.py
EN: Invariants of metrics/08_residuals.json (technical report §8): every figure and table covers exactly the
    listings in the model (01's unique listings); the R² of the predicted-vs-actual figure is the LightGBM OOF R²
    of 07 (both are the same OOF predictions); the error bands and the quartile groups partition the listings.
TR: metrics/08_residuals.json'un değişmezleri (teknik rapor §8): her figür ve tablo tam olarak modele giren
    ilanları kapsıyor (01'in tekil ilanları); tahmin-gerçek figürünün R²'si 07'deki LightGBM OOF R²'si (ikisi de
    aynı OOF tahminleri); hata bantları ve çeyrek grupları ilanları bölüyor.
"""
import pytest


@pytest.fixture(scope="module")
def res(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("08_residuals")


@pytest.fixture(scope="module")
def n_listings(metrics):
    """EN: Listings in the model (01). / TR: Modele giren ilan (01)."""
    return metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_every_count_is_the_listings(res, n_listings):
    """EN: n of each figure = its points = error table n = listings. / TR: Her figürün n'i = noktaları = ilan."""
    d = res["domain"]
    assert d["pred_vs_true"]["n"] == len(d["pred_vs_true"]["points"]) == n_listings
    assert d["residual_scatter"]["n"] == len(d["residual_scatter"]["points"]) == n_listings
    assert res["report"]["err_n"] == n_listings


def test_r2_is_the_model_comparison_r2(res, metrics):
    """EN: Same OOF predictions → same R². / TR: Aynı OOF tahminleri → aynı R²."""
    assert res["domain"]["pred_vs_true"]["r2"] == metrics("07_model_comparison")["domain"]["model_compare"]["lightgbm"]["R2"]


def test_error_bands_partition_the_listings(res):
    """
    EN: The |error| bands are contiguous, their counts add up to n and their shares to 100%.
    TR: |hata| bantları bitişik, sayıları n'e, payları %100'e toplanıyor.
    """
    bands = res["report"]["err_bands"]
    assert all(a[1] == b[0] for a, b in zip(bands, bands[1:])) and bands[-1][1] is None
    assert sum(b[2] for b in bands) == res["report"]["err_n"]
    assert sum(b[3] for b in bands) == pytest.approx(100, abs=0.01)
    assert res["report"]["err_over20"] + res["report"]["err_under20"] == bands[-1][2]


def test_quartile_groups_partition_the_listings(res, n_listings):
    """EN: Price and predicted-price quartiles each hold every listing once. / TR: Her çeyrek grubu her ilanı bir kez tutar."""
    ed = res["error_drivers"]
    assert sum(r[1] for r in ed["lira_quartile"]) == n_listings
    assert sum(r[1] for r in ed["pred_quartile"]) == n_listings
