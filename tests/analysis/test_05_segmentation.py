"""
test_05_segmentation.py
EN: Invariants of metrics/05_segmentation.json (technical report §5 market structure): the two brands add up to the
    listings and to meta.brands; the price tables cover the listings (segments fully, mileage with its cut-off,
    body and age within them); the silhouette scan covers consecutive k with scores in [−1, 1].
TR: metrics/05_segmentation.json'un değişmezleri (teknik rapor §5 piyasa yapısı): iki marka ilanlara ve
    meta.brands'e toplanıyor; fiyat tabloları ilanları kapsıyor (segmentler tamamen, km kesimiyle birlikte, kasa ve
    yaş içinde); silhouette taraması ardışık k'leri [−1, 1] skorlarla kapsıyor.
"""
import pytest


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("05_segmentation")


@pytest.fixture(scope="module")
def n(metrics):
    """EN: Listings in the model. / TR: Modele giren ilan."""
    return metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_brands_add_up(doc, metrics, n):
    """EN: bmw + audi = listings = meta.brands; δ ∈ [−1, 1], p ∈ [0, 1]. / TR: bmw + audi = ilan = meta.brands."""
    bc = doc["domain"]["brand_compare"]
    assert bc["bmw_n"] + bc["audi_n"] == n
    assert {"bmw": bc["bmw_n"], "audi": bc["audi_n"]} == metrics("01_dedup_leakage")["meta"]["brands"]
    assert -1 <= bc["cliffs_delta"] <= 1 and 0 <= bc["mwu_p"] <= 1
    assert bc["bmw_median"] > 0 and bc["audi_median"] > 0


def test_price_tables_cover_listings(doc, n):
    """
    EN: Segments sum to the listings (cheapest first); mileage bins + the cut-off's outside = listings; body and
        age tables stay within them.
    TR: Segmentler ilanlara toplanıyor (en ucuz önce); km kovaları + kesim dışı = ilan; kasa ve yaş tabloları içinde.
    """
    d = doc["domain"]
    ladder = d["segment_ladder"]
    assert sum(r[2] for r in ladder) == n and [r[1] for r in ladder] == sorted(r[1] for r in ladder)
    assert sum(r[2] for r in d["km_price"]) + d["km_price_scope"]["outside"] == n
    assert sum(r[2] for r in d["body_median"]) <= n and sum(r[2] for r in d["age_depreciation"]) <= n


def test_k_selection(doc):
    """
    EN: The silhouette scan covers consecutive k from 2, each score in [−1, 1].
    TR: Silhouette taraması 2'den başlayan ardışık k'leri kapsıyor, her skor [−1, 1].
    """
    sil = doc["methodology"]["kmeans_selection"]["silhouette"]
    assert [k for k, _ in sil] == list(range(2, 2 + len(sil))) and len(sil) >= 2
    assert all(-1 <= s <= 1 for _, s in sil)
