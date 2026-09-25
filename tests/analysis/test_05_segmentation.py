"""
test_05_segmentation.py
EN: Invariants of metrics/05_segmentation.json (technical report §5 market structure): the two brands add up to the
    listings and to meta.brands; the price tables cover the listings (segments fully, mileage with its cut-off,
    body and age within them); the clusters are 0..k−1 for the chosen k and cover every PCA point; the k search
    holds the chosen k; the PCA axes are ordered and their loadings sorted by size.
TR: metrics/05_segmentation.json'un değişmezleri (teknik rapor §5 piyasa yapısı): iki marka ilanlara ve
    meta.brands'e toplanıyor; fiyat tabloları ilanları kapsıyor (segmentler tamamen, km kesimiyle birlikte, kasa ve
    yaş içinde); kümeler seçilen k için 0..k−1 ve her PCA noktasını kapsıyor; k araması seçilen k'yi içeriyor; PCA
    eksenleri sıralı, yükleri büyüklüğe göre dizili.
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


def test_clusters(doc, n):
    """
    EN: Clusters 0..k−1 for the chosen k, their sizes cover every PCA point, names unique, three distinguishing axes.
    TR: Seçilen k için 0..k−1 kümeler, boyları her PCA noktasını kapsıyor, adlar tekil, üç ayırt edici eksen.
    """
    d, k = doc["domain"], doc["methodology"]["kmeans_selection"]["chosen_k"]
    km = d["kmeans"]
    assert [c["cluster"] for c in km] == list(range(k))
    assert sum(c["n"] for c in km) == len(d["pca_scatter"]) == len(d["pca_scatter_13"]) == n
    assert {p[2] for p in d["pca_scatter"]} <= set(range(k))
    assert len({c["name"] for c in km}) == k
    for c in km:
        assert 0 <= c["heavy_damage_pct"] <= 100 and len(c["distinguishing"]) == 3
        assert all(s in ("+", "-") for _, s in c["distinguishing"])


def test_k_selection(doc):
    """EN: Elbow and silhouette cover the same k, including the chosen one. / TR: Aynı k'ler, seçilen dahil."""
    ks = doc["methodology"]["kmeans_selection"]
    ks_elbow = [k for k, _ in ks["elbow"]]
    assert ks_elbow == [k for k, _ in ks["silhouette"]] and ks["chosen_k"] in ks_elbow
    assert all(-1 <= s <= 1 for _, s in ks["silhouette"])


def test_pca_axes(doc):
    """
    EN: Explained variance falls from axis to axis and sums to at most 100; loadings sorted by |w|.
    TR: Açıklanan varyans eksenden eksene düşüyor, toplamı en çok 100; yükler |w|'ye göre sıralı.
    """
    axes = doc["methodology"]["pca_axes"]
    var = [a["var_pct"] for a in axes]
    assert var == sorted(var, reverse=True) and sum(var) <= 100
    for a in axes:
        w = [abs(x) for _, x in a["top"]]
        assert w == sorted(w, reverse=True)
