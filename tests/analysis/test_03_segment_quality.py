"""
test_03_segment_quality.py
EN: Invariants of metrics/03_segment_quality.json (technical report §3 derived segment): the resolution paths and
    the series × segment matrix cover every listing; the disagreement with the raw segment is the sum of its
    cross cells and its share is the ratio; the G segment's body and series counts add up to it; every listing
    outside the series map is resolved from the model name, and the families that span segments are among those.
    Key-agnostic about the path ids (they come from analysis/lib/segment_rule.py).
TR: metrics/03_segment_quality.json'un değişmezleri (teknik rapor §3 türetilen segment): çözüm yolları ve seri ×
    segment matrisi her ilanı kapsıyor; ham segmentle uyuşmazlık çapraz hücrelerinin toplamı ve payı oran; G
    segmentinin kasa ve seri sayımları ona toplanıyor; seri haritası dışındaki her ilan model adından çözülüyor ve
    birden çok segmente yayılan aileler bunların arasında. Yol kimliklerinden bağımsız (analysis/lib/segment_rule.py).
"""
import pytest


@pytest.fixture(scope="module")
def sq(metrics):
    """EN: The published block. / TR: Yayımlanan blok."""
    return metrics("03_segment_quality")["error_drivers"]["segment_quality"]


@pytest.fixture(scope="module")
def n_listings(metrics):
    """EN: Listings in the model (01_dedup_leakage). / TR: Modele giren ilan (01_dedup_leakage)."""
    return metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_paths_cover_listings(sq, metrics, n_listings):
    """EN: Paths and the matrix both sum to the listings. / TR: Yollar ve matris ilanlara toplanıyor."""
    assert sum(sq["paths"].values()) == n_listings
    assert sum(r[3] for r in metrics("03_segment_quality")["domain"]["series_segment_matrix"]) == n_listings
    assert sq["map_size"] > 0


def test_mismatch_adds_up(sq, n_listings):
    """EN: differs = Σ cross; pct = differs / raw known. / TR: farklı = Σ çapraz; pay = farklı / ham dolu."""
    mm = sq["mismatch"]
    assert sum(r[2] for r in mm["cross"]) == mm["differs"] <= mm["raw_known"] <= n_listings
    assert mm["pct"] == pytest.approx(round(100 * mm["differs"] / mm["raw_known"], 2))


def test_g_segment(sq):
    """EN: Body and series counts add up to G; MPV is its body count. / TR: Kasa ve seri sayımları G'ye toplanıyor."""
    g = sq["g_segment"]
    assert sum(n for _, n in g["body"]) == sum(n for _, n in g["series"]) == g["n"]
    assert g["mpv_n"] == dict(g["body"]).get("MPV", 0) <= g["n"]


def test_model_name_families(sq):
    """
    EN: Each family's count is the sum of its segments; together they are every listing outside the largest path
        (the series map); multi-segment series are among the families.
    TR: Her ailenin sayısı segmentlerinin toplamı; hepsi birlikte en büyük yol (seri haritası) dışındaki ilanlar;
        birden çok segmentli seriler aileler arasında.
    """
    fams = sq["from_model_name"]
    assert all(n == sum(c for _, c in segs) for _, n, segs in fams)
    assert sum(n for _, n, _ in fams) == sum(sq["paths"].values()) - max(sq["paths"].values())
    assert set(sq["multi_segment_series"]) <= {name for name, _, _ in fams}
