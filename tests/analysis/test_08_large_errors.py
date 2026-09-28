"""
test_08_large_errors.py
EN: Invariants of metrics/08_large_errors.json (technical report §8 where the large errors come from): the
    overall counts add up (big = over + under, the share is the ratio); every breakdown covers all listings (by
    comparables, age, age each year, F/S segment, snapshot, live vs gone, per-model buckets); the per-model buckets
    are contiguous and hold every model; the lira ranking's top-N splits into under and over and its sub-counts fit
    in it; the spec-outlier shares are the ratios.
TR: metrics/08_large_errors.json'un değişmezleri (teknik rapor §8 büyük hataların kaynağı): genel sayımlar
    toplanıyor (büyük = fazla + düşük, pay oran); her kırılım bütün ilanları kapsıyor (emsal, yaş, tek tek yaş,
    F/S segment, tarama, canlı/kaybolan, model kovaları); model kovaları bitişik ve her modeli tutuyor; lira
    sıralamasının ilk N'i düşük ve fazla diye bölünüyor ve alt sayımları içine sığıyor; spec aykırı payları oranlar.
"""
import pytest


@pytest.fixture(scope="module")
def ed(metrics):
    """EN: The error_drivers section. / TR: error_drivers bölümü."""
    return metrics("08_large_errors")["error_drivers"]


@pytest.fixture(scope="module")
def n(metrics):
    """EN: Listings in the model. / TR: Modele giren ilan."""
    return metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_overall(ed, n):
    """EN: big = over + under; share = big / n. / TR: büyük = fazla + düşük; pay = büyük / n."""
    o = ed["overall"]
    assert o["n"] == n and o["n_big"] == o["n_over"] + o["n_under"]
    assert o["big_pct"] == pytest.approx(round(100 * o["n_big"] / n, 2))


def test_breakdowns_cover_listings(ed, n, metrics):
    """EN: Every breakdown sums to the listings. / TR: Her kırılım ilanlara toplanıyor."""
    assert sum(b["n"] for b in ed["by_model_year_n"]) == n
    assert sum(b["n"] for b in ed["by_snapshot"]) == n
    assert sum(r[1] for r in ed["age_sensitivity"]) == n
    assert sum(g["n"] for g in ed["by_age"].values()) == n
    assert sum(g["n"] for g in ed["by_segment_FS"].values()) == n
    live = ed["live_vs_gone"]
    assert live["live_n"] + live["gone_n"] == n
    assert live["last_snapshot"] == metrics("01_dedup_leakage")["meta"]["snapshots"][-1]


def test_per_model_buckets(ed, n):
    """
    EN: Buckets are contiguous and hold every model and listing of the per-model table.
    TR: Kovalar bitişik; model başına tablonun her modelini ve ilanını tutuyor.
    """
    b = ed["per_model_buckets"]
    assert all(x["hi"] + 1 == y["lo"] for x, y in zip(b, b[1:]))
    assert sum(x["n_models"] for x in b) == len(ed["per_model_error"])
    assert sum(x["n_listings"] for x in b) == sum(r[0] for r in ed["per_model_error"]) == n


def test_lira_ranking(ed):
    """
    EN: top-N = under + over; the top-quartile and performance-family counts fit in it; the symmetric-sign
        expectation is an ordered interval inside [0, N].
    TR: ilk N = düşük + fazla; en pahalı çeyrek ve performans ailesi sayımları içine sığıyor; simetrik işaret
        beklentisi [0, N] içinde sıralı bir aralık.
    """
    ls = ed["lira_scaled"]
    assert ls["top_n_under"] + ls["top_n_over"] == ls["top_n"]
    assert max(ls["top_n_q4"], ls["top_n_perf"]) <= ls["top_n"]
    null = ls["under_if_symmetric"]
    assert 0 <= null["lo"] <= null["mean"] <= null["hi"] <= ls["top_n"] and null["draws"] > 0


def test_spec_outliers(ed, n):
    """EN: Shares are count / listings. / TR: Paylar sayım / ilan."""
    so = ed["spec_outliers"]
    assert so["pct"] == pytest.approx(round(100 * so["n"] / n, 2))
    assert so["blind_spot"]["pct"] == pytest.approx(round(100 * so["blind_spot"]["listings"] / n, 2))


def test_example_compares(ed):
    """
    EN: Each example's comparison is computed on the data: its series holds it, the other listings of its name are
        its model count minus itself, and a named comparison group has listings and a positive median.
    TR: Her örneğin karşılaştırması veriden hesaplanır: serisi onu içerir, adının öteki ilanları model sayısı eksi
        kendisi, adı geçen karşılaştırma grubunun ilanı ve pozitif medyanı var.
    """
    assert len(ed["examples"]) == 3
    for e in ed["examples"]:
        c = e["compare"]
        assert c["series_n"] >= 1 and len(c["others"]) == e["n_model"] - 1
        if c["peer"] is not None:
            assert c["peer"]["n"] >= 1 and c["peer"]["median"] > 0

