"""
test_01_unspecified_panels.py
EN: Invariants of metrics/01_unspecified_panels.json (technical report §1, "Belirtilmemiş" panels): the counts
    fit inside each other (all 13 unspecified ≤ at least one ≤ matched listings ≤ listings, panels = 13 per matched
    listing); the published share is the ratio of the published counts; the panel structure the report quotes is
    self-consistent; and the share, read here from the raw JSONL, equals the mean of 02_missingness' per-flag
    unspecified shares, read from the database's NULLs — two sources, one number.
TR: metrics/01_unspecified_panels.json'un değişmezleri (teknik rapor §1, "Belirtilmemiş" paneller): sayımlar
    birbirinin içine sığıyor (13'ü de belirtilmemiş ≤ en az biri ≤ eşleşen ilan ≤ ilan, eşleşen ilan başına 13
    panel); yayımlanan pay, yayımlanan sayımların oranı; raporun andığı panel yapısı kendi içinde tutarlı; ve
    burada ham JSONL'den okunan pay, 02_missingness'in veritabanı NULL'larından okuduğu bayrak başına
    belirtilmemiş paylarının ortalamasına eşit — iki kaynak, tek sayı.
"""
import pytest


@pytest.fixture(scope="module")
def bl(metrics):
    """EN: The published block. / TR: Yayımlanan blok."""
    return metrics("01_unspecified_panels")["error_drivers"]["unspecified"]


def test_counts_nest(bl):
    """EN: all-unspecified ≤ at least one ≤ matched ≤ listings. / TR: hepsi ≤ en az biri ≤ eşleşen ≤ ilan."""
    assert 0 <= bl["all_unspecified"] <= bl["at_least_one"] <= bl["matched_listings"] <= bl["listings"]
    assert bl["duplicates_skipped"] >= 0


def test_panels_per_listing(bl):
    """EN: Every matched listing has every panel's status. / TR: Eşleşen her ilanın her panelinin durumu var."""
    assert bl["n_panels"] == bl["structure"]["panel"] * bl["matched_listings"]
    assert bl["unspecified_panels"] >= bl["all_unspecified"] * bl["structure"]["panel"]


def test_share_is_the_ratio(bl):
    """EN: belirtilmemis_pct = unspecified / all panels. / TR: belirtilmemis_pct = belirtilmemiş / bütün panel."""
    assert bl["unspecified_pct"] == pytest.approx(100 * bl["unspecified_panels"] / bl["n_panels"], abs=0.05)


def test_structure_is_consistent(bl):
    """
    EN: 3 flags per panel; single panels + grouped panels = all panels; five answers per panel.
    TR: Panel başına 3 bayrak; tek paneller + gruplu paneller = bütün paneller; panel başına beş cevap.
    """
    y = bl["structure"]
    assert y["flags"] == 3 * y["panel"]
    assert y["single_panels"] + sum(y["grouped_panels"].values()) == y["panel"]
    assert y["n_answers"] == 5


def test_share_matches_the_database(bl, metrics):
    """
    EN: The raw-JSONL share equals the mean of the database's per-flag NULL shares (same listings, 3 flags per
        panel, one denominator).
    TR: Ham JSONL payı, veritabanının bayrak başına NULL paylarının ortalamasına eşit (aynı ilanlar, panel başına 3
        bayrak, tek payda).
    """
    flags = [p for c, p in metrics("02_missingness")["methodology"]["sistematik_missing"]["belirtilmemis"]["kolonlar"]
             if c.endswith(("_degisen", "_boyali", "_lokal"))]
    assert len(flags) == bl["structure"]["flags"]
    assert sum(flags) / len(flags) == pytest.approx(bl["unspecified_pct"], abs=0.1)
