"""
test_01_dedup_leakage.py
EN: Invariants of metrics/01_dedup_leakage.json (technical report §1: rows → listings, price changes, plate scope,
    collection filters, content duplicates): the counts fit inside each other and add up (brands and fuels to the
    listings, cuts + rises + returned to the changed ones), the TR plate row is the training set, the data sits
    inside the collection filters (max age = last snapshot year − first model year), and the duplicate definitions
    nest (strict ≤ loose, strict ≤ no-price) with shares that are the ratio of the counts.
TR: metrics/01_dedup_leakage.json'un değişmezleri (teknik rapor §1: satır → ilan, fiyat değişimi, plaka kapsamı,
    toplama filtreleri, içerik tekrarları): sayımlar birbirinin içine sığar ve toplanır (markalar ve yakıtlar
    ilanlara, indirim + zam + dönüş değişenlere), TR plaka satırı eğitim kümesidir, veri toplama filtrelerinin
    içindedir (en büyük yaş = son tarama yılı − ilk model yılı) ve tekrar tanımları iç içedir (katı ≤ gevşek,
    katı ≤ fiyatsız), paylar sayımların oranıdır.
"""
import pytest


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("01_dedup_leakage")


def test_rows_and_listings(doc):
    """
    EN: rows ≥ listings; brands add up to the listings; snapshots sorted and unique.
    TR: satır ≥ ilan; markalar ilanlara toplanır; taramalar sıralı ve tekil.
    """
    meta = doc["meta"]
    assert meta["n_raw"] >= meta["n_dedup"] > 0
    assert sum(meta["brands"].values()) == meta["n_dedup"]
    assert meta["snapshots"] == sorted(set(meta["snapshots"]))


def test_price_changes_add_up(doc):
    """
    EN: cuts + rises + returned = changed ≤ seen again ≤ listings; shares are the ratios.
    TR: indirim + zam + dönüş = değişen ≤ yeniden görülen ≤ ilan; paylar oranlardır.
    """
    pc = doc["error_drivers"]["price_changes"]
    assert pc["cuts"] + pc["rises"] + pc["returned"] == pc["changed"] <= pc["seen_again"] <= pc["listings"]
    assert pc["listings"] == doc["meta"]["n_dedup"]
    assert pc["seen_again_pct"] == round(100 * pc["seen_again"] / pc["listings"], 1)
    assert pc["changed_pct"] == round(100 * pc["changed"] / pc["listings"], 1)


def test_plates_match_training(doc):
    """
    EN: The TR plate row holds every snapshot row and every listing, and that is the training set; the dropped
        listings are the unknown-plate ones never seen with a TR plate.
    TR: TR plaka satırı her tarama satırını ve her ilanı tutar, eğitim kümesi de odur; atılanlar hiç TR plakayla
        görülmemiş bilinmeyen plakalı ilanlardır.
    """
    ps, meta = doc["error_drivers"]["plate_scope"], doc["meta"]
    tr = next(r for r in ps["distribution"] if r["plate"] == "(TR) Türkiye")
    assert tr["rows"] == meta["n_raw"] and tr["listings"] == ps["in_training"] == meta["n_dedup"]
    other = sum(r["listings"] for r in ps["distribution"] if r is not tr)
    assert ps["dropped_listings"] == other - ps["listings_with_both"] >= 0


def test_filters_fit_the_data(doc):
    """
    EN: The data sits inside the collection filters: price and km under their caps, no year before the first,
        no electric car, fuels add up to the listings, max age = last snapshot year − first model year.
    TR: Veri toplama filtrelerinin içinde: fiyat ve km tavanın altında, ilk yıldan eski yıl yok, elektrikli yok,
        yakıtlar ilanlara toplanır, en büyük yaş = son tarama yılı − ilk model yılı.
    """
    sc, meta = doc["error_drivers"]["scope"], doc["meta"]
    me = sc["measured"]
    assert sc["price_min"] < me["price_max"] <= sc["price_max"]
    assert 0 <= me["at_cap"] <= me["within_1pct_of_cap"]
    assert me["km_max"] <= sc["max_km"]
    assert me["year_min"] >= sc["min_year"] and me["in_min_year"] >= 0
    assert me["electric"] == 0
    assert sum(me["fuel"].values()) == meta["n_dedup"]
    assert sc["max_age"] == int(meta["snapshots"][-1][:4]) - sc["min_year"]


def test_content_duplicates_nest(doc):
    """
    EN: strict ≤ loose, strict ≤ no-price, non-round-km ≤ no-price, groups ≤ strict extras; each share is
        count / listings; at most five most-repeated rows, sorted, each repeated at least twice.
    TR: katı ≤ gevşek, katı ≤ fiyatsız, yuvarlak olmayan km ≤ fiyatsız, grup ≤ katı fazla; her pay sayım / ilan;
        en çok beş en sık tekrar satırı, sıralı, her biri en az iki kez.
    """
    cd, n = doc["methodology"]["content_duplicates"], doc["meta"]["n_dedup"]
    assert cd["strict_extra"] <= cd["loose_extra"] and cd["strict_extra"] <= cd["no_price_extra"]
    assert cd["no_price_non_round_km_extra"] <= cd["no_price_extra"]
    assert 0 < cd["n_duplicate_groups"] <= cd["strict_extra"]
    for k in ("strict", "loose", "no_price"):
        assert cd[f"{k}_pct"] == pytest.approx(round(100 * cd[f"{k}_extra"] / n, 2))
    reps = [r["repeats"] for r in cd["most_repeated"]]
    assert len(reps) <= 5 and reps == sorted(reps, reverse=True) and min(reps) >= 2
