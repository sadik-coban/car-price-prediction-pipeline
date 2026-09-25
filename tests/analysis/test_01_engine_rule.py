"""
test_01_engine_rule.py
EN: Invariants of metrics/01_engine_rule.json (technical report §1, engine size and power from a range to one
    number; figures 29 and 30), for both measures: exact + range + no-value listings = all listings; the published
    (lower, upper) pairs add up to those counts (exact on the diagonal, ranges above it); the listings with a
    reference are a subset of the closed ranges; the ranges the report quotes are real pairs; the chosen candidate
    has the smallest median gap; and the position quartiles are ordered.
TR: metrics/01_engine_rule.json'un değişmezleri (teknik rapor §1, motor hacmi ve gücü aralıktan tek sayıya; Figür
    29 ve 30), iki ölçü için: kesin + aralıklı + değersiz ilan = bütün ilanlar; yayımlanan (alt, üst) çiftleri bu
    sayımlara toplanıyor (kesin köşegende, aralık üstünde); referanslı ilanlar kapalı aralıkların alt kümesi;
    raporun andığı aralıklar gerçek çiftler; seçilen aday en küçük medyan farkını veriyor; konum çeyrekleri sıralı.
"""
import pytest

MEASURES = ["engine_cc", "power_hp"]


@pytest.fixture(scope="module")
def rule(metrics):
    """EN: The published block. / TR: Yayımlanan blok."""
    return metrics("01_engine_rule")["error_drivers"]["engine_rule"]


def ranges(m):
    """EN: {(low, up): n} of the range pairs. / TR: Aralık çiftlerinin {(alt, üst): n} sözlüğü."""
    return {(lo, up): n for lo, up, is_range, n in m["pairs"] if is_range}


@pytest.mark.parametrize("p", MEASURES)
def test_listings_split(rule, p):
    """EN: exact + range + no value = listings. / TR: kesin + aralıklı + değersiz = ilan."""
    m = rule[p]
    assert m["exact"] + m["ranged"] + m["no_value"] == m["listings"]
    assert 0 <= m["open_ended"] <= m["ranged"]


@pytest.mark.parametrize("p", MEASURES)
def test_pairs_add_up(rule, p):
    """
    EN: Exact pairs sit on the diagonal and sum to the exact count; range pairs sit above it and sum to the closed
        ranges.
    TR: Kesin çiftler köşegende ve kesin sayısına toplanıyor; aralık çiftleri köşegenin üstünde ve kapalı aralıklara
        toplanıyor.
    """
    m = rule[p]
    exact = [r for r in m["pairs"] if not r[2]]
    rng = [r for r in m["pairs"] if r[2]]
    assert all(lo == up for lo, up, _, _ in exact)
    assert all(lo < up for lo, up, _, _ in rng)
    assert sum(r[3] for r in exact) == m["exact"]
    assert sum(r[3] for r in rng) == m["ranged"] - m["open_ended"]
    assert len({(lo, up, r) for lo, up, r, _ in m["pairs"]}) == len(m["pairs"])


@pytest.mark.parametrize("p", MEASURES)
def test_reference_is_a_subset(rule, p):
    """EN: Listings with a reference ⊆ closed ranges. / TR: Referanslı ilanlar ⊆ kapalı aralıklar."""
    m = rule[p]
    assert 0 < m["with_reference"] <= m["ranged"] - m["open_ended"]
    assert m["reference_models"] > 0


@pytest.mark.parametrize("p", MEASURES)
def test_quoted_ranges_are_pairs(rule, p):
    """
    EN: The example range and every per-range winner are published range pairs with at least that many listings.
    TR: Örnek aralık ve aralık başına her kazanan, en az o kadar ilanlı yayımlanmış aralık çiftleri.
    """
    m, rg = rule[p], ranges(rule[p])
    ok = m["example_range"]
    assert rg[(ok["low"], ok["up"])] >= ok["n"]
    for lo, up, n, _, _ in m["range_winners"]:
        assert rg[(lo, up)] >= n >= 100


@pytest.mark.parametrize("p", MEASURES)
def test_chosen_is_best(rule, p):
    """
    EN: The chosen candidate has the strictly smallest median gap; percentiles rise and the 50th is the median.
    TR: Seçilen adayın medyan farkı kesin olarak en küçük; yüzdelikler artıyor ve 50. yüzdelik medyan.
    """
    m = rule[p]
    med = {k: v["median"] for k, v in m["candidates"].items()}
    assert set(med) == {"lower", "mid", "upper"}
    assert all(med[m["chosen"]] < v for k, v in med.items() if k != m["chosen"])
    for v in m["candidates"].values():
        assert v["percentiles"] == sorted(v["percentiles"])
        assert v["percentiles"][2] == v["median"]


@pytest.mark.parametrize("p", MEASURES)
def test_position_is_ordered(rule, p):
    """EN: lower quartile ≤ median ≤ upper quartile. / TR: alt çeyrek ≤ medyan ≤ üst çeyrek."""
    m = rule[p]
    q1, q3 = m["position_quartiles"]
    assert q1 <= m["position_median"] <= q3
    assert 0 <= m["inside_pct"] <= 100
