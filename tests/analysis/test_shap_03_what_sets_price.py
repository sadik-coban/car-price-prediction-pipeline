"""
test_shap_03_what_sets_price.py
EN: Invariants of metrics/shap/03_what_sets_price.json (SHAP report §3 what sets the price): one row per listing;
    the grouped table is sorted with shares summing to 100 and a typical effect for each of its groups; the five
    fold rankings are orderings of the same groups; the report's direction claims hold (age and mileage lower the
    price, power raises it); the mileage windows are ordered and the figures it lists exist.
TR: metrics/shap/03_what_sets_price.json'un değişmezleri (SHAP raporu §3 fiyatı ne belirliyor): ilan başına bir
    satır; gruplu tablo sıralı, payları 100'e toplanıyor ve her grubunun tipik etkisi var; beş fold sıralaması aynı
    grupların sıralamaları; raporun yön iddiaları tutuyor (yaş ve km fiyatı düşürür, güç yükseltir); km pencereleri
    sıralı ve andığı figürler var.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def shap(metrics):
    """EN: The shap section. / TR: shap bölümü."""
    return metrics("shap/03_what_sets_price")["shap"]


def test_table(shap, metrics):
    """EN: n = listings; sorted; shares sum to 100; a typical effect per group. / TR: Sıralı tablo, paylar 100."""
    t = shap["lightgbm_tfidf_svd"]
    assert shap["n"] == metrics("01_dedup_leakage")["meta"]["n_dedup"]
    assert [r[1] for r in t] == sorted((r[1] for r in t), reverse=True)
    assert sum(r[2] for r in t) == pytest.approx(100, abs=0.5)
    assert set(shap["typical"]) == {r[0] for r in t}


def test_fold_rankings(shap):
    """EN: Five orderings of the table's groups. / TR: Tablonun gruplarının beş sıralaması."""
    groups = {r[0] for r in shap["lightgbm_tfidf_svd"]}
    assert len(shap["fold_ranks"]) == 5 and all(set(f) == groups and len(f) == len(groups) for f in shap["fold_ranks"])


def test_directions(shap):
    """EN: Claim: age and mileage lower the price, power raises it. / TR: İddia: yaş ve km düşürür, güç yükseltir."""
    d = shap["direction"]
    assert d["vehicle_age"] < 0 and d["gb_mileage"] < 0 and d["power_hp_val"] > 0


def test_mileage_windows_and_figures(shap):
    """
    EN: Window centres rise; one rate between each pair of centres; the listed figures exist.
    TR: Pencere merkezleri artan; her iki merkez arasında bir oran; listelenen figürler var.
    """
    df = shap["dep_facts"]
    c = df["km_centers"]
    assert all(a < b for a, b in zip(c, c[1:])) and len(df["km_rates"]) == len(c) - 1
    assert all((ROOT / "reports" / "figures" / f).exists() for f in shap["figures"])
