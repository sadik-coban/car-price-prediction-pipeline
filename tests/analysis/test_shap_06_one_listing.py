"""
test_shap_06_one_listing.py
EN: Invariants of metrics/shap/06_one_listing.json (SHAP report §6, one listing explained): the waterfall's items
    are 02_oof_shap's groups; the visible bars plus the folded rest are all items and fit the chart; the top rows
    and the rest cover every item; the listing has a price and a prediction; labels come in both languages; the
    figures it lists exist.
TR: metrics/shap/06_one_listing.json'un değişmezleri (SHAP raporu §6, tek ilanın açıklaması): şelalenin öğeleri
    02_oof_shap'ın grupları; görünen çubuklar ile katlanan kalan bütün öğeler ve grafiğe sığıyor; üst satırlar ile
    kalan her öğeyi kapsıyor; ilanın fiyatı ve tahmini var; etiketler iki dilde; andığı figürler var.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def case(metrics):
    """EN: The shap_case section. / TR: shap_case bölümü."""
    return metrics("shap/06_one_listing")["shap_case"]


def test_items(case, metrics):
    """
    EN: items = 02's groups; visible + other = items; visible fits the chart; top + rest = items.
    TR: öğe = 02'nin grupları; görünen + öteki = öğe; görünen grafiğe sığıyor; üst + kalan = öğe.
    """
    assert case["n_items"] == len(metrics("shap/02_oof_shap")["oof_shap"]["groups"])
    assert len(case["visible"]) + case["n_other"] == case["n_items"]
    assert len(case["visible"]) <= case["wf_max"]
    assert len(case["top"]) + case["n_rest"] == case["n_items"]


def test_listing_and_labels(case):
    """EN: Price and prediction > 0; bilingual labels. / TR: Fiyat ve tahmin > 0; iki dilli etiketler."""
    assert case["price"] > 0 and case["pred"] > 0
    assert all(set(r[2]) == {"tr", "en"} for r in case["top"])
    assert set(case["km_label"]) == {"tr", "en"}


def test_figures_exist(case):
    """EN: The listed figures exist. / TR: Listelenen figürler var."""
    assert all((ROOT / "reports" / "figures" / f).exists() for f in case["figures"])
