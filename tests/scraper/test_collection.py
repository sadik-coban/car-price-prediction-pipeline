"""
test_collection.py
EN: Tests of scraper/collection.py and collection_config.json — the values are the ones scraper/main.py and
    getlistofcars.py had written in code until 2026-09-24, and the list URL is character for character the one
    the scraper always sent. Only collection.py is imported (getlistofcars.py would load .env); no network.
TR: scraper/collection.py ve collection_config.json testleri — değerler scraper/main.py ile getlistofcars.py'de
    2026-09-24'e kadar kodda yazılı olanlar, liste adresi de scraper'ın hep gönderdiğiyle karakteri karakterine
    aynı. Yalnız collection.py import edilir (getlistofcars.py .env'i yüklerdi); ağ yok.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraper"))
import collection as C  # noqa: E402

BASE = "https://example.test"


def test_values_match_the_old_code():
    """EN: The config holds the values that were in code. / TR: Ayar dosyası kodda yazılı olan değerleri taşır."""
    assert C.CONFIG["brands"] == {"audi": {"min": 300000, "max": 6500000, "step": 100000},
                                  "bmw": {"min": 300000, "max": 6500000, "step": 100000},
                                  "DEFAULT": {"min": 200000, "max": 8000000, "step": 100000}}
    assert C.CONFIG["selected_brands"] == ["audi", "bmw"]
    assert C.CONFIG["not_selected"] == ["volkswagen", "skoda", "seat", "volvo", "mercedes-benz", "renault", "fiat"]
    assert C.CONFIG["listing_query"] == {"category": "otomobil", "fuels": ["Benzin", "Dizel", "Hibrit", "LPG"],
                                         "max_km": 700000, "min_year": 2005}


@pytest.mark.parametrize("brand, b_min, b_max", [("audi", 300000, 400000), ("bmw", 6400000, 6500000)])
def test_list_url_is_unchanged(brand, b_min, b_max):
    """
    EN: The URL equals the f-string getlistofcars.py used before 2026-09-24, written out here.
    TR: Adres, getlistofcars.py'nin 2026-09-24'ten önce kullandığı f-string'in açık yazılmış hâline eşit.
    """
    old = (f"{BASE}/ikinci-el/otomobil/{brand}?currency=TL&fuel=Benzin&fuel=Dizel&fuel=Hibrit&fuel=LPG"
           f"&maxPrice={b_max}&maxkm=700000&minPrice={b_min}&minYear=2005&page=")
    assert C.list_url(BASE, brand, b_min, b_max, 2005) == old
