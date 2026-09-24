"""
collection.py
EN: What the scraper collects, read from collection_config.json (per-brand price ranges, selected brands, the
    listing search query), and the list-page URL built from it. No network, no .env and nothing else happens
    at import, so tests and the analysis can use it (getlistofcars.py loads .env when imported).
TR: Scraper'ın ne topladığı, collection_config.json'dan okunur (marka başına fiyat aralıkları, seçili markalar,
    ilan arama sorgusu) ve ondan kurulan liste sayfası adresi. Import'ta ağ yok, .env yok, başka hiçbir şey
    olmaz; testler ve analiz kullanabilir (getlistofcars.py import edilince .env'i yükler).
"""
import json
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("collection_config.json").read_text(encoding="utf-8"))


def list_url(base_url, brand, b_min, b_max, min_year):
    """
    EN: The search URL of one price block, without the page number (it is appended by the caller).
        Parameter order is the site's, as the scraper always sent it.
    TR: Tek fiyat bloğunun arama adresi, sayfa numarası olmadan (çağıran sona ekler). Parametre sırası
        sitenin sırası; scraper hep böyle gönderdi.
    """
    q = CONFIG["listing_query"]
    fuels = "".join(f"&fuel={f}" for f in q["fuels"])
    return (f"{base_url}/ikinci-el/{q['category']}/{brand}?currency=TL{fuels}"
            f"&maxPrice={b_max}&maxkm={q['max_km']}&minPrice={b_min}&minYear={min_year}&page=")
