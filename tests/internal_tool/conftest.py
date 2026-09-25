"""
conftest.py (tests/internal_tool)
EN: A tiny made-up data folder for the internal tool's tests, shaped like data/: raw/<brand>/<snapshot>/details.jsonl
    (a TR-plated ad seen in two snapshots, a blue-plate ad, one without a plate field, a failed page and a broken
    line) and cars.duckdb / cars_gold.duckdb with a small car_listings (silver keeps NULLs and
    kb_paint_change_summary; gold has NULL → false/0 and not that column). No real data is read.
TR: İç aracın testleri için data/ biçiminde küçük uydurma bir veri klasörü: raw/<marka>/<tarama>/details.jsonl (iki
    taramada görülen TR plakalı bir ilan, mavi plakalı bir ilan, plaka alanı olmayan bir ilan, başarısız bir sayfa ve
    bozuk bir satır) ile küçük bir car_listings'li cars.duckdb / cars_gold.duckdb (silver NULL'ları ve
    kb_paint_change_summary'yi tutar; gold'da NULL → false/0 ve o kolon yok). Gerçek veri okunmaz.
"""
import json

import duckdb
import pytest


def raw_record(ad, plate, year, seller, price, km, damage, text):
    """EN: One raw record like the scraper writes. / TR: Kazıyıcının yazdığı gibi tek ham kayıt."""
    rec = {"brand": "audi", "url": f"https://example.invalid/ilan/{ad}", "scraped_at": "2026-01-18T16:59:10+00:00",
           "search_date": "2026-01-18_19-56", "Fiyat": price, "Ilan_Basligi": f"İlan {ad}",
           "Konum": "Göztepe Mh. Bağcılar, İstanbul", "Aciklama_HTML": text, "Agir_Hasar": False,
           "Hasar_Listesi": damage, "KısaBilgi - İlan No": f"Kopyalandı\n\n      {ad}", "KısaBilgi - Seri": "A3",
           "KısaBilgi - Yıl": year, "KısaBilgi - Kilometre": km, "KısaBilgi - Kimden": seller,
           "Genel Bakış - Motor Hacmi": "1401 - 1600 cm3"}
    if plate is not None:
        rec["Genel Bakış - Plaka Uyruğu"] = plate
    return rec


RAW = {
    "audi/2026-01-18_19-56": [
        raw_record(1001, "(TR) Türkiye", "2024", "Galeriden", "1.250.000 TL", "12.000 km",
                   ["Tavan: Orjinal", "Motor Kaputu: Boyalı"], "<p>Temiz&nbsp;araç</p>"),
        raw_record(1002, "Mavi plakalı", "2024", "Galeriden", "2.000.000 TL", "5.000 km",
                   ["Tavan: Orjinal", "Motor Kaputu: Orjinal"], "<p>YABANCIDAN&nbsp;YABANCIYA. MA PLAKALIDIR.</p>"),
        {"brand": "audi", "url": "https://example.invalid/x", "scraped_at": "2026-01-18T17:00:00+00:00",
         "search_date": "2026-01-18_19-56", "error": "Read timed out"},
    ],
    "bmw/2026-03-21_22-18": [
        raw_record(1001, "(TR) Türkiye", "2024", "Galeriden", "1.200.000 TL", "13.000 km",
                   ["Tavan: Orjinal", "Motor Kaputu: Boyalı"], "<p>Temiz araç</p>"),
        raw_record(1003, None, "2019", "Sahibinden", "900.000 TL", "150.000 km",
                   ["Tavan: Değişmiş", "Motor Kaputu: Belirtilmemiş"], "<p>Sahibinden</p>"),
    ],
}

DB_SCHEMA = ("id BIGINT, ad_id BIGINT, brand VARCHAR, series VARCHAR, model VARCHAR, location VARCHAR, price DOUBLE, "
             "kb_year BIGINT, kb_mileage BIGINT, gb_plate_origin VARCHAR, is_heavy_damaged BOOLEAN, "
             "count_changed BIGINT, tavan_degisen BIGINT, tavan_boyali BIGINT, tavan_lokal BIGINT, url VARCHAR, "
             "description_text VARCHAR, scraped_at TIMESTAMPTZ, search_date DATE")
# EN: (id, ad, price, plate, date, heavy, changed, tavan flags) | TR: (id, ilan, fiyat, plaka, tarih, ağır, değişen, tavan)
DB_ROWS = [(1, 1, 1_000_000, "(TR) Türkiye", "2026-01-18", None, None, (None, None, None)),
           (2, 1, 1_100_000, "(TR) Türkiye", "2026-03-21", False, 0, (0, 0, 0)),
           (3, 2, 2_000_000, None, "2026-03-21", True, 1, (1, 0, 0)),
           (4, 3, 3_000_000, "(TR) Türkiye", "2026-01-18", None, 0, (0, 1, 0))]


def write_db(path, gold):
    """EN: A small car_listings DB, silver or gold shaped. / TR: Silver ya da gold biçimli küçük bir car_listings DB."""
    con = duckdb.connect(str(path))
    try:
        con.execute(f"CREATE TABLE car_listings ({DB_SCHEMA}{'' if gold else ', kb_paint_change_summary VARCHAR'})")
        for rid, ad, price, plate, date, heavy, changed, (d, b, lok) in DB_ROWS:
            if gold:
                heavy, changed, d, b, lok = (heavy or False), (changed or 0), (d or 0), (b or 0), (lok or 0)
            values = [rid, ad, "audi", "A3", "A3 Sportback", "Göztepe Mh. Bağcılar, İstanbul", price, 2020, 50_000,
                      plate, heavy, changed, d, b, lok, f"https://example.invalid/ilan/{ad}", f"Metin {rid}",
                      "2026-01-18 19:59:00+03", date] + ([] if gold else ["Tamamı orjinal"])
            con.execute(f"INSERT INTO car_listings VALUES ({', '.join('?' * len(values))})", values)
    finally:
        con.close()


@pytest.fixture(scope="session")
def tool_data(tmp_path_factory):
    """EN: The made-up data folder (built once). / TR: Uydurma veri klasörü (bir kez kurulur)."""
    root = tmp_path_factory.mktemp("tool_data")
    for rel, records in RAW.items():
        folder = root / "raw" / rel
        folder.mkdir(parents=True)
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        if rel.startswith("bmw"):
            lines.insert(1, "{broken json")
        (folder / "details.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_db(root / "cars.duckdb", gold=False)
    write_db(root / "cars_gold.duckdb", gold=True)
    return root
