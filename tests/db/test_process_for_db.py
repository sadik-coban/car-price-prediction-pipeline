"""
test_process_for_db.py
EN: Tests of db/lib/process_for_db.py — every parser on the formats that occur in the real data (all
    records were counted; no invented format is tested), the semi-raw rules (unknown stays None), the
    description heading, the damage diagram, whole records and reading a JSONL file. Records are fake.
TR: db/lib/process_for_db.py testleri — her ayrıştırıcı gerçek veride görülen biçimlerle (bütün kayıtlar
    sayıldı; uydurma biçim sınanmaz), yarı ham kurallar (bilinmeyen None kalır), açıklama başlığı, hasar şeması,
    tam kayıtlar ve JSONL dosyası okuma. Kayıtlar sahte.
"""
import json
import unicodedata
from datetime import date, datetime, timezone

import pytest
from conftest import MISSING, damage_list, raw_record

from lib import process_for_db as P

NONE4 = (None, None, None, None)


@pytest.mark.parametrize("text, expected", [
    ("1595 cc", (1595, None, None, False)),
    ("1401 - 1600 cm3", (None, 1401, 1600, True)),
    ("1200 cm3' e kadar", (None, None, 1200, True)),
    ("-", NONE4),
    (None, NONE4),
])
def test_engine_cc(text, expected):
    """EN: The engine-volume formats of the real data. / TR: Gerçek verideki motor hacmi biçimleri."""
    assert P._engine_cc(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("102 hp", (102, None, None, False)),
    ("101 - 125 HP", (None, 101, 125, True)),
    ("50 HP'ye kadar", (None, None, 50, True)),
    ("601 HP ve üzeri", (None, 601, None, True)),
    ("-", NONE4),
    (None, NONE4),
])
def test_power_hp(text, expected):
    """EN: The engine-power formats of the real data. / TR: Gerçek verideki motor gücü biçimleri."""
    assert P._power_hp(text) == expected


@pytest.mark.parametrize("text, expected", [("Evet", True), ("Hayır", False), ("Belirtilmemiş", None), (None, None)])
def test_yes_no(text, expected):
    """EN: Unknown stays unknown. / TR: Bilinmeyen bilinmeyen kalır."""
    assert P._yes_no(text) is expected


@pytest.mark.parametrize("text, expected", [("İlk Sahibiyim", True), ("İlk Sahibi Değilim", False),
                                            ("-", None), (None, None)])
def test_first_owner(text, expected):
    """EN: A missing answer is None, not "no". / TR: Cevap yoksa None, "hayır" değil."""
    assert P._bool_first_owner(text) is expected


@pytest.mark.parametrize("text, expected", [("Takasa Uygun", True), ("Takasa Uygun Değil", False),
                                            ("-", None), (None, None)])
def test_trade(text, expected):
    """EN: "Değil" means no; missing is None. / TR: "Değil" hayır demek; yoksa None."""
    assert P._bool_takasa(text) is expected


@pytest.mark.parametrize("text, expected", [("365.000 km", 365000), ("5 km", 5), ("350 nm", 350),
                                            ("220 km/s", 220), ("4000 rpm", 4000), ("1.234 TL", 1234),
                                            ("4", 4), ("-", None), (None, None)])
def test_int_with_unit(text, expected):
    """EN: Single values with a unit; the dot is a thousands separator. / TR: Birimli tek değer; nokta binlik."""
    assert P._int_with_unit(text) == expected


@pytest.mark.parametrize("text, expected", [("6,5 lt", 6.5), ("7 lt", 7.0), ("8,2 sn", 8.2), ("9 sn", 9.0),
                                            (None, None)])
def test_decimal_with_unit(text, expected):
    """EN: The comma is the decimal separator. / TR: Virgül ondalık ayırıcı."""
    assert P._decimal_with_unit(text) == expected


def test_ad_id_price_year():
    """EN: Ad number after the button text, price, year. / TR: Düğme metninden sonra ilan no, fiyat, yıl."""
    assert P._ad_id("Kopyalandı\n\r\n                                   10000001") == 10000001
    assert P._price_tl("1.250.000 TL") == 1250000 and P._price_tl("950.000 TL") == 950000
    assert P._year("2015") == 2015


def test_dates():
    """EN: Listing date (Turkish month), snapshot name, fetch time. / TR: İlan tarihi, tarama adı, çekim anı."""
    assert P._listing_date("26 Kasım 2025") == date(2025, 11, 26)
    assert P._listing_date("3 Ağustos 2025") == date(2025, 8, 3)
    assert P._search_date("2026-01-18_19-56") == datetime(2026, 1, 18, 19, 56)
    assert P._scraped_at("2026-01-18T16:59:10.760527+00:00") == datetime(2026, 1, 18, 16, 59, 10, 760527,
                                                                         tzinfo=timezone.utc)
    assert P._year_range("2015 - 2019") == (2015, 2019) and P._year_range(None) == (None, None)


def test_strip_html_keeps_case_and_normalises():
    """EN: Visible text, case kept, NFC. / TR: Görünen metin, harf büyüklüğü korunur, NFC."""
    decomposed = unicodedata.normalize("NFD", "YAPILMIŞTIR Çok")
    assert P._strip_html(f"<div><p>{decomposed}</p></div>") == "YAPILMIŞTIR Çok"


@pytest.mark.parametrize("html, expected", [
    ("<h5>Açıklama</h5><div><p>Araç sorunsuzdur</p></div>", "Araç sorunsuzdur"),
    ("<h5>Açıklama</h5><div>- Araç sorunsuzdur</div>", "- Araç sorunsuzdur"),
    ("<h5>Açıklama</h5><div>-Değişensiz</div>", "-Değişensiz"),
    ("<h5>Açıklama</h5><div>: Araç sorunsuzdur</div>", ": Araç sorunsuzdur"),
    ("<h5>Açıklama</h5><div>----- Bilgi</div>", "----- Bilgi"),
    ("<h5>Açıklama</h5><div>Açıklama kısmına bakınız</div>", "Açıklama kısmına bakınız"),
    ("<h5>Açıklama</h5><div></div>", None),
    (None, None),
])
def test_description_removes_only_the_heading(html, expected):
    """
    EN: Only the page heading "Açıklama" and the one space after it go; the seller's text stays as written.
    TR: Yalnız sayfa başlığı "Açıklama" ve arkasındaki tek boşluk gider; satıcının metni yazıldığı gibi kalır.
    """
    assert P._description(html) == expected


def test_damage_mappings_file():
    """EN: 13 panels, 5 statuses, in the site's labels. / TR: 13 parça, 5 durum, sitenin etiketleriyle."""
    assert len(P.DAMAGE_PART_MAP) == 13 and len(set(P.DAMAGE_PART_MAP.values())) == 13
    assert P.DAMAGE_STATUS_MAP == {"Orjinal": "original", "Belirtilmemiş": "unspecified", "Boyalı": "painted",
                                   "Lokal boyalı": "local_painted", "Değişmiş": "changed"}


def test_parse_damage_list():
    """EN: Each panel gets its status. / TR: Her parça kendi durumunu alır."""
    out = P._parse_damage_list(damage_list(Tavan="Belirtilmemiş", Motor_Kaputu="Değişmiş",
                                           Sol_Ön_Kapı="Boyalı", Sağ_Ön_Kapı="Lokal boyalı"))
    assert out["roof_status"] == "unspecified" and out["engine_hood_status"] == "changed"
    assert out["door_fl_status"] == "painted" and out["door_fr_status"] == "local_painted"
    assert out["trunk_lid_status"] == "original" and len(out) == 13


def test_process_record_full():
    """EN: A whole record → the expected silver row. / TR: Tam kayıt → beklenen silver satırı."""
    s = P.process_record(raw_record())
    assert (s["ad_id"], s["price"], s["brand"], s["kb_year"], s["kb_mileage"]) == (10000001, 1250000, "audi", 2015, 120000)
    assert (s["engine_cc"], s["engine_cc_is_range"], s["power_hp"], s["torque_nm"]) == (1968, False, 150, 320)
    assert (s["accel_0_100"], s["fuel_cons_avg"], s["mtv_yearly"], s["seat_count"]) == (8.6, 4.9, 8629, 5)
    assert (s["is_heavy_damaged"], s["is_first_owner"], s["kb_trade_available"]) == (False, False, True)
    assert s["plate_origin"] == "(TR) Türkiye" and s["description_text"] == "Araç sorunsuzdur."
    assert s["kb_paint_change_summary"] == "Tamamı orjinal" and s["listing_date"] == date(2025, 11, 26)
    assert all(s[c] == "original" for c in P.DAMAGE_PART_MAP.values())


@pytest.mark.parametrize("key", ["Fiyat", "Hasar_Listesi", "KısaBilgi - İlan No"])
def test_process_record_drops_empty_pages(key):
    """EN: No price / damage list / ad number → dropped. / TR: Fiyat / hasar listesi / ilan no yoksa atılır."""
    assert P.process_record(raw_record(**{key: MISSING})) is None


@pytest.mark.parametrize("value, expected", [("Evet", True), ("Hayır", False), ("Belirtilmemiş", None),
                                             (MISSING, None)])
def test_process_record_heavy_damage(value, expected):
    """
    EN: Heavy damage comes from "KısaBilgi - Ağır Hasarlı", not from the scraper's flag (False when unknown).
    TR: Ağır hasar "KısaBilgi - Ağır Hasarlı"dan gelir, scraper'ın bayrağından değil (bilinmezken False).
    """
    rec = raw_record(**{"KısaBilgi - Ağır Hasarlı": value, "Agir_Hasar": expected is True})
    assert P.process_record(rec)["is_heavy_damaged"] is expected


def test_process_record_unknowns_stay_none():
    """EN: Missing counts / first owner / summary stay None. / TR: Eksik sayaç / ilk sahip / özet None kalır."""
    s = P.process_record(raw_record(**{"Degisen_Parca_Sayisi": None, "Genel Bakış - Aracın ilk sahibiyim": MISSING,
                                       "KısaBilgi - Boya-değişen": "-"}))
    assert (s["count_changed"], s["is_first_owner"], s["kb_paint_change_summary"]) == (None, None, None)
    assert s["count_painted"] == 0


@pytest.mark.parametrize("summary", ["Tamamı orjinal", "2 değişen, 3 boyalı", "Belirtilmemiş", "Tamamı boyalı"])
def test_process_record_paint_summary_is_raw(summary):
    """EN: The summary line is kept exactly as written. / TR: Özet satırı yazıldığı gibi tutulur."""
    assert P.process_record(raw_record(**{"KısaBilgi - Boya-değişen": summary}))["kb_paint_change_summary"] == summary


def test_process_record_fuel_fallback():
    """EN: Missing KısaBilgi fuel → the Yakıt Tüketimi tab. / TR: KısaBilgi yakıtı yoksa Yakıt Tüketimi sekmesi."""
    s = P.process_record(raw_record(**{"KısaBilgi - Ort. Yakıt Tüketimi": MISSING,
                                       "Yakıt Tüketimi - Ortalama Yakıt Tüketimi": "6,1 lt"}))
    assert s["fuel_cons_avg"] == 6.1


def test_jsonl_to_silver_df(tmp_path):
    """EN: Kept records become rows; empty pages are left out. / TR: Tutulan kayıtlar satır olur; boş sayfalar dışarıda."""
    path = tmp_path / "details.jsonl"
    lines = [raw_record(10000001), raw_record(10000002, Fiyat=MISSING), raw_record(10000003)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + "\n", encoding="utf-8")
    df = P.jsonl_to_silver_df(path)
    assert list(df["ad_id"]) == [10000001, 10000003]
    assert "roof_status" in df.columns


def test_jsonl_broken_line_names_file_and_line(tmp_path):
    """EN: An unreadable line stops with its file and line. / TR: Okunamayan satır dosya ve satırıyla durur."""
    path = tmp_path / "details.jsonl"
    path.write_text(json.dumps(raw_record(), ensure_ascii=False) + "\n{bozuk\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"details\.jsonl:2"):
        P.jsonl_to_silver_df(path)
