"""
process_for_db.py
EN: arabam.com details.jsonl → silver: one flat row per listing and snapshot. A library, never run on its own
    (build_duckdb.py calls jsonl_to_silver_df). The DB it feeds is SEMI-RAW:
      - a value the page does not give stays None (NULL) — nothing is guessed. "Belirtilmemiş" (unspecified)
        is not turned into "no" / "original"; the analysis decides that;
      - text is kept as written; only the page's own section heading "Açıklama" is cut off the description.
    One parser per real format; every format was counted over all records, and
    tests/db/test_process_for_db.py pins each one.
    Where the fields come from:
      - KısaBilgi (kb_) and Genel Bakış (gb_) are two tabs of the page and are kept separately: they can differ
        (body type differs in 10,658 records);
      - damage: the damage diagram (Hasar_Listesi, 13 panels → *_status) and the scraper's counts, which are
        counted from the same diagram; heavy damage from "KısaBilgi - Ağır Hasarlı" (Evet / Hayır /
        Belirtilmemiş). The scraper's Agir_Hasar flag is not used: it is False also when the page says nothing;
      - kb_paint_change_summary: the one-line "Boya-değişen" summary, kept as raw text.
    Raw keys that are not read, and why: the twins "Genel Bakış / Motor ve Performans - Motor Hacmi / Motor
    Gücü", "Yakıt Tüketimi - Yakıt Deposu" and "KısaBilgi - Marka" equal the key that is read wherever both are
    filled and never fill a gap; "Motor ve Performans - Çekiş" differs from kb_drivetrain only in wording
    ("4x4" ↔ "4WD (Sürekli)"); "error" never occurs; "Agir_Hasar" is replaced (above).
    first_scraped_at / first_search_date are copies of scraped_at / search_date; build_duckdb does not write them.
TR: arabam.com details.jsonl → silver: ilan ve tarama başına tek düz satır. Tek başına koşulmayan bir kütüphane
    (build_duckdb.py jsonl_to_silver_df'i çağırır). Beslediği DB YARI HAM:
      - sayfanın vermediği değer None (NULL) kalır, tahmin edilmez. "Belirtilmemiş" "hayır" / "orijinal"e
        çevrilmez; o kararı analiz verir;
      - metin yazıldığı gibi kalır; açıklamadan yalnız sayfanın kendi bölüm başlığı "Açıklama" kesilir.
    Her gerçek biçim için bir ayrıştırıcı; bütün biçimler tüm kayıtlarda sayıldı ve her biri
    tests/db/test_process_for_db.py'de sabitlendi.
    Alanların kaynağı:
      - KısaBilgi (kb_) ve Genel Bakış (gb_) sayfanın iki sekmesi, ayrı tutulur: farklı olabiliyorlar (kasa
        tipi 10.658 kayıtta farklı);
      - hasar: hasar şeması (Hasar_Listesi, 13 parça → *_status) ve scraper'ın aynı şemadan saydığı sayaçlar;
        ağır hasar "KısaBilgi - Ağır Hasarlı"dan (Evet / Hayır / Belirtilmemiş). Scraper'ın Agir_Hasar bayrağı
        kullanılmaz: sayfa hiçbir şey demediğinde de False;
      - kb_paint_change_summary: tek satırlık "Boya-değişen" özeti, ham metin olarak.
    Okunmayan ham anahtarlar ve nedeni: "Genel Bakış / Motor ve Performans - Motor Hacmi / Motor Gücü",
    "Yakıt Tüketimi - Yakıt Deposu" ve "KısaBilgi - Marka" ikizleri, ikisinin de dolu olduğu her kayıtta okunan
    anahtarla aynı ve hiçbir boşluğu doldurmuyor; "Motor ve Performans - Çekiş" kb_drivetrain'den yalnız yazımda
    ayrılıyor ("4x4" ↔ "4WD (Sürekli)"); "error" hiç yok; "Agir_Hasar"ın yerini yukarıdaki alan aldı.
    first_scraped_at / first_search_date, scraped_at / search_date'in kopyası; build_duckdb bunları yazmaz.
"""
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# EN: the damage diagram's labels (13 panels, 5 statuses), shared with analysis/01_unspecified_panels.py
# TR: hasar şemasının etiketleri (13 parça, 5 durum); analysis/01_unspecified_panels.py ile ortak
_MAPPINGS = json.loads((Path(__file__).with_name("damage_mappings.json")).read_text(encoding="utf-8"))
DAMAGE_PART_MAP = _MAPPINGS["panels"]          # site label → silver status column | site etiketi → silver kolonu
DAMAGE_STATUS_MAP = _MAPPINGS["statuses"]      # site label → enum | site etiketi → enum

TR_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4,
    "Mayıs": 5, "Haziran": 6, "Temmuz": 7, "Ağustos": 8,
    "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}
# EN: the description's first line is the page's section heading, not the seller's text
# TR: açıklamanın ilk satırı sayfanın bölüm başlığı, satıcının metni değil
DESCRIPTION_HEADING = re.compile(r"^Açıklama ?")


# =====================================================================
# Helpers | Yardımcılar
# =====================================================================

def _is_blank(val):
    """
    EN: None, an empty / whitespace string, or arabam.com's "-" placeholder.
    TR: None, boş / yalnız boşluk metin ya da arabam.com'un "-" yer tutucusu.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip() in ("", "-")
    return False


def _str(val):
    """
    EN: Stripped text; blank → None.
    TR: Kırpılmış metin; boş → None.
    """
    if _is_blank(val):
        return None
    return str(val).strip()


def _opt_int(val):
    """
    EN: Keys that are already int | null in the JSONL (Tramer_Tutari, the damage counts). None stays None.
    TR: JSONL'de zaten int | null olan anahtarlar (Tramer_Tutari, hasar sayaçları). None, None kalır.
    """
    if val is None:
        return None
    return int(val)


def _tr_lower(s):
    """
    EN: Turkish-safe lower case. Python's "İ".lower() gives "i" + a combining dot (U+0307), so
        "İlk Sahibiyim".lower().startswith("ilk") was False (first owner was always False until 2026-09-23).
    TR: Türkçe güvenli küçültme. Python'da "İ".lower() "i" + birleşik nokta (U+0307) verir; bu yüzden
        "İlk Sahibiyim".lower().startswith("ilk") False dönüyordu (2026-09-23'e kadar ilk sahip hep False).
    """
    return str(s).replace("İ", "i").replace("I", "ı").lower()


# =====================================================================
# Format parsers (each one for one real JSONL format) | Biçim ayrıştırıcıları (her biri tek gerçek biçim)
# =====================================================================

def _int_with_unit(val):
    """
    EN: '365.000 km', '1995 cc', '60 lt', '215 km/s', '350 nm', '1.198 TL' → int. Every non-digit is removed;
        the dot is arabam.com's thousands separator. Every such field holds a single value (no ranges).
    TR: '365.000 km', '1995 cc', '60 lt', '215 km/s', '350 nm', '1.198 TL' → int. Rakam dışı her şey silinir;
        nokta arabam.com'da binlik ayırıcı. Bu alanların hepsi tek değer taşır (aralık yok).
    """
    if _is_blank(val):
        return None
    digits = re.sub(r"\D", "", str(val))
    return int(digits) if digits else None


def _decimal_with_unit(val):
    """
    EN: '7,9 lt', '10,9 sn', '7 lt' → float. The comma is the Turkish decimal separator.
    TR: '7,9 lt', '10,9 sn', '7 lt' → float. Virgül Türkçe ondalık ayırıcı.
    """
    if _is_blank(val):
        return None
    m = re.search(r"[\d.,]+", str(val))
    if not m:
        return None
    token = m.group(0)
    if "," in token:
        # EN: a comma → decimal separator; a dot, if any, is the thousands separator
        # TR: virgül varsa ondalık ayırıcı; nokta varsa binlik
        token = token.replace(".", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _year(val):
    """
    EN: A 4-digit year → int.
    TR: 4 haneli yıl → int.
    """
    if _is_blank(val):
        return None
    s = str(val).strip()
    return int(s) if s.isdigit() and len(s) == 4 else None


def _ad_id(val):
    """
    EN: 'Kopyalandı\\n\\r\\n' + spaces + '12345678' → 12345678 (the page's "copied" button text comes first).
    TR: 'Kopyalandı\\n\\r\\n' + boşluklar + '12345678' → 12345678 (önce sayfanın "kopyalandı" düğme metni gelir).
    """
    if _is_blank(val):
        return None
    digits = re.sub(r"\D", "", str(val))
    return int(digits) if digits else None


def _price_tl(val):
    """
    EN: '299.900 TL' → 299900.
    TR: '299.900 TL' → 299900.
    """
    if _is_blank(val):
        return None
    digits = re.sub(r"\D", "", str(val))
    return int(digits) if digits else None


def _listing_date(val):
    """
    EN: '26 Kasım 2025' → date(2025, 11, 26).
    TR: '26 Kasım 2025' → date(2025, 11, 26).
    """
    if _is_blank(val):
        return None
    parts = str(val).strip().split()
    if len(parts) != 3:
        return None
    day_s, month_s, year_s = parts
    month = TR_MONTHS.get(month_s)
    if not (month and day_s.isdigit() and year_s.isdigit()):
        return None
    try:
        return datetime(int(year_s), month, int(day_s)).date()
    except ValueError:
        return None


def _scraped_at(val):
    """
    EN: ISO-8601 with time zone (the moment the record was fetched) → datetime.
    TR: Saat dilimli ISO-8601 (kaydın çekildiği an) → datetime.
    """
    if _is_blank(val):
        return None
    return pd.to_datetime(val).to_pydatetime()


def _search_date(val):
    """
    EN: '2026-01-18_19-56' (the snapshot folder name) → datetime.
    TR: '2026-01-18_19-56' (tarama klasörünün adı) → datetime.
    """
    if _is_blank(val):
        return None
    try:
        return datetime.strptime(str(val).strip(), "%Y-%m-%d_%H-%M")
    except ValueError:
        return None


def _engine_cc(val):
    """
    EN: Engine volume, two formats. Returns (cc, low, high, is_range):
          '1995 cc'            → (1995, None, None, False)      exact | kesin
          '1601 - 1800 cm3'    → (None, 1601, 1800, True)       bucket | kova
          "1200 cm3' e kadar"  → (None, None, 1200, True)       open below | alttan açık
          '-' or blank         → (None, None, None, None)
    TR: Motor hacmi, iki biçim. (cc, alt, üst, aralık_mı) döndürür (örnekler yukarıda).
    """
    if _is_blank(val):
        return None, None, None, None
    s = str(val).strip()
    # EN: 'cc' is exact, 'cm3' a bucket ('cc' never occurs inside 'cm3') | TR: 'cc' kesin, 'cm3' kova
    if "cc" in s and "cm3" not in s:
        digits = re.sub(r"\D", "", s)
        if digits:
            return int(digits), None, None, False
        return None, None, None, None
    if "kadar" in s:
        nums = re.findall(r"\d+", s)
        if nums:
            return None, None, int(nums[0]), True
        return None, None, None, None
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return None, int(nums[0]), int(nums[1]), True
    if len(nums) == 1:
        return None, int(nums[0]), int(nums[0]), True
    return None, None, None, None


def _power_hp(val):
    """
    EN: Engine power, two formats. Returns (hp, low, high, is_range):
          '150 hp'            → (150, None, None, False)        lower-case hp = exact | küçük hp = kesin
          '101 - 125 HP'      → (None, 101, 125, True)          upper-case HP = bucket | büyük HP = kova
          "50 HP'ye kadar"    → (None, None, 50, True)          open below | alttan açık
          '601 HP ve üzeri'   → (None, 601, None, True)         open above | üstten açık
          '-' or blank        → (None, None, None, None)
        Until 2026-09-23 the two open buckets were written as exact values.
    TR: Motor gücü, iki biçim. (hp, alt, üst, aralık_mı) döndürür (örnekler yukarıda).
        2026-09-23'e kadar iki açık uçlu kova kesin değer yazılıyordu.
    """
    if _is_blank(val):
        return None, None, None, None
    s = str(val).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return None, None, None, None
    low_s = _tr_lower(s)
    if "kadar" in low_s:
        return None, None, int(nums[0]), True
    if "üzeri" in low_s:
        return None, int(nums[0]), None, True
    is_range = " - " in s or s.endswith("HP")
    if is_range:
        if len(nums) >= 2:
            return None, int(nums[0]), int(nums[1]), True
        return None, int(nums[0]), int(nums[0]), True
    return int(nums[0]), None, None, False


def _year_range(val):
    """
    EN: '2003 - 2008' → (2003, 2008).
    TR: '2003 - 2008' → (2003, 2008).
    """
    if _is_blank(val):
        return None, None
    nums = re.findall(r"\d{4}", str(val))
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return None, None


def _yes_no(val):
    """
    EN: 'Evet' → True, 'Hayır' → False; 'Belirtilmemiş', blank or missing → None (unknown stays unknown).
    TR: 'Evet' → True, 'Hayır' → False; 'Belirtilmemiş', boş ya da yok → None (bilinmeyen bilinmeyen kalır).
    """
    return {"Evet": True, "Hayır": False}.get(_str(val))


def _bool_takasa(val):
    """
    EN: 'Takasa Uygun' → True · 'Takasa Uygun Değil' → False · missing / '-' → None.
        Until 2026-09-23 the word "uygun" was searched, so "Takasa Uygun Değil" was True too (8,466 raw
        records), and listings without the field were written False.
    TR: 'Takasa Uygun' → True · 'Takasa Uygun Değil' → False · yok / '-' → None.
        2026-09-23'e kadar "uygun" kelimesi aranıyordu; "Takasa Uygun Değil" de True oluyordu (8.466 ham
        kayıt) ve alanı olmayan ilanlar False yazılıyordu.
    """
    if _is_blank(val):
        return None
    s = _tr_lower(str(val).strip())
    if "değil" in s:
        return False
    return True if "uygun" in s else None


def _bool_first_owner(val):
    """
    EN: 'İlk Sahibiyim' → True; 'İlk Sahibi Değilim' → False; missing / '-' → None (was False until 2026-09-24).
    TR: 'İlk Sahibiyim' → True; 'İlk Sahibi Değilim' → False; yok / '-' → None (2026-09-24'e kadar False).
    """
    if _is_blank(val):
        return None
    s = _tr_lower(str(val).strip())
    return s.startswith("ilk sahib") and "değil" not in s


def _strip_html(val):
    """
    EN: The visible text of the listing HTML, Unicode NFC. Case is NOT changed: Python's .lower() is not
        Turkish-aware and destroys information ("YAPILMIŞTIR" → "yapilmiştir"); lower-casing is the analysis'
        job (analysis/lib/text_flags.lower_tr).
    TR: İlan HTML'inin görünen metni, Unicode NFC. Büyük/küçük harf DEĞİŞMEZ: Python'un .lower()'ı Türkçe
        bilmez ve bilgi yok eder ("YAPILMIŞTIR" → "yapilmiştir"); küçültme analizin işi
        (analysis/lib/text_flags.lower_tr).
    """
    if _is_blank(val):
        return None
    txt = BeautifulSoup(str(val), "html.parser").get_text(" ", strip=True)
    return unicodedata.normalize("NFC", txt) or None


def _description(val):
    """
    EN: The seller's description: the visible text without the page's section heading. Every listing's HTML
        starts with <h5>Açıklama</h5>, so the text starts with "Açıklama " — only that word and the one space
        after it are removed; the seller's own "-" or ":" stays. Only the heading → None.
    TR: Satıcının açıklaması: sayfanın bölüm başlığı olmadan görünen metin. Her ilanın HTML'i <h5>Açıklama</h5>
        ile başladığı için metin "Açıklama " ile başlar; yalnız bu kelime ve arkasındaki tek boşluk silinir,
        satıcının kendi "-" ya da ":" işareti kalır. Yalnız başlık → None.
    """
    txt = _strip_html(val)
    if txt is None:
        return None
    return DESCRIPTION_HEADING.sub("", txt, count=1) or None


# =====================================================================
# Damage diagram | Hasar şeması — 13 panels × 5 statuses | 13 parça × 5 durum
# =====================================================================

def _parse_damage_list(items):
    """
    EN: Hasar_Listesi ("Panel: Status" lines) → dict of the 13 status columns (None when not listed).
    TR: Hasar_Listesi ("Parça: Durum" satırları) → 13 durum kolonunun dict'i (listede yoksa None).
    """
    out = {col: None for col in DAMAGE_PART_MAP.values()}
    if not isinstance(items, list):
        return out
    for entry in items:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        part, status = entry.split(":", 1)
        col = DAMAGE_PART_MAP.get(part.strip())
        enum = DAMAGE_STATUS_MAP.get(status.strip())
        if col and enum:
            out[col] = enum
    return out


# =====================================================================
# Record → silver row | Kayıt → silver satırı
# =====================================================================

def process_record(raw):
    """
    EN: One JSONL record → one flat silver dict. A record without a price, a damage list or an ad number is
        dropped (returns None). In the real data the 20 dropped records lack all three (empty pages).
    TR: Tek JSONL kaydı → tek düz silver dict. Fiyatı, hasar listesi ya da ilan numarası olmayan kayıt atılır
        (None döner). Gerçek veride atılan 20 kaydın üçü de yok (boş sayfalar).
    """
    fiyat = raw.get("Fiyat")
    if _is_blank(fiyat):
        return None
    hasar = raw.get("Hasar_Listesi")
    if not hasar:
        return None
    ad_id = _ad_id(raw.get("KısaBilgi - İlan No"))
    if ad_id is None:
        return None

    cc, cc_lo, cc_hi, cc_range = _engine_cc(raw.get("KısaBilgi - Motor Hacmi"))
    hp, hp_lo, hp_hi, hp_range = _power_hp(raw.get("KısaBilgi - Motor Gücü"))
    py_start, py_end = _year_range(raw.get("Genel Bakış - Üretim Yılı (İlk/Son)"))
    scraped_at = _scraped_at(raw.get("scraped_at"))
    search_date = _search_date(raw.get("search_date"))

    rec = {
        # Identity | Kimlik
        "ad_id":      ad_id,
        "brand":      _str(raw.get("brand")),
        "series":     _str(raw.get("KısaBilgi - Seri")),
        "model":      _str(raw.get("KısaBilgi - Model")),
        "ad_title":   _str(raw.get("Ilan_Basligi")),
        "url":        _str(raw.get("url")),
        "eids_model": _str(raw.get("EIDS_Model")),
        "location":   _str(raw.get("Konum")),
        "price":      _price_tl(fiyat),

        # Dates | Tarihler
        "listing_date":      _listing_date(raw.get("KısaBilgi - İlan Tarihi")),
        "scraped_at":        scraped_at,
        "search_date":       search_date,
        # EN: copies; build_duckdb does not write them | TR: kopya; build_duckdb bunları yazmaz
        "first_scraped_at":  scraped_at,
        "first_search_date": search_date,

        # EN: KısaBilgi (kb_) and Genel Bakış (gb_) separately — the two tabs can differ
        # TR: KısaBilgi (kb_) ve Genel Bakış (gb_) ayrı — iki sekme farklı olabiliyor
        "kb_year":            _year(raw.get("KısaBilgi - Yıl")),
        "gb_year":            _year(raw.get("Genel Bakış - Yıl")),
        "kb_mileage":         _int_with_unit(raw.get("KısaBilgi - Kilometre")),
        "gb_mileage":         _int_with_unit(raw.get("Genel Bakış - Kilometre")),
        "kb_transmission":    _str(raw.get("KısaBilgi - Vites Tipi")),
        "gb_transmission":    _str(raw.get("Genel Bakış - Vites Tipi")),
        "kb_fuel":            _str(raw.get("KısaBilgi - Yakıt Tipi")),
        "gb_fuel":            _str(raw.get("Genel Bakış - Yakıt Tipi")),
        "kb_body_type":       _str(raw.get("KısaBilgi - Kasa Tipi")),
        "gb_body_type":       _str(raw.get("Genel Bakış - Kasa Tipi")),
        "kb_color":           _str(raw.get("KısaBilgi - Renk")),
        "gb_color":           _str(raw.get("Genel Bakış - Renk")),
        "kb_drivetrain":      _str(raw.get("KısaBilgi - Çekiş")),
        "gb_drivetrain":      _str(raw.get("Genel Bakış - Çekiş")),
        "kb_condition":       _str(raw.get("KısaBilgi - Araç Durumu")),
        "gb_condition":       _str(raw.get("Genel Bakış - Araç Durumu")),
        "kb_seller_type":     _str(raw.get("KısaBilgi - Kimden")),
        "gb_seller_type":     _str(raw.get("Genel Bakış - Kimden")),
        "kb_trade_available": _bool_takasa(raw.get("KısaBilgi - Takasa Uygun")),
        "gb_trade_available": _bool_takasa(raw.get("Genel Bakış - Takasa Uygun")),

        "transmission_brand": _str(raw.get("Genel Bakış - Şanzıman")),

        # Genel Bakış only | Yalnız Genel Bakış'ta
        "warranty_status":    _str(raw.get("Genel Bakış - Garanti Durumu")),
        "usage_type":         _str(raw.get("Genel Bakış - Araç Türü")),
        "is_first_owner":     _bool_first_owner(raw.get("Genel Bakış - Aracın ilk sahibiyim")),
        "segment":            _str(raw.get("Genel Bakış - Sınıfı")),
        "plate_origin":       _str(raw.get("Genel Bakış - Plaka Uyruğu")),

        # Engine (cc exact / cm3 bucket) | Motor (cc kesin / cm3 kova)
        "engine_cc":           cc,
        "engine_cc_low":       cc_lo,
        "engine_cc_high":      cc_hi,
        "engine_cc_is_range":  cc_range,

        "power_hp":            hp,
        "power_hp_low":        hp_lo,
        "power_hp_high":       hp_hi,
        "power_hp_is_range":   hp_range,

        # Performance | Performans
        "torque_nm":      _int_with_unit(raw.get("Motor ve Performans - Tork")),
        "cylinder_count": _int_with_unit(raw.get("Motor ve Performans - Silindir Sayısı")),
        "max_speed_kmh":  _int_with_unit(raw.get("Motor ve Performans - Maksimum Hız")),
        "accel_0_100":    _decimal_with_unit(raw.get("Motor ve Performans - Hızlanma (0-100)")),
        "rpm_max":        _int_with_unit(raw.get("Motor ve Performans - Maksimum Güç")),
        "rpm_min":        _int_with_unit(raw.get("Motor ve Performans - Minimum Güç")),

        # Fuel | Yakıt
        # EN: when the KısaBilgi value is missing the same value is on the "Yakıt Tüketimi" tab (equal wherever
        #     both exist; 2,932 records have only the second)
        # TR: KısaBilgi değeri yoksa aynı değer "Yakıt Tüketimi" sekmesinde (ikisinin de olduğu her kayıtta
        #     aynı; 2.932 kayıtta yalnız ikincisi var)
        "fuel_cons_avg":     _decimal_with_unit(raw.get("KısaBilgi - Ort. Yakıt Tüketimi")
                                                if not _is_blank(raw.get("KısaBilgi - Ort. Yakıt Tüketimi"))
                                                else raw.get("Yakıt Tüketimi - Ortalama Yakıt Tüketimi")),
        "city_fuel_cons":    _decimal_with_unit(raw.get("Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi")),
        "highway_fuel_cons": _decimal_with_unit(raw.get("Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi")),
        "fuel_tank":         _int_with_unit(raw.get("KısaBilgi - Yakıt Deposu")),

        # Dimensions | Boyut
        "length_mm":         _int_with_unit(raw.get("Boyut ve Kapasite - Uzunluk")),
        "width_mm":          _int_with_unit(raw.get("Boyut ve Kapasite - Genişlik")),
        "height_mm":         _int_with_unit(raw.get("Boyut ve Kapasite - Yükseklik")),
        "weight_kg":         _int_with_unit(raw.get("Boyut ve Kapasite - Ağırlık")),
        "curb_weight_kg":    _int_with_unit(raw.get("Boyut ve Kapasite - Boş Ağırlığı")),
        "trunk_capacity_lt": _int_with_unit(raw.get("Boyut ve Kapasite - Bagaj Hacmi")),
        "wheelbase_mm":      _int_with_unit(raw.get("Boyut ve Kapasite - Aks Aralığı")),
        "seat_count":        _int_with_unit(raw.get("Boyut ve Kapasite - Koltuk Sayısı")),
        "front_tire_spec":   _str(raw.get("Boyut ve Kapasite - Ön Lastik")),

        # Costs (TL, no kuruş) | Mali (TL, kuruşsuz)
        "mtv_yearly":            _int_with_unit(raw.get("Genel Bakış - Yıllık MTV")),
        "kasko_avg":             _int_with_unit(raw.get("Genel Bakış - Ortalama Kasko")),
        "traffic_insurance_avg": _int_with_unit(raw.get("Genel Bakış - Ortalama Trafik Sigortası")),

        # Production years | Üretim aralığı
        "production_year_start": py_start,
        "production_year_end":   py_end,

        # Damage | Hasar
        "is_heavy_damaged":        _yes_no(raw.get("KısaBilgi - Ağır Hasarlı")),
        "tramer_fee":              _opt_int(raw.get("Tramer_Tutari")),
        "count_changed":           _opt_int(raw.get("Degisen_Parca_Sayisi")),
        "count_painted":           _opt_int(raw.get("Boyali_Parca_Sayisi")),
        "count_local_painted":     _opt_int(raw.get("Lokal_Boyali_Parca_Sayisi")),
        "kb_paint_change_summary": _str(raw.get("KısaBilgi - Boya-değişen")),

        "description_text": _description(raw.get("Aciklama_HTML")),
    }
    rec.update(_parse_damage_list(hasar))
    return rec


def jsonl_to_silver_df(path):
    """
    EN: One details.jsonl → silver DataFrame (dropped records left out). Lines are read one by one with
        json.loads, so no value is type-converted by pandas; an unreadable line stops with ValueError naming
        the file and line (before 2026-09-24 one broken record silently dropped the whole snapshot).
    TR: Tek details.jsonl → silver DataFrame (atılan kayıtlar dışarıda). Satırlar json.loads ile tek tek okunur,
        böylece hiçbir değerin türünü pandas değiştirmez; okunamayan satır dosyayı ve satırı söyleyen
        ValueError ile durur (2026-09-24'ten önce tek bozuk kayıt bütün taramayı sessizce düşürüyordu).
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: unreadable JSON line | okunamayan JSON satırı ({e.msg})") from e
            rec = process_record(raw)
            if rec is not None:
                rows.append(rec)
    return pd.DataFrame(rows)
