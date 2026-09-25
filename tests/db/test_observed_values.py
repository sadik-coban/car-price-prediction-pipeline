"""
test_observed_values.py
EN: The raw → DB code expects only what the data showed, and handles everything it showed — both ways, against the
    register of observed values (db/observed_values.json; no data needed, the fast gate reads the JSON):
      (a) every value the code names occurs in the register — no assumed case ("Yabancı plakalı" was one);
      (b) every value in the register is handled on purpose — no silent default.
    Covered: the dropped plate values, the damage labels, the yes/no text maps, and every observed form of the fields
    the parsers read strictly (each parses without UnknownValue, and each pattern the code has is used by the data).
TR: Ham → DB kodu yalnız verinin gösterdiğini bekler ve gösterdiği her şeyi ele alır — iki yönde, gözlenen değerler
    kaydına göre (db/observed_values.json; veri gerekmez, hızlı kapı JSON'u okur):
      (a) kodun andığı her değer kayıtta geçiyor — varsayılan durum yok ("Yabancı plakalı" böyleydi);
      (b) kayıttaki her değer bilerek ele alınıyor — sessiz varsayılan yok.
    Kapsam: atılan plaka değerleri, hasar etiketleri, evet/hayır metin haritaları ve ayrıştırıcıların katı okuduğu
    alanların gözlenen her biçimi (her biri UnknownValue olmadan ayrışır ve koddaki her kalıp veride kullanılıyor).
"""
import pytest
from conftest import raw_record

import build_duckdb as BD
from lib import observed_values as OV, process_for_db as P

REGISTER = OV.load()
FIELDS = REGISTER["raw"]["fields"]
# EN: the only plate value the analysis keeps (analysis/lib/common.py) | TR: analizin tuttuğu tek plaka değeri
TR_PLATE = "(TR) Türkiye"


def values(field):
    """EN: The observed values of a raw field. / TR: Bir ham alanın gözlenen değerleri."""
    return [v for v, _ in FIELDS[field].get("values", [])]


def test_plates_both_ways():
    """
    EN: (a) every dropped plate value occurs; (b) every observed value is dropped or is the TR plate.
    TR: (a) atılan her plaka değeri geçiyor; (b) gözlenen her değer ya atılıyor ya TR plakası.
    """
    seen = set(values("Genel Bakış - Plaka Uyruğu"))
    assert set(BD.FOREIGN_PLATES) <= seen, set(BD.FOREIGN_PLATES) - seen
    assert seen - set(BD.FOREIGN_PLATES) == {TR_PLATE}


def test_damage_labels_both_ways():
    """EN: The mapped labels are exactly the observed ones. / TR: Eşlenen etiketler tam olarak gözlenenler."""
    damage = REGISTER["raw"]["damage"]
    assert set(P.DAMAGE_PART_MAP) == {v for v, _ in damage["parts"]}
    assert set(P.DAMAGE_STATUS_MAP) == {v for v, _ in damage["states"]}
    assert damage["malformed"] == []


@pytest.mark.parametrize("field, table", [("KısaBilgi - Ağır Hasarlı", P.HEAVY_DAMAGE_TEXT),
                                          ("KısaBilgi - Takasa Uygun", P.TRADE_TEXT),
                                          ("Genel Bakış - Takasa Uygun", P.TRADE_TEXT),
                                          ("Genel Bakış - Aracın ilk sahibiyim", P.FIRST_OWNER_TEXT)])
def test_text_maps_both_ways(field, table):
    """
    EN: (a) every text the map knows occurs; (b) every non-blank observed text is in the map.
    TR: (a) haritanın bildiği her metin geçiyor; (b) boş olmayan gözlenen her metin haritada.
    """
    seen = {v for v in values(field) if not P._is_blank(v)}
    assert set(table) <= seen, set(table) - seen
    assert seen <= set(table), seen - set(table)


@pytest.mark.parametrize("field, parse, patterns", [
    ("KısaBilgi - Motor Hacmi", P._engine_cc, (P.ENGINE_CC_EXACT, P.ENGINE_CC_BUCKET, P.ENGINE_CC_BELOW)),
    ("KısaBilgi - Motor Gücü", P._power_hp, (P.POWER_HP_EXACT, P.POWER_HP_BUCKET, P.POWER_HP_BELOW,
                                             P.POWER_HP_ABOVE)),
    ("Genel Bakış - Üretim Yılı (İlk/Son)", P._year_range, (P.YEAR_RANGE,)),
    ("KısaBilgi - İlan Tarihi", P._listing_date, (P.LISTING_DATE,)),
])
def test_every_observed_form_parses(field, parse, patterns):
    """
    EN: (b) every observed value parses without UnknownValue; (a) every pattern of the parser is used by the data.
    TR: (b) gözlenen her değer UnknownValue olmadan ayrışır; (a) ayrıştırıcının her kalıbı veride kullanılıyor.
    """
    observed = values(field)
    assert observed, f"{field}: no values in the register | kayıtta değer yok"
    for value in observed:
        parse(value)
    strings = [v.strip() for v in observed if isinstance(v, str)]
    unused = [p.pattern for p in patterns if not any(p.match(s) for s in strings)]
    assert not unused, f"patterns no observed value uses | hiçbir gözlenen değerin kullanmadığı kalıplar: {unused}"


def test_month_names_both_ways():
    """EN: The month table is exactly the observed months. / TR: Ay tablosu tam olarak gözlenen aylar."""
    months = {v.split()[1] for v in values("KısaBilgi - İlan Tarihi")}
    assert set(P.TR_MONTHS) == months


def test_fake_records_hold_only_observed_values():
    """
    EN: A fake test record with a value the data never showed fails at once — the old fixture MTV "8.629 TL" (made
        up; the data has 23 MTV amounts) would; an observed one passes.
    TR: Verinin hiç göstermediği bir değer taşıyan sahte test kaydı hemen düşer — eski fixture MTV'si "8.629 TL"
        (uydurma; veride 23 MTV tutarı var) düşerdi; gözlenen bir değer geçer.
    """
    with pytest.raises(AssertionError, match="8.629 TL"):
        raw_record(**{"Genel Bakış - Yıllık MTV": "8.629 TL"})
    assert raw_record(**{"Genel Bakış - Yıllık MTV": "1.198 TL"})["Genel Bakış - Yıllık MTV"] == "1.198 TL"
