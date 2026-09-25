"""
test_tool_sources.py
EN: The explorer's loaders on the made-up data folder: raw records are flattened with their values untouched (the
    damage list opened into "Hasar - <Part>" columns, a failed page kept, a broken line counted and skipped, ad_id
    from the ad-number digits), number-like fields get a parsed "(sayı)" column and ranges stay text, one record can
    be read back whole, the ad text comes aligned and as plain text; the DBs load without the ad text, the
    timestamp as text, and leave no open handle (the file can be replaced right after, as a rebuild does).
TR: Gezginin yükleyicileri uydurma veri klasöründe: ham kayıtlar değerlerine dokunulmadan düzleştirilir (hasar
    listesi "Hasar - <Parça>" kolonlarına açılır, başarısız sayfa tutulur, bozuk satır sayılıp atlanır, ad_id ilan no
    rakamlarından), sayıya benzeyen alanlara ayrıştırılmış "(sayı)" kolonu eklenir, aralıklar metin kalır, tek kayıt
    bütünüyle geri okunabilir, ilan metni hizalı ve düz metin gelir; DB'ler ilan metni olmadan, zaman damgası metin
    olarak yüklenir ve açık handle bırakmaz (dosya hemen ardından, yeniden kurulumdaki gibi, değiştirilebilir).
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import sources as S  # noqa: E402


@pytest.mark.parametrize("text, number", [("400.000 TL", 400000.0), ("12,1 sn", 12.1), ("1595 cc", 1595.0),
                                          ("2005", 2005.0), ("183 km/s", 183.0), ("-5", -5.0),
                                          ("1401 - 1600 cm3", None), ("205/55 R16", None), ("04 Aralık 2025", None),
                                          ("İkinci El", None), ("", None), (None, None), (7, None)])
def test_parse_number(text, number):
    """EN: One number with an optional unit only. / TR: Yalnız isteğe bağlı birimli tek sayı."""
    assert S.parse_number(text) == number


def test_load_raw(tool_data):
    """EN: Records, stats, flattening and ids. / TR: Kayıtlar, bilgi, düzleştirme ve kimlikler."""
    df, stats = S.load("raw", tool_data)
    assert stats == {"files": 2, "records": 5, "broken": 1} and len(df) == 5
    assert set(df[S.SNAPSHOT_DIR]) == {"2026-01-18_19-56", "2026-03-21_22-18"} and set(df[S.BRAND_DIR]) == {"audi", "bmw"}
    assert sorted(df["ad_id"].dropna().astype(int)) == [1001, 1001, 1002, 1003]
    assert df["error"].notna().sum() == 1 and df.loc[df["error"].notna(), "ad_id"].isna().all()
    assert set(df["Hasar - Tavan"].dropna()) == {"Orjinal", "Değişmiş"} and "Hasar_Listesi" not in df.columns
    assert S.RAW_TEXT not in df.columns
    assert df["Genel Bakış - Plaka Uyruğu"].isna().sum() == 2


def test_number_columns(tool_data):
    """EN: Parsed next to the text; ranges stay text. / TR: Metnin yanında ayrıştırılmış; aralıklar metin kalır."""
    df, _ = S.load("raw", tool_data)
    assert sorted(df["Fiyat (sayı)"].dropna()) == [900000, 1200000, 1250000, 2000000]
    assert list(df.columns).index("Fiyat (sayı)") == list(df.columns).index("Fiyat") + 1
    assert "KısaBilgi - Yıl (sayı)" in df.columns and "Genel Bakış - Motor Hacmi (sayı)" not in df.columns
    assert df["Fiyat"].dropna().iloc[0].endswith("TL")


def test_raw_record_and_texts(tool_data):
    """EN: A whole record by reference; aligned plain texts. / TR: Referansla tüm kayıt; hizalı düz metinler."""
    df, _ = S.load("raw", tool_data)
    files = S.data_files("raw", tool_data)
    blue = df.index[df["Genel Bakış - Plaka Uyruğu"] == "Mavi plakalı"][0]
    record = S.read_raw_record(files, df.loc[blue, S.RAW_REF])
    assert record["Genel Bakış - Plaka Uyruğu"] == "Mavi plakalı" and "MA PLAKALIDIR" in record[S.RAW_TEXT]
    texts = S.texts("raw", tool_data)
    assert len(texts) == len(df) and texts[blue] == "YABANCIDAN YABANCIYA. MA PLAKALIDIR."
    assert S.html_to_text("<p>a&nbsp;b</p><div>c</div>") == "a b\n c"
    assert "\xa0" not in texts[0] and S.html_to_text(None) is None


def test_load_db(tool_data):
    """EN: No ad text, timestamp as text, nullable types. / TR: İlan metni yok, zaman damgası metin, nullable tipler."""
    silver, stats = S.load("silver", tool_data)
    gold, _ = S.load("gold", tool_data)
    assert stats["records"] == 4 and list(silver["id"]) == [1, 2, 3, 4]
    assert S.DB_TEXT not in silver.columns and "kb_paint_change_summary" in silver.columns
    assert "kb_paint_change_summary" not in gold.columns
    assert isinstance(silver["scraped_at"].iloc[0], str)
    assert silver["is_heavy_damaged"].isna().sum() == 2 and gold["is_heavy_damaged"].isna().sum() == 0
    assert list(S.texts("silver", tool_data)) == ["Metin 1", "Metin 2", "Metin 3", "Metin 4"]


def test_db_handle_is_closed(tool_data, tmp_path):
    """
    EN: After a load the DB file can be replaced (a rebuild's os.replace fails on Windows while a handle is open).
    TR: Yüklemeden sonra DB dosyası değiştirilebilir (Windows'ta açık handle varken yeniden kurulumun os.replace'i
        başarısız olur).
    """
    shutil.copy(tool_data / "cars.duckdb", tmp_path / "cars.duckdb")
    shutil.copy(tool_data / "cars_gold.duckdb", tmp_path / "other.duckdb")
    S.load("silver", tmp_path)
    S.texts("silver", tmp_path)
    os.replace(tmp_path / "other.duckdb", tmp_path / "cars.duckdb")
    assert "kb_paint_change_summary" not in S.load("silver", tmp_path)[0].columns


def test_missing_and_fingerprint(tool_data, tmp_path):
    """EN: Missing data raises; the key changes with the file. / TR: Eksik veri hata verir; anahtar dosyayla değişir."""
    with pytest.raises(FileNotFoundError):
        S.load("gold", tmp_path)
    with pytest.raises(ValueError):
        S.data_files("other", tmp_path)
    path = tmp_path / "f.txt"
    path.write_text("a")
    before = S.fingerprint([path])
    path.write_text("ab")
    assert S.fingerprint([path]) != before
