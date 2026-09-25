"""
test_tool_details.py
EN: The explorer's listing window as data, on the made-up data folder: every field lands in exactly one group in a
    fixed group order (raw and DB), the damage table reads raw states as written and the DB flags with NULL as
    unknown (gold: original), the ad's snapshots come in order with the price change, a table click picks the row
    (row selection first, then a cell), the header finds its values, and the start page's file state says when a
    file is missing.
TR: Gezginin ilan penceresi veri olarak, uydurma veri klasöründe: her alan sabit grup sırasıyla tam bir gruba düşer
    (ham ve DB), hasar tablosu ham durumları yazıldığı gibi, DB bayraklarını NULL'u bilinmiyor sayarak okur (gold:
    orijinal), ilanın taramaları sırayla ve fiyat farkıyla gelir, tablodaki tıklama satırı seçer (önce satır seçimi,
    sonra hücre), üst bilgi değerlerini bulur ve başlangıç sayfasının dosya durumu eksik dosyayı söyler.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import charts, details as D, sources as S  # noqa: E402


@pytest.fixture(scope="module")
def frames(tool_data):
    """EN: The three sources loaded once. / TR: Üç kaynak bir kez yüklenir."""
    return {s: S.load(s, tool_data)[0] for s in ("raw", "silver", "gold")}


def raw_record(frames, tool_data, plate):
    """EN: The raw record with a given plate value. / TR: Belirli plaka değerli ham kayıt."""
    raw = frames["raw"]
    idx = raw.index[raw["Genel Bakış - Plaka Uyruğu"] == plate][0]
    return S.read_raw_record(S.data_files("raw", tool_data), raw.loc[idx, S.RAW_REF])


def test_group_fields_raw(frames, tool_data):
    """EN: Each raw field once, groups in order. / TR: Her ham alan bir kez, gruplar sırayla."""
    record = raw_record(frames, tool_data, "Mavi plakalı")
    groups = D.group_fields(record, "raw")
    placed = [f for _, items in groups for f, _ in items]
    assert sorted(placed) == sorted(record) and len(placed) == len(set(placed))
    names = [g for g, _ in groups]
    assert names == [g for g in D.RAW_GROUP_ORDER if g in names] and names[0] == "Kayıt"
    assert ("Genel Bakış - Plaka Uyruğu", "Mavi plakalı") in dict(groups)["Genel Bakış"]


def test_group_fields_db(frames):
    """EN: Each DB column once; damage flags under Hasar. / TR: Her DB kolonu bir kez; hasar bayrakları Hasar'da."""
    fields = frames["silver"].iloc[0].to_dict()
    groups = dict(D.group_fields(fields, "silver"))
    placed = [f for items in groups.values() for f, _ in items]
    assert sorted(placed) == sorted(fields)
    assert {f for f, _ in groups["Hasar"]} >= {"tavan_degisen", "is_heavy_damaged", "kb_paint_change_summary"}
    assert {f for f, _ in groups["Kimlik ve fiyat"]} >= {"id", "ad_id", "price", "url"}


def test_damage_table(frames, tool_data):
    """EN: Raw as written; silver unknown, gold original. / TR: Ham yazıldığı gibi; silver bilinmiyor, gold orijinal."""
    raw = D.damage_table(raw_record(frames, tool_data, "Mavi plakalı"), "raw")
    assert raw.to_dict("records") == [{"parça": "Tavan", "durum": "Orjinal"}, {"parça": "Motor Kaputu", "durum": "Orjinal"}]
    silver = D.damage_table(frames["silver"].iloc[0].to_dict(), "silver")
    gold = D.damage_table(frames["gold"].iloc[0].to_dict(), "gold")
    assert silver.set_index("parça").loc["Tavan", "durum"] == charts.UNKNOWN
    assert gold.set_index("parça").loc["Tavan", "durum"] == "orijinal"
    assert D.damage_table(frames["silver"].iloc[2].to_dict(), "silver").set_index("parça").loc["Tavan", "durum"] \
        == "değişen"


def test_history(frames):
    """EN: Snapshots in order with the price change. / TR: Taramalar sırayla, fiyat farkıyla."""
    raw = D.history(frames["raw"], "raw", 1001)
    assert list(raw["tarama"]) == ["2026-01-18", "2026-03-21"] and list(raw["fiyat"]) == [1_250_000, 1_200_000]
    assert pd.isna(raw["fiyat farkı"][0]) and raw["fiyat farkı"][1] == -50_000
    db = D.history(frames["silver"], "silver", 1)
    assert list(db["fiyat"]) == [1_000_000, 1_100_000] and db["fiyat farkı"][1] == 100_000
    assert D.history(frames["raw"], "raw", None).empty and D.history(frames["raw"], "raw", pd.NA).empty


def test_picked_position():
    """EN: Row selection first, then a cell. / TR: Önce satır seçimi, sonra hücre."""
    assert D.picked_position([3], [(5, "price")]) == 3
    assert D.picked_position([], [(5, "price")]) == 5
    assert D.picked_position([], [{"row": 2, "column": "url"}]) == 2
    assert D.picked_position([], []) is None


def test_header(frames, tool_data):
    """EN: Header values found; blanks → None. / TR: Üst bilgi bulunur; boşlar None."""
    head = D.header(raw_record(frames, tool_data, "Mavi plakalı"), "raw")
    assert head["price"] == "2.000.000 TL" and head["year"] == "2024" and head["url"].startswith("https://")
    assert D.header({"ad_title": " ", "price": 5.0}, "silver")["title"] is None


def test_file_state(tool_data, tmp_path):
    """EN: Present files counted; a missing source says so. / TR: Var olan dosyalar sayılır; eksik kaynak söylenir."""
    state = {r["kaynak"]: r for r in S.file_state(tool_data)}
    assert state[S.SOURCES["raw"]]["dosya"] == 2 and state[S.SOURCES["raw"]]["son tarama"] == "2026-03-21_22-18"
    assert all(r["var"] for r in state.values())
    empty = S.file_state(tmp_path)
    assert not any(r["var"] for r in empty) and all(r["değişme"] is None for r in empty)
