"""
test_missing.py
notebooks/lib/missing.py: birlikte eksik payı, birlikte boş bloklar ve kb/gb ikiz tablosu küçük, elle kurulmuş
tablolarla. Gerçek veri okunmaz.
"""
import numpy as np
import pandas as pd
import pytest

from lib import missing

NA = np.nan


def test_co_missing_share():
    table = pd.DataFrame({"a": [NA, NA, 1, 1], "b": [NA, 1, NA, 1], "c": [1, 1, 1, 1]})
    assert missing.co_missing_share(table, ["a", "b"]) == pytest.approx(1 / 3)   # ikisi birden 1, en az biri 3
    assert missing.co_missing_share(table, ["a", "a"]) == 1.0
    assert missing.co_missing_share(table, ["c"]) == 0.0                          # hiç boş yok


def test_comissing_blocks():
    table = pd.DataFrame({
        "a": [NA, NA, 1, 1, 1, 1], "b": [NA, NA, 1, 1, 1, 1],       # hep birlikte boş
        "c": [1, 1, NA, NA, NA, 1], "d": [1, 1, NA, NA, NA, 1], "e": [1, 1, NA, NA, NA, 1],
        "f": [NA, 1, 1, NA, 1, 1]})                                   # kimseyle birlikte değil
    blocks = missing.comissing_blocks(table, ["a", "b", "c", "d", "e", "f"], max_dist=0.02)
    assert blocks == [["c", "d", "e"], ["a", "b"]]                    # büyükten küçüğe, tek kolon blok değil


def test_comissing_blocks_needs_two_columns():
    assert missing.comissing_blocks(pd.DataFrame({"a": [NA, 1]}), ["a"], max_dist=0.02) == []


def twin_frame():
    """İkiz çiftleri: yıl birebir aynı, çekişte gb çoğu boş, kasa tipinde değerler farklı, ağır hasar aynı."""
    return pd.DataFrame({
        "kb_year": [2015, 2016, 2017, 2018, 2019], "gb_year": [2015, 2016, 2017, 2018, 2019],
        "kb_drivetrain": ["Arkadan İtiş"] * 5, "gb_drivetrain": ["Arkadan İtiş", None, None, None, None],
        "kb_body_type": ["Sedan", "Coupe", "Sedan", None, "Sedan"],
        "gb_body_type": ["Sedan, 5 Koltuk", "Coupe", "Sedan", None, "Sedan"],
        "kb_is_heavy_damaged": pd.array([True, False, pd.NA, pd.NA, False], dtype="boolean"),
        "is_heavy_damaged": pd.array([True, False, pd.NA, pd.NA, False], dtype="boolean"),
    })


def test_twin_table():
    used = {"gb_year", "kb_drivetrain", "kb_body_type", "is_heavy_damaged"}
    twins = missing.twin_table(twin_frame(), used, by_hand={"gb_body_type": "kb daha genel"}).set_index("kb")
    assert twins.loc["kb_year", ["kept", "dropped", "reason"]].tolist() == ["gb_year", "kb_year", "identical"]
    assert twins.loc["kb_drivetrain", ["kept", "dropped", "reason"]].tolist() == ["kb_drivetrain", "gb_drivetrain",
                                                                                   "empty"]
    assert twins.loc["kb_drivetrain", "gb_missing_pct"] == 80.0
    assert twins.loc["kb_body_type", "reason"] == "by_hand"
    assert twins.loc["kb_body_type", "same_pct"] == 80.0                # ikisi de boş olan satır aynı sayılır
    assert twins.loc["kb_is_heavy_damaged", ["gb", "dropped", "reason"]].tolist() == [
        "is_heavy_damaged", "kb_is_heavy_damaged", "identical"]


def test_twin_table_when_neither_side_is_used():
    twins = missing.twin_table(twin_frame(), {"kb_drivetrain", "kb_body_type", "is_heavy_damaged"},
                               by_hand={"gb_body_type": "kb daha genel"}).set_index("kb")
    assert pd.isna(twins.loc["kb_year", "kept"]) and twins.loc["kb_year", "dropped"] == "gb_year"


def test_twin_table_stops_without_a_reason():
    # kasa tipinin elle gerekçesi yoksa: iki taraf aynı değil, atılan taraf çoğu dolu -> gerekçe yok
    with pytest.raises(ValueError, match="gb_body_type"):
        missing.twin_table(twin_frame(), {"gb_year", "kb_drivetrain", "kb_body_type", "is_heavy_damaged"}, by_hand={})
