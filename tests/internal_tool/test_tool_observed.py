"""
test_tool_observed.py
EN: The internal tool's view of the register of observed values: one row per raw field with how it is kept, value
    and format tables with shares summing to 100, empty tables for count-only fields, the silver and series tables —
    on the real register file (in git, no data needed).
TR: İç aracın gözlenen değerler kaydı görünümü: ham alan başına nasıl tutulduğuyla bir satır, payları 100'e toplanan
    değer ve biçim tabloları, yalnız sayılan alanlarda boş tablo, silver ve seri tabloları — gerçek kayıt dosyasında
    (git'te, veri gerekmez).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import observed as O  # noqa: E402

REG = O.load()


def test_fields_table():
    """EN: One row per raw field; kinds as documented. / TR: Ham alan başına bir satır; türler belgelendiği gibi."""
    t = O.fields_table(REG)
    assert len(t) == len(REG["raw"]["fields"]) and set(t["tutulan"]) <= {"değerler", "biçimler", "yalnız sayı"}
    row = t.set_index("alan").loc["Genel Bakış - Plaka Uyruğu"]
    assert row["tutulan"] == "değerler" and row["kayıt"] + row["eksik"] == REG["raw"]["records"]


@pytest.mark.parametrize("field, kinds", [("Genel Bakış - Plaka Uyruğu", {"değerler"}), ("Fiyat", {"biçimler"}),
                                          ("KısaBilgi - Motor Hacmi", {"değerler", "biçimler"}), ("url", set())])
def test_field_tables(field, kinds):
    """EN: The kept tables; shares sum to 100. / TR: Tutulan tablolar; paylar 100'e toplanır."""
    tables = O.field_tables(REG, field)
    assert set(tables) == kinds
    for frame in tables.values():
        assert frame["%"].sum() == pytest.approx(100, abs=0.1)


def test_silver_and_series():
    """EN: NULL shows as "(boş)"; one row per pair. / TR: NULL "(boş)" görünür; çift başına bir satır."""
    plate = O.silver_table(REG, "gb_plate_origin")
    assert "(boş)" in set(plate["değer"])
    series = O.series_table(REG)
    assert len(series) == sum(len(m) for m in REG["silver"]["series_models"].values())
