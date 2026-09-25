"""
test_tool_filters.py
EN: The explorer's condition engine on a small frame: column kinds and their operators, every operator (empty =
    NULL or blank; "in" with the empty marker; "not_in" keeps empty rows unless it is chosen; ranges drop NULLs and
    allow open ends; text search blind to case and Turkish letters and to non-breaking spaces), AND-joining with the
    rows left after each step, conditions on a column the source lacks reported instead of applied, the readable
    summary, and the row sets (latest row per ad, the analysis set as load_clean, one snapshot).
TR: Gezginin koşul motoru küçük bir çerçevede: kolon türleri ve operatörleri, her operatör (boş = NULL ya da boş
    metin; boş işaretiyle "şunlardan biri"; "hiçbiri" boş satırları, boş seçilmedikçe tutar; aralık NULL'u atar ve
    açık uca izin verir; büyük/küçük harfe, Türkçe harflere ve bölünmez boşluğa duyarsız metin araması), her adımdan
    sonra kalan satırlarla VE birleşimi, kaynağın sahip olmadığı kolondaki koşulun uygulanmayıp bildirilmesi, okunur
    özet ve satır kümeleri (ilan başına son satır, load_clean gibi analiz kümesi, tek tarama).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import filters as F  # noqa: E402

C = F.Condition


@pytest.fixture
def df():
    """EN: Five rows covering every kind and empties. / TR: Her türü ve boşları kapsayan beş satır."""
    return pd.DataFrame({
        "ad_id": pd.array([1, 1, 2, 3, None], dtype="Int64"),
        "plate": ["(TR) Türkiye", "(TR) Türkiye", "Mavi plakalı", None, " "],
        "price": [1_000_000.0, 1_100_000.0, 2_000_000.0, None, 0.0],
        "heavy": pd.array([True, False, None, True, False], dtype="boolean"),
        "text": ["YABANCIDAN\xa0YABANCIYA", "temiz", "MA PLAKALIDIR", None, "İstanbul"],
        "snap": pd.to_datetime(["2026-01-18", "2026-03-21", "2026-03-21", "2026-01-18", "2026-01-18"]),
        "gb_plate_origin": ["(TR) Türkiye", "(TR) Türkiye", None, "(TR) Türkiye", "(TR) Türkiye"],
    })


def rows(df, *conds):
    """EN: Row numbers kept by the conditions. / TR: Koşulların tuttuğu satır numaraları."""
    return list(F.apply(df, list(conds))[0].nonzero()[0])


def test_kinds_and_ops(df):
    """EN: Kinds from dtypes and distinct counts. / TR: Türler dtype ve farklı değer sayısından."""
    kinds = {c: F.column_kind(df[c]) for c in ("ad_id", "plate", "price", "heavy", "snap")}
    assert kinds == {"ad_id": "number", "plate": "category", "price": "number", "heavy": "bool", "snap": "date"}
    assert F.column_kind(df["text"], max_categories=2) == "text"
    assert "in" in F.ops_for(df["text"], "text") and F.ops_for(df["heavy"], "bool") == F.OPS["bool"]


def test_empty_and_in(df):
    """EN: Empty = NULL or blank; the empty marker in "in"/"not_in". / TR: Boş = NULL ya da boş metin; boş işareti."""
    assert rows(df, C("plate", "is_null")) == [3, 4]
    assert rows(df, C("plate", "not_null")) == [0, 1, 2]
    assert rows(df, C("plate", "in", ("Mavi plakalı",))) == [2]
    assert rows(df, C("plate", "in", ("Mavi plakalı", F.NULL))) == [2, 3, 4]
    assert rows(df, C("plate", "not_in", ("(TR) Türkiye",))) == [2, 3, 4]
    assert rows(df, C("plate", "not_in", ("(TR) Türkiye", F.NULL))) == [2]


def test_between_and_bool(df):
    """EN: Ranges drop NULL, open ends; booleans. / TR: Aralık NULL'u atar, açık uç; mantıksal."""
    assert rows(df, C("price", "between", (1_000_000, 1_500_000))) == [0, 1]
    assert rows(df, C("price", "between", (None, 1_000_000))) == [0, 4]
    assert rows(df, C("price", "between", (None, None))) == [0, 1, 2, 4]
    assert rows(df, C("snap", "between", (pd.Timestamp("2026-03-01"), None))) == [1, 2]
    assert rows(df, C("heavy", "is_true")) == [0, 3] and rows(df, C("heavy", "is_false")) == [1, 4]
    assert rows(df, C("heavy", "is_null")) == [2]


def test_text_search(df):
    """EN: Case / Turkish-letter / space blind contains; regex; equals. / TR: Harf ve boşluk duyarsız içerir."""
    assert rows(df, C("text", "contains", ("yabancıdan yabancıya",))) == [0]
    assert rows(df, C("text", "contains", ("plakalı",))) == [2]
    assert rows(df, C("text", "contains", ("istanbul",))) == [4]
    assert rows(df, C("text", "not_contains", ("plaka",))) == [0, 1, 3, 4]
    assert rows(df, C("text", "regex", (r"^ma\s",))) == [2]
    assert rows(df, C("text", "equals", ("temiz",))) == [1]
    assert F.fold("PLAKALIDIR  İstanbul\xa0ŞÇ") == "plakalidir istanbul sc"


def test_and_steps_and_missing(df):
    """EN: AND with counts; missing columns reported; disabled ignored. / TR: VE ve sayılar; eksik kolon bildirilir."""
    conds = [C("plate", "in", ("(TR) Türkiye",)), C("heavy", "is_true"), C("nope", "is_null"),
             C("price", "between", (0, 1)), ]
    conds[3].enabled = False
    mask, steps, missing = F.apply(df, conds)
    assert list(mask.nonzero()[0]) == [0]
    assert [n for _, n in steps] == [2, 1] and [c.column for c in missing] == ["nope"]


def test_describe(df):
    """EN: Readable phrases. / TR: Okunur ifadeler."""
    assert F.describe(C("plate", "in", ("Mavi plakalı",))) == "plate ∈ {Mavi plakalı}"
    assert F.describe(C("price", "between", (1_000_000.0, None))) == "price 1.000.000–…"
    assert F.describe(C("text", "contains", ("ma plaka",))) == "text içerir “ma plaka”"
    assert F.describe(C("plate", "is_null")) == "plate boş"
    off = C("heavy", "is_true", enabled=False)
    assert F.summary([C("plate", "is_null"), off, C("heavy", "is_false")]) == "plate boş VE heavy hayır"
    assert F.summary([]) == "koşul yok"


def test_row_sets(df):
    """EN: latest per ad (no-id rows kept), analysis set, snapshot. / TR: İlan başına son, analiz kümesi, tarama."""
    assert list(F.row_set(df, "all", "snap").nonzero()[0]) == [0, 1, 2, 3, 4]
    assert list(F.row_set(df, "latest", "snap").nonzero()[0]) == [1, 2, 3, 4]
    assert list(F.row_set(df, "snapshot", "snap", "2026-01-18").nonzero()[0]) == [0, 3, 4]
    # EN: price > 0 and TR plate first, then the latest row per ad | TR: önce fiyat > 0 ve TR plaka, sonra son satır
    assert list(F.row_set(df, "analysis", "snap").nonzero()[0]) == [1]
    with pytest.raises(ValueError):
        F.row_set(df, "other", "snap")


def test_latest_ties_take_the_later_row():
    """EN: Same ad and date: the later row wins. / TR: Aynı ilan ve tarih: sonraki satır kazanır."""
    d = pd.DataFrame({"ad_id": [5, 5], "snap": ["a", "a"]})
    assert list(F.latest_per_ad(d, "ad_id", "snap")) == [False, True]
