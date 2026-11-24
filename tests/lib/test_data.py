"""
test_data.py
notebooks/lib/data.py: satır seçimi (TR plakası, fiyat > 0, ilan başına en son satır), türetilen kolonlar ve segment
kuralı küçük, elle kurulmuş tablolarla; segment kuralının gözlenen değerler kaydıyla (db/observed_values.json) iki
yönlü uyumu. Gerçek veri okunmaz.
"""
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from lib import data

REGISTER = json.loads((data.ROOT / "db" / "observed_values.json").read_text(encoding="utf-8"))
FLAGS = [f"{part}_{suffix}" for part in list(data.SINGLE_PANELS.values())
         + [p for panels in data.PANEL_GROUPS.values() for p in panels] for suffix in data.FLAG_KINDS.values()]


def raw_frame(**overrides):
    """İki satırlık ham tablo; her değer verinin gösterdiği bir değer. Bayraklar 0, verilenler üstüne yazılır."""
    frame = pd.DataFrame({
        "ad_id": [1, 2], "search_date": pd.to_datetime(["2026-01-18", "2026-06-27"]),
        "brand": ["bmw", "audi"], "series": ["3 Serisi", "A4"], "model": ["320i Sport Line", "A4 Sedan 2.0 TDI"],
        "gb_year": [2015, 2026], "gb_mileage": [180000, 10000], "price": [1250000.0, 2500000.0],
        "power_hp_low": [170.0, 126.0], "power_hp_up": [170.0, 150.0],
        "engine_cc_low": [1598.0, 1801.0], "engine_cc_up": [1598.0, 2000.0],
        "is_heavy_damaged": pd.array([pd.NA, True], dtype="boolean"),
        "kb_body_type": ["Sedan", None], "kb_drivetrain": ["Arkadan İtiş", "Önden Çekiş"],
        "kb_transmission": ["Otomatik", "Düz"], "kb_fuel": ["Benzin", "Dizel"],
        **{flag: pd.array([0, 0], dtype="Int64") for flag in FLAGS},
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame


def test_add_features_engine_age_and_heavy_damage():
    out = data.add_features(raw_frame())
    assert out["power_hp_val"].tolist() == [170.0, 138.0]          # iki sınırın ortalaması
    assert out["engine_cc_val"].tolist() == [1598.0, 2000.0]       # üst sınır
    assert out["vehicle_age"].tolist() == [11, 0]
    assert out["is_heavy_damaged"].tolist() == [0, 1]              # boş kayıt 0
    assert out["snap"].tolist() == ["2026-01-18", "2026-06-27"]


def test_add_features_damage_states_and_counts():
    raw = raw_frame(kaput_degisen=pd.array([1, pd.NA], dtype="Int64"),
                    kaput_boyali=pd.array([1, 0], dtype="Int64"),
                    tavan_lokal=pd.array([pd.NA, 1], dtype="Int64"),
                    door_fl_boyali=pd.array([1, pd.NA], dtype="Int64"),
                    door_rr_boyali=pd.array([1, 0], dtype="Int64"),
                    bumper_rear_degisen=pd.array([0, 1], dtype="Int64"))
    out = data.add_features(raw)
    assert out["hood_state"].tolist() == ["changed", "original"]   # değişen boyalıdan önce gelir
    assert out["roof_state"].tolist() == ["original", "local"]
    assert out["door_painted"].tolist() == [2, 0]                  # boş bayrak 0 sayılır
    assert out["bumper_changed"].tolist() == [0, 1]


def test_add_features_fills_categories_and_leaves_input_alone():
    raw = raw_frame()
    out = data.add_features(raw)
    assert out["kb_body_type"].tolist() == ["Sedan", "missing"]
    assert raw["kb_body_type"].isna().sum() == 1                   # girdi değişmedi
    assert raw["engine_cc_up"].tolist() == [1598.0, 2000.0] and "segment" not in raw.columns
    assert set(data.FEATURES) <= set(out.columns)


@pytest.mark.parametrize("series, model, segment", [
    ("3 Serisi", "320i Sport Line", "D"), ("A4", "A4 Sedan 2.0 TDI", "D"), ("M Serisi", "M5", "E"),
    ("M Serisi", "M235i xDrive", "C"), ("RS", "RS 6", "E"), ("S", "S3 2.0 TFSI Quattro", "C"),
    ("i Serisi", "i8 Premium Techno", "S"), ("Z Serisi", "Z4 2.0i sDrive", "S")])
def test_segment_of(series, model, segment):
    assert data.segment_of(series, model) == segment


@pytest.mark.parametrize("series, model, resolved", [
    ("3 Serisi", "320i Sport Line", ("D", "series_map")), ("M Serisi", "M5", ("E", "base_class")),
    ("RS", "RS 6", ("E", "base_class")), ("i Serisi", "i8 Premium Techno", ("S", "model_table")),
    ("M Serisi", "Z4 M Coupe", ("S", "model_table")), ("Yeni Seri", "X1", (None, None))])
def test_resolve_segment_gives_the_path(series, model, resolved):
    assert data.resolve_segment(series, model) == resolved


def test_segments_stop_on_an_unresolved_pair():
    # durmanın kendisini sınar: "Yeni Seri" hiç görülmemiş bir seri
    assert data.segments(["3 Serisi"], ["320i Sport Line"]) == ["D"]
    with pytest.raises(ValueError, match="Yeni Seri"):
        data.segments(["3 Serisi", "Yeni Seri"], ["320i Sport Line", "X1"])


def test_segment_rule_names_only_observed_series():
    """(a) kuralın andığı her seri veride geçiyor; her MODEL_SEG öneki kendi serisinin bir araç modeline uyuyor."""
    observed = {s for s, _ in REGISTER["silver"]["columns"]["series"]["values"] if s is not None}
    named = set(data.SEGMENT_MAP) | set(data.PERF_BASE) | set(data.MODEL_SEG)
    assert named <= observed, sorted(named - observed)
    for series, prefixes in data.MODEL_SEG.items():
        models = [m for m, _ in REGISTER["silver"]["series_models"][series]]
        for prefix in prefixes:
            assert any(str(m).startswith(prefix) for m in models), (series, prefix)


def test_every_observed_pair_gets_a_segment():
    """(b) veride görülen her (seri, araç modeli) çifti bir segmente çözülüyor."""
    unresolved = [(s, m) for s, models in REGISTER["silver"]["series_models"].items() for m, _ in models
                  if data.segment_of(s, m) is None]
    assert not unresolved, unresolved[:10]


def test_tr_plate_is_observed():
    observed = {v for v, _ in REGISTER["silver"]["columns"]["gb_plate_origin"]["values"]}
    assert data.TR_PLATE in observed


@pytest.fixture
def small_db(tmp_path, monkeypatch):
    """Küçük bir car_listings tablosu: tekrar görülen ilan, mavi plaka, boş plaka, sıfır fiyat."""
    path = tmp_path / "cars.duckdb"
    rows = pd.DataFrame({
        "ad_id": [30, 10, 10, 20, 40, 50],
        "search_date": pd.to_datetime(["2026-01-18", "2026-01-18", "2026-03-21", "2026-01-27", "2026-01-18",
                                       "2026-01-18"]),
        "price": [900000.0, 1000000.0, 950000.0, 0.0, 800000.0, 700000.0],
        "gb_plate_origin": [data.TR_PLATE, data.TR_PLATE, data.TR_PLATE, data.TR_PLATE, "Mavi plakalı", None],
    })
    with duckdb.connect(str(path)) as con:
        con.register("rows_view", rows)
        con.execute("CREATE TABLE car_listings AS SELECT * FROM rows_view")
    monkeypatch.setattr(data, "DB_PATH", path)
    return path


def test_load_rows_keeps_tr_plates_with_a_price(small_db):
    rows = data.load_rows(latest_only=False)
    assert rows["ad_id"].tolist() == [10, 10, 30]                  # ad_id, sonra tarih sırası
    assert rows["search_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-18", "2026-03-21", "2026-01-18"]


def test_load_rows_takes_the_latest_row_per_listing(small_db):
    rows = data.load_rows()
    assert rows["ad_id"].tolist() == [10, 30]
    assert rows["price"].tolist() == [950000.0, 900000.0]          # ilan 10'un son taramadaki fiyatı
    assert np.all(np.diff(rows["ad_id"]) > 0)
