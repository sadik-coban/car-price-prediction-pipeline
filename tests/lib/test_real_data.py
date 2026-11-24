"""
test_real_data.py
notebooks/lib fonksiyonları gerçek veride (data/cars.duckdb ve data/raw, salt okunur). Yavaş katman: `data` işareti.
Eski betiklerin durma koşullarından lib'e düşenler burada sınanır: ilan başına tek satır, her satırın segmenti ve onu
çözen yol, görülmemiş plaka ya da segment değeri yok, ham hasar cevapları veritabanının bayraklarıyla bire bir aynı.
Defterlerin veri hakkında söylediği ama kendisi göstermediği cümleler de burada sınanır (motor değeri kolonları).
"""
import json

import pytest

from lib import cleaning, data

pytestmark = pytest.mark.data


def test_listings_are_the_latest_row_of_each_ad_in_ad_id_order():
    rows = data.load_rows(latest_only=False)
    listings = data.load_rows()
    assert listings["ad_id"].is_unique and listings["ad_id"].is_monotonic_increasing
    assert set(listings["ad_id"]) == set(rows["ad_id"])
    latest = rows.groupby("ad_id")["search_date"].max()
    assert (listings.set_index("ad_id")["search_date"] == latest).all()


def test_every_row_gets_its_features():
    out = data.add_features(data.load_rows(latest_only=False))      # segment çözülemezse durur
    assert out["segment"].notna().all()
    assert set(data.FEATURES) <= set(out.columns)


def test_every_plate_value_has_a_kind():
    plates = data.query("SELECT gb_plate_origin FROM car_listings")["gb_plate_origin"]
    assert set(cleaning.plate_kind(plates)) <= {"tr", "blue", "empty"}


def test_every_segment_value_has_an_id():
    segments = data.query("SELECT gb_segment FROM car_listings")["gb_segment"]
    assert set(cleaning.raw_segment(segments).dropna()) <= set(cleaning.RAW_SEGMENTS.values())


def test_segment_paths_agree_with_the_segments():
    listings = data.add_features(data.load_rows())
    resolved = [data.resolve_segment(s, m) for s, m in zip(listings["series"], listings["model"])]
    assert [segment for segment, _ in resolved] == listings["segment"].tolist()
    assert {path for _, path in resolved} == {"series_map", "base_class", "model_table"}


def test_engine_values_are_positive_where_present():
    # hedonik regresyon yalnız boş güç ve hacmi dışarıda bırakır; sıfır ya da eksi bir değer sessizce girerdi
    listings = data.add_features(data.load_rows())
    for column in ["power_hp_val", "engine_cc_val"]:
        assert (listings[column].dropna() > 0).all(), column


@pytest.mark.parametrize("measure", ["engine_cc", "power_hp"])
def test_database_engine_value_is_the_midpoint_of_the_bounds(measure):
    # 1. defter, 2.1: veritabanının *_val kolonu iki sınır varsa orta nokta, tek sınır varsa o sınır, sınır yoksa boş
    rows = data.query(f"SELECT {measure}_low AS low, {measure}_up AS up, {measure}_val AS val, "
                      f"{measure}_is_range AS is_range FROM car_listings")
    low, up, val = rows["low"], rows["up"], rows["val"]
    is_range = rows["is_range"].fillna(False).astype(bool)
    both = low.notna() & up.notna()
    assert (val[both] == (low[both] + up[both]) / 2).all()
    assert (val[low.notna() & up.isna()] == low[low.notna() & up.isna()]).all()
    assert (val[low.isna() & up.notna()] == up[low.isna() & up.notna()]).all()
    assert val[low.isna() & up.isna()].isna().all()
    assert (low[is_range & both] < up[is_range & both]).all()        # aralıkta alt sınır üst sınırdan küçük
    assert (low[~is_range & both] == up[~is_range & both]).all()     # kesin değerde alt = üst = değer
    assert not (~is_range & (low.isna() != up.isna())).any()         # tek sınır yalnız aralıkta olur


def test_engine_rule_leaves_a_value_empty_only_where_the_site_gave_no_bound():
    # hacim kuralı üst sınırı alır: yalnız alt sınırı olan bir hacim ("… cm3 ve üzeri") boş kalırdı. Site bu biçimi
    # göstermedi; gösterirse önce veritabanı kurulumu durur, biçim tanıtılıp kural uyarlanmazsa bu test yakalar
    listings = data.add_features(data.load_rows(latest_only=False))
    for measure in ["engine_cc", "power_hp"]:
        no_bound = listings[f"{measure}_low"].isna() & listings[f"{measure}_up"].isna()
        assert (listings[f"{measure}_val"].isna() == no_bound).all(), measure


def test_raw_damage_answers_equal_the_database_flags():
    maps = json.loads(data.DAMAGE_MAPPINGS.read_text(encoding="utf-8"))
    damage = cleaning.read_damage_lists(maps["panels"], maps["statuses"])
    assert not damage.duplicated(["ad_id", "snap"]).any()
    listings = data.add_features(data.load_rows())
    matched = listings.merge(damage, on=["ad_id", "snap"], how="inner", validate="one_to_one")
    assert len(matched) == len(listings)
    rebuilt = cleaning.flags_from_statuses(matched[list(maps["panels"].values())], maps["statuses"])
    assert (rebuilt == matched[rebuilt.columns].fillna(0).astype(int)).all().all()
