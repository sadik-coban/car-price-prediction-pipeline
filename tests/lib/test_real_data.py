"""
test_real_data.py
notebooks/lib fonksiyonları gerçek veride (data/cars.duckdb ve data/raw, salt okunur). Yavaş katman: `data` işareti.
Eski betiklerin durma koşullarından lib'e düşenler burada sınanır: ilan başına tek satır, her satırın segmenti,
görülmemiş plaka değeri yok, ham hasar cevapları veritabanının bayraklarıyla bire bir aynı.
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


def test_raw_damage_answers_equal_the_database_flags():
    maps = json.loads(data.DAMAGE_MAPPINGS.read_text(encoding="utf-8"))
    damage = cleaning.read_damage_lists(maps["panels"], maps["statuses"])
    assert not damage.duplicated(["ad_id", "snap"]).any()
    listings = data.add_features(data.load_rows())
    matched = listings.merge(damage, on=["ad_id", "snap"], how="inner", validate="one_to_one")
    assert len(matched) == len(listings)
    rebuilt = cleaning.flags_from_statuses(matched[list(maps["panels"].values())], maps["statuses"])
    assert (rebuilt == matched[rebuilt.columns].fillna(0).astype(int)).all().all()
