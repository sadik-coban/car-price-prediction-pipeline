"""
test_real_data_errors.py
4. defterin metninde gösterilmeden kullanılan iki veri cümlesi, gerçek veride (data işareti):
  - 3.4'teki il gruplaması `location` kolonunun "ilçe, il" biçimine dayanıyor: her ilanda tam bir virgül var ve
    virgülden sonraki kısım boş değil.
  - 6.3: ileri testin eğitim kümesinde son taramadan önce kalkmış ilanlar var (son taramada hiç görülmeyen ilanlar);
    eğitim fiyatlarının bir kısmı bu ilanların eski fiyatları.
"""
import pytest

from lib import data, forward


@pytest.mark.data
def test_location_is_district_comma_city():
    location = data.add_features(data.load_rows())["location"]
    assert location.notna().all()
    assert (location.str.count(",") == 1).all()
    assert (location.str.split(",").str[-1].str.strip() != "").all()


@pytest.mark.data
def test_forward_training_holds_listings_gone_before_the_last_snapshot():
    rows = data.add_features(data.load_rows(latest_only=False))
    train, _, _, test_snap = forward.split(rows)
    in_last = set(rows.loc[rows["snap"] == test_snap, "ad_id"])
    gone = ~train["ad_id"].isin(in_last)
    assert 0 < gone.sum() < len(train)
