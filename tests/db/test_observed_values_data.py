"""
test_observed_values_data.py
EN: The register of observed values (db/observed_values.json) still describes the real data: a fresh scan of
    data/raw and data/cars.duckdb gives the same register (the generation stamp left out); and the build with its
    register gate and non-guessing parsers still gives the current DB — the real raw data is built into a temp
    folder (data/ is not touched) and its tables equal data/cars.duckdb row for row. Marked `data` (reads the real
    data, not in the fast gate: python tools/verify.py --data); skipped in a clone without data/.
TR: Gözlenen değerler kaydı (db/observed_values.json) hâlâ gerçek veriyi anlatıyor: data/raw ve data/cars.duckdb'nin
    taze taraması aynı kaydı verir (üretim damgası hariç); kayıt kapılı ve tahmin etmeyen ayrıştırıcılı kurulum hâlâ
    bugünkü DB'yi verir — gerçek ham veri geçici bir klasöre kurulur (data/'ya dokunulmaz) ve tabloları
    data/cars.duckdb ile satır satır aynıdır. `data` işaretli (gerçek veriyi okur, hızlı kapıda değil:
    python tools/verify.py --data); data/ olmayan bir klonda atlanır.
"""
import duckdb
import pytest

import build_duckdb as BD
from lib import observed_values as OV

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not OV.DB_PATH.exists() or not OV.raw_files(), reason="no local data | yerel veri yok")]


def test_register_matches_the_data():
    """EN: Fresh scan = the register file. / TR: Taze tarama = kayıt dosyası."""
    assert OV.differences(OV.load(), OV.collect()) == [], "rerun | yeniden yaz: python tools/observed_values.py"


def test_rebuild_equals_the_current_db(tmp_path):
    """
    EN: The real raw data built into a temp folder = data/cars.duckdb (car_listings, price_history,
        duplicate_ad_ids; EXCEPT ALL both ways is empty).
    TR: Gerçek ham verinin geçici klasöre kurulumu = data/cars.duckdb (car_listings, price_history, duplicate_ad_ids;
        iki yönde EXCEPT ALL boş).
    """
    out = tmp_path / "cars.duckdb"
    BD.build(out)
    con = duckdb.connect(str(out), read_only=True)
    try:
        con.execute(f"ATTACH '{OV.DB_PATH.as_posix()}' AS cur (READ_ONLY)")
        for table in ("car_listings", "price_history", "duplicate_ad_ids"):
            for a, b in ((table, f"cur.{table}"), (f"cur.{table}", table)):
                n = con.execute(f"SELECT COUNT(*) FROM (SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b})").fetchone()[0]
                assert n == 0, f"{a} has {n} rows not in {b}"
    finally:
        con.close()
