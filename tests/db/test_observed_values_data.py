"""
test_observed_values_data.py
EN: The register of observed values (db/observed_values.json) still describes the real data: a fresh scan of
    data/raw and data/cars.duckdb gives the same register (the generation stamp left out). Marked `data` (reads the
    real data, not in the fast gate: python tools/verify.py --data); skipped in a clone without data/.
TR: Gözlenen değerler kaydı (db/observed_values.json) hâlâ gerçek veriyi anlatıyor: data/raw ve data/cars.duckdb'nin
    taze taraması aynı kaydı verir (üretim damgası hariç). `data` işaretli (gerçek veriyi okur, hızlı kapıda değil:
    python tools/verify.py --data); data/ olmayan bir klonda atlanır.
"""
import pytest

from lib import observed_values as OV

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not OV.DB_PATH.exists() or not OV.raw_files(), reason="no local data | yerel veri yok")]


def test_register_matches_the_data():
    """EN: Fresh scan = the register file. / TR: Taze tarama = kayıt dosyası."""
    assert OV.differences(OV.load(), OV.collect()) == [], "rerun | yeniden yaz: python tools/observed_values.py"
