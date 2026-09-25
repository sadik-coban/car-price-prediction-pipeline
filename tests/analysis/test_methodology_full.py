"""
test_methodology_full.py
EN: The heavy methodology tests (marker `full`; minutes; not in the fast gate). They train models on the real data
    in a separate process (methodology_checks.py) and check the evaluation setup itself: folds without ad_id
    overlap, no signal found in a shuffled target, bit-identical reruns. Run with:
        python -m pytest -m full tests/analysis
TR: Ağır metodoloji testleri (`full` işareti; dakikalar; hızlı kapıda değil). Gerçek veride ayrı bir süreçte
    (methodology_checks.py) model eğitir ve değerlendirme kurulumunun kendisini sınar: ad_id örtüşmesiz
    fold'lar, karıştırılmış hedefte sinyal bulunmaması, bit-birebir aynı yeniden koşum. Koşum:
        python -m pytest -m full tests/analysis
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKS = Path(__file__).with_name("methodology_checks.py")
pytestmark = [pytest.mark.full,
              pytest.mark.skipif(not (ROOT / "data" / "cars.duckdb").exists(),
                                 reason="data/cars.duckdb not in this clone | bu klonda yok")]


def run(check):
    """EN: Runs one check in its own process. Returns: its JSON result. / TR: Bir denetimi kendi sürecinde koşar."""
    r = subprocess.run([sys.executable, str(CHECKS), check], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_folds_have_no_ad_id_overlap():
    """
    EN: Every listing is in exactly one fold and no ad_id appears in two folds (dedup before the split).
    TR: Her ilan tam bir fold'da ve hiçbir ad_id iki fold'da değil (bölmeden önce tekilleştirme).
    """
    out = run("folds")
    assert out["rows"] == out["unique_ad_ids"] == out["rows_in_folds"] == out["distinct_rows_in_folds"]
    assert out["ad_id_overlap"] == 0


def test_shuffled_target_has_no_signal():
    """
    EN: With the price shuffled across listings the OOF R² stays near 0 (≤ 0.05); a leak would lift it.
    TR: Fiyat ilanlar arasında karıştırılınca OOF R² 0 civarında kalır (≤ 0,05); bir sızıntı onu yükseltirdi.
    """
    assert run("shuffled")["r2_log_shuffled"] <= 0.05


def test_rerun_is_bit_identical():
    """EN: Same data, same seed → same predictions and tree counts. / TR: Aynı veri, aynı tohum → aynı tahmin ve ağaç."""
    out = run("determinism")
    assert out["max_abs_diff"] == 0.0 and out["trees_equal"]
