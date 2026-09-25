"""
test_tool_population.py
EN: The internal tool's explorer on the REAL data (marked `data`, not in the fast gate: python tools/verify.py
    --data). Raw: every scraped record is there (45,335), 38 blue-plate records, 138 without a plate field. Silver:
    the explorer's analysis set is the analyses' set — as many rows as 01_dedup_leakage's n_dedup and the median
    price of 04_target — and gold gives the same set with no unknown heavy damage. Only counts are compared; no id
    is printed.
TR: İç aracın veri gezgini GERÇEK veride (`data` işaretli, hızlı kapıda değil: python tools/verify.py --data). Ham:
    kazınan her kayıt orada (45.335), 38 mavi plakalı kayıt, plaka alanı olmayan 138 kayıt. Silver: gezginin analiz
    kümesi analizlerin kümesidir — 01_dedup_leakage'ın n_dedup'u kadar satır ve 04_target'ın medyan fiyatı — ve gold
    aynı kümeyi bilinmeyen ağır hasar olmadan verir. Yalnız sayılar karşılaştırılır; hiçbir kimlik basılmaz.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import filters as F, sources as S  # noqa: E402

DATA = ROOT / "data"
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not (DATA / "cars.duckdb").exists() or not S.data_files("raw", DATA),
                                 reason="no local data | yerel veri yok")]


def metric(name):
    """EN: A metrics file. / TR: Bir metrik dosyası."""
    return json.loads((ROOT / "metrics" / f"{name}.json").read_text(encoding="utf-8"))


def test_raw_counts():
    """EN: All records; blue plates; missing plate field. / TR: Bütün kayıtlar; mavi plaka; plaka alanı yok."""
    raw, stats = S.load("raw", DATA)
    plate = "Genel Bakış - Plaka Uyruğu"
    assert stats["broken"] == 0 and len(raw) == 45_335
    assert int(F.apply(raw, [F.Condition(plate, "in", ("Mavi plakalı",))])[0].sum()) == 38
    assert int(F.apply(raw, [F.Condition(plate, "is_null")])[0].sum()) == 138


def test_analysis_set_is_the_analyses_set():
    """EN: n_dedup rows and 04_target's median; gold alike. / TR: n_dedup satır ve 04_target medyanı; gold da öyle."""
    n_dedup = metric("01_dedup_leakage")["meta"]["n_dedup"]
    median = metric("04_target")["domain"]["price_dist"]["median"]
    for source in ("silver", "gold"):
        df, _ = S.load(source, DATA)
        base = F.row_set(df, "analysis", "search_date")
        assert int(base.sum()) == n_dedup, source
        assert float(df.loc[base, "price"].median()) == median, source
    gold, _ = S.load("gold", DATA)
    assert int(F.apply(gold, [F.Condition("is_heavy_damaged", "is_null")])[0].sum()) == 0
