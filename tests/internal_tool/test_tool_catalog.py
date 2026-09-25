"""
test_tool_catalog.py
EN: The internal tool's script catalog: long lists are shortened for display (and nothing else changes), source
    text is served only from analysis/ and tests/, a script's files and run facts are found, the places a script is
    used come from the map, and stamping the map clears the builder-drift notice.
TR: İç aracın betik kataloğu: uzun listeler gösterim için kısaltılır (başka hiçbir şey değişmez), kaynak metin yalnız
    analysis/ ve tests/ altından verilir, bir betiğin dosyaları ve koşum bilgileri bulunur, bir betiğin kullanıldığı
    yerler eşlemeden gelir ve eşlemeyi damgalamak derleyici sapması uyarısını kaldırır.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import catalog  # noqa: E402


def test_shorten():
    """EN: Lists over the limit become text; the rest is kept. / TR: Sınırı aşan liste metin olur; gerisi kalır."""
    doc = {"a": list(range(60)), "b": {"c": [1, 2], "d": [[0] * 51]}, "e": "x"}
    out = catalog.shorten(doc)
    assert out["a"] == "[60 öğeli liste · list of 60]" and out["b"]["c"] == [1, 2]
    assert out["b"]["d"] == ["[51 öğeli liste · list of 51]"] and out["e"] == "x"
    assert doc["a"] == list(range(60))


@pytest.mark.parametrize("stem", ["builders/build_site_data", "../secrets", "db/build_duckdb", "", "/analysis/x"])
def test_read_source_refuses_other_folders(stem):
    """EN: Only analysis/ and tests/. / TR: Yalnız analysis/ ve tests/."""
    with pytest.raises(ValueError):
        catalog.read_source(stem)


def test_read_source_and_files():
    """EN: A script's source and test are served. / TR: Bir betiğin kaynağı ve testi verilir."""
    files = catalog.files_of("shap/02_oof_shap.py")
    assert files["source"] == "analysis/shap/02_oof_shap" and files["test_source"] == "tests/analysis/test_shap_02_oof_shap"
    assert "save_metrics" in catalog.read_source(files["source"])
    assert "def test_" in catalog.read_source(files["test_source"])


def test_meta_summary():
    """EN: Run facts from _meta; None without metrics. / TR: _meta'dan koşum bilgileri; metrik yoksa None."""
    doc = {"_meta": {"generated_at": "2026-09-25T10:00:00+03:00", "data_until": "2026-06-27", "run_id": None,
                     "db": {"sha256": "abc", "rows": 5}}}
    s = catalog.meta_summary(doc, silver_sha="abc")
    assert s == {"generated_at": "2026-09-25T10:00:00+03:00", "data_until": "2026-06-27", "run_id": None,
                 "db_rows": 5, "stale": None, "db_matches": True}
    assert catalog.meta_summary(doc, silver_sha="zzz")["db_matches"] is False
    assert catalog.meta_summary(doc)["db_matches"] is None and catalog.meta_summary(None) is None


def test_real_metrics_are_current():
    """EN: A real metrics file: code unchanged since it ran. / TR: Gerçek bir metrik: koştuğundan beri kod değişmedi."""
    s = catalog.meta_summary(catalog.read_json(catalog.files_of("08_residuals.py")["metrics"]))
    assert s["stale"] == [] and s["generated_at"]


def test_sections_of():
    """EN: 07_lofo feeds technical §6 and the business note. / TR: 07_lofo teknik §6'yı ve karar notunu besler."""
    places = [(r, i) for r, i, _ in catalog.sections_of("07_lofo.py", catalog.load_map())]
    assert ("technical", 6) in places and ("business", 1) in places


def test_stamp_clears_drift(tmp_path):
    """EN: After stamping, no drift. / TR: Damgadan sonra sapma yok."""
    path = tmp_path / "section_map.json"
    shutil.copy(catalog.MAP_PATH, path)
    m = json.loads(path.read_text(encoding="utf-8"))
    m["reviewed_against"] = {}
    path.write_text(json.dumps(m), encoding="utf-8")
    assert catalog.map_drift(catalog.load_map(path)) == list(catalog.REVIEWED)
    catalog.stamp(path)
    assert catalog.map_drift(catalog.load_map(path)) == []
