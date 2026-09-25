"""
test_provenance.py
EN: Stale metrics: every metrics file records in _meta.source the sha256 of the script that wrote it and of the
    analysis/lib modules it had loaded (analysis/lib/common.py::save_metrics). If one of those files has changed
    since, the metrics no longer come from the current code — the fix is to rerun that script, not to edit
    anything here. Hashes only; no code is parsed.
TR: Bayat metrik: her metrik dosyası _meta.source'ta onu yazan betiğin ve yüklediği analysis/lib modüllerinin
    sha256'sını tutar (analysis/lib/common.py::save_metrics). O dosyalardan biri o zamandan beri değiştiyse
    metrikler artık bugünkü koddan gelmiyor — çözüm o betiği yeniden koşmak, burada bir şey düzenlemek değil.
    Yalnız hash; kod ayrıştırılmaz.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import analysis_coverage as AC  # noqa: E402

FILES = sorted((ROOT / "metrics").rglob("*.json"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT / "metrics").with_suffix("").as_posix())
def test_metrics_come_from_current_code(path):
    """
    EN: _meta.source exists and every hash in it matches the file today.
    TR: _meta.source var ve içindeki her hash bugünkü dosyayla aynı.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    stale = AC.stale_sources(doc)
    script = doc["_meta"]["script"]
    assert stale is not None, f"no _meta.source (written before provenance) | kaynak izi yok: rerun python analysis/{script}"
    assert not stale, (f"stale metrics | bayat metrik — changed since | o zamandan beri değişen: {stale}; "
                       f"rerun | yeniden koş: python analysis/{script}")


def test_every_script_file_is_recorded():
    """
    EN: The script named in _meta.script is the one hashed in _meta.source (no mix-up between files).
    TR: _meta.script'te adı geçen betik _meta.source'ta hash'lenen betik (dosyalar karışmamış).
    """
    for path in FILES:
        meta = json.loads(path.read_text(encoding="utf-8"))["_meta"]
        if meta.get("source"):
            assert list(meta["source"]["script"]) == [f"analysis/{meta['script']}"], path.name
