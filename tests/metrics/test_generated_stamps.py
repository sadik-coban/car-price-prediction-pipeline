"""
test_generated_stamps.py
EN: Every JSON the pipeline generates says when it was written, in one format: a top-level _meta.generated_at
    with local time, its UTC offset and second precision (e.g. 2026-09-25T04:03:29+03:00). Covered: every
    metrics file (analysis/lib/common.py), the OOF info (analysis/lib/cv.py), site_data.json and
    column_labels.json (builders/build_site_data.py). The S3 manifest's stamp is checked in
    tests/db/test_publish_data_to_s3.py. data/ is not in git, so its files are skipped in a clone without them.
TR: Boru hattının ürettiği her JSON ne zaman yazıldığını tek biçimde söyler: üst düzeyde _meta.generated_at,
    yerel saat, UTC farkı ve saniye hassasiyeti (ör. 2026-09-25T04:03:29+03:00). Kapsam: her metrik dosyası
    (analysis/lib/common.py), OOF bilgisi (analysis/lib/cv.py), site_data.json ve column_labels.json
    (builders/build_site_data.py). S3 manifest'inin damgası tests/db/test_publish_data_to_s3.py'de sınanır.
    data/ git'te değil; o dosyalar olmayan bir klonda atlanır.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "builders"))
from report_lib import metrics_view as MV  # noqa: E402

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
METRICS = sorted((ROOT / "metrics").rglob("*.json"))
DATA = [ROOT / "data" / "site_data.json", ROOT / "data" / "serving" / "column_labels.json",
        ROOT / "data" / "analysis" / "oof_info.json"]


def check(stamp):
    """
    EN: Asserts one stamp has the shared format and parses as a tz-aware time.
    TR: Bir damganın ortak biçimde olduğunu ve saat dilimli bir zaman olarak ayrıştığını doğrular.
    """
    assert isinstance(stamp, str) and STAMP.match(stamp), f"stamp format | damga biçimi: {stamp!r}"
    assert datetime.fromisoformat(stamp).tzinfo is not None


@pytest.mark.parametrize("path", METRICS + DATA, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_generated_json_is_stamped(path):
    """
    EN: The file has _meta.generated_at in the shared format.
    TR: Dosyada ortak biçimde _meta.generated_at var.
    """
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT).as_posix()} not built in this clone | bu klonda yok")
    doc = json.loads(path.read_text(encoding="utf-8"))
    check((doc.get("_meta") or {}).get("generated_at"))


def test_builder_stamp_format():
    """
    EN: The builders' stamp helper writes the shared format.
    TR: Derleyicilerin damga yardımcısı ortak biçimi yazar.
    """
    check(MV.generated_stamp())


def test_site_keeps_the_metrics_time_apart():
    """
    EN: site_data.json keeps the newest metrics stamp as meta.metrics_generated_at, and it is not later than the
        build itself.
    TR: site_data.json en yeni metrik damgasını meta.metrics_generated_at olarak tutar ve bu, derlemenin
        kendisinden sonra değildir.
    """
    path = ROOT / "data" / "site_data.json"
    if not path.exists():
        pytest.skip("data/site_data.json not built in this clone | bu klonda yok")
    doc = json.loads(path.read_text(encoding="utf-8"))
    built, newest = doc["_meta"]["generated_at"], doc["meta"]["metrics_generated_at"]
    check(newest)
    assert "generated_at" not in doc["meta"]
    assert datetime.fromisoformat(newest) <= datetime.fromisoformat(built)
