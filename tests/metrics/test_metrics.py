"""
test_metrics.py
EN: The metrics layer of the fast gate: every metrics/*.json comes from the same database and model run (the
    builders' consistency gate), has the same files, keys and key types as the baseline, and the same values
    apart from the documented exemptions. An intended change is taken into the baseline with a reason:
    python tools/snapshot_metrics.py --accept "<reason>".
TR: Hızlı kapının metrik katmanı: her metrics/*.json aynı veritabanı ve model koşumundan geliyor (derleyicilerin
    tutarlılık kapısı), referansla aynı dosyalara, anahtarlara ve anahtar tiplerine sahip, değerleri de
    belgelenmiş istisnalar dışında aynı. Bilinçli bir değişiklik referansa gerekçeyle alınır:
    python tools/snapshot_metrics.py --accept "<gerekçe>".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "builders"))
import snapshot_metrics as SM  # noqa: E402
from report_lib import metrics_view as MV  # noqa: E402

HINT = ('if intended: python tools/snapshot_metrics.py --accept "<reason>" | bilinçliyse gerekçeyle onayla')


@pytest.fixture(scope="module")
def diff():
    """EN: Current metrics vs the baseline, computed once. / TR: Güncel metrikler ile referans, bir kez hesaplanır."""
    return SM.compare()


def test_consistency_gate():
    """
    EN: Every metrics file comes from the same database and, if model-derived, the same run.
    TR: Her metrik dosyası aynı veritabanından ve modelden türüyorsa aynı koşumdan geliyor.
    """
    try:
        MV.load_view(ROOT)
    except SystemExit as e:
        pytest.fail(str(e))


def test_same_metrics_files(diff):
    """EN: No metrics file appeared or vanished. / TR: Hiçbir metrik dosyası eklenmedi ya da kaybolmadı."""
    assert not diff["files"], "\n".join(diff["files"] + [HINT])


def test_same_keys_and_types(diff):
    """EN: No key added, dropped or retyped. / TR: Hiçbir anahtar eklenmedi, düşmedi ya da tipi değişmedi."""
    assert not diff["shape"], "\n".join(diff["shape"][:40] + [f"... {len(diff['shape'])} total", HINT])


def test_same_values(diff):
    """EN: Every non-exempt value is unchanged. / TR: İstisna dışı her değer aynı."""
    assert not diff["values"], "\n".join(diff["values"][:40] + [f"... {len(diff['values'])} total", HINT])


def test_exemptions_are_documented_and_used():
    """
    EN: Every exemption has an EN and a TR reason and still matches at least one current key (no stale entry).
    TR: Her istisnanın EN ve TR gerekçesi var ve hâlâ en az bir güncel anahtara uyuyor (bayat girdi yok).
    """
    exemptions = SM.load_exemptions()
    shape, _ = SM.snapshot()
    for e in exemptions:
        assert e["reason"].get("en") and e["reason"].get("tr"), e
        assert any(SM.is_exempt(name, key, [e]) for name, keys in shape.items() for key in keys), \
            f"exemption matches nothing | hiçbir anahtara uymuyor: {e['file']} {e['path']}"
