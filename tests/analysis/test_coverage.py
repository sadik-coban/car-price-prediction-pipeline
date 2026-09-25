"""
test_coverage.py
EN: The analysis coverage gate (tools/analysis_coverage.py): every script in run_all.ORDER has a complete card,
    a real test, its metrics in the baseline and metrics written by the current code — or is on the legacy list
    with its listed source. No numbered script sits on disk outside ORDER. The legacy list only shrinks: a name
    that was not on the first list (2026-09-24) is refused. So a new analysis cannot count as done without a
    card and a test.
TR: Analiz kapsam kapısı (tools/analysis_coverage.py): run_all.ORDER'daki her betiğin tam kartı, gerçek testi,
    referanstaki metriği ve bugünkü kodun yazdığı metriği var — ya da listelenen kaynağıyla eski betik listesinde.
    ORDER dışında diskte numaralı betik yok. Eski betik listesi yalnız kısalır: ilk listede (2026-09-24) olmayan
    ad reddedilir. Böylece yeni bir analiz kartsız ve testsiz "bitti" sayılamaz.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import analysis_coverage as AC  # noqa: E402

# EN: the first legacy list (2026-09-24); it may only lose names | TR: ilk eski betik listesi; yalnız ad kaybedebilir
FIRST_LEGACY = frozenset({
    "01_dedup_leakage.py", "01_engine_rule.py", "01_unspecified_panels.py", "02_missingness.py", "03_association.py",
    "03_brand_ablation.py", "03_segment_quality.py", "04_target.py", "05_segmentation.py", "06_hedonic.py",
    "07_lofo.py", "07_text_flag.py", "08_large_errors.py", "10_free_text.py", "shap/02_oof_shap.py",
    "shap/03_what_sets_price.py", "shap/04_variants.py", "shap/06_one_listing.py"})


@pytest.fixture(scope="module")
def rows():
    """EN: The coverage matrix. / TR: Kapsam matrisi."""
    return AC.matrix()


def test_every_cell_filled(rows):
    """EN: No script has an empty cell. / TR: Hiçbir betiğin boş hücresi yok."""
    problems = [f"{r['script']}: {m}" for r in rows for m in r["problems"]]
    assert not problems, "analysis coverage gaps | analiz kapsam boşlukları:\n" + "\n".join(problems)


def test_no_script_outside_order():
    """EN: Every numbered script on disk is in ORDER, and back. / TR: Diskteki her numaralı betik ORDER'da, tersi de."""
    assert set(AC.on_disk()) == set(AC.order()), (
        f"not in ORDER | ORDER'da yok: {sorted(set(AC.on_disk()) - set(AC.order()))} · "
        f"missing on disk | diskte yok: {sorted(set(AC.order()) - set(AC.on_disk()))}")


def test_legacy_list_only_shrinks():
    """
    EN: legacy.json names only scripts from the first list, and only scripts that still exist.
    TR: legacy.json yalnız ilk listedeki ve hâlâ var olan betikleri anar.
    """
    listed = set(json.loads(AC.LEGACY.read_text(encoding="utf-8"))["scripts"])
    assert listed <= FIRST_LEGACY, f"legacy list may only shrink | liste yalnız kısalır: new {sorted(listed - FIRST_LEGACY)}"
    assert listed <= set(AC.order()), f"legacy entry for a removed script | silinmiş betik: {sorted(listed - set(AC.order()))}"
