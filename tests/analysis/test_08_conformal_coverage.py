"""
test_08_conformal_coverage.py
EN: Invariants of metrics/08_conformal_coverage.json (technical report §8, pre-registered plan
    plans/08-mondrian-coverage). The calibration never sees the scored fold (checked on the script's own functions:
    changing one fold's errors cannot change that fold's q). The four predicted-price bands are equal-sized, so their
    coverages average to the overall one for each arm; the overall coverage of each arm sits at the target within
    3 binomial SE (cross-calibrated, so this is measured, not true by construction); the band bounds increase; the
    served q is positive. Per-band coverage of the single-q arm is NOT required to hit the target: the cheap band's
    under-coverage is a real finding.
TR: metrics/08_conformal_coverage.json'un değişmezleri (teknik rapor §8, ön kayıtlı plan plans/08-mondrian-coverage).
    Kalibrasyon puanlanan katı hiç görmez (betiğin kendi fonksiyonlarında sınanır: bir katın hatalarını değiştirmek o
    katın q'sunu değiştiremez). Tahmin fiyatının dört bandı eşit büyüklükte; her kolda kapsamalarının ortalaması
    geneli verir; her kolun genel kapsaması hedefte, 3 binom SH içinde (çapraz kalibreli, yani tanım gereği değil
    ölçülmüş); bant sınırları artar; servis edilen q pozitif. Tek q kolunun bant kapsamasının hedefi tutması
    BEKLENMEZ: ucuz banttaki düşük kapsama gerçek bir bulgu.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "08_conformal_coverage.py"


@pytest.fixture(scope="module")
def conf(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("08_conformal_coverage")


@pytest.fixture(scope="module")
def functions():
    """
    EN: The script's pure functions (cells 1–3), without loading any data.
    TR: Betiğin saf işlevleri (1–3. hücreler), veri yüklemeden.
    """
    # EN: "lib" is also db/lib (other tests import it): load analysis/lib for this exec only, then put things back
    # TR: "lib" aynı zamanda db/lib (başka testler import eder): yalnız bu exec için analysis/lib yüklenir, sonra geri konur
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "lib" or k.startswith("lib.")}
    sys.path.insert(0, str(ROOT / "analysis"))
    try:
        head = SCRIPT.read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "conformal_functions"}
        exec(compile(head, str(SCRIPT), "exec"), ns)
    finally:
        sys.path.remove(str(ROOT / "analysis"))
        for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ns


def test_calibration_never_sees_the_scored_fold(functions):
    """
    EN: Blowing up fold 0's errors changes no q given to fold 0 (both arms), but does change the q of other folds.
    TR: 0. katın hatalarını büyütmek 0. kata verilen hiçbir q'yu değiştirmez (iki kol), öteki katlarınkini değiştirir.
    """
    rng = np.random.default_rng(0)
    n = 4000
    err, fold, band = rng.random(n), np.arange(n) % 5, rng.integers(0, 4, n)
    qg, qb = functions["cross_conformal"](err, fold, band)
    err2 = err.copy()
    err2[fold == 0] *= 100
    qg2, qb2 = functions["cross_conformal"](err2, fold, band)
    assert np.array_equal(qg[fold == 0], qg2[fold == 0]) and np.array_equal(qb[fold == 0], qb2[fold == 0])
    assert not np.array_equal(qg[fold != 0], qg2[fold != 0])


def test_conformal_quantile_rank(functions):
    """EN: The ceil((n+1)·0.9)-th smallest: 1..19 → 18. / TR: En küçükten ceil((n+1)·0,9)'uncu: 1..19 → 18."""
    assert functions["conformal_quantile"](np.arange(1, 20), 0.9) == 18


def test_bands_average_to_overall(conf):
    """
    EN: Four equal bands; for each arm the mean band coverage is the overall coverage.
    TR: Dört eşit bant; her kolda bant kapsamalarının ortalaması genel kapsama.
    """
    m = conf["report"]["mondrian"]
    rows = m["by_band"]
    assert [r[0] for r in rows] == ["Q1", "Q2", "Q3", "Q4"] and len({r[1] for r in rows}) == 1
    assert sum(r[2] for r in rows) / 4 == pytest.approx(conf["report"]["conformal_all"], abs=0.02)
    assert conf["domain"]["conformal"]["by_quantile"] == [[r[0], r[2]] for r in rows]
    assert conf["domain"]["conformal"]["mondrian_by_quantile"] == [[r[0], r[3]] for r in rows]


def test_overall_coverage_at_target(conf, metrics):
    """EN: Each arm: |coverage − target| ≤ 3 binomial SE. / TR: Her kol: |kapsama − hedef| ≤ 3 binom SH."""
    n = metrics("01_dedup_leakage")["meta"]["n_dedup"]
    target = conf["domain"]["conformal"]["coverage_target"]
    se = 100 * math.sqrt(target / 100 * (1 - target / 100) / n)
    mondrian_all = sum(r[3] for r in conf["report"]["mondrian"]["by_band"]) / 4      # four equal bands | dört eşit bant
    for cov in (conf["report"]["conformal_all"], mondrian_all):
        assert abs(cov - target) <= 3 * se


def test_accept_band_is_the_plan(conf):
    """
    EN: The published acceptance band is the pre-registered one (plans/08-mondrian-coverage, H1).
    TR: Yayımlanan kabul aralığı ön kayıttaki (plans/08-mondrian-coverage, H1).
    """
    plan = (ROOT / "plans" / "08-mondrian-coverage" / "analysis_plan.json").read_text(encoding="utf-8")
    lo, hi = conf["report"]["mondrian"]["accept_band"]
    assert f">= {lo} and <= {hi} (percent)" in plan


def test_band_bounds_and_q(conf):
    """
    EN: Three increasing cut points; the served q positive.
    TR: Artan üç kesim noktası; servis edilen q pozitif.
    """
    b = conf["report"]["q_bounds"]
    assert len(b) == 3 and b == sorted(b) and len(set(b)) == 3
    assert conf["report"]["conformal_q"] > 0
