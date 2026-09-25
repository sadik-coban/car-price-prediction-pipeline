"""
test_08_conformal_coverage.py
EN: Invariants of metrics/08_conformal_coverage.json (technical report §8). q is the 90th percentile of the OOF
    log errors and coverage is measured on the same OOF, so the overall coverage must sit at the target within
    sampling noise (3 binomial standard errors) — a gross miss means a bug. The four predicted-price quartiles
    are equal-sized, so their coverages must average to the overall one; their bounds must increase. Per-quartile
    coverage is NOT required to hit the target: the cheap quartile's under-coverage (~81%) is a real finding.
TR: metrics/08_conformal_coverage.json'un değişmezleri (teknik rapor §8). q, OOF log hatalarının 90. yüzdeliği ve
    kapsama aynı OOF'ta ölçülüyor; genel kapsama örnekleme gürültüsü içinde (3 binom standart hatası) hedefte
    olmalı — büyük sapma hata demektir. Tahmin edilen fiyatın dört çeyreği eşit büyüklükte; kapsamalarının
    ortalaması geneli vermeli, sınırları artmalı. Çeyrek kapsamasının hedefi tutması BEKLENMEZ: ucuz çeyrekteki
    düşük kapsama (~%81) gerçek bir bulgu.
"""
import math

import pytest


@pytest.fixture(scope="module")
def conf(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("08_conformal_coverage")


def test_overall_coverage_at_target(conf, metrics):
    """EN: |coverage − target| ≤ 3 binomial SE (n = listings). / TR: |kapsama − hedef| ≤ 3 binom SE (n = ilan)."""
    n = metrics("01_dedup_leakage")["meta"]["n_dedup"]
    target = conf["domain"]["conformal"]["coverage_target"]
    se = 100 * math.sqrt(target / 100 * (1 - target / 100) / n)
    assert abs(conf["report"]["conformal_all"] - target) <= 3 * se


def test_quartile_coverages_average_to_overall(conf):
    """
    EN: Four equal predicted-price quartiles: their mean coverage is the overall coverage.
    TR: Tahmin edilen fiyatın dört eşit çeyreği: kapsamalarının ortalaması genel kapsama.
    """
    q = conf["report"]["conformal_by_pred"]
    assert [r[0] for r in q] == ["Q1", "Q2", "Q3", "Q4"]
    assert sum(r[1] for r in q) / 4 == pytest.approx(conf["report"]["conformal_all"], abs=0.05)


def test_quartile_bounds_increase(conf):
    """EN: Three increasing cut points. / TR: Artan üç kesim noktası."""
    b = conf["report"]["q_bounds"]
    assert len(b) == 3 and b == sorted(b) and len(set(b)) == 3


def test_q_is_positive(conf):
    """EN: q > 0 (a log-error percentile). / TR: q > 0 (bir log-hata yüzdeliği)."""
    assert conf["report"]["conformal_q"] > 0
