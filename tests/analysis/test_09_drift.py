"""
test_09_drift.py
EN: Invariants of metrics/09_drift.json (technical report §9): every pair of snapshots is compared once; the
    distances are in range; and the published significance (how many pairs have p < 0.05, how many survive Holm,
    which) is recomputed here from the published disjoint-part p-values — the report's claim must follow from
    the numbers next to it.
TR: metrics/09_drift.json'un değişmezleri (teknik rapor §9): her tarama çifti bir kez karşılaştırılıyor; uzaklıklar
    aralıkta; yayımlanan anlamlılık (kaç çiftte p < 0,05, kaçı Holm'dan geçiyor, hangileri) burada yayımlanan
    ayrık-kısım p-değerlerinden yeniden hesaplanıyor — raporun iddiası yanındaki sayılardan çıkmalı.
"""
from math import comb

import pytest

ALPHA = 0.05


@pytest.fixture(scope="module")
def drift(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("09_drift")


def test_every_pair_once(drift):
    """EN: k snapshots → C(k, 2) pairs in each pair table. / TR: k tarama → her çift tablosunda C(k, 2) çift."""
    d = drift["domain"]["drift"]
    k = len(d["table"]) + 1                                   # the first snapshot is the reference | ilk tarama referans
    assert len(d["all_pairs"]) == len(d["ortusme"]) == drift["report"]["drift_holm"]["n_tests"] == comb(k, 2)
    assert len({r[0] for r in d["all_pairs"]}) == comb(k, 2)


def test_distances_in_range(drift):
    """EN: KS D in [0, 1], p in [0, 1], PSI ≥ 0, EMD ≥ 0. / TR: KS D [0, 1], p [0, 1], PSI ≥ 0, EMD ≥ 0."""
    for _pair, ks, p, psi, emd in drift["domain"]["drift"]["all_pairs"]:
        assert 0 <= ks <= 1 and 0 <= p <= 1 and psi >= 0 and emd >= 0


def test_holm_follows_from_p_values(drift):
    """
    EN: n_sig = pairs with p < α; Holm step-down on the same p-values gives n_holm and the pairs.
    TR: n_sig = p < α olan çiftler; aynı p-değerlerinde Holm adım adım n_holm'u ve çiftleri verir.
    """
    rows = drift["domain"]["drift"]["ortusme"]
    ps = sorted((r[3], r[0]) for r in rows)
    n_holm = 0
    for i, (p, _pair) in enumerate(ps):
        if p > ALPHA / (len(ps) - i):
            break
        n_holm += 1
    h = drift["report"]["drift_holm"]
    assert h["n_sig"] == sum(1 for r in rows if r[3] < ALPHA)
    assert h["n_holm"] == n_holm and h["pairs"] == [pair for _p, pair in ps[:n_holm]]
    assert h["n_holm"] <= h["n_sig"] <= h["n_tests"]


def test_overlap_shares_in_range(drift):
    """EN: The overlap share of a pair is a percentage. / TR: Bir çiftin örtüşme payı bir yüzde."""
    assert all(0 <= r[1] <= 100 for r in drift["domain"]["drift"]["ortusme"])
