"""
test_09_drift.py
EN: Invariants of metrics/09_drift.json (technical report §9): every pair of snapshots is compared once, in order;
    the sizes of the difference are in range (KS in [0, 1], PSI ≥ 0, EMD ≥ 0) and the shared-listing share is a
    percentage. No p-value is published (the snapshots share listings; a test would assume independence).
TR: metrics/09_drift.json'un değişmezleri (teknik rapor §9): her tarama çifti bir kez ve sırayla karşılaştırılıyor;
    farkın büyüklükleri aralıkta (KS [0, 1], PSI ≥ 0, EMD ≥ 0) ve ortak ilan payı bir yüzde. p-değeri
    yayımlanmıyor (taramalar ilan paylaşıyor; test bağımsızlık varsayardı).
"""
from math import comb

import pytest


@pytest.fixture(scope="module")
def drift(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("09_drift")


def test_every_pair_once(drift, metrics):
    """
    EN: k snapshots → C(k, 2) distinct pairs, each earlier → later.
    TR: k tarama → C(k, 2) farklı çift, her biri önceki → sonraki.
    """
    snaps = [s[5:] for s in metrics("01_dedup_leakage")["meta"]["snapshots"]]
    pairs = [r[0] for r in drift["domain"]["drift"]["all_pairs"]]
    assert len(set(pairs)) == len(pairs) == comb(len(snaps), 2)
    for p in pairs:
        a, b = p.split("→")
        assert snaps.index(a) < snaps.index(b)


def test_distances_in_range(drift):
    """
    EN: KS in [0, 1], PSI ≥ 0, EMD ≥ 0, shared share in [0, 100]; no p-value column.
    TR: KS [0, 1], PSI ≥ 0, EMD ≥ 0, ortak pay [0, 100]; p-değeri kolonu yok.
    """
    for row in drift["domain"]["drift"]["all_pairs"]:
        assert len(row) == 5
        _pair, ks, psi, emd, shared = row
        assert 0 <= ks <= 1 and psi >= 0 and emd >= 0 and 0 <= shared <= 100
