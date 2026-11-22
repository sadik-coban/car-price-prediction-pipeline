"""
test_08_forward_test.py
EN: Invariants of metrics/08_forward_test.json (technical report §8, the error on listings that arrive later): q, the band
    q's and the band edges come from the training numbers only; training ends before the test snapshot and the 95%
    interval sits around the MAPE; every listing of the data is either in training or in the test (the test is the
    listings first seen in the last snapshot); band sizes add up to the test n and coverages are percentages.
TR: metrics/08_forward_test.json'un değişmezleri (teknik rapor §8, sonradan gelen ilanlardaki hata): q, bant q'ları ve
    bant sınırları yalnız eğitim sayılarından gelir; eğitim test taramasından önce biter ve %95 aralık MAPE'nin iki
    yanında; verideki her ilan ya eğitimde ya testte (test, son taramada ilk kez görülen ilanlar); bant büyüklükleri
    test n'ine toplanır ve kapsamalar yüzde.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


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
        head = (ROOT / "analysis" / "08_forward_test.py").read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "forward_test_functions"}
        exec(compile(head, "08_forward_test.py", "exec"), ns)
    finally:
        sys.path.remove(str(ROOT / "analysis"))
        for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ns


def test_forward_coverage_calibrates_on_training_only(functions):
    """
    EN: q, the band q's and the band edges come from the training numbers only: changing every test price leaves
        them as they were (only the coverage moves).
    TR: q, bant q'ları ve bant sınırları yalnız eğitim sayılarından gelir: bütün test fiyatlarını değiştirmek onları
        olduğu gibi bırakır (yalnız kapsama değişir).
    """
    rng = np.random.default_rng(0)
    tr_price = rng.uniform(5e5, 5e6, 4000)
    tr_oof = np.log1p(tr_price) + rng.normal(0, .1, 4000)
    te_log = np.log1p(rng.uniform(5e5, 5e6, 800))
    te_price = np.expm1(te_log + rng.normal(0, .1, 800))
    nocomp = rng.random(800) < .1
    a = functions["forward_coverage"](tr_price, tr_oof, te_price, te_log, nocomp)
    b = functions["forward_coverage"](tr_price, tr_oof, te_price * 3, te_log, nocomp)
    assert (a["q"], a["q_band"], a["bounds"]) == (b["q"], b["q_band"], b["bounds"])
    assert a["coverage_global"] != b["coverage_global"]


@pytest.fixture(scope="module")
def ft(metrics):
    """EN: The forward test block. / TR: İleri test bloğu."""
    return metrics("08_forward_test")["methodology"]["forward_test"]


def test_train_before_test(ft):
    """
    EN: Training ends on the snapshot before the test one; lo ≤ MAPE ≤ hi; trees > 0.
    TR: Eğitim test taramasından önceki taramada biter; alt ≤ MAPE ≤ üst; ağaç > 0.
    """
    assert ft["train_until"] < ft["test"] and ft["trees"] > 0
    assert 0 < ft["ci_lo"] <= ft["MAPE"] <= ft["ci_hi"]


def test_every_listing_is_train_or_test(ft, metrics):
    """
    EN: Every listing is seen before the last snapshot or first seen in it, so training + test = all listings; the
        test snapshot is the last one in the data.
    TR: Her ilan ya son taramadan önce görülmüş ya da ilk kez onda görülmüş; yani eğitim + test = bütün ilanlar; test
        taraması verideki sonuncusu.
    """
    meta = metrics("01_dedup_leakage")["meta"]
    assert ft["n_train"] + ft["n"] == meta["n_dedup"]
    assert ft["test"] == meta["snapshots"][-1] and ft["train_until"] == meta["snapshots"][-2]


def test_coverage_adds_up(ft):
    """
    EN: Band sizes add up to the test n; the no-comparable listings are part of it; coverages are percentages.
    TR: Bant büyüklükleri test n'ine toplanır; emsalsiz ilanlar onun parçası; kapsamalar yüzde.
    """
    c = ft["coverage"]
    assert c["n"] == ft["n"] == sum(b[1] for b in c["bands"])
    assert 0 < c["no_comparable"]["n"] < c["n"]
    vals = [c["coverage_global"], c["coverage_band"], c["no_comparable"]["coverage_global"],
            c["no_comparable"]["coverage_band"]] + [x for b in c["bands"] for x in b[2:]]
    assert all(0 <= x <= 100 for x in vals) and len(c["q_band"]) == 4 and c["q"] > 0
