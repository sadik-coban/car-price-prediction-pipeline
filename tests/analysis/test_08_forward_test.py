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


# ── Scale model | Ölçek modeli ─────────────────────────────────────────────────────────────────────
def test_forward_scale_calibrates_on_training_only(functions):
    """
    EN: The scale arm takes no test price at all, and its k comes from the training listings only: another test set
        (other cars, other predictions) gives the same k. A test listing's comparables are counted in training.
    TR: Ölçek kolu test fiyatını hiç almaz ve k'sı yalnız eğitim ilanlarından gelir: başka bir test kümesi (başka
        araçlar, başka tahminler) aynı k'yı verir. Test ilanının emsali eğitimde sayılır.
    """
    import inspect

    import pandas as pd
    rng = np.random.default_rng(0)

    def cars(n):
        """EN: n synthetic cars. / TR: n yapay araç."""
        return pd.DataFrame({"brand": rng.choice(["a", "b"], n), "kb_fuel": rng.choice(["x", "y"], n),
                             "vehicle_age": rng.integers(0, 25, n), "gb_mileage": rng.uniform(0, 3e5, n)})

    n = 1500
    tr_X, tr_price = cars(n), rng.uniform(5e5, 5e6, n)
    tr_oof = np.log1p(tr_price) + rng.normal(0, .1, n)
    tr_cells, fold = rng.choice(["m1", "m2", "m3"], n), np.arange(n) % 5
    forward_scale = functions["forward_scale"]
    assert not [p for p in inspect.signature(forward_scale).parameters if p.startswith("test_price")]
    a = forward_scale(tr_X, tr_price, tr_oof, tr_cells, fold, cars(300), rng.uniform(13, 15, 300),
                      np.array(["m1"] * 150 + ["never"] * 150))
    b = forward_scale(tr_X, tr_price, tr_oof, tr_cells, fold, cars(200), rng.uniform(13, 15, 200), np.array(["m2"] * 200))
    assert a["k"] == b["k"] and a["k"] > 0
    assert a["comparables"][:150].tolist() == [int((tr_cells == "m1").sum())] * 150 and not a["comparables"][150:].any()
    assert np.allclose(a["r"], a["k"] * a["s"]) and np.all((a["p"] >= 0) & (a["p"] <= 1))


def test_arms_side_by_side(ft):
    """
    EN: The three arms on the same test listings: the q arms repeat the published coverages; every listing is inside
        or missed above or missed below; the width fifths partition the test set; the scale model's k is positive.
    TR: Aynı test ilanlarında üç kol: q kolları yayımlanan kapsamaları yineler; her ilan ya içeride ya üstte ya altta
        kaçırılmış; genişlik beşte birleri test kümesini bölümler; ölçek modelinin k'sı pozitif.
    """
    c = ft["coverage"]
    arms = {r[0]: r for r in c["arms"]}
    assert list(arms) == ["global", "band", "scale"] and all(len(r) == len(c["arm_columns"]) for r in c["arms"])
    assert arms["global"][1] == c["coverage_global"] and arms["band"][1] == c["coverage_band"]
    assert arms["global"][5] == c["no_comparable"]["coverage_global"]
    assert all(r[1] + r[6] + r[7] == pytest.approx(100, abs=0.02) for r in c["arms"])
    assert sum(r[1] for r in c["by_width_fifth"]) == ft["n"] and [r[0] for r in c["by_width_fifth"]] == [1, 2, 3, 4, 5]
    assert c["scale"]["k"] > 0 and 0 <= c["scale"]["floor_hits"] <= ft["n"]


def test_forward_hit_probability(ft):
    """
    EN: Ten reliability points with rising predictions; the shown values partition the test listings; the signed
        error's mean and median are published.
    TR: Öngörüsü artan on güvenilirlik noktası; gösterilen değerler test ilanlarını bölümler; işaretli hatanın
        ortalaması ve medyanı yayımlanır.
    """
    cf = ft["confidence"]
    assert len(cf["deciles"]) == 10 and [p for p, _ in cf["deciles"]] == sorted(p for p, _ in cf["deciles"])
    assert sum(r[1] for r in cf["by_shown"]) == ft["n"] and all(0 <= r[0] <= 9 for r in cf["by_shown"])
    assert 0 <= cf["mean_observed"] <= 100 and set(ft["signed_error"]) == {"mean", "median"}
