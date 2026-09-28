"""
test_09_backtest.py
EN: Invariants of metrics/09_backtest.json (technical report §9): in the leak-free arms every test snapshot comes
    after the training snapshot(s) and every row's 95% interval is ordered around its MAPE; the cumulative arm's first
    block is the single arm's first block (the protocol says it is the same experiment); each paired comparison is
    cumulative minus single on at most the listings both tested; the in-sample arm ends on all listings in the model
    with the headline MAPE of 07 (the script asserts the predictions are the headline OOF itself).
TR: metrics/09_backtest.json'un değişmezleri (teknik rapor §9): sızıntısız kollarda her test taraması eğitim
    taramasından sonra ve her satırın %95 aralığı MAPE'sinin iki yanında sıralı; cumulative kolun ilk bloğu single
    kolun ilk bloğu (protokol aynı deney diyor); her eşli karşılaştırma, ikisinin de test ettiği ilanlarda cumulative
    eksi single; insample kol modele giren bütün ilanlarla ve 07'nin manşet MAPE'siyle bitiyor (betik tahminlerin
    manşet OOF'un kendisi olduğunu assert eder).
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
        head = (ROOT / "analysis" / "09_backtest.py").read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "backtest_functions"}
        exec(compile(head, "09_backtest.py", "exec"), ns)
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
def bt(metrics):
    """EN: The backtest block. / TR: Backtest bloğu."""
    return metrics("09_backtest")["methodology"]["backtest"]


def test_train_before_test(bt):
    """
    EN: Snapshot dates are MM-DD of one year, so text order is time order; lo ≤ MAPE ≤ hi; trees > 0.
    TR: Tarih metni sırası zaman sırası; alt ≤ MAPE ≤ üst; ağaç > 0.
    """
    assert bt["columns"]["forward"] == ["train", "test", "MAPE", "n", "ci_lo", "ci_hi", "trees"]
    for train, test, mape, n, lo, hi, trees in bt["single"] + bt["cumulative"]:
        assert train.lstrip("→") < test and n > 0 and trees > 0
        assert 0 < lo <= mape <= hi, (train, test)


def test_cumulative_first_block_is_single(bt):
    """EN: Training on the first snapshot only is the same in both arms. / TR: Yalnız ilk taramada eğitim iki kolda aynı."""
    first = bt["single"][0][0]
    single = [r for r in bt["single"] if r[0] == first]
    cumulative = [[r[0].lstrip("→")] + r[1:] for r in bt["cumulative"] if r[0] == "→" + first]
    assert single and cumulative == single


def test_paired_is_cumulative_minus_single(bt):
    """
    EN: One row per test snapshot whose two trainings differ; difference = cumulative − single; n within both arms.
    TR: İki eğitimi farklı her test taraması için bir satır; fark = cumulative − single; n iki kolun içinde.
    """
    n_of = {(r[0].lstrip("→"), r[1]): r[3] for r in bt["cumulative"]}
    n_single = {(r[0], r[1]): r[3] for r in bt["single"]}
    first = bt["single"][0][0]
    expected = [(r[0].lstrip("→"), r[1]) for r in bt["cumulative"] if r[0] != "→" + first]
    assert [(r[1], r[0]) for r in bt["paired"]] == expected
    for test, s_train, c_train, n, s_mape, c_mape, diff, lo, hi in bt["paired"]:
        assert c_train == "→" + s_train and lo < hi
        assert diff == pytest.approx(c_mape - s_mape, abs=0.011)
        assert 0 < n <= min(n_of[(s_train, test)], n_single[(s_train, test)])


def test_insample_ends_on_the_headline(bt, metrics):
    """
    EN: The last accumulated block is every listing, with the headline LightGBM MAPE.
    TR: Son biriken blok bütün ilanlar, manşet LightGBM MAPE'siyle.
    """
    assert bt["insample"][-1][2] == metrics("01_dedup_leakage")["meta"]["n_dedup"]
    headline = metrics("07_model_comparison")["domain"]["model_compare"]["lightgbm"]["MAPE"]
    assert bt["insample"][-1][1] == pytest.approx(headline, abs=0.005)


def test_horizon_on_a_fixed_test_set(bt):
    """
    EN: The fixed test set is the cumulative arm's last test set; one row per earlier single snapshot, each
        difference = its MAPE − the latest one's.
    TR: Sabit test kümesi cumulative kolun son test kümesi; önceki her tek tarama için bir satır, her fark = kendi
        MAPE'si − en yenisininki.
    """
    hz = bt["horizon"]
    last = bt["cumulative"][-1]
    assert hz["test"] == last[1] and hz["n"] == last[3]
    mape = {r[0]: r[1] for r in hz["rows"]}
    latest = [t for t in mape if t not in {r[0] for r in hz["vs_latest"]}]
    assert len(latest) == 1
    for train, diff, lo, hi in hz["vs_latest"]:
        assert diff == pytest.approx(mape[train] - mape[latest[0]], abs=0.011) and lo < hi
    assert all(lo <= m <= hi for _t, m, lo, hi in hz["rows"])


def test_forward_coverage_rows(bt):
    """
    EN: The pre-registered setups: every single row and the cumulative rows whose training differs; band sizes add
        up to the test n; coverages are percentages.
    TR: Ön kayıtlı kurulumlar: her single satırı ve eğitimi farklı cumulative satırları; bant büyüklükleri test n'ine
        toplanır; kapsamalar yüzde.
    """
    first = bt["single"][0][0]
    want = [(r[0], r[1]) for r in bt["single"]] + [(r[0], r[1]) for r in bt["cumulative"] if r[0] != "→" + first]
    fc = bt["forward_coverage"]
    assert [(c["train"], c["test"]) for c in fc] == want
    n_of = {(r[0], r[1]): r[3] for r in bt["single"] + bt["cumulative"]}
    for c in fc:
        assert c["n"] == n_of[(c["train"], c["test"])] == sum(b[1] for b in c["bands"])
        vals = [c["coverage_global"], c["coverage_band"]] + [x for b in c["bands"] for x in b[2:]]
        assert all(0 <= x <= 100 for x in vals) and len(c["q_band"]) == 4 and c["q"] > 0
