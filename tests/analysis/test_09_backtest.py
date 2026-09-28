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
import pytest


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
