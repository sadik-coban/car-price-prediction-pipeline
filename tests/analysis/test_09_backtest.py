"""
test_09_backtest.py
EN: Invariants of metrics/09_backtest.json (technical report §9): in the leak-free arms every test snapshot comes
    after the training snapshot(s); the cumulative arm's first block is the single arm's first block (the
    protocol says it is the same experiment); the in-sample arm ends on all listings in the model.
TR: metrics/09_backtest.json'un değişmezleri (teknik rapor §9): sızıntısız kollarda her test taraması eğitim
    taramasından sonra; cumulative kolun ilk bloğu single kolun ilk bloğu (protokol aynı deney diyor); insample
    kol modele giren bütün ilanlarla bitiyor.
"""
import pytest


@pytest.fixture(scope="module")
def bt(metrics):
    """EN: The backtest block. / TR: Backtest bloğu."""
    return metrics("09_backtest")["methodology"]["backtest"]


def test_train_before_test(bt):
    """EN: Snapshot dates are MM-DD of one year, so text order is time order. / TR: Tarih metni sırası zaman sırası."""
    for train, test, mape, n in bt["single"]:
        assert train < test and mape > 0 and n > 0
    for train, test, mape, n in bt["cumulative"]:
        assert train.lstrip("→") < test and mape > 0 and n > 0


def test_cumulative_first_block_is_single(bt):
    """EN: Training on the first snapshot only is the same in both arms. / TR: Yalnız ilk taramada eğitim iki kolda aynı."""
    first = bt["single"][0][0]
    single = [r for r in bt["single"] if r[0] == first]
    cumulative = [[r[0].lstrip("→")] + r[1:] for r in bt["cumulative"] if r[0] == "→" + first]
    assert single and cumulative == single


def test_insample_ends_on_all_listings(bt, metrics):
    """EN: The last accumulated block is every listing. / TR: Son biriken blok bütün ilanlar."""
    assert bt["insample"][-1][2] == metrics("01_dedup_leakage")["meta"]["n_dedup"]
