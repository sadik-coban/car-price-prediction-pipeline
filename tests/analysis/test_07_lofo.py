"""
test_07_lofo.py
EN: Invariants of metrics/07_lofo.json (technical report §6 LOFO): one row per removed feature or group, groups
    (upper-case names) marked as such, sorted by ΔRMSE; a tree list of five folds for the baseline and for every
    removal; the baseline's trees are 07_model_comparison's CV trees (same model, same folds); the single rows are
    features the model keeps.
TR: metrics/07_lofo.json'un değişmezleri (teknik rapor §6 LOFO): çıkarılan her öznitelik ya da grup için bir satır,
    gruplar (büyük harfli adlar) öyle işaretli, ΔRMSE'ye göre sıralı; taban ve her çıkarma için beş fold'luk ağaç
    listesi; tabanın ağaçları 07_model_comparison'ın CV ağaçları (aynı model, aynı fold'lar); tekil satırlar modelin
    tuttuğu öznitelikler.
"""
import pytest


@pytest.fixture(scope="module")
def met(metrics):
    """EN: The methodology section. / TR: Methodology bölümü."""
    return metrics("07_lofo")["methodology"]


def test_rows(met, metrics):
    """
    EN: Groups are the upper-case names; single rows are kept features; one row per name.
    TR: Gruplar büyük harfli adlar; tekil satırlar tutulan öznitelikler; ad başına bir satır.
    """
    rows = met["lofo"]
    assert len({r[0] for r in rows}) == len(rows)
    assert all((r[2] == "group") == r[0].isupper() for r in rows)
    kept = set(metrics("02_missingness")["methodology"]["feature_kept"])
    assert {r[0] for r in rows if r[2] == "single"} <= kept


def test_sorted_by_delta(met):
    """EN: Largest ΔRMSE first. / TR: En büyük ΔRMSE önce."""
    d = [r[1] for r in met["lofo"]]
    assert d == sorted(d, reverse=True)


def test_trees(met, metrics):
    """
    EN: Five positive tree counts for the baseline and each removal; the baseline equals 07's CV trees.
    TR: Taban ve her çıkarma için beş pozitif ağaç sayısı; taban 07'nin CV ağaçlarına eşit.
    """
    trees = met["lofo_trees"]
    assert set(trees) == {r[0] for r in met["lofo"]} | {"baseline"}
    assert all(len(v) == 5 and all(t > 0 for t in v) for v in trees.values())
    assert trees["baseline"] == metrics("07_model_comparison")["meta"]["repro"]["cv_trees"]
