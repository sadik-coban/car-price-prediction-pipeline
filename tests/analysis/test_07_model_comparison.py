"""
test_07_model_comparison.py
EN: Invariants of metrics/07_model_comparison.json (technical report §7): every compared model has every error
    measure and the measures are ordered as they must be (RMSE ≥ MAE ≥ MedAE > 0, 0 < R² < 1); one early-stopped
    tree count per fold; the comparable-listing shares add up; the baseline comparison is internally consistent.
TR: metrics/07_model_comparison.json'un değişmezleri (teknik rapor §7): karşılaştırılan her modelin her hata ölçüsü
    var ve ölçüler olması gereken sırada (RMSE ≥ MAE ≥ MedAE > 0, 0 < R² < 1); fold başına bir erken durdurma ağaç
    sayısı; emsal payları toplanıyor; taban karşılaştırması kendi içinde tutarlı.
"""
import pytest

MODELS = ("lightgbm", "catboost", "catboost_svd", "catboost_native")
MEASURES = ("MAE", "MedAE", "RMSE", "MAPE", "R2")
N_FOLDS = 5


@pytest.fixture(scope="module")
def mc(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("07_model_comparison")


def test_every_model_has_every_measure(mc):
    """EN: Four models × five measures. / TR: Dört model × beş ölçü."""
    compare = mc["domain"]["model_compare"]
    assert all(set(compare[m]) >= set(MEASURES) for m in MODELS)


def test_error_measures_are_ordered(mc):
    """
    EN: RMSE ≥ MAE ≥ MedAE > 0 (true for any error sample), MAPE > 0, 0 < R² < 1.
    TR: RMSE ≥ MAE ≥ MedAE > 0 (her hata örneğinde doğru), MAPE > 0, 0 < R² < 1.
    """
    for m in MODELS:
        e = mc["domain"]["model_compare"][m]
        assert e["RMSE"] >= e["MAE"] >= e["MedAE"] > 0, m
        assert e["MAPE"] > 0 and 0 < e["R2"] < 1, m


def test_one_tree_count_per_fold(mc):
    """EN: Early stopping picked a positive tree count in each of the 5 folds. / TR: 5 fold'un her birinde pozitif ağaç sayısı."""
    trees = mc["meta"]["repro"]["cv_trees"]
    assert len(trees) == N_FOLDS and all(t > 0 for t in trees)


def test_comparable_shares_add_up(mc):
    """
    EN: thin + comped = 100% and singletons are a subset of thin groups.
    TR: thin + comped = %100 ve tekil ilanlar thin grupların alt kümesi.
    """
    d = mc["domain"]["dealer_coverage"]
    assert d["thin_pct"] + d["comped_pct"] == pytest.approx(100, abs=0.02)
    assert 0 <= d["singleton_pct"] <= d["thin_pct"]


def test_baseline_gap_is_consistent(mc):
    """
    EN: gap = baseline MAE − model MAE, improvement % = gap / baseline MAE.
    TR: fark = taban MAE − model MAE, iyileşme % = fark / taban MAE.
    """
    b = mc["error_drivers"]["baseline_equal_terms"]
    assert b["gap_tl"] == pytest.approx(b["baseline_mae"] - b["model_mae"], abs=1)
    assert b["improvement_pct"] == pytest.approx(100 * b["gap_tl"] / b["baseline_mae"], abs=0.05)
