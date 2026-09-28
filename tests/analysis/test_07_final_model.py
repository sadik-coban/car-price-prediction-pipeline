"""
test_07_final_model.py
EN: Invariants of metrics/07_final_model.json (technical report §7, the served models): the final LightGBM uses
    the median of the CV fold tree counts.
TR: metrics/07_final_model.json'un değişmezleri (teknik rapor §7, servis modelleri): son LightGBM CV fold ağaç
    sayılarının medyanını kullanıyor.
"""
import statistics

import pytest


@pytest.fixture(scope="module")
def final(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("07_final_model")


def test_tree_count_is_cv_median(final, metrics):
    """EN: final trees = int(median of the 5 fold tree counts). / TR: son ağaç = int(5 fold ağaç sayısının medyanı)."""
    cv_trees = metrics("07_model_comparison")["meta"]["repro"]["cv_trees"]
    assert final["meta"]["repro"]["final_lgb_trees"] == int(statistics.median(cv_trees))
