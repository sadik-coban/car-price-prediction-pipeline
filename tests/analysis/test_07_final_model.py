"""
test_07_final_model.py
EN: Invariants of metrics/07_final_model.json (technical report §7, the served models): the final LightGBM uses
    the median of the CV fold tree counts; every sample prediction's published deviation equals |prediction −
    actual| / actual; the samples are the ones the script promises (OOF residual within ±5%).
TR: metrics/07_final_model.json'un değişmezleri (teknik rapor §7, servis modelleri): son LightGBM CV fold ağaç
    sayılarının medyanını kullanıyor; her örnek tahminin yayımlanan sapması |tahmin − gerçek| / gerçek; örnekler
    betiğin söz verdikleri (OOF artığı ±%5 içinde).
"""
import statistics

import pytest

MAX_ABS_RESID = 5          # the script's promise | betiğin sözü


@pytest.fixture(scope="module")
def final(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("07_final_model")


def test_tree_count_is_cv_median(final, metrics):
    """EN: final trees = int(median of the 5 fold tree counts). / TR: son ağaç = int(5 fold ağaç sayısının medyanı)."""
    cv_trees = metrics("07_model_comparison")["meta"]["repro"]["cv_agac"]
    assert final["meta"]["repro"]["final_lgb_agac"] == int(statistics.median(cv_trees))


def test_sample_deviation_matches_prediction(final):
    """EN: lgb_sapma_pct = |LightGBM − actual| / actual, 1 decimal. / TR: lgb_sapma_pct = |LightGBM − gerçek| / gerçek."""
    for s in final["domain"]["final_results"]["ornek_tahminler"]:
        assert s["lgb_sapma_pct"] == pytest.approx(abs(s["lightgbm_tahmin"] - s["gercek"]) / s["gercek"] * 100,
                                                   abs=0.051), s["arac"]


def test_samples_keep_the_oof_promise(final):
    """
    EN: Every sample was predicted within ±5% out of fold, and the headline sample is one of them.
    TR: Her örnek fold dışında ±%5 içinde tahmin edilmiş ve manşet örnek onlardan biri.
    """
    res = final["domain"]["final_results"]
    assert res["ornek_tahminler"] and all(abs(s["oof_artik_pct"]) <= MAX_ABS_RESID for s in res["ornek_tahminler"])
    assert res["ornek_tahmin"] in res["ornek_tahminler"]
