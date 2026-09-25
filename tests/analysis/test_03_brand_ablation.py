"""
test_03_brand_ablation.py
EN: Invariants of metrics/03_brand_ablation.json (technical report §3 brand ablation): each arm's errors are in
    the usual order (RMSE ≥ MAE ≥ MedAE > 0, R² in (0, 1)); the brand + series + model arm is the headline model of
    07_model_comparison exactly (same features, folds and parameters); the plumbing gate is consistent (identical
    OOF ⇔ zero gap, and a changed OOF means brand split somewhere); and the report's claim holds: brand alone is
    worse than series + model.
TR: metrics/03_brand_ablation.json'un değişmezleri (teknik rapor §3 marka ablasyonu): her kolun hataları olağan
    sırada (RMSE ≥ MAE ≥ MedAE > 0, R² (0, 1) aralığında); marka + seri + model kolu 07_model_comparison'ın manşet
    modeliyle birebir aynı (aynı öznitelik, fold ve parametre); tesisat kapısı tutarlı (aynı OOF ⇔ sıfır fark,
    değişen OOF markanın bir yerde bölme yaptığı demek); ve raporun iddiası tutuyor: yalnız marka, seri + modelden
    kötü.
"""
import pytest

ARMS = ("brand_only", "series_model", "brand_series_model")


@pytest.fixture(scope="module")
def ba(metrics):
    """EN: The published block. / TR: Yayımlanan blok."""
    return metrics("03_brand_ablation")["domain"]["brand_ablation"]


def test_metric_order(ba):
    """EN: RMSE ≥ MAE ≥ MedAE > 0, 0 < R² < 1, MAPE > 0. / TR: RMSE ≥ MAE ≥ MedAE > 0, 0 < R² < 1, MAPE > 0."""
    for arm in ARMS:
        m = ba[arm]
        assert m["RMSE"] >= m["MAE"] >= m["MedAE"] > 0 and 0 < m["R2"] < 1 and m["MAPE"] > 0, arm


def test_full_arm_is_the_headline(ba, metrics):
    """EN: brand + series + model = 07's LightGBM. / TR: marka + seri + model = 07'nin LightGBM'i."""
    assert ba["brand_series_model"] == metrics("07_model_comparison")["domain"]["model_compare"]["lightgbm"]


def test_plumbing(ba):
    """
    EN: Identical OOF ⇔ zero gap; a changed OOF means brand won at least one split.
    TR: Aynı OOF ⇔ sıfır fark; değişen OOF markanın en az bir bölme kazandığı demek.
    """
    val = ba["validation"]
    assert val["oof_same"] == (val["max_gap_tl"] == 0)
    assert val["brand_split_cv"] >= 0 and (val["oof_same"] or val["brand_split_cv"] > 0)


def test_brand_alone_is_worse(ba):
    """EN: Claim: brand only has a larger error than series + model. / TR: İddia: yalnız marka daha çok hata yapar."""
    assert ba["brand_only"]["MAE"] > ba["series_model"]["MAE"]
    assert ba["brand_only"]["MAPE"] > ba["series_model"]["MAPE"]
