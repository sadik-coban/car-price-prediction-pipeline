"""
test_shap_02_oof_shap.py
EN: Invariants of metrics/shap/02_oof_shap.json (SHAP report §1–2, OOF SHAP): the gate held (the refitted fold
    models reproduce the stored OOF predictions within the thresholds, and the MAPE is 07_model_comparison's); one
    row per listing and per group; the grouped table is sorted, its shares sum to 100, it covers the same groups as
    the separate table, and grouping never raises importance (mean |Σφ| ≤ mean Σ|φ|, triangle inequality).
TR: metrics/shap/02_oof_shap.json'un değişmezleri (SHAP raporu §1–2, OOF SHAP): kapı tuttu (yeniden kurulan fold
    modelleri saklı OOF tahminlerini eşikler içinde üretiyor, MAPE 07_model_comparison'ınki); ilan ve grup başına bir
    satır; gruplu tablo sıralı, payları 100'e toplanıyor, ayrı tabloyla aynı grupları kapsıyor ve gruplamak önemi hiç
    artırmıyor (ort. |Σφ| ≤ ort. Σ|φ|, üçgen eşitsizliği).
"""
import pytest


@pytest.fixture(scope="module")
def oof(metrics):
    """EN: The oof_shap section. / TR: oof_shap bölümü."""
    return metrics("shap/02_oof_shap")["oof_shap"]


def test_gate(oof, metrics):
    """EN: max gap ≤ its threshold; |MAPE refit − published| ≤ its threshold. / TR: Kapı eşikleri tutuyor."""
    g = oof["gate"]
    assert g["max_gap_tl"] <= g["threshold_max_tl"] and g["mean_gap_tl"] <= g["max_gap_tl"]
    assert abs(g["mape_refit"] - g["mape_published"]) <= g["threshold_mape"]
    assert g["mape_published"] == metrics("07_model_comparison")["domain"]["model_compare"]["lightgbm"]["MAPE"]


def test_sizes(oof, metrics):
    """EN: n = listings; one row per group in both tables. / TR: n = ilan; iki tabloda grup başına bir satır."""
    assert oof["n"] == metrics("01_dedup_leakage")["meta"]["n_dedup"]
    assert len(oof["groups"]) == len(oof["global"]) == len(oof["global_separate"])


def test_global_table(oof):
    """EN: Sorted by mean |SHAP|, shares sum to 100, same groups. / TR: Ort. |SHAP|'e göre sıralı, paylar 100, aynı gruplar."""
    g = oof["global"]
    assert [r[1] for r in g] == sorted((r[1] for r in g), reverse=True)
    assert sum(r[2] for r in g) == pytest.approx(100, abs=0.5)
    assert {r[0] for r in g} == set(oof["groups"]) == {r[0] for r in oof["global_separate"]}


def test_grouping_never_raises_importance(oof):
    """EN: separate ≥ grouped for every group. / TR: Her grupta ayrı ≥ gruplu."""
    sep = {r[0]: r[1] for r in oof["global_separate"]}
    assert all(sep[name] >= value - 1e-9 for name, value, _ in oof["global"])
