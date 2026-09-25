"""
test_shap_04_variants.py
EN: Invariants of metrics/shap/04_variants.json (SHAP report §4, the served models): SHAP adds up to the
    prediction (additivity error ~0); every listing is explained; the final-model table in the SHAP report is the
    one on the site (domain.shap) and so is CatBoost (SVD); the deterministic tables are sorted with shares summing
    to 100. The native CatBoost tables flip between two states run to run (a documented residue), so only their
    shape is checked.
TR: metrics/shap/04_variants.json'un değişmezleri (SHAP raporu §4, servis edilen modeller): SHAP tahmine toplanıyor
    (toplanabilirlik hatası ~0); her ilan açıklanıyor; SHAP raporundaki final model tablosu sitedekiyle (domain.shap)
    aynı, CatBoost (SVD) de öyle; deterministik tablolar sıralı ve payları 100'e toplanıyor. Native CatBoost
    tabloları koşumdan koşuma iki durum arasında gidip geliyor (belgelenmiş kalıntı); onların yalnız biçimi sınanır.
"""
import pytest


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("shap/04_variants")


def test_additivity_and_size(doc, metrics):
    """EN: Additivity error ~0; n used = listings. / TR: Toplanabilirlik hatası ~0; kullanılan n = ilan."""
    assert abs(doc["shap_final"]["add_err"]) < 1e-6
    assert doc["domain"]["shap"]["n_used"] == metrics("01_dedup_leakage")["meta"]["n_dedup"]


def test_report_tables_are_the_site_tables(doc):
    """EN: final table = domain.shap LightGBM; CatBoost (SVD) the same in both. / TR: Tablolar sitedekiyle aynı."""
    fin, site = doc["shap_final"], doc["domain"]["shap"]
    assert fin["final_model_table"] == site["lightgbm_tfidf_svd"]
    assert fin["catboost_tfidf_svd"] == site["catboost_tfidf_svd"]


@pytest.mark.parametrize("key", ["final_model_table", "final_model_table_separate", "catboost_tfidf_svd",
                                 "catboost_tfidf_svd_separate"])
def test_deterministic_tables(doc, key):
    """EN: Sorted by mean |SHAP|, shares sum to 100. / TR: Ort. |SHAP|'e göre sıralı, paylar 100."""
    t = doc["shap_final"][key]
    assert [r[1] for r in t] == sorted((r[1] for r in t), reverse=True)
    assert sum(r[2] for r in t) == pytest.approx(100, abs=0.5)


def test_native_tables_shape(doc):
    """EN: Native tables are [name, value, share] rows (values flip run to run). / TR: Yalnız biçim."""
    for key in ("catboost_native", "catboost_native_separate", "catboost_native_combined"):
        assert all(len(r) == 3 and isinstance(r[0], str) and r[1] >= 0 for r in doc["shap_final"][key]), key
