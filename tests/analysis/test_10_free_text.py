"""
test_10_free_text.py
EN: Invariants of metrics/10_free_text.json (technical report §10 free text): the extraction coverage fits
    inside the texts and the listings and its share is the ratio; the frozen ablation's ΔR² is the gap between its
    two R² values; the "no model or series name" flag agrees with the published feature lists; the LLM classes are
    the three published ids.
TR: metrics/10_free_text.json'un değişmezleri (teknik rapor §10 serbest metin): çıkarım kapsamı metinlere ve
    ilanlara sığıyor, payı oran; dondurulmuş ablasyonun ΔR²'si iki R² değerinin farkı; "model ya da seri adı yok"
    bayrağı yayımlanan öznitelik listeleriyle uyuşuyor; LLM sınıfları yayımlanan üç kimlik.
"""
import pytest


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("10_free_text")


def test_coverage(doc, metrics):
    """EN: in model ≤ texts; listings = model's; share = ratio. / TR: modelde ≤ metin; ilan = modelinki; pay = oran."""
    ts = doc["error_drivers"]["text_source"]
    assert 0 < ts["llm_listings_in_model"] <= ts["llm_texts"]
    assert ts["listings"] == metrics("01_dedup_leakage")["meta"]["n_dedup"]
    assert ts["llm_coverage_pct"] == pytest.approx(round(100 * ts["llm_listings_in_model"] / ts["listings"], 1))


def test_ablation_delta(doc):
    """EN: ΔR² = R²(+text) − R²(structured); both in (0, 1). / TR: ΔR² = R²(+metin) − R²(yapısal); ikisi (0, 1)."""
    ab = doc["report"]["text_ablation"]["ablation"]
    assert 0 < ab["r2_structured"] < 1 and 0 < ab["r2_structured_plus_text"] < 1
    assert ab["delta_r2"] == pytest.approx(ab["r2_structured_plus_text"] - ab["r2_structured"], abs=1e-4)


def test_identity_flag_matches_features(doc):
    """
    EN: The model/series flag is true exactly when model or series is among the ablation features.
    TR: Model/seri bayrağı, model ya da seri ablasyon öznitelikleri arasındaysa doğru.
    """
    ts = doc["error_drivers"]["text_source"]
    cols = ts["ablation_categorical"] + ts["ablation_numeric"]
    assert ts["ablation_model_series"] == any(c in ("model", "series") for c in cols)
    assert ts["ablation_trees"] > 0


def test_llm_classes(doc):
    """EN: The classes are among the three published ids. / TR: Sınıflar yayımlanan üç kimlik arasında."""
    llm = doc["report"]["text_ablation"]["llm"]
    assert llm["classes"] and set(llm["classes"]) <= {"damage", "maintenance", "modification"}
    assert llm["library"] and llm["model"]
