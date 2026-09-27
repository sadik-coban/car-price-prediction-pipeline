"""
test_10_free_text.py
EN: Invariants of metrics/10_free_text.json (technical report §1 and §10, pre-registered plan
    plans/10-text-contribution): the description features are fitted on the training fold only; the published
    gap is the text arm minus the base arm; the base arm is this run's model (08's log R², 07's MAPE, the same
    listings and 5 folds); the text arm's settings are the plan's.
TR: metrics/10_free_text.json'un değişmezleri (teknik rapor §1 ve §10, ön kayıtlı plan plans/10-text-contribution):
    açıklama öznitelikleri yalnız eğitim fold'unda kurulur; yayımlanan fark metin kolu eksi taban kol; taban kol bu
    koşunun modeli (08'in log R²'si, 07'nin MAPE'si, aynı ilanlar ve 5 fold); metin kolunun ayarları planınki.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "10_free_text.py"


@pytest.fixture(scope="module")
def functions():
    """
    EN: The script's pure functions (cells 1–3), without loading any data.
    TR: Betiğin saf işlevleri (1–3. hücreler), veri yüklemeden.
    """
    # EN: "lib" is also db/lib (other tests import it): load analysis/lib for this exec only, then put things back
    # TR: "lib" aynı zamanda db/lib (başka testler import eder): yalnız bu exec için analysis/lib yüklenir, sonra geri konur
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "lib" or k.startswith("lib.")}
    sys.path.insert(0, str(ROOT / "analysis"))
    try:
        head = SCRIPT.read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "free_text_functions"}
        exec(compile(head, str(SCRIPT), "exec"), ns)
    finally:
        sys.path.remove(str(ROOT / "analysis"))
        for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ns


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published text ablation. / TR: Yayımlanan metin ablasyonu."""
    return metrics("10_free_text")["report"]["text_ablation"]


def test_text_features_fit_on_train_only(functions):
    """
    EN: A word that occurs only in the validation part never enters the vocabulary, and both parts get the same
        SVD columns.
    TR: Yalnız doğrulama kısmında geçen kelime sözlüğe girmez; iki kısım da aynı SVD kolonlarını alır.
    """
    texts = pd.Series(["alfa beta gama", "alfa delta", "beta gama delta", "zeta zeta zeta"])
    tr, va = [0, 1, 2], [3]
    dtr, dva, vocab = functions["description_features"](texts, tr, va, size=2, min_df=1)
    assert "zeta" not in vocab and "alfa" in vocab
    assert dtr.shape == (3, 2) and dva.shape == (1, 2) and list(dtr.columns) == list(dva.columns)


def test_gap_is_text_minus_base(doc):
    """
    EN: ΔR² = R²(text) − R²(base); the closed share is ΔR² over the base's unexplained variance.
    TR: ΔR² = R²(metin) − R²(taban); kapatılan pay, ΔR²'nin tabanın açıklayamadığı varyansa oranı.
    """
    assert 0 < doc["r2_log_base"] < 1 and 0 < doc["r2_log_text"] < 1
    assert doc["delta_r2"] == pytest.approx(doc["r2_log_text"] - doc["r2_log_base"], abs=1e-4)
    assert doc["unexplained_share_pct"] == pytest.approx(
        doc["delta_r2"] / (1 - doc["r2_log_base"]) * 100, abs=0.5)


def test_base_is_the_model(doc, metrics):
    """
    EN: The base arm is this run's model: 08's log R², 07's LightGBM MAPE and MAE, the model's listings, 5 folds.
    TR: Taban kol bu koşunun modeli: 08'in log R²'si, 07'nin LightGBM MAPE ve MAE'si, modelin ilanları, 5 fold.
    """
    lgb = metrics("07_model_comparison")["domain"]["model_compare"]["lightgbm"]
    assert doc["r2_log_base"] == pytest.approx(metrics("08_residuals")["report"]["model_r2_log"], abs=1e-4)
    assert doc["mape_base"] == pytest.approx(lgb["MAPE"], abs=0.01)
    assert doc["mae_base"] == pytest.approx(lgb["MAE"], abs=1)
    assert doc["n"] == metrics("01_dedup_leakage")["meta"]["n_dedup"] and doc["folds"] == 5
    assert 0 <= doc["empty_descriptions"] < doc["n"]


def test_settings_are_the_plan(doc):
    """
    EN: The text arm's settings are the pre-registered ones (word TF-IDF 1–2 grams, min_df 5, SVD 50).
    TR: Metin kolunun ayarları ön kayıttakiler (kelime TF-IDF 1–2 gram, min_df 5, SVD 50).
    """
    assert doc["tfidf"] == {"analyzer": "word", "ngram_range": [1, 2], "min_df": 5}
    assert doc["text_svd"] == 50
