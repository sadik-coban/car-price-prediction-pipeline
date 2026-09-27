"""
10_free_text.py
EN: Technical report §10 — what the seller's free text adds to the model, measured live on every run
    (pre-registered plan plans/10-text-contribution). The base arm is this run's model OOF (07_model_comparison,
    read with cv.load_oof); the text arm uses the same 5 folds, features and LightGBM settings plus the
    description as word TF-IDF (1–2 grams) reduced by SVD, both fitted on each fold's training part only. Both
    arms are scored on log R², MAPE and MAE. Replaces the ablation frozen from the archived text run (2026-09-27,
    the rule that nothing from the archive reaches the live chain); no LLM is run.
TR: Teknik rapor §10 — satıcının serbest metni modele ne katıyor; her koşuda canlı ölçülür (ön kayıtlı plan
    plans/10-text-contribution). Taban kol bu koşunun model OOF'u (07_model_comparison, cv.load_oof ile okunur);
    metin kolu aynı 5 fold'u, öznitelikleri ve LightGBM ayarlarını kullanır, üstüne açıklama kelime TF-IDF'i
    (1–2 gram) ve SVD ile; ikisi de her fold'un yalnız eğitim kısmında kurulur. İki kol log R², MAPE ve MAE ile
    ölçülür. Arşivdeki metin koşumundan dondurulmuş ablasyonun yerini aldı (2026-09-27, arşivden canlı zincire
    hiçbir şey girmez kuralı); LLM koşulmaz.
Output / Çıktı: metrics/10_free_text.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

from lib import text_flags as TF
from lib.common import FEATURES, load_clean, save_metrics
from lib.cv import N_JOBS, SEED, fit_lgb_fold, fold_matrices, load_oof, make_folds, price_metrics

# EN: the text arm's settings, fixed in the pre-registered plan | TR: metin kolunun ayarları, ön kayıtlı planda sabit
TFIDF = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 5}
TEXT_SVD = 50


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def description_features(texts, tr, va, size=TEXT_SVD, min_df=TFIDF["min_df"]):
    """
    EN: One fold's description features: word TF-IDF (1–2 grams) and TruncatedSVD, both fitted on the training
        part only (a word seen only in the validation part never enters the vocabulary).
        texts: the lower-cased descriptions (a Series); tr, va: row positions; size: SVD components.
        Returns: (train frame, valid frame, vocabulary).
    TR: Bir fold'un açıklama öznitelikleri: kelime TF-IDF (1–2 gram) ve TruncatedSVD, ikisi de yalnız eğitim
        kısmında kurulur (yalnız doğrulama kısmında geçen kelime sözlüğe girmez).
        texts: küçültülmüş açıklamalar (Series); tr, va: satır konumları; size: SVD bileşen sayısı.
        Döndürür: (eğitim tablosu, doğrulama tablosu, sözlük).
    """
    vec = TfidfVectorizer(analyzer=TFIDF["analyzer"], ngram_range=TFIDF["ngram_range"], min_df=min_df)
    m_tr, m_va = vec.fit_transform(texts.iloc[tr]), vec.transform(texts.iloc[va])
    n = min(size, m_tr.shape[1] - 1)
    svd = TruncatedSVD(n, random_state=SEED)
    names = [f"desc_{i}" for i in range(n)]
    return (pd.DataFrame(svd.fit_transform(m_tr), columns=names), pd.DataFrame(svd.transform(m_va), columns=names),
            set(vec.vocabulary_))


def text_arm_oof(X, yl, texts, folds, n_jobs=N_JOBS):
    """
    EN: 5-fold OOF LightGBM with the model's features (cv.fold_matrices) plus the description features, the same
        folds and settings as the model (cv.fit_lgb_fold). Returns: OOF log predictions.
    TR: Modelin öznitelikleri (cv.fold_matrices) artı açıklama öznitelikleriyle 5-fold OOF LightGBM; modelle aynı
        fold'lar ve ayarlar (cv.fit_lgb_fold). Döndürür: OOF log tahminleri.
    """
    pred = np.zeros(len(X))
    for tr, va in folds:
        Xtr, Xva, catc = fold_matrices(X, tr, va)
        dtr, dva, _ = description_features(texts, tr, va)
        mdl = fit_lgb_fold(pd.concat([Xtr, dtr], axis=1), yl[tr], pd.concat([Xva, dva], axis=1), yl[va], catc, n_jobs)
        pred[va] = mdl.predict(pd.concat([Xva, dva], axis=1))
    return pred


def ablation(price, yl, base_log, text_log):
    """
    EN: Both arms on log R², MAPE and MAE (price metrics on expm1 of the log prediction, as in 07) and their
        gaps; the share of the base's unexplained log variance the text closes.
    TR: İki kol log R², MAPE ve MAE ile (fiyat ölçüleri log tahminin expm1'i üzerinden, 07'deki gibi) ve farkları;
        metnin tabanın açıklayamadığı log varyanstan kapattığı pay.
    """
    b, t = price_metrics(price, np.expm1(base_log)), price_metrics(price, np.expm1(text_log))
    r2b, r2t = r2_score(yl, base_log), r2_score(yl, text_log)
    return {"r2_log_base": r2b, "r2_log_text": r2t, "delta_r2": r2t - r2b,
            "mape_base": b["MAPE"], "mape_text": t["MAPE"], "mae_base": b["MAE"], "mae_text": t["MAE"],
            "unexplained_share_pct": (r2t - r2b) / (1 - r2b) * 100}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published as report.text_ablation (technical report §1 and §10).
    TR: report.text_ablation olarak yayımlanır (teknik rapor §1 ve §10).
    """
    a = res["ablation"]
    return {"report": {"text_ablation": {
        "r2_log_base": round(a["r2_log_base"], 4), "r2_log_text": round(a["r2_log_text"], 4),
        "delta_r2": round(a["delta_r2"], 4),
        "mape_base": round(a["mape_base"], 2), "mape_text": round(a["mape_text"], 2),
        "mae_base": round(a["mae_base"]), "mae_text": round(a["mae_text"]),
        "unexplained_share_pct": round(a["unexplained_share_pct"], 1),
        "n": res["n"], "folds": res["folds"], "empty_descriptions": res["empty"],
        "text_svd": TEXT_SVD, "tfidf": {"analyzer": TFIDF["analyzer"], "ngram_range": list(TFIDF["ngram_range"]),
                                        "min_df": TFIDF["min_df"]},
        "note": ("Taban = bu koşunun model OOF'u (07_model_comparison); metin kolu aynı fold'lar ve ayarlar + açıklama "
                 "kelime TF-IDF'i ve SVD, yalnız eğitim fold'unda kurulur. Ön kayıt: plans/10-text-contribution.")}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, _ = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
texts = TF.descriptions(listings)
price = listings["price"].values.astype(float)
yl = np.log1p(price)
folds = make_folds(len(listings))
text_log = text_arm_oof(listings[FEATURES], yl, texts, folds)
res = {"ablation": ablation(price, yl, oof["lgb_log"].values, text_log), "n": len(listings), "folds": len(folds),
       "empty": int((texts.str.strip() == "").sum())}
print({k: round(v, 4) for k, v in res["ablation"].items()})

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("10_free_text", to_metrics(res)))
