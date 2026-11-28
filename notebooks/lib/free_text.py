"""
Serbest metin kolu: satıcının ilan açıklaması tahmin modeline girdi olarak verilince kat dışı hata ne kadar değişir.
Açıklama kelime TF-IDF'ine (1-2 kelimelik parçalar, en az 5 ilanda geçenler) ve SVD ile 50 kolona çevrilir; ikisi de
her katta yalnız eğitim kısmından kurulur. Öteki her şey temel koşumla aynı: katlar, öznitelikler, LightGBM ayarları ve
erken durdurma. Metinde kelime ya da kalıp aranmaz; açıklama yalnız sayısal kolonlara çevrilir.
Küçük harfe çevirme (lower_tr) eski zincirin analysis/lib/text_flags.py dosyasındaki lower_tr'nin aynısı; her koşumda
güncel açıklamalara uygulanır ve çıktısı yalnız bu kolda kullanılır.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from lib import folds, model

TFIDF = dict(analyzer="word", ngram_range=(1, 2), min_df=5)
TEXT_SVD = 50                                   # açıklamanın SVD boyutu
_TO_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_COMBINING_DOT = "̇"                       # küçük harfe çevirmede artakalan birleşik nokta


def lower_tr(text):
    """Türkçeye uygun küçük harf: I -> ı, İ -> i, artakalan birleşik noktalar silinir, NFC."""
    lowered = str(text).translate(_TO_LOWER).lower().replace(_COMBINING_DOT, "")
    return unicodedata.normalize("NFC", lowered)


def descriptions(listings):
    """Her ilanın küçük harfe çevrilmiş açıklaması (description_text; boşsa boş metin). Döndürür: seri."""
    return listings["description_text"].fillna("").astype(str).map(lower_tr)


def description_columns(train_texts, other_texts, size=TEXT_SVD):
    """
    Bir katın açıklama kolonları: kelime TF-IDF'i ve SVD, ikisi de yalnız eğitim kısmının metinlerinden kurulur
    (yalnız öteki kısımda geçen kelime sözlüğe girmez). Döndürür: (eğitim, öteki) tabloları; kolonlar desc_0, desc_1 ...
    """
    vectorizer = TfidfVectorizer(**TFIDF)
    train_terms, other_terms = vectorizer.fit_transform(train_texts), vectorizer.transform(other_texts)
    svd = TruncatedSVD(min(size, train_terms.shape[1] - 1), random_state=folds.SEED)
    columns = [f"desc_{i}" for i in range(svd.n_components)]
    return (pd.DataFrame(svd.fit_transform(train_terms), columns=columns),
            pd.DataFrame(svd.transform(other_terms), columns=columns))


def oof_with_description(table, texts, cv_folds):
    """
    Açıklamalı kat dışı koşum: her katta tahmin modelinin matrisleri (model.fold_matrices) ve açıklama kolonları
    (description_columns) yan yana; LightGBM model.LGB_PARAMS ile, katın doğrulama kısmında model.EARLY_STOP tur
    iyileşmeyince durur (temel koşumla aynı). table: data.FEATURES ve price; texts: tabloyla hizalı açıklamalar.
    Döndürür: {"pred_log": kat dışı log tahminler, "trees": kat başına ağaç sayısı}.
    """
    y = np.log1p(table["price"].to_numpy(dtype=float))
    texts = pd.Series(np.asarray(texts, dtype=object)).astype(str)
    pred, trees = np.zeros(len(table)), []
    for train, valid in cv_folds:
        X_train, X_valid, categorical = model.fold_matrices(table, train, valid)
        text_train, text_valid = description_columns(texts.iloc[train], texts.iloc[valid])
        X_train, X_valid = pd.concat([X_train, text_train], axis=1), pd.concat([X_valid, text_valid], axis=1)
        regressor = lgb.LGBMRegressor(**model.LGB_PARAMS)
        regressor.fit(X_train, y[train], categorical_feature=categorical, eval_X=(X_valid,), eval_y=(y[valid],),
                      callbacks=[lgb.early_stopping(model.EARLY_STOP, verbose=False)])
        pred[valid] = regressor.predict(X_valid)
        trees.append(int(regressor.best_iteration_ or model.LGB_PARAMS["n_estimators"]))
    return {"pred_log": pred, "trees": trees}
