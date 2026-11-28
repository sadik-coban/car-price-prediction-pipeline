"""
test_free_text.py
notebooks/lib/free_text.py küçük, uydurma metinlerle: Türkçeye uygun küçük harf; boş açıklama boş metin; açıklama
kolonları yalnız eğitim kısmından kurulur; açıklamalı koşumda bir katın tahmini, temel koşumun kat matrisi ve o katın
açıklama kolonlarıyla, öteki katlarda kurulan modelden; fiyatı taşıyan bir metin hatayı küçültür. Gerçek veri okunmaz.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import folds, free_text, model
from test_interval import made_up


def test_lower_tr():
    assert free_text.lower_tr("İSTANBUL IŞIK") == "istanbul ışık"
    assert free_text.lower_tr("KIRMIZI İç") == "kırmızı iç"
    assert free_text.lower_tr("i̇") == "i"                                      # artakalan birleşik nokta
    assert free_text.descriptions(pd.DataFrame({"description_text": ["BOYASIZ", None]})).tolist() == ["boyasız", ""]


def test_description_columns_come_from_the_training_part():
    train = pd.Series(["temiz araç bakımlı", "temiz araç hasarsız", "bakımlı araç"] * 10)
    other_a = pd.Series(["temiz araç", "sadece yepyeni kelimeler"])
    train_a, out_a = free_text.description_columns(train, other_a, size=10)
    train_b, _ = free_text.description_columns(train, pd.Series(["başka bir metin"] * 4), size=10)
    assert np.array_equal(train_a.to_numpy(), train_b.to_numpy())                    # öteki kısım eğitimi değiştirmez
    assert list(train_a.columns) == [f"desc_{i}" for i in range(7)]                  # 8 terim: boyut 7'ye kırpılır
    assert out_a.shape == (2, 7) and np.allclose(out_a.iloc[1], 0)                   # sözlükte olmayan kelimeler


def test_description_arm_is_the_base_setup_plus_text():
    table = made_up(500, seed=5)
    cv = folds.make_folds(len(table))
    log_price = np.log(table["price"])
    level = np.digitize(log_price, np.quantile(log_price, [0.2, 0.4, 0.6, 0.8]))
    words = np.array(["çok ucuz", "ucuz fiyat", "orta fiyat", "pahalı fiyat", "çok pahalı"])
    texts = pd.Series([f"ilan {w}" for w in words[level]])
    told = free_text.oof_with_description(table, texts, cv)
    y = np.log1p(table["price"].to_numpy())
    train, valid = cv[0]
    X_train, X_valid, categorical = model.fold_matrices(table, train, valid)
    text_train, text_valid = free_text.description_columns(texts.iloc[train], texts.iloc[valid])
    alone = lgb.LGBMRegressor(**{**model.LGB_PARAMS, "n_estimators": told["trees"][0]})
    alone.fit(pd.concat([X_train, text_train], axis=1), y[train], categorical_feature=categorical)
    assert np.allclose(alone.predict(pd.concat([X_valid, text_valid], axis=1)), told["pred_log"][valid])
    base = model.oof_lightgbm(table, cv)
    rmse = [model.price_metrics(table["price"], np.expm1(run["pred_log"]))["rmse"] for run in (base, told)]
    assert rmse[1] < rmse[0] and len(told["trees"]) == folds.N_FOLDS
