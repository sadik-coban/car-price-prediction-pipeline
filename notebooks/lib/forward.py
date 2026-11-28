"""
İleri test: son taramadan önce görülen ilanlarla servis edilen tahmin modeli gibi eğitilen bir tahmin modeli, son
taramada ilk kez görülen ilanları fiyatlar. Test taraması hiçbir şeyi belirlemez: ne tahmin modelini, ne ağaç
sayısını, ne aralığın s'sini ya da k'sını. Yöntem eski zincirin analysis/08_forward_test.py dosyasından; kod bu defter
için yeniden yazıldı.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import data, folds, interval, model
from lib.errors import PRICE_CAP

MIN_TEST = 30          # test kümesinde en az bu kadar ilan olmalı


def split(rows, min_test=MIN_TEST):
    """
    rows: her taramanın satırları (data.load_rows(latest_only=False) + data.add_features; kolonlar ad_id, snap).
    Eğitim: son taramadan önce görülen her ilan, o taramalardaki son satırıyla. Test: son taramada ilk kez görülen
    ilanlar, ilan başına tek satır. İkisi de ad_id sırasıyla. Testte min_test'ten az ilan varsa durur.
    Döndürür: (eğitim, test, eğitimin son taraması, test taraması).
    """
    snaps = sorted(rows["snap"].unique())
    ordered = rows.sort_values(["ad_id", "search_date"], kind="stable")
    train = ordered[ordered["snap"].isin(snaps[:-1])].drop_duplicates("ad_id", keep="last").reset_index(drop=True)
    last = ordered[(ordered["snap"] == snaps[-1]) & ~ordered["ad_id"].isin(train["ad_id"])]
    test = last.drop_duplicates("ad_id", keep="last").reset_index(drop=True)
    if len(test) < min_test:
        raise ValueError(f"son taramada yeni ilan çok az: {len(test)}")
    return train, test, snaps[-2], snaps[-1]


def fit_predict(train_table, test_table, trees):
    """
    Eğitim ilanlarıyla, kat matrisleriyle aynı kurulumda (kodlar ve TF-IDF + SVD yalnız eğitim ilanlarından)
    LightGBM, trees ağaç, erken durdurma yok; test ilanlarını tahmin eder. train_table: data.FEATURES ve price;
    test_table: data.FEATURES (fiyatı kullanılmaz). Döndürür: test ilanlarının log tahmini.
    """
    both = pd.concat([train_table[data.FEATURES], test_table[data.FEATURES]], ignore_index=True)
    train, test = np.arange(len(train_table)), np.arange(len(train_table), len(both))
    X_train, X_test, categorical = model.fold_matrices(both, train, test)
    regressor = lgb.LGBMRegressor(**{**model.LGB_PARAMS, "n_estimators": trees})
    regressor.fit(X_train, np.log1p(train_table["price"].to_numpy(dtype=float)), categorical_feature=categorical)
    return regressor.predict(X_test)


def forward_run(train_table, test_table, train_cells, test_cells):
    """
    İleri testin ağır kısmı. Eğitim kümesinde beş katlı kat dışı koşum (model.oof_lightgbm, folds.make_folds); ağaç
    sayısı katların medyanı; sonra bütün eğitim kümesiyle tek eğitim ve test tahmini (fit_predict). Aralık eğitim
    kümesinin kendi kat dışı hatalarından: iç içe s ve skorlar (interval.nested_scale; eğitim ilanının emsali kendi
    iç katı sayılmadan), test ilanlarının s'si bütün eğitim kümesiyle eğitilmiş ölçek modelinden; test ilanının emsali
    eğitim kümesinde aynı hücredeki ilan sayısı. test_table'ın fiyatı hiçbir yere girmez.
    train_cells, test_cells: ilan başına hücre (araç modeli + model yılı). Döndürür: {"cv_trees", "trees",
    "train_pred_log", "train_fold" (1-5), "test_pred_log", "scores", "test_s", "test_comparables"}.
    """
    cv_folds = folds.make_folds(len(train_table))
    run = model.oof_lightgbm(train_table[data.FEATURES + ["price"]], cv_folds)
    trees = int(np.median(run["trees"]))
    test_log = fit_predict(train_table, test_table, trees)
    fold = np.zeros(len(train_table), dtype=int)
    for k, (_, valid) in enumerate(cv_folds, 1):
        fold[valid] = k
    train_pred = np.clip(np.expm1(run["pred_log"]), 0, PRICE_CAP)
    train_err = np.abs(np.log1p(train_table["price"].to_numpy(dtype=float)) - np.log1p(train_pred))
    log_pred = np.r_[np.log1p(train_pred), np.log1p(np.clip(np.expm1(test_log), 0, PRICE_CAP))]
    cells = pd.Series(np.asarray(train_cells, dtype=object)).astype(str)
    test_counts = pd.Series(np.asarray(test_cells, dtype=object)).astype(str).map(cells.value_counts())
    comparables = np.r_[interval.other_fold_counts(train_cells, fold), test_counts.fillna(0).to_numpy(dtype=int)]
    n = len(train_table)
    s_train = interval.nested_scale(train_table, train_err, log_pred[:n], comparables[:n], fold)
    both = pd.concat([train_table[data.FEATURES], test_table[data.FEATURES]], ignore_index=True)
    X_train, X_test, categorical = interval.scale_matrices(both, np.arange(n), np.arange(n, len(both)), log_pred,
                                                           comparables)
    test_s = interval.predict_scale(interval.fit_scale(X_train, train_err, categorical), X_test)
    return {"cv_trees": run["trees"], "trees": trees, "train_pred_log": run["pred_log"], "train_fold": fold,
            "test_pred_log": test_log, "scores": train_err / s_train, "test_s": test_s,
            "test_comparables": comparables[n:]}


def cluster_bootstrap(groups, values, n_boot, seed):
    """
    Grupları (ör. araç modellerini) yerine koyarak yeniden örnekleyen bootstrap: her çekilişte gruplar çekilir, her
    dizinin ortalaması çekilen grupların ilanları üzerinden alınır. values: aynı ilanlar üzerinde bir ya da daha çok
    dizi; hepsi aynı çekilişleri paylaşır (eşli). Döndürür: (n_boot, dizi sayısı) ortalamalar.
    """
    _, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    n_groups = inverse.max() + 1
    per_group = np.bincount(inverse).astype(float)
    sums = [np.bincount(inverse, weights=np.asarray(v, dtype=float)) for v in values]
    weights = np.random.default_rng(seed).multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    return np.column_stack([(weights @ total) / (weights @ per_group) for total in sums])
