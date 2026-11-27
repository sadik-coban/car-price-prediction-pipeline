"""
Tahmin modeli: bir katın matrisleri, LightGBM'in ve iki CatBoost'un kat dışı koşumu, bütün ilanlarla son LightGBM,
hata ölçüleri, medyan tabanı ve iki koşumun farkı için eşli bootstrap. Ayarlar burada adlı sabitler; defter onları
gösterir. Hedef log(1 + fiyat); kat dışı tahminin ₺ karşılığı exp(tahmin) - 1.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from lib import cache, data, folds

N_JOBS = 16       # iş parçacığı: LightGBM'in sonucunu değiştirmiyor, ama son modelin metnine yazılıyor; sabit
TEXT_SVD = {"model": 150, "series": 20}       # metin kolonu -> SVD boyutu
TFIDF = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5)
LGB_PARAMS = dict(n_estimators=900, learning_rate=0.05, num_leaves=128, min_child_samples=20, random_state=folds.SEED,
                  verbose=-1, deterministic=True, force_row_wise=True, n_jobs=N_JOBS)
EARLY_STOP = 40                               # doğrulama hatası bu kadar tur iyileşmezse durur
CB_PARAMS = dict(iterations=500, learning_rate=0.06, depth=8, loss_function="RMSE", random_seed=folds.SEED, verbose=0,
                 thread_count=N_JOBS, allow_writing_files=False)


def category_codes(train, other):
    """
    Kategorik bir kolonun kodları, yalnız eğitim kısmından: değerler eğitimde ilk görüldükleri sırayla 0, 1, 2 ...;
    eğitimde görülmemiş değer -1. Boş değerde durur (önce 'missing' ile doldurulmalı).
    Döndürür: (eğitim kodları, öteki kısmın kodları).
    """
    train, other = pd.Series(np.asarray(train, dtype=object)), pd.Series(np.asarray(other, dtype=object))
    if train.isna().any() or other.isna().any():
        raise ValueError("kategorik kolonda boş değer var; önce 'missing' ile doldurun")
    codes = {value: i for i, value in enumerate(train.unique())}
    return train.map(codes).to_numpy(dtype=int), other.map(codes).fillna(-1).to_numpy(dtype=int)


def text_columns(train, other, name, size):
    """
    Bir metin kolonunun (araç modeli ya da seri adı) sayısal kolonları, yalnız eğitim kısmında kurulur: karakter 3-5'li
    TF-IDF, sonra size boyutlu SVD (TF-IDF'in terim sayısı yetmezse terim sayısının bir eksiği).
    Döndürür: (eğitim, öteki) tabloları; kolonlar <name>_0, <name>_1, ...
    """
    vectorizer = TfidfVectorizer(**TFIDF)
    train_terms, other_terms = vectorizer.fit_transform(train), vectorizer.transform(other)
    svd = TruncatedSVD(min(size, train_terms.shape[1] - 1), random_state=folds.SEED)
    columns = [f"{name}_{i}" for i in range(svd.n_components)]
    return (pd.DataFrame(svd.fit_transform(train_terms), columns=columns),
            pd.DataFrame(svd.transform(other_terms), columns=columns))


def fold_matrices(table, train, valid, drop=(), added=()):
    """
    Bir katın tahmin modeli matrisleri. table: data.FEATURES (ve added) kolonlu ilanlar; train, valid: satır konumları.
    Kolonlar sırayla: kategorikler (category_codes), sayısallar olduğu gibi, araç modeli ve seri adlarının SVD kolonları
    (text_columns), en sonda added kolonları (kategorik gibi kodlanır). drop'taki öznitelikler girmez.
    Bilinmeyen bir öznitelik adında ya da tabloda olmayan bir added kolonunda durur.
    Döndürür: (eğitim matrisi, doğrulama matrisi, kategorik kolonların adları).
    """
    unknown = [c for c in drop if c not in data.FEATURES] + [c for c in added if c in data.FEATURES or c not in table]
    if unknown:
        raise ValueError(f"bilinmeyen öznitelik ya da kolon: {unknown}")
    part_train, part_valid = table.iloc[train].reset_index(drop=True), table.iloc[valid].reset_index(drop=True)
    numeric = [c for c in data.NUM if c not in drop]
    categorical = [c for c in data.CAT if c not in drop] + list(added)
    coded = {c: category_codes(part_train[c], part_valid[c]) for c in categorical}
    blocks_train = [pd.DataFrame({c: coded[c][0] for c in categorical if c not in added}), part_train[numeric]]
    blocks_valid = [pd.DataFrame({c: coded[c][1] for c in categorical if c not in added}), part_valid[numeric]]
    for name, size in TEXT_SVD.items():
        if name not in drop:
            text_train, text_valid = text_columns(part_train[name], part_valid[name], name, size)
            blocks_train.append(text_train)
            blocks_valid.append(text_valid)
    blocks_train.append(pd.DataFrame({c: coded[c][0] for c in added}, index=part_train.index))
    blocks_valid.append(pd.DataFrame({c: coded[c][1] for c in added}, index=part_valid.index))
    return pd.concat(blocks_train, axis=1), pd.concat(blocks_valid, axis=1), categorical


def oof_lightgbm(table, cv_folds, drop=(), added=()):
    """
    LightGBM'in kat dışı koşumu. Her katta matrisler fold_matrices ile yalnız eğitim kısmından kurulur; LightGBM
    log(1 + fiyat) üzerinde LGB_PARAMS ile eğitilir ve katın doğrulama kısmında EARLY_STOP tur iyileşmeyince durur.
    table: data.FEATURES, added ve price kolonları. Döndürür: {"pred_log": kat dışı log tahminler,
    "trees": kat başına erken durdurmanın seçtiği ağaç sayısı}.
    """
    y = np.log1p(table["price"].to_numpy(dtype=float))
    pred, trees = np.zeros(len(table)), []
    for train, valid in cv_folds:
        X_train, X_valid, categorical = fold_matrices(table, train, valid, drop, added)
        regressor = lgb.LGBMRegressor(**LGB_PARAMS)
        regressor.fit(X_train, y[train], categorical_feature=categorical, eval_X=(X_valid,), eval_y=(y[valid],),
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        pred[valid] = regressor.predict(X_valid)
        trees.append(int(regressor.best_iteration_ or LGB_PARAMS["n_estimators"]))
    return {"pred_log": pred, "trees": trees}


def oof_catboost(table, cv_folds):
    """
    CatBoost'un kat dışı koşumu, LightGBM'le aynı kat matrislerinde (fold_matrices; kategorikler CatBoost'a kategorik
    olarak bildirilir). CB_PARAMS ile, erken durdurma yok. Döndürür: kat dışı log tahminler.
    """
    y = np.log1p(table["price"].to_numpy(dtype=float))
    pred = np.zeros(len(table))
    for train, valid in cv_folds:
        X_train, X_valid, categorical = fold_matrices(table, train, valid)
        positions = [X_train.columns.get_loc(c) for c in categorical]
        regressor = CatBoostRegressor(**CB_PARAMS)
        regressor.fit(Pool(X_train, y[train], cat_features=positions))
        pred[valid] = regressor.predict(Pool(X_valid, cat_features=positions))
    return pred


def oof_catboost_native(table, cv_folds):
    """
    CatBoost'un kat dışı koşumu, kendi metin işleyişiyle: ham öznitelik kolonları (data.FEATURES); kategorikler
    kategorik, araç modeli ve seri adları metin özniteliği olarak verilir (TF-IDF ve SVD yok). CB_PARAMS ile.
    Döndürür: kat dışı log tahminler.
    """
    X = table[data.FEATURES]
    y = np.log1p(table["price"].to_numpy(dtype=float))
    roles = dict(cat_features=[X.columns.get_loc(c) for c in data.CAT],
                 text_features=[X.columns.get_loc(c) for c in data.TEXT])
    pred = np.zeros(len(table))
    for train, valid in cv_folds:
        regressor = CatBoostRegressor(**CB_PARAMS)
        regressor.fit(Pool(X.iloc[train], y[train], **roles))
        pred[valid] = regressor.predict(Pool(X.iloc[valid], **roles))
    return pred


def fit_final(table, trees):
    """
    Son tahmin modeli: bütün ilanlarla, kat matrisleriyle aynı kurulumda (kodlar ve SVD bütün ilanlardan) LightGBM;
    trees ağaç, erken durdurma yok. Döndürür: eğitilmiş LGBMRegressor.
    """
    everything = np.arange(len(table))
    X, _, categorical = fold_matrices(table, everything, everything)        # doğrulama kısmı kullanılmaz
    regressor = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": trees})
    regressor.fit(X, np.log1p(table["price"].to_numpy(dtype=float)), categorical_feature=categorical)
    return regressor


def base_oof(listings):
    """
    Temel koşumun (3. defter, 1. bölüm: bütün öznitelikler, ortak beş kat) kat dışı tahminleri, önbellekten; kayıt yoksa
    koşar. listings: data.add_features çıktısı. Döndürür: ilan başına log tahmin, ₺ tahmin ve kat (1-5).
    """
    table = listings[data.FEATURES + ["price"]]
    cv_folds = folds.make_folds(len(table))
    run = cache.load_or_run(oof_lightgbm, table, cv_folds)
    fold = np.zeros(len(table), dtype=int)
    for k, (_, valid) in enumerate(cv_folds, 1):
        fold[valid] = k
    return pd.DataFrame({"pred_log": run["pred_log"], "pred_price": np.expm1(run["pred_log"]), "fold": fold},
                        index=listings.index)


def price_metrics(actual, pred):
    """Fiyat tahminlerinin hata ölçüleri (ikisi de ₺): MAE, MedAE, RMSE, MAPE (%) ve R²."""
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return {"mae": float(mean_absolute_error(actual, pred)), "medae": float(median_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "mape": float(np.mean(np.abs((actual - pred) / actual)) * 100), "r2": float(r2_score(actual, pred))}


def median_baseline(key, year, price, cv_folds):
    """
    Kat dışı medyan tabanı: her katın ilanına, eğitim katlarında aynı (anahtar, model yılı) hücresindeki ilanların
    medyan fiyatı. Hücre eğitim kısmında yoksa merdiven iner: anahtarın bütün yılları, o da yoksa bütün eğitim ilanları.
    Model yılı boş bir ilanda durur. Döndürür: (tahminler, ilan başına basamak: 'key_year', 'key' ya da 'overall').
    """
    frame = pd.DataFrame({"key": np.asarray(key, dtype=object), "year": np.asarray(year),
                          "price": np.asarray(price, dtype=float)})
    if frame["year"].isna().any():
        raise ValueError("model yılı boş ilan var")
    pred, rung = np.zeros(len(frame)), np.empty(len(frame), dtype=object)
    for train, valid in cv_folds:
        part, scored = frame.iloc[train], frame.iloc[valid]
        cell = part.groupby(["key", "year"])["price"].median()
        cell = cell.reindex(pd.MultiIndex.from_frame(scored[["key", "year"]])).to_numpy()
        by_key = part.groupby("key")["price"].median().reindex(scored["key"]).to_numpy()
        pred[valid] = np.where(~np.isnan(cell), cell, np.where(~np.isnan(by_key), by_key, part["price"].median()))
        rung[valid] = np.where(~np.isnan(cell), "key_year", np.where(~np.isnan(by_key), "key", "overall"))
    return pred, rung


def paired_bootstrap(actual, base, variants, n_resamples, seed):
    """
    Eşli bootstrap, ilanlar üzerinden: her yeniden örneklemde ilanlar yerine koyarak çekilir; çekiliş bütün varyantlar
    ve taban için aynıdır. Her örneklemde her varyantın RMSE'si (₺) ve MAPE'si (puan) eksi tabanınki.
    actual: gerçek fiyat; base: tabanın ₺ tahmini; variants: {ad: ₺ tahmin}. Tahminler sabit tutulur, modeller yeniden
    kurulmaz. Döndürür: {ad: {"rmse": farklar, "mape": farklar}}, her biri n_resamples uzunluğunda.
    """
    actual = np.asarray(actual, dtype=float)
    squared = {name: (actual - np.asarray(p)) ** 2 for name, p in variants.items()}
    pct = {name: np.abs((actual - np.asarray(p)) / actual) * 100 for name, p in variants.items()}
    base_squared, base_pct = (actual - np.asarray(base)) ** 2, np.abs((actual - np.asarray(base)) / actual) * 100
    out = {name: {"rmse": np.empty(n_resamples), "mape": np.empty(n_resamples)} for name in variants}
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        rows = rng.integers(0, len(actual), size=len(actual))
        base_rmse, base_mape = np.sqrt(base_squared[rows].mean()), base_pct[rows].mean()
        for name in variants:
            out[name]["rmse"][i] = np.sqrt(squared[name][rows].mean()) - base_rmse
            out[name]["mape"][i] = pct[name][rows].mean() - base_mape
    return out
