"""
cv.py
EN: The shared 5-fold cross-validation used by every model script (03_brand_ablation, 07_model_comparison,
    07_lofo, shap/02_oof_shap): the same folds, the same in-fold text features (TF-IDF + SVD fitted on the
    training part only) and the same LightGBM settings, so an out-of-fold (OOF) number means the same thing
    everywhere. Also the OOF artefact that 07_model_comparison writes to data/analysis/ and the later scripts
    read; reading it stops if it was built from a different database.
TR: Her model betiğinin kullandığı ortak 5-fold çapraz doğrulama (03_brand_ablation, 07_model_comparison,
    07_lofo, shap/02_oof_shap): aynı fold'lar, aynı fold içi metin öznitelikleri (TF-IDF + SVD yalnız eğitim
    kısmında kurulur) ve aynı LightGBM ayarı; böylece bir OOF sayısı her yerde aynı anlama gelir. Ayrıca
    07_model_comparison'ın data/analysis/'e yazdığı, sonraki betiklerin okuduğu OOF artefaktı; başka bir
    veritabanından üretilmişse okuma durur.
"""
import hashlib
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold

from .common import ANALYSIS_DIR, CAT, FEATURES, NUM, _db_fingerprint

SEED = 42
N_FOLDS = 5
N_JOBS = int(os.environ.get("N_JOBS") or (os.cpu_count() or 4))
TEXT_SVD = (("model", 150), ("series", 20))      # text column, SVD size | metin kolonu, SVD boyutu
# EN: deterministic=True alone is not enough with several threads; it needs force_row_wise
# TR: çok iş parçacığında deterministic=True tek başına yetmez; force_row_wise ile gelmeli
LGB_PARAMS = dict(n_estimators=900, learning_rate=0.05, num_leaves=128, min_child_samples=20,
                  random_state=SEED, verbose=-1, deterministic=True, force_row_wise=True)
EARLY_STOP = 40
CB_PARAMS = dict(iterations=500, learning_rate=0.06, depth=8, loss_function="RMSE", random_seed=SEED, verbose=0)
PRICE_CAP = 1.5e7
LARGE_ERROR_PCT = 20.0          # |OOF residual| above this is a large error | bunun üstü büyük hata
OOF_FILE = ANALYSIS_DIR / "oof.parquet"
OOF_INFO = ANALYSIS_DIR / "oof_info.json"
# EN: lightgbm 4.6 renamed eval_set; the call is still valid | TR: lightgbm 4.6 eval_set adını değiştirdi; çağrı geçerli
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")


def catboost_device():
    """
    EN: "GPU" if a CUDA device is visible, else "CPU". GPU and CPU CatBoost do not grow the same trees, so the
        device is recorded in meta.repro.
    TR: CUDA aygıtı görünüyorsa "GPU", yoksa "CPU". GPU ve CPU CatBoost aynı ağacı üretmez; aygıt bu yüzden
        meta.repro'ya yazılır.
    """
    try:
        import torch
        return "GPU" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "GPU" if os.path.exists("/proc/driver/nvidia/version") else "CPU"


def catboost_task(device):
    """
    EN: CatBoost task arguments for the device. / TR: Aygıt için CatBoost görev argümanları.
    """
    return dict(task_type="GPU", devices="0") if device == "GPU" else {}


def make_folds(n):
    """
    EN: The 5 (train, valid) position splits every script uses (shuffled, seed 42). Row order must be fixed
        (load_clean orders by ad_id), otherwise fold membership changes between runs.
    TR: Her betiğin kullandığı 5 (eğitim, doğrulama) konum bölmesi (karıştırılmış, tohum 42). Satır sırası
        sabit olmalı (load_clean ad_id'ye göre sıralar), yoksa fold üyeliği koşudan koşuya değişir.
    """
    return list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))


def fold_matrices(X, tr, va, cols=None, use_text=True):
    """
    EN: The model matrices of one fold. Categories are coded from the training part (unseen value → −1);
        model/series become TF-IDF (char 3–5) + SVD fitted on the training part only (no leakage).
        Column order: categorical, numeric, model_*, series_*.
        Returns: (train matrix, valid matrix, categorical column names).
    TR: Bir fold'un model matrisleri. Kategoriler eğitim kısmından kodlanır (görülmemiş değer → −1);
        model/seri yalnız eğitim kısmında kurulan TF-IDF (karakter 3–5) + SVD olur (sızıntı yok).
        Kolon sırası: kategorik, sayısal, model_*, series_*.
        Döndürür: (eğitim matrisi, doğrulama matrisi, kategorik kolon adları).
    """
    cols = cols or FEATURES
    numc, catc = [c for c in NUM if c in cols], [c for c in CAT if c in cols]
    parts_tr, parts_va = [X.iloc[tr][numc].reset_index(drop=True)], [X.iloc[va][numc].reset_index(drop=True)]
    ct, cv = pd.DataFrame(index=range(len(tr))), pd.DataFrame(index=range(len(va)))
    for c in catc:
        codes = {v: i for i, v in enumerate(X.iloc[tr][c].unique())}
        ct[c] = X.iloc[tr][c].map(codes).values
        cv[c] = X.iloc[va][c].map(codes).fillna(-1).astype(int).values
    if use_text:
        for txt, size in TEXT_SVD:
            if txt not in cols:
                continue
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5)
            m_tr, m_va = vec.fit_transform(X.iloc[tr][txt]), vec.transform(X.iloc[va][txt])
            n = min(size, m_tr.shape[1] - 1)
            svd = TruncatedSVD(n, random_state=SEED)
            names = [f"{txt}_{i}" for i in range(n)]
            parts_tr.append(pd.DataFrame(svd.fit_transform(m_tr), columns=names))
            parts_va.append(pd.DataFrame(svd.transform(m_va), columns=names))
    return pd.concat([ct] + parts_tr, axis=1), pd.concat([cv] + parts_va, axis=1), catc


def fit_lgb_fold(Xtr, ytr, Xva, yva, catc, n_jobs=N_JOBS):
    """
    EN: One fold's LightGBM on log price, stopped early on the fold's own validation part (40 rounds).
        Returns: the fitted LGBMRegressor.
    TR: Bir fold'un log fiyat üzerindeki LightGBM'i; fold'un kendi doğrulama kısmında erken durdurulur (40 tur).
        Döndürür: eğitilmiş LGBMRegressor.
    """
    mdl = lgb.LGBMRegressor(n_jobs=n_jobs, **LGB_PARAMS)
    mdl.fit(Xtr, ytr, categorical_feature=catc, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return mdl


def lgb_oof(X, yl, folds, cols=None, n_jobs=N_JOBS):
    """
    EN: 5-fold OOF LightGBM. X: listings[FEATURES]; yl: log1p(price); cols: feature subset (None = all).
        Returns: {"pred_log": OOF log predictions, "iters": trees chosen per fold,
                  "splits": per fold {feature: number of splits}}.
    TR: 5-fold OOF LightGBM. X: listings[FEATURES]; yl: log1p(fiyat); cols: öznitelik alt kümesi (None = hepsi).
        Döndürür: {"pred_log": OOF log tahminleri, "iters": fold başına seçilen ağaç sayısı,
                   "splits": fold başına {öznitelik: bölme sayısı}}.
    """
    pred, iters, splits = np.zeros(len(X)), [], []
    for tr, va in folds:
        Xtr, Xva, catc = fold_matrices(X, tr, va, cols)
        mdl = fit_lgb_fold(Xtr, yl[tr], Xva, yl[va], catc, n_jobs)
        pred[va] = mdl.predict(Xva)
        iters.append(int(mdl.best_iteration_ or LGB_PARAMS["n_estimators"]))
        booster = mdl.booster_
        splits.append(dict(zip(booster.feature_name(), booster.feature_importance("split").tolist())))
    return {"pred_log": pred, "iters": iters, "splits": splits}


def price_metrics(actual, pred):
    """
    EN: MAE, MedAE, RMSE, MAPE (%) and R² of price predictions (both in ₺).
    TR: Fiyat tahminlerinin MAE, MedAE, RMSE, MAPE (%) ve R²'si (ikisi de ₺).
    """
    return dict(MAE=mean_absolute_error(actual, pred), MedAE=median_absolute_error(actual, pred),
                RMSE=np.sqrt(mean_squared_error(actual, pred)),
                MAPE=np.mean(np.abs((actual - pred) / actual)) * 100, R2=r2_score(actual, pred))


def round_metrics(m):
    """
    EN: The published rounding: R² 4 decimals, MAE/MedAE/RMSE whole ₺, MAPE 2 decimals.
    TR: Yayımlanan yuvarlama: R² 4 hane, MAE/MedAE/RMSE tam ₺, MAPE 2 hane.
    """
    return {k: round(float(v), 4 if k == "R2" else 0 if k in ("MAE", "MedAE", "RMSE") else 2) for k, v in m.items()}


def conformal_q(y_log, pred_price, level=0.90):
    """
    EN: Half-width of the conformal interval: the level-quantile of |log1p(price) − log1p(OOF prediction)|,
        the prediction clipped to [0, 15M]. Interval = expm1(prediction_log ± q).
    TR: Conformal aralığın yarı genişliği: |log1p(fiyat) − log1p(OOF tahmin)|'in level yüzdeliği; tahmin
        [0, 15M]'ye kırpılır. Aralık = expm1(tahmin_log ± q).
    """
    return float(np.quantile(np.abs(y_log - np.log1p(np.clip(pred_price, 0, PRICE_CAP))), level))


def residual_pct(price, oof_price):
    """
    EN: OOF residual in % of the actual price: (actual − prediction) / actual · 100, the prediction clipped to
        [0, 15M]. Negative = the model priced above the listing.
    TR: Gerçek fiyatın yüzdesi olarak OOF artık: (gerçek − tahmin) / gerçek · 100, tahmin [0, 15M]'ye kırpılır.
        Negatif = model ilanın üstünde fiyatladı.
    """
    return (price - np.clip(oof_price, 0, PRICE_CAP)) / price * 100


def large_errors(resid):
    """
    EN: Large-error masks at |residual| > 20% (residual rounded to 2 decimals, as the reports print it).
        Returns: (over-predicted, under-predicted).
    TR: |artık| > %20'de büyük hata maskeleri (artık, raporların bastığı gibi 2 haneye yuvarlanır).
        Döndürür: (fazla tahmin, düşük tahmin).
    """
    r = np.round(resid, 2)
    return r < -LARGE_ERROR_PCT, r > LARGE_ERROR_PCT


def save_oof(frame, info):
    """
    EN: Writes the OOF artefact (data/analysis/oof.parquet + oof_info.json) with the database fingerprint.
        run_id is derived from the predictions and the database, so an identical rerun gets the same id.
        Returns: run_id.
    TR: OOF artefaktını (data/analysis/oof.parquet + oof_info.json) veritabanı parmak iziyle yazar.
        run_id tahminlerden ve veritabanından türetilir; aynı yeniden koşum aynı kimliği alır.
        Döndürür: run_id.
    """
    db = _db_fingerprint()
    h = hashlib.sha256(np.ascontiguousarray(frame["lgb"].values, dtype=np.float64).tobytes())
    h.update(db["sha256"].encode())
    run_id = h.hexdigest()[:12]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OOF_FILE, index=False)
    OOF_INFO.write_text(json.dumps({"run_id": run_id, "db": db, **info}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return run_id


def load_oof(listings):
    """
    EN: Reads the OOF artefact and checks it belongs to this database and these listings (same fingerprint,
        same row count, same prices in the same order); otherwise stops and says which script to run.
        Returns: (frame, info) — info has run_id and the per-fold tree counts.
    TR: OOF artefaktını okur; bu veritabanına ve bu ilanlara ait olduğunu sınar (aynı parmak izi, aynı satır
        sayısı, aynı sırada aynı fiyatlar); değilse durur ve hangi betiğin koşulacağını söyler.
        Döndürür: (tablo, bilgi) — bilgide run_id ve fold başına ağaç sayıları.
    """
    rerun = "run first | önce koşun: python analysis/07_model_comparison.py"
    if not (OOF_FILE.exists() and OOF_INFO.exists()):
        raise SystemExit(f"OOF artefact missing | OOF artefaktı yok — {rerun}")
    info = json.loads(OOF_INFO.read_text(encoding="utf-8"))
    frame = pd.read_parquet(OOF_FILE)
    if info["db"] != _db_fingerprint():
        raise SystemExit(f"OOF artefact is from another database | OOF başka veritabanından — {rerun}")
    if len(frame) != len(listings) or not np.array_equal(frame["price"].values, listings["price"].values.astype(float)):
        raise SystemExit(f"OOF rows do not match the listings | OOF satırları ilanlarla uyuşmuyor — {rerun}")
    return frame, info
