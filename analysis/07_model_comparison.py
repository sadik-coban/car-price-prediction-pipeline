"""
07_model_comparison.py
EN: Technical report §7 — how good the model is, against what. The headline LightGBM (in-fold TF-IDF + SVD
    text features) and two CatBoost variants (the same inputs; CatBoost's own text engine) are scored with
    the same 5-fold OOF, next to two median baselines on the same folds: series + year (a rough search) and
    model + year (what a dealer does). Also how many listings have comparables in their (model, year) cell.
    Writes the OOF artefact (data/analysis/oof.parquet) that 07_final_model, 08_* and shap/* read.
TR: Teknik rapor §7 — model ne kadar iyi, neye göre. Manşet LightGBM (fold içi TF-IDF + SVD metin
    öznitelikleri) ve iki CatBoost varyantı (aynı girdi; CatBoost'un kendi metin motoru) aynı 5-fold OOF ile
    ölçülür; yanında aynı fold'larda iki medyan tabanı: seri + yıl (kaba arama) ve model + yıl (galerinin
    yaptığı). Ayrıca kaç ilanın (model, yıl) hücresinde emsali olduğu. 07_final_model, 08_* ve shap/*'ın
    okuduğu OOF artefaktını (data/analysis/oof.parquet) yazar.
Output / Çıktı: metrics/07_model_comparison.json · data/analysis/oof.parquet · data/analysis/oof_info.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

from lib.common import CAT, FEATURES, TEXT, load_clean, save_metrics
from lib.cv import (CB_PARAMS, N_JOBS, PRICE_CAP, SEED, catboost_device, catboost_task, fold_matrices, lgb_oof,
                make_folds, price_metrics, round_metrics, save_oof)

RUNGS = ["model+yıl", "model", "global"]
MIN_COMPS = 5                 # a (model, year) cell with ≥5 listings has ≥4 comparables | ≥5 ilan = ≥4 emsal


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def catboost_svd_oof(X, yl, folds, task):
    """
    EN: 5-fold OOF CatBoost on the same fold matrices as LightGBM (TF-IDF + SVD text) — the fair comparison.
        Returns: OOF log predictions.
    TR: LightGBM ile aynı fold matrislerinde (TF-IDF + SVD metin) 5-fold OOF CatBoost — adil karşılaştırma.
        Döndürür: OOF log tahminleri.
    """
    pred = np.zeros(len(X))
    for tr, va in folds:
        Xtr, Xva, catc = fold_matrices(X, tr, va)
        ci = [Xtr.columns.get_loc(c) for c in catc]
        mdl = CatBoostRegressor(**CB_PARAMS, **task)
        mdl.fit(Pool(Xtr, yl[tr], cat_features=ci))
        pred[va] = mdl.predict(Pool(Xva, cat_features=ci))
    return pred


def catboost_native_oof(X, yl, folds, task):
    """
    EN: 5-fold OOF CatBoost with its own text engine on the raw model/series strings (no TF-IDF/SVD).
        Returns: OOF log predictions.
    TR: Ham model/seri metni üzerinde kendi metin motoruyla 5-fold OOF CatBoost (TF-IDF/SVD yok).
        Döndürür: OOF log tahminleri.
    """
    cat_idx, txt_idx = [X.columns.get_loc(c) for c in CAT], [X.columns.get_loc(c) for c in TEXT]
    pred = np.zeros(len(X))
    for tr, va in folds:
        mdl = CatBoostRegressor(**CB_PARAMS, **task)
        mdl.fit(Pool(X.iloc[tr], yl[tr], cat_features=cat_idx, text_features=txt_idx))
        pred[va] = mdl.predict(Pool(X.iloc[va], cat_features=cat_idx, text_features=txt_idx))
    return pred


def median_baseline(listings, keycol, folds):
    """
    EN: OOF median price of the listing's (key, model year) cell, from the training folds only. If the cell
        is not in the training part the ladder steps down: (key, all years) → overall median.
        Returns: (OOF predictions, the rung used per listing).
    TR: İlanın (anahtar, model yılı) hücresinin yalnız eğitim fold'larından OOF medyan fiyatı. Hücre eğitim
        kısmında yoksa merdiven iner: (anahtar, tüm yıllar) → genel medyan.
        Döndürür: (OOF tahminleri, ilan başına kullanılan basamak).
    """
    key = listings[keycol].fillna("missing").astype(str).values
    year = pd.to_numeric(listings["gb_year"], errors="coerce").values
    price = listings["price"].values.astype(float)
    pred, rung = np.zeros(len(listings)), np.empty(len(listings), dtype=object)
    for tr, va in folds:
        t = pd.DataFrame({"s": key[tr], "y": year[tr], "p": price[tr]})
        cell = t.dropna(subset=["y"]).groupby(["s", "y"])["p"].median().to_dict()
        by_key, overall = t.groupby("s")["p"].median().to_dict(), float(t["p"].median())
        for i in va:
            if not np.isnan(year[i]) and (key[i], year[i]) in cell:
                pred[i], rung[i] = cell[(key[i], year[i])], f"{keycol}+yıl"
            elif key[i] in by_key:
                pred[i], rung[i] = by_key[key[i]], keycol
            else:
                pred[i], rung[i] = overall, "global"
    return pred, rung


def rung_breakdown(price, pred, rung):
    """
    EN: Per ladder rung: listings, share, MAPE, MAE, R² (None for a rung with <2 listings or no spread).
        Returns: [[rung, n, %, MAPE, MAE, R2], ...].
    TR: Merdiven basamağı başına: ilan, pay, MAPE, MAE, R² (<2 ilanlı ya da yayılımsız basamakta None).
        Döndürür: [[basamak, n, %, MAPE, MAE, R2], ...].
    """
    out = []
    for r in RUNGS:
        m = rung == r
        if not m.any():
            continue
        mape = float(np.mean(np.abs((price[m] - pred[m]) / price[m])) * 100)
        mae = float(np.mean(np.abs(price[m] - pred[m])))
        r2 = r2_score(price[m], pred[m]) if (m.sum() > 1 and np.ptp(price[m]) > 0) else None
        out.append([r, int(m.sum()), round(100 * float(m.mean()), 2), round(mape, 2), round(mae, 0),
                    round(float(r2), 4) if r2 is not None else None])
    return out


def comparable_coverage(listings, rung):
    """
    EN: (model, model year) cell sizes counting the listing itself: singletons, thin (<5), comped (≥5), the
        number of cells, and how often the dealer baseline had to step down the ladder.
    TR: İlanın kendisi dahil (model, model yılı) hücre boyutları: tekil, ince (<5), emsalli (≥5), hücre sayısı
        ve galeri tabanının merdivenden ne sıklıkla indiği.
    """
    size = listings.groupby(["model", "gb_year"], dropna=False)["price"].transform("size")
    return {"singleton_pct": round(100 * float((size == 1).mean()), 2),
            "thin_pct": round(100 * float((size < MIN_COMPS).mean()), 2),
            "comped_pct": round(100 * float((size >= MIN_COMPS).mean()), 2),
            "groups": int(listings.groupby(["model", "gb_year"], dropna=False).ngroups),
            "cv_fallback_pct": round(100 * float((rung != "model+yıl").mean()), 2)}


def same_listing_comparison(price, baseline, rung, model_pred):
    """
    EN: Baseline and model MAE on the same listings: only those where the dealer baseline found the
        listing's own (model, year) cell in the training folds (the headline baseline MAE mixes in the
        ladder's fallback rungs).
    TR: Taban ve modelin MAE'si aynı ilanlarda: yalnız galeri tabanının ilanın kendi (model, yıl) hücresini
        eğitim fold'larında bulduğu ilanlar (manşetteki taban MAE'si merdivenin geri düşme basamaklarını da
        karıştırır).
    """
    m = rung == "model+yıl"
    base_mae = float(np.mean(np.abs(price[m] - baseline[m])))
    model_mae = float(np.mean(np.abs(price[m] - model_pred[m])))
    return {"n": int(m.sum()), "share": float(m.mean()), "base_mae": base_mae, "model_mae": model_mae}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.model_compare, dealer_coverage, model_yil_medyani, final_results,
        meta.repro) and the report inputs (error_drivers.taban_esit_kosul).
    TR: Site ağacında (domain.model_compare, dealer_coverage, model_yil_medyani, final_results, meta.repro)
        ve rapor girdilerinde (error_drivers.taban_esit_kosul) yayımlanır.
    """
    lgb_m, svd_m, nat_m = (round_metrics(res[k]) for k in ("lgb", "cb_svd", "cb_native"))
    naive_m, dealer_m = round_metrics(res["naive"]), round_metrics(res["dealer"])
    sl = res["same_listing"]
    return {
        "error_drivers": {"taban_esit_kosul": {
            "ilan": sl["n"], "pct": round(100 * sl["share"], 2), "taban_mae": round(sl["base_mae"], 0),
            "model_mae": round(sl["model_mae"], 0), "fark_tl": round(sl["base_mae"] - sl["model_mae"], 0),
            "iyilesme_pct": round(100 * (sl["base_mae"] - sl["model_mae"]) / sl["base_mae"], 1),
            "not": "Emsali olan ilanlar = taban merdiveninin model+yıl basamağı (fold içi)."}},
        "meta": {"repro": {"seed": SEED, "catboost_device": res["device"], "lgb_deterministic": True,
                           "row_order": "ORDER BY ad_id", "n_jobs": N_JOBS, "cv_agac": res["iters"]}},
        "domain": {
            "model_compare": {
                "lightgbm": lgb_m, "catboost": svd_m, "catboost_svd": svd_m, "catboost_native": nat_m,
                "not": ("CatBoost iki şekilde: TF-IDF-SVD beslemeli (LightGBM ile aynı girdi, adil "
                        "karşılaştırma) ve kendi native text motoru. Ana karşılaştırma TF-IDF-SVD üzerinden.")},
            "dealer_coverage": {
                **res["coverage"],
                "not": ("(model, gb_year) grup boyutları, ilanın kendisi dahil. comped = grubunda en az 5 ilan olanların "
                        "oranı (yani en az 4 emsal); thin = 5'ten az; singleton = grupta tek ilan (emsalsiz). Taban "
                        "bu ilanlarda tanımsız değil: fold içinde (model) medyanına, o da yoksa genel medyana iner.")},
            "model_yil_medyani": {
                "taban": dealer_m, "model": lgb_m,
                "kapsama": [[r[0], r[1], r[2]] for r in res["rungs"]], "metrik_kirilim": res["rungs"],
                "fallback_pct": res["coverage"]["cv_fallback_pct"],
                "merdiven": ["(model, yıl) medyanı", "(model) medyanı — tüm yıllar", "global medyan"],
                "not": ("Galerinin yaptığı iş: aynı model + aynı yıl, medyana bak. Fallback'lı OOF median tabanı — "
                        "medyanlar HER fold'da yalnız train'de (sızıntısız, modelle aynı 5-fold). Test'te o "
                        "(model, yıl) hücresi train'de yoksa merdiven iner: (model) tüm yıllar → global. "
                        "taban = tüm satırların OOF metrikleri; metrik_kirilim = [basamak, n, %, MAPE, MAE, R2] "
                        "(fallback basamaklarında hata belirgin artar — emsalsizde taban zaten zayıf). "
                        "Model metrikleri kıyas için; aynı protokol → fark gerçek kazanç.")},
            "final_results": {
                "model_karsilastirma": {
                    "naive": naive_m, "dealer": dealer_m,
                    "lightgbm_tfidf_svd": lgb_m, "catboost_tfidf_svd": svd_m, "catboost_native": nat_m,
                    "kazanan": "lightgbm" if res["lgb"]["MAPE"] < res["cb_svd"]["MAPE"] else "catboost",
                    "not": ("İKİ medyan tabanı + ÜÇ model varyantı, hepsi aynı 5-fold OOF ile. Tabanlar: naive "
                            "(seri+yıl medyanı = kaba arama), dealer (model+yıl medyanı — basamak-başına fallback "
                            "için bkz. domain.model_yil_medyani). Modeller: LightGBM (TF-IDF+SVD), CatBoost "
                            "(TF-IDF+SVD, LightGBM ile aynı girdi=adil), CatBoost (kendi native text motoru). "
                            "kazanan yalnız MODEL varyantları arasında seçilir. Metrikler sızıntısız OOF; "
                            "final modeller tüm veriyle eğitildi.")},
                "egitim": {"n_arac": res["n"], "n_feature": len(FEATURES), "hedef": "log1p(price)"},
                "not": "Metrikler 5-fold OOF (sızıntısız). Final modeller tüm veriyle eğitildi (production)."}},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
X = listings[FEATURES].copy()
price = listings["price"].values.astype(float)
yl = np.log1p(price)
folds = make_folds(len(X))
device = catboost_device()
headline = lgb_oof(X, yl, folds)
cb_svd = catboost_svd_oof(X, yl, folds, catboost_task(device))
cb_native = catboost_native_oof(X, yl, folds, catboost_task(device))
naive_pred, _ = median_baseline(listings, "series", folds)
dealer_pred, dealer_rung = median_baseline(listings, "model", folds)
oof_price = np.expm1(headline["pred_log"])
res = {"n": len(X), "device": device, "iters": headline["iters"],
       "lgb": price_metrics(price, oof_price),
       "cb_svd": price_metrics(price, np.expm1(cb_svd)), "cb_native": price_metrics(price, np.expm1(cb_native)),
       "naive": price_metrics(price, naive_pred), "dealer": price_metrics(price, dealer_pred),
       "rungs": rung_breakdown(price, dealer_pred, dealer_rung), "coverage": comparable_coverage(listings, dealer_rung),
       "same_listing": same_listing_comparison(price, dealer_pred, dealer_rung, np.clip(oof_price, 0, PRICE_CAP))}
fold_of = np.zeros(len(X), dtype=np.int8)
for k, (_tr, va) in enumerate(folds, 1):
    fold_of[va] = k
oof = pd.DataFrame({"price": price, "lgb": oof_price, "lgb_log": headline["pred_log"], "cb_svd_log": cb_svd,
                    "cb_native_log": cb_native, "fold": fold_of})
print({k: round(res[k]["MAPE"], 2) for k in ("lgb", "cb_svd", "cb_native", "naive", "dealer")}, "trees | ağaç", res["iters"])

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
run_id = save_oof(oof, {"cv_iters": headline["iters"], "splits": headline["splits"]})
print("written | yazıldı:", save_metrics("07_model_comparison", to_metrics(res), run_id=run_id), "· run_id", run_id)
