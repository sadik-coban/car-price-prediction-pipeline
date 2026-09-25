"""
07_final_model.py
EN: Technical report §7 — the served models. LightGBM is trained on all listings with the median of the
    tree counts early stopping chose in the 5-fold CV (so the served model has the measured setting), next to
    the two CatBoost variants. Writes the serving files (native model formats + encoders.pkl + README) and
    predicts three sample listings (the best-predicted undamaged listing in each price third).
    Needs 07_model_comparison's OOF artefact (tree counts, OOF errors, conformal q).
TR: Teknik rapor §7 — servis edilen modeller. LightGBM bütün ilanlarla, 5-fold CV'de erken durdurmanın
    seçtiği ağaç sayılarının medyanıyla eğitilir (servis edilen model ölçülen ayarla aynı olsun diye); yanında
    iki CatBoost varyantı. Servis dosyalarını (yerel model biçimleri + encoders.pkl + README) yazar ve üç örnek
    ilanı tahmin eder (her fiyat üçte birinde en iyi tahmin edilen hasarsız ilan).
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar (ağaç sayıları, OOF hataları, conformal q).
Output / Çıktı: metrics/07_final_model.json · data/serving/final_model.pkl · data/serving/serve/*
"""

# %% [1] Setup | Kurulum
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from lib import segment_rule as SR
from lib.common import CAT, FEATURES, NUM, ROOT, TEXT, load_clean, save_metrics
from lib.cv import (CB_PARAMS, LGB_PARAMS, N_JOBS, PRICE_CAP, SEED, TEXT_SVD, catboost_device, catboost_task,
                conformal_q, load_oof)

SERVING_DIR = ROOT / "data" / "serving"
SERVE_DIR = SERVING_DIR / "serve"
TIER_NAMES = ["economy", "mid", "premium"]
MAX_ABS_RESID = 5                     # a sample must be predicted within ±5% OOF | örnek OOF'ta ±%5 içinde olmalı
CONFORMAL_DEF = ("OOF log-hatalarinin %90 yuzdeligi; aralik = expm1(tahmin_log ± q), "
                 "alt uc 0, ust uc log1p(1.5e7) ile kirpilir")


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def full_matrix(listings):
    """
    EN: The training matrix on all listings, built like a CV fold but fitted on everything: category codes
        (first-seen order), numeric columns, TF-IDF + SVD of model/series.
        Returns: (matrix, {"tfidf": {text: {vec, svd, n}}, "cat_maps": {column: {value: code}}}).
    TR: Bütün ilanlarla eğitim matrisi; bir CV fold'u gibi kurulur ama hepsine uydurulur: kategori kodları
        (ilk görülme sırası), sayısal kolonlar, model/seri'nin TF-IDF + SVD'si.
        Döndürür: (matris, {"tfidf": {metin: {vec, svd, n}}, "cat_maps": {kolon: {değer: kod}}}).
    """
    cols, cat_maps, tfidf = {}, {}, {}
    for c in CAT:
        cat_maps[c] = {v: i for i, v in enumerate(listings[c].unique())}
        cols[c] = listings[c].map(cat_maps[c]).values
    for c in NUM:
        cols[c] = listings[c].values
    for txt, size in TEXT_SVD:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5)
        m = vec.fit_transform(listings[txt])
        n = min(size, m.shape[1] - 1)
        svd = TruncatedSVD(n, random_state=SEED)
        emb = svd.fit_transform(m)
        tfidf[txt] = {"vec": vec, "svd": svd, "n": n}
        cols.update({f"{txt}_{i}": emb[:, i] for i in range(n)})
    return pd.DataFrame(cols), {"tfidf": tfidf, "cat_maps": cat_maps}


def train_models(X, Xf, yl, trees, task):
    """
    EN: The three served models on all listings: LightGBM (trees = CV median, no early stopping), CatBoost
        on the same matrix, CatBoost with its own text engine on the raw columns.
        Returns: {"lgb", "cb", "cb_native", "cat_idx"} (cat_idx = categorical positions in the matrix).
    TR: Bütün ilanlarla servis edilen üç model: LightGBM (ağaç = CV medyanı, erken durdurma yok), aynı
        matriste CatBoost, ham kolonlarda kendi metin motoruyla CatBoost.
        Döndürür: {"lgb", "cb", "cb_native", "cat_idx"} (cat_idx = matristeki kategorik konumlar).
    """
    final_lgb = lgb.LGBMRegressor(n_jobs=N_JOBS, **{**LGB_PARAMS, "n_estimators": trees})
    final_lgb.fit(Xf, yl, categorical_feature=CAT)
    cat_idx = [list(Xf.columns).index(c) for c in CAT]
    final_cb = CatBoostRegressor(**CB_PARAMS, **task)
    final_cb.fit(Pool(Xf, yl, cat_features=cat_idx))
    native = CatBoostRegressor(**CB_PARAMS, **task)
    native.fit(Pool(X, yl, cat_features=[X.columns.get_loc(c) for c in CAT],
                    text_features=[X.columns.get_loc(c) for c in TEXT]))
    return {"lgb": final_lgb, "cb": final_cb, "cb_native": native, "cat_idx": cat_idx}


def predict_one(listings, X, idx, models, enc, which):
    """
    EN: Price prediction of one listing by one served model, built the way serving builds it (unseen
        category → −1, text through the stored TF-IDF + SVD), clipped to [0, 15M].
        which: "lgb", "cb" or "cb_native".
    TR: Bir ilanın bir servis modeliyle fiyat tahmini; servisin kurduğu gibi kurulur (görülmemiş kategori → −1,
        metin saklanan TF-IDF + SVD'den), [0, 15M]'ye kırpılır. which: "lgb", "cb" ya da "cb_native".
    """
    if which == "cb_native":
        pool = Pool(X.iloc[[idx]], cat_features=[X.columns.get_loc(c) for c in CAT],
                    text_features=[X.columns.get_loc(c) for c in TEXT])
        return float(np.expm1(np.clip(models["cb_native"].predict(pool)[0], 0, np.log1p(PRICE_CAP))))
    rec = listings.iloc[idx]
    row = {c: enc["cat_maps"][c].get(str(rec[c]), -1) for c in CAT}
    row.update({c: float(rec[c]) for c in NUM})
    for txt in TEXT:
        t = enc["tfidf"][txt]
        emb = t["svd"].transform(t["vec"].transform([str(rec[txt])]))
        row.update({f"{txt}_{i}": emb[0, i] for i in range(t["n"])})
    Xr = pd.DataFrame([row])[enc["feat_cols"]]
    pl = models["lgb"].predict(Xr)[0] if which == "lgb" else models["cb"].predict(Pool(Xr, cat_features=models["cat_idx"]))[0]
    return float(np.expm1(np.clip(pl, 0, np.log1p(PRICE_CAP))))


def pick_examples(price, oof_price, heavy):
    """
    EN: One sample per price third: among undamaged listings predicted within ±5% OOF, the one with the
        smallest |OOF residual %| — best cases, which the report presents as such (typical error is MAPE).
        Returns: [(row index, tier name, OOF residual %), ...].
    TR: Her fiyat üçte biri için bir örnek: OOF'ta ±%5 içinde tahmin edilen hasarsız ilanlar arasında
        |OOF artık %|'si en küçük olan — en iyi durumlar; rapor onları öyle sunar (tipik hata MAPE'dir).
        Döndürür: [(satır indeksi, dilim adı, OOF artık %), ...].
    """
    resid_pct = (price - oof_price) / price * 100
    abs_r = np.abs(resid_pct)
    q = np.quantile(price, [1 / 3, 2 / 3])
    out = []
    for lo, hi, name in zip([-np.inf, q[0], q[1]], [q[0], q[1], np.inf], TIER_NAMES):
        c = np.where((price > lo) & (price <= hi) & (~heavy) & (abs_r <= MAX_ABS_RESID))[0]
        if len(c):
            i = int(c[np.argmin(abs_r[c])])
            out.append((i, name, float(resid_pct[i])))
    return out[:3]


def serving_readme(trees):
    """
    EN: The serving README text (which file is what, how to build the input exactly like training).
    TR: Servis README metni (hangi dosya ne, girdi eğitimle birebir aynı nasıl kurulur).
    """
    return f"""# SERVING DOSYALARI

## Modeller (native format)
- lightgbm_tfidf_svd.txt : LightGBM, {trees} ağaç (CV'nin erken durdurmayla seçtiği ağaç sayılarının
  medyanı; raporlanan metrikler bu ayarın 5-fold OOF'undan). Yükle: lgb.Booster(model_file=...)
- catboost_tfidf_svd.cbm : CatBoost, TF-IDF+SVD girdi, 500 iterasyon. CatBoostRegressor().load_model(...)
- catboost_native.cbm    : CatBoost, ham model/seri metni (TF-IDF/SVD YOK), 500 iterasyon.

## encoders.pkl (önişleme — model değil)
tfidf (model/seri için vectorizer + SVD + boyut) · cat_maps (kategori → tamsayı kod) · feat_cols (sütun
sırası) · CAT / NUM / TEXT · cat_idx · SEGMENT_MAP / PERF_BASE / MODEL_SEG / PERF_RE (segment kuralı) ·
FINAL_LGB_AGAC · CONFORMAL_Q + CONFORMAL_TANIM (%90 aralık).

## Girdi nasıl kurulur (eğitimle BİREBİR aynı olmalı)
Kaynak: cars.duckdb `car_listings`, eğitimde TR plakalı ilanlar.
- vehicle_age = ilan yılı − gb_year (eğitimde tarama yılı; alt sınır 0).
- gb_mileage = gb_mileage.
- power_hp_val = ortalama(power_hp_low, power_hp_up), eksik uç atlanır.
- engine_cc_val = **engine_cc_up** (üst sınır). DB'deki `engine_cc_val` kolonu DEĞİL: o aralığın orta
  noktası. Üst sınırın seçilme gerekçesi teknik raporun §1'inde.
- is_heavy_damaged = is_heavy_damaged, boşsa 0.
- roof_state / hood_state / trunk_state (tavan / kaput / bagaj) = değişen → 'changed', yoksa boyalı →
  'painted', yoksa lokal → 'local', yoksa 'original' (DB kolonları tavan_*, kaput_*, bagaj_*).
- door_{{changed,painted,local}} = 4 kapının ilgili bayraklarının toplamı; fender_* = 4 çamurluk;
  bumper_* = 2 tampon (DB kolonları door_fl_degisen, …, bumper_rear_lokal).
- segment = segment_of(series, model): önce MODEL_SEG[series] içinde model adı öneki, sonra
  SEGMENT_MAP[series], sonra PERF_BASE[series] + PERF_RE (ör. M3 → '3 Serisi' → D). Çözülemezse
  eğitim DURUR; tüketici de hata vermeli (varsayılan segment YOK). DB'deki gb_segment kullanılmaz.
- CAT ve TEXT kolonlarında eksik → 'missing'; CAT kodları cat_maps'ten, sözlükte olmayan değer → -1.
- TEXT (model, series): tfidf[txt]['vec'].transform → tfidf[txt]['svd'].transform → {{txt}}_0 … {{txt}}_(n−1),
  n = tfidf[txt]['n'].
- Sütunları feat_cols sırasına diz.

## Çıktı
- tahmin_log = model.predict(X) ; fiyat = expm1(clip(tahmin_log, 0, log1p(1.5e7))).
- %90 aralık = expm1(clip(tahmin_log ± CONFORMAL_Q)) (bkz. CONFORMAL_TANIM). Kapsama tüm veride %90, en
  ucuz çeyrekte daha düşük (teknik rapor §8).

## Native-text CatBoost
Ham model/series string'i doğrudan ver (TF-IDF/SVD yok): X = df[TEXT+CAT+NUM];
Pool(X, cat_features=[X.columns.get_loc(c) for c in CAT], text_features=[X.columns.get_loc(c) for c in TEXT]).
encoders.pkl'deki cat_idx bu model için DEĞİL: o, feat_cols sırasındaki TF-IDF/SVD modellerinin indeksi.

## Neden bu formatlar
Modeller native (.txt/.cbm) — sürüm-dayanıklı. TF-IDF/SVD'nin native formatı yok → pickle.
"""


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree: meta.repro.final_lgb_trees and domain.final_results sample predictions.
    TR: Site ağacında yayımlanır: meta.repro.final_lgb_trees ve domain.final_results örnek tahminleri.
    """
    samples = [{"vehicle": s["model"], "segment": s["segment"], "price_band": s["tier"], "age": s["age"], "km": s["km"],
                "actual": round(s["actual"], 0), "lightgbm_pred": round(s["lgb"], 0),
                "catboost_pred": round(s["cb"], 0), "catboost_native_pred": round(s["cb_native"], 0),
                "lgb_dev_pct": round(abs(s["lgb"] - s["actual"]) / s["actual"] * 100, 1),
                "oof_resid_pct": round(s["oof_resid"], 1)} for s in res["samples"]]
    return {"meta": {"repro": {"final_lgb_trees": res["trees"]}},
            "domain": {"final_results": {"example_predictions": samples,
                                         "example_prediction": samples[0] if samples else None}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
X = listings[FEATURES].copy()
price = listings["price"].values.astype(float)
yl = np.log1p(price)
trees = int(np.median(oof_info["cv_iters"]))
q = conformal_q(yl, oof["lgb"].values)
Xf, enc = full_matrix(listings)
enc["feat_cols"] = list(Xf.columns)
models = train_models(X, Xf, yl, trees, catboost_task(catboost_device()))
heavy = pd.to_numeric(listings["is_heavy_damaged"], errors="coerce").fillna(0).values.astype(bool)
samples = []
for i, tier, resid in pick_examples(price, np.clip(oof["lgb"].values, 0, PRICE_CAP), heavy):
    rec = listings.iloc[i]
    samples.append({"model": str(rec["model"]), "segment": str(rec["segment"]), "tier": tier, "age": int(rec["vehicle_age"]),
                    "km": int(rec["gb_mileage"]), "actual": float(price[i]), "oof_resid": resid,
                    **{w: predict_one(listings, X, i, models, enc, w) for w in ("lgb", "cb", "cb_native")}})
res = {"trees": trees, "samples": samples}
print(f"trees | ağaç {trees} · q {q:.4f} ·", [(s["tier"], round(s["lgb"])) for s in samples])

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
SERVE_DIR.mkdir(parents=True, exist_ok=True)
with open(SERVING_DIR / "final_model.pkl", "wb") as fh:
    pickle.dump({"final_lgb": models["lgb"], "final_cb": models["cb"], "tfidf": enc["tfidf"], "cat_maps": enc["cat_maps"],
                 "feat_cols": enc["feat_cols"], "CAT": CAT, "NUM": NUM, "TEXT": TEXT, "cat_idx": models["cat_idx"]}, fh)
models["lgb"].booster_.save_model(str(SERVE_DIR / "lightgbm_tfidf_svd.txt"))
models["cb"].save_model(str(SERVE_DIR / "catboost_tfidf_svd.cbm"))
models["cb_native"].save_model(str(SERVE_DIR / "catboost_native.cbm"))
with open(SERVE_DIR / "encoders.pkl", "wb") as fh:
    pickle.dump({"tfidf": enc["tfidf"], "cat_maps": enc["cat_maps"], "feat_cols": enc["feat_cols"],
                 "CAT": CAT, "NUM": NUM, "TEXT": TEXT, "cat_idx": models["cat_idx"],
                 "SEGMENT_MAP": SR.SEGMENT_MAP, "PERF_BASE": SR.PERF_BASE, "MODEL_SEG": SR.MODEL_SEG,
                 "PERF_RE": SR.PERF_RE.pattern, "FINAL_LGB_AGAC": trees, "CONFORMAL_Q": q,
                 "CONFORMAL_TANIM": CONFORMAL_DEF}, fh)
(SERVE_DIR / "README.md").write_text(serving_readme(trees), encoding="utf-8")
print("written | yazıldı:", save_metrics("07_final_model", to_metrics(res), run_id=oof_info["run_id"]))
