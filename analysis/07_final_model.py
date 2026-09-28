"""
07_final_model.py
EN: Technical report §7 — the served models. LightGBM is trained on all listings with the median of the
    tree counts early stopping chose in the 5-fold CV (so the served model has the measured setting), next to
    the two CatBoost variants. Writes the serving files (native model formats + encoders.pkl + README).
    2026-09-29 (second simplification list): the three best-case sample predictions left §7, so they are not
    computed. Needs 07_model_comparison's OOF artefact (tree counts, OOF errors, conformal q).
TR: Teknik rapor §7 — servis edilen modeller. LightGBM bütün ilanlarla, 5-fold CV'de erken durdurmanın
    seçtiği ağaç sayılarının medyanıyla eğitilir (servis edilen model ölçülen ayarla aynı olsun diye); yanında
    iki CatBoost varyantı. Servis dosyalarını (yerel model biçimleri + encoders.pkl + README) yazar.
    2026-09-29 (ikinci sadeleştirme listesi): üç "en iyi durum" örnek tahmini §7'den çıktı, hesaplanmıyor.
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
from lib.cv import (CB_PARAMS, LGB_PARAMS, N_JOBS, SEED, TEXT_SVD, catboost_device, catboost_task,
                conformal_q, load_oof)

SERVING_DIR = ROOT / "data" / "serving"
SERVE_DIR = SERVING_DIR / "serve"
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
    EN: Published in the site tree: meta.repro.final_lgb_trees (technical report §7 names the served model's rule).
    TR: Site ağacında yayımlanır: meta.repro.final_lgb_trees (teknik rapor §7 servis edilen modelin kuralını anar).
    """
    return {"meta": {"repro": {"final_lgb_trees": res["trees"]}}}


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
res = {"trees": trees}
print(f"trees | ağaç {trees} · q {q:.4f}")

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
