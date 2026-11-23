"""
07_final_model.py
EN: Technical report §7 — the served models. LightGBM is trained on all listings with the median of the
    tree counts early stopping chose in the 5-fold CV (so the served model has the measured setting), next to
    the two CatBoost variants. Writes the serving files (native model formats + encoders.pkl + README).
    2026-09-29 (second simplification list): the three best-case sample predictions left §7, so they are not
    computed. Needs 07_model_comparison's OOF artefact (tree counts, OOF errors, conformal q).
    2026-11-23 (docs/decisions.md): the served interval is per car. encoders.pkl also carries the scale model s
    (lib.conformal, trained on every listing's OOF error with the main model's cat_maps codes), k and the sorted
    nested scores (hit probability), the comparables counts per (model, age) and the last snapshot date. The main
    model file must not change by this. serve_predict() is the serving recipe run from the bundle alone; the script
    stops if it disagrees with lib.conformal, and writes worked examples for the API's parity check.
TR: Teknik rapor §7 — servis edilen modeller. LightGBM bütün ilanlarla, 5-fold CV'de erken durdurmanın
    seçtiği ağaç sayılarının medyanıyla eğitilir (servis edilen model ölçülen ayarla aynı olsun diye); yanında
    iki CatBoost varyantı. Servis dosyalarını (yerel model biçimleri + encoders.pkl + README) yazar.
    2026-09-29 (ikinci sadeleştirme listesi): üç "en iyi durum" örnek tahmini §7'den çıktı, hesaplanmıyor.
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar (ağaç sayıları, OOF hataları, conformal q).
    2026-11-23 (docs/decisions.md): servis edilen aralık araca göre. encoders.pkl ayrıca ölçek modeli s'yi
    (lib.conformal; her ilanın OOF hatasıyla, ana modelin cat_maps kodlarıyla eğitilir), k'yı ve sıralı iç içe
    skorları (isabet olasılığı), (model, yaş) başına emsal sayılarını ve son tarama tarihini taşır. Ana model
    dosyası bununla değişmemeli. serve_predict() servis tarifinin yalnız paketten koşan hali; lib.conformal ile
    uyuşmazsa betik durur. API'nin parite denetimi için işlenmiş örnekler yazar.
Output / Çıktı: metrics/07_final_model.json · data/serving/final_model.pkl · data/serving/serve/*
"""

# %% [1] Setup | Kurulum
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from lib import conformal as CF
from lib import segment_rule as SR
from lib.common import CAT, FEATURES, NUM, ROOT, TEXT, generated_stamp, load_clean, save_metrics
from lib.cv import (CB_PARAMS, LGB_PARAMS, N_JOBS, PRICE_CAP, SEED, TEXT_SVD, catboost_device, catboost_task,
                conformal_q, load_oof)

SERVING_DIR = ROOT / "data" / "serving"
SERVE_DIR = SERVING_DIR / "serve"
CONFORMAL_DEF = ("Araca göre %90 aralık: s = max(CONFORMAL_SCALE_MODEL(kodlu girdiler, tahmin_log, emsal), "
                 "CONFORMAL_S_FLOOR); aralık = expm1(clip(tahmin_log ± CONFORMAL_K·s)), alt uç 0, üst uç "
                 "log1p(1.5e7). İsabet olasılığı = CONFORMAL_SCORES içinde log(1 + CONFIDENCE_WITHIN) / s'nin "
                 "altında ya da ona eşit olanların payı. CONFORMAL_Q eski tarifin tek payıdır (OOF log "
                 "hatalarının %90 yüzdeliği; aralık = expm1(tahmin_log ± q)); yalnız eski backend okur.")
MANY_COMPARABLES = 100          # a worked example needs a cell this full | işlenmiş örnek bu kadar dolu bir hücre ister


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


def comparable_counts(listings):
    """
    EN: Comparables as served: listings per (model, age) cell. The analysis counts them per (model, year); the two
        are the same cells while age = snapshot year − model year for every listing, and this stops if they are not.
        Returns: {(model, age): listings}.
    TR: Servisteki haliyle emsal: (model, yaş) hücresi başına ilan. Analiz onları (model, yıl) başına sayar; her
        ilanda yaş = tarama yılı − model yılı olduğu sürece ikisi aynı hücrelerdir, değilse burada durur.
        Döndürür: {(model, yaş): ilan sayısı}.
    """
    by_age = listings.groupby(["model", "vehicle_age"]).size()
    n_year = len(listings.groupby(["model", "gb_year"]).size())
    n_both = len(listings[["model", "gb_year", "vehicle_age"]].drop_duplicates())
    if not len(by_age) == n_year == n_both:
        raise ValueError(f"(model, age) cells {len(by_age)} ≠ (model, year) cells {n_year} | hücreler farklı")
    return {(str(m), int(a)): int(n) for (m, a), n in by_age.items()}


def interval_bundle(scale_model, scale_cols, k, scores, counts, data_until):
    """
    EN: The served interval's part of encoders.pkl: the scale model as LightGBM text, its column order, k, the floor,
        the sorted nested scores, the hit tolerance, the comparables counts and the last snapshot date.
    TR: Servis edilen aralığın encoders.pkl parçası: LightGBM metni olarak ölçek modeli, sütun sırası, k, alt sınır,
        sıralı iç içe skorlar, isabet toleransı, emsal sayıları ve son tarama tarihi.
    """
    return {"CONFORMAL_SCALE_MODEL": scale_model.booster_.model_to_string(),
            "CONFORMAL_SCALE_COLS": {"columns": list(scale_cols), "categorical": list(CAT)},
            "CONFORMAL_K": float(k), "CONFORMAL_S_FLOOR": CF.S_FLOOR,
            "CONFORMAL_SCORES": np.sort(np.asarray(scores, dtype=float)), "CONFIDENCE_WITHIN": CF.WITHIN,
            "COMPARABLE_COUNTS": counts, "DATA_UNTIL": data_until}


def serve_scale(bundle, coded, log_pred, comparables):
    """
    EN: s, half-width and hit probability from the bundle alone, for rows already coded like the main model's
        matrix (CAT codes + NUM). coded: DataFrame; log_pred, comparables: one value per row.
        Returns: {"s", "r", "p"} arrays.
    TR: Yalnız paketten s, yarı genişlik ve isabet olasılığı; ana modelin matrisi gibi kodlanmış satırlar için (CAT
        kodları + NUM). coded: DataFrame; log_pred, comparables: satır başına bir değer.
        Döndürür: {"s", "r", "p"} dizileri.
    """
    X = coded.reset_index(drop=True).copy()
    X[CF.LOG_PRED], X[CF.COMPARABLES] = np.asarray(log_pred, dtype=float), np.asarray(comparables)
    booster = lgb.Booster(model_str=bundle["CONFORMAL_SCALE_MODEL"])
    s = np.clip(booster.predict(X[bundle["CONFORMAL_SCALE_COLS"]["columns"]]), bundle["CONFORMAL_S_FLOOR"], None)
    cut = np.log1p(bundle["CONFIDENCE_WITHIN"]) / s
    return {"s": s, "r": bundle["CONFORMAL_K"] * s,
            "p": np.searchsorted(bundle["CONFORMAL_SCORES"], cut, side="right") / len(bundle["CONFORMAL_SCORES"])}


def serve_predict(bundle, booster, car):
    """
    EN: The serving recipe on one car, from the bundle and the main model's booster only (what the API does, see the
        serving README): CAT codes from cat_maps (unseen → −1), NUM as given, model/series through TF-IDF + SVD,
        columns in feat_cols order → log prediction; comparables looked up by (model, age); then serve_scale.
        car: {feature: value} for FEATURES. Returns: {"price", "log_pred", "comparables", "s", "price_min",
        "price_max", "lower_percent", "upper_percent", "hit_probability", "shown"}.
    TR: Tek araçta servis tarifi, yalnız paketten ve ana modelin booster'ından (API'nin yaptığı; servis README'sine
        bakın): CAT kodları cat_maps'ten (görülmemiş → −1), NUM verildiği gibi, model/seri TF-IDF + SVD'den, sütunlar
        feat_cols sırasında → log tahmin; emsal (model, yaş) ile aranır; sonra serve_scale.
        car: FEATURES için {öznitelik: değer}. Döndürür: {"price", "log_pred", "comparables", "s", "price_min",
        "price_max", "lower_percent", "upper_percent", "hit_probability", "shown"}.
    """
    row = {c: bundle["cat_maps"][c].get(car[c], -1) for c in bundle["CAT"]}
    row.update({c: float(car[c]) for c in bundle["NUM"]})
    for txt in bundle["TEXT"]:
        e = bundle["tfidf"][txt]
        emb = e["svd"].transform(e["vec"].transform([car[txt]]))[0]
        row.update({f"{txt}_{i}": float(emb[i]) for i in range(e["n"])})
    frame = pd.DataFrame([row])
    log_pred = float(np.clip(booster.predict(frame[bundle["feat_cols"]])[0], 0, np.log1p(PRICE_CAP)))
    comparables = bundle["COMPARABLE_COUNTS"].get((str(car["model"]).strip(), int(car["vehicle_age"])), 0)
    sc = serve_scale(bundle, frame[bundle["CAT"] + bundle["NUM"]], [log_pred], [comparables])
    s, r, p = float(sc["s"][0]), float(sc["r"][0]), float(sc["p"][0])
    lo, hi = CF.interval(np.expm1(log_pred), r)
    return {"price": int(round(float(np.expm1(log_pred)))), "log_pred": log_pred, "comparables": int(comparables),
            "s": s, "price_min": int(round(float(lo))), "price_max": int(round(float(hi))),
            "lower_percent": round(float(np.expm1(-r)) * 100, 1), "upper_percent": round(float(np.expm1(r)) * 100, 1),
            "hit_probability": p, "shown": int(min(np.floor(p * 10 + 1e-9), CF.SHOWN_MAX))}


def parity_examples(listings, log_all, bundle, booster):
    """
    EN: Worked examples for the API's parity check, each a full input with the served output (serve_predict):
        many_comparables = the dearest predicted listing among cells with at least MANY_COMPARABLES listings;
        no_comparable = the same car at the nearest older age its model never has; model_typo = the same car with the
        last character of the model name dropped; unseen_category = the same car with a fuel value the training data
        never had (its own body type), which codes to −1. The last three are made-up inputs, not listings.
        Returns: [{"id", "input": {feature: value}, "output": {...}}, ...].
    TR: API'nin parite denetimi için işlenmiş örnekler; her biri tam bir girdi ve servis çıktısı (serve_predict):
        many_comparables = en az MANY_COMPARABLES ilanlı hücreler içinde tahmini en yüksek ilan; no_comparable = aynı
        araç, modelinin hiç görülmediği en yakın daha büyük yaşta; model_typo = aynı araç, model adının son karakteri
        düşmüş; unseen_category = aynı araç, eğitim verisinde hiç olmayan bir yakıt değeriyle (kendi kasa tipi), kodu
        −1 olur. Son üçü uydurma girdidir, ilan değil.
        Döndürür: [{"id", "input": {öznitelik: değer}, "output": {...}}, ...].
    """
    counts = bundle["COMPARABLE_COUNTS"]
    cell_n = np.array([counts[(m, int(a))] for m, a in zip(listings["model"], listings["vehicle_age"])])
    full = np.flatnonzero(cell_n >= MANY_COMPARABLES)
    base = {c: (v.item() if hasattr(v, "item") else v)
            for c, v in listings.iloc[full[np.argmax(log_all[full])]][FEATURES].items()}
    ages = {a for m, a in counts if m == base["model"]}
    older = next(a for a in range(int(base["vehicle_age"]) + 1, int(base["vehicle_age"]) + 100) if a not in ages)
    cars = {"many_comparables": base, "no_comparable": {**base, "vehicle_age": older},
            "model_typo": {**base, "model": base["model"][:-1]},
            "unseen_category": {**base, "kb_fuel": base["kb_body_type"]}}
    assert base["kb_body_type"] not in bundle["cat_maps"]["kb_fuel"], "fuel example is a seen value | örnek görülmüş"
    out = [{"id": name, "input": car, "output": serve_predict(bundle, booster, car)} for name, car in cars.items()]
    assert [e["output"]["comparables"] > 0 for e in out] == [True, False, False, True], "examples changed | örnekler"
    return out


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
FINAL_LGB_AGAC · CONFORMAL_TANIM (aralığın tarifi) · CONFORMAL_SCALE_MODEL (ölçek modeli s, LightGBM metni:
lgb.Booster(model_str=...)) · CONFORMAL_SCALE_COLS (s'nin sütun sırası ve kategorik sütunları) · CONFORMAL_K ·
CONFORMAL_S_FLOOR · CONFORMAL_SCORES (sıralı skorlar, isabet olasılığı için) · CONFIDENCE_WITHIN (0.10) ·
COMPARABLE_COUNTS ({{(model, yaş): ilan sayısı}}) · DATA_UNTIL (verideki son tarama) · CONFORMAL_Q (eski tek pay;
yalnız eski backend okur).

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
- emsal = COMPARABLE_COUNTS.get((model.strip(), vehicle_age), 0): eğitimde aynı model adı ve yaştaki ilan sayısı.
  Model dizesi eğitimdekiyle BİREBİR aranır; yazım farkı emsali 0 yapar ve aralığı genişletir.

## Çıktı
- tahmin_log = clip(model.predict(X), 0, log1p(1.5e7)) ; fiyat = expm1(tahmin_log).
- s girdisi: CONFORMAL_SCALE_COLS["columns"] sırasında CAT kodları + NUM (yukarıdaki satırın aynısı, metin sütunları
  hariç) + log_pred = tahmin_log + comparables = emsal.
- s = max(ölçek_modeli.predict(...), CONFORMAL_S_FLOOR) ; r = CONFORMAL_K · s.
- %90 aralık = expm1(clip(tahmin_log ± r)), alt uç 0, üst uç log1p(1.5e7). Kapsama yaklaşık %90 ve eğitim verisinin
  dönemi (DATA_UNTIL'e kadar) için; sonradan gelen ilanlarda birkaç puan düşük (teknik rapor §8).
- isabet olasılığı = searchsorted(CONFORMAL_SCORES, log(1 + CONFIDENCE_WITHIN) / s, side="right") / len(CONFORMAL_SCORES):
  ilan fiyatının tahminin ±%10 yakınında olma olasılığı. Sitede "10'da N": N = min(floor(10·p), {CF.SHOWN_MAX}).
- parity_examples.json: dört işlenmiş örnek (girdi + bu tarifin çıktısı). API aynı girdiyle aynı çıktıyı vermeli.

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
        report.served_interval: what the bundle serves for the interval (k, floor, tolerance, the size of the
        comparables table) and the worked examples' outputs.
    TR: Site ağacında yayımlanır: meta.repro.final_lgb_trees (teknik rapor §7 servis edilen modelin kuralını anar).
        report.served_interval: paketin aralık için servis ettiği şey (k, alt sınır, tolerans, emsal tablosunun
        boyutu) ve işlenmiş örneklerin çıktıları.
    """
    return {"meta": {"repro": {"final_lgb_trees": res["trees"]}},
            "report": {"served_interval": res["served_interval"]}}


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

# EN: served interval — s on every listing's OOF error (comparables without the own fold), k from the nested scores
# TR: servis edilen aralık — s her ilanın OOF hatasıyla (emsal kendi katı sayılmadan), k iç içe skorlardan
fold = oof["fold"].values
log_oof = np.log1p(np.clip(oof["lgb"].values, 0, PRICE_CAP))
abs_err = np.abs(yl - log_oof)
other = CF.other_fold_counts(listings["model"].astype(str) + "|" + listings["gb_year"].astype(str), fold)
scores, _ = CF.nested_scores(X.reset_index(drop=True), abs_err, log_oof, other, fold)
scale_X = CF.scale_inputs(Xf[CAT + NUM], log_oof, other)
scale_model = CF.fit_scale(scale_X, abs_err, CAT)
counts = comparable_counts(listings)
bundle = {"tfidf": enc["tfidf"], "cat_maps": enc["cat_maps"], "feat_cols": enc["feat_cols"], "CAT": CAT, "NUM": NUM,
          "TEXT": TEXT, **interval_bundle(scale_model, scale_X.columns, CF.conformal_quantile(scores), scores, counts,
                                          str(listings["snap"].max()))}
# EN: the bundle's recipe must give lib.conformal's s on every listing | TR: paketin tarifi her ilanda lib.conformal'ın s'sini vermeli
served = serve_scale(bundle, Xf[CAT + NUM], log_oof, other)
if not np.allclose(served["s"], CF.predict_scale(scale_model, scale_X), rtol=1e-12, atol=0):
    raise ValueError("the served s differs from lib.conformal | servis edilen s lib.conformal'dan farklı")
log_all = np.clip(models["lgb"].predict(Xf), 0, np.log1p(PRICE_CAP))
examples = parity_examples(listings, log_all, bundle, models["lgb"].booster_)
res = {"trees": trees,
       "served_interval": {
           "k": round(bundle["CONFORMAL_K"], 4), "s_floor": CF.S_FLOOR, "within_pct": round(CF.WITHIN * 100),
           "n_scores": int(len(scores)), "comparable_cells": len(counts), "comparable_total": int(sum(counts.values())),
           "data_until": bundle["DATA_UNTIL"],
           "examples": [{"id": e["id"], "price": e["output"]["price"], "comparables": e["output"]["comparables"],
                         "lower_percent": e["output"]["lower_percent"], "upper_percent": e["output"]["upper_percent"],
                         "hit_probability": round(e["output"]["hit_probability"] * 100, 1), "shown": e["output"]["shown"]}
                        for e in examples]}}
print(f"trees | ağaç {trees} · q {q:.4f} · k {bundle['CONFORMAL_K']:.4f} · cells | hücre {len(counts)}",
      [(e["id"], e["output"]["comparables"], e["output"]["lower_percent"], e["output"]["upper_percent"],
        e["output"]["shown"]) for e in examples])

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
SERVE_DIR.mkdir(parents=True, exist_ok=True)
with open(SERVING_DIR / "final_model.pkl", "wb") as fh:
    pickle.dump({"final_lgb": models["lgb"], "final_cb": models["cb"], "tfidf": enc["tfidf"], "cat_maps": enc["cat_maps"],
                 "feat_cols": enc["feat_cols"], "CAT": CAT, "NUM": NUM, "TEXT": TEXT, "cat_idx": models["cat_idx"]}, fh)
models["lgb"].booster_.save_model(str(SERVE_DIR / "lightgbm_tfidf_svd.txt"))
models["cb"].save_model(str(SERVE_DIR / "catboost_tfidf_svd.cbm"))
models["cb_native"].save_model(str(SERVE_DIR / "catboost_native.cbm"))
with open(SERVE_DIR / "encoders.pkl", "wb") as fh:
    pickle.dump({**bundle, "cat_idx": models["cat_idx"],
                 "SEGMENT_MAP": SR.SEGMENT_MAP, "PERF_BASE": SR.PERF_BASE, "MODEL_SEG": SR.MODEL_SEG,
                 "PERF_RE": SR.PERF_RE.pattern, "FINAL_LGB_AGAC": trees, "CONFORMAL_Q": q,
                 "CONFORMAL_TANIM": CONFORMAL_DEF}, fh)
(SERVE_DIR / "README.md").write_text(serving_readme(trees), encoding="utf-8")
(SERVE_DIR / "parity_examples.json").write_text(
    json.dumps({"_meta": {"script": "07_final_model.py", "generated_at": generated_stamp()}, "examples": examples},
               ensure_ascii=False, indent=1), encoding="utf-8")
print("written | yazıldı:", save_metrics("07_final_model", to_metrics(res), run_id=oof_info["run_id"]))
