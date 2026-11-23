"""
08_forward_test.py
EN: Technical report §8 — the error on listings that arrive later. A model trained like the served one on every listing
    seen before the last snapshot (each at its latest price) prices the listings first seen in the last snapshot. The
    test snapshot sets nothing: not the model, the tree count, q or the band edges. The setup is the headline one:
      - cv.fold_matrices: category codes and model/series TF-IDF+SVD built on the training part only;
      - cv.LGB_PARAMS; log1p price;
      - trees = median of the early-stopping iterations of a 5-fold CV on the training set, then one fit on the whole
        training set without early stopping (as 07_final_model).
    Reported:
      - MAPE (cv.price_metrics on expm1 of the log prediction) with a 95% interval from a bootstrap that resamples
        whole models (a model's listings are not independent);
      - the interval recipes applied forward, everything calibrated on the training set's own OOF errors: one q for
        all, one q per predicted-price band, and the scale model (lib.conformal: s and k from the training set's nested
        OOF scores, the served s trained on the whole training set; a training listing's comparables are counted
        without its inner fold, a test listing's in the whole training set). Coverage overall, per band and on the
        listings whose (model, year) never occurs in training; width, interval score and the direction of the misses
        per arm; coverage per fifth of the scale width; the signed log error;
      - the hit probability (±10%) of the scale model on the test listings: predicted against observed.
    2026-11-22 (owner's decision, docs/decisions.md): drift and the other time analyses left the analysis. This single
    test replaced 09_backtest: it is that script's last cumulative setup, with the same functions. Heavy: it trains
    (analysis/lib/records.py HEAVY), so a notebook reads its record.
TR: Teknik rapor §8 — sonradan gelen ilanlardaki hata. Servis edilen model gibi eğitilen bir model, son taramadan önce
    görülen bütün ilanlarla (her biri son fiyatıyla) eğitilir ve son taramada ilk kez görülen ilanları fiyatlar. Test
    taraması hiçbir şeyi belirlemez: ne modeli, ne ağaç sayısını, ne q'yu, ne bant sınırlarını. Kurulum manşetteki:
      - cv.fold_matrices: kategori kodları ve model/seri TF-IDF+SVD yalnız eğitim kısmında kurulur;
      - cv.LGB_PARAMS; log1p fiyat;
      - ağaç = eğitim kümesinde 5 katlı CV'nin erken durdurma turlarının medyanı, sonra bütün eğitim kümesinde erken
        durdurmasız tek eğitim (07_final_model gibi).
    Raporlanan:
      - MAPE (log tahminin expm1'i üzerinde cv.price_metrics) ve bütün modelleri yeniden örnekleyen bootstrap'ten %95
        aralık (bir modelin ilanları bağımsız değil);
      - aralık tarifleri ileriye uygulanır, her şey eğitim kümesinin kendi OOF hatalarıyla kalibre edilir: herkese tek
        q, tahmin fiyatı bandı başına q ve ölçek modeli (lib.conformal: s ve k eğitim kümesinin iç içe OOF
        skorlarından, servis edilen s bütün eğitim kümesiyle; eğitim ilanının emsali kendi iç katı sayılmadan, test
        ilanınınki bütün eğitim kümesinde sayılır). Kapsama genel, bant başına ve (model, yıl)'ı eğitimde hiç geçmeyen
        ilanlarda; kol başına genişlik, aralık skoru ve kaçırmanın yönü; ölçek genişliğinin beşte birlerinde kapsama;
        işaretli log hata;
      - ölçek modelinin test ilanlarındaki isabet olasılığı (±%10): öngörülen ve gerçekleşen.
    2026-11-22 (kullanıcı kararı, docs/decisions.md): kayma ve öteki zaman analizleri analizden çıktı. Bu tek test
    09_backtest'in yerini aldı: o betiğin son biriken kurulumu, aynı fonksiyonlarla. Ağır: model eğitir
    (analysis/lib/records.py HEAVY); defter kaydını okur.
Output / Çıktı: metrics/08_forward_test.json
"""

# %% [1] Setup | Kurulum
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import conformal as CF
from lib.common import FEATURES, load_clean, save_metrics
from lib.cv import LGB_PARAMS, N_JOBS, PRICE_CAP, SEED, fold_matrices, lgb_oof, make_folds, price_metrics

MIN_TEST = 30
BANDS = ["Q1", "Q2", "Q3", "Q4"]
N_BOOT, LEVEL = 1000, 0.95


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def latest_per_ad(rows):
    """
    EN: One row per ad_id: its latest snapshot within rows, ordered by ad_id (as load_clean orders).
    TR: ad_id başına tek satır: rows içindeki son taraması, ad_id'ye göre sıralı (load_clean gibi).
    """
    return (rows.sort_values(["ad_id", "search_date"], kind="stable").drop_duplicates("ad_id", keep="last")
            .reset_index(drop=True))


def mape(price, pred_log):
    """
    EN: MAPE (%) of expm1(log prediction), as the headline (cv.price_metrics). Returns: float, 2 decimals.
    TR: expm1(log tahmin)'in MAPE'si (%), manşetteki gibi (cv.price_metrics). Döndürür: float, 2 hane.
    """
    return round(float(price_metrics(price, np.expm1(pred_log))["MAPE"]), 2)


def fit_forward(train):
    """
    EN: Trains like the served model: a 5-fold CV with early stopping on the training set picks the tree count
        (median of the folds), then one fit on the whole set. Returns: (predict(test_frame) → log predictions,
        trees, the training set's OOF log predictions from that CV).
    TR: Servis edilen model gibi eğitir: eğitim kümesinde erken durdurmalı 5 katlı CV ağaç sayısını seçer (katların
        medyanı), sonra bütün kümede tek eğitim. Döndürür: (predict(test_tablosu) → log tahminler, ağaç, o CV'nin
        eğitim kümesindeki OOF log tahminleri).
    """
    X, yl = train[FEATURES].reset_index(drop=True), np.log1p(train["price"].values.astype(float))
    cv_run = lgb_oof(X, yl, make_folds(len(X)))
    trees = int(np.median(cv_run["iters"]))

    def predict(test):
        """EN: Log predictions for test rows. / TR: Test satırları için log tahminler."""
        both = pd.concat([X, test[FEATURES].reset_index(drop=True)], ignore_index=True)
        tr, te = np.arange(len(X)), np.arange(len(X), len(both))
        Xtr, Xte, catc = fold_matrices(both, tr, te)
        mdl = lgb.LGBMRegressor(n_jobs=N_JOBS, **{**LGB_PARAMS, "n_estimators": trees})
        mdl.fit(Xtr, yl, categorical_feature=catc)
        return mdl.predict(Xte)
    return predict, trees, cv_run["pred_log"]


def test_rows(rows, seen, snap):
    """
    EN: The listings of snap the model never saw (new ad_ids), one row each. Stops if fewer than MIN_TEST.
    TR: snap'in modelin hiç görmediği ilanları (yeni ad_id), her biri tek satır. MIN_TEST'ten azsa durur.
    """
    te = rows[(rows["snap"] == snap) & (~rows["ad_id"].isin(seen))].drop_duplicates("ad_id", keep="last")
    if len(te) < MIN_TEST:
        raise ValueError(f"only {len(te)} new listings in {snap} | yeni ilan çok az")
    return te.reset_index(drop=True)


def cluster_boot(groups, *apes, n_boot=N_BOOT, seed=SEED):
    """
    EN: Bootstrap draws of MAPE that resample whole models (groups) with replacement; several APE arrays over the
        same listings share the same draws (paired). Returns: array (n_boot, len(apes)) of MAPEs.
    TR: Bütün modelleri (grupları) yerine koyarak yeniden örnekleyen MAPE bootstrap çekilişleri; aynı ilanlar
        üzerindeki birden çok APE dizisi aynı çekilişleri paylaşır (eşli). Döndürür: MAPE'lerin (n_boot, len(apes)) dizisi.
    """
    codes, inv = np.unique(groups, return_inverse=True)
    n_per = np.bincount(inv).astype(float)
    sums = [np.bincount(inv, weights=a) for a in apes]
    w = np.random.default_rng(seed).multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n_boot)
    return np.column_stack([(w @ s) / (w @ n_per) for s in sums])


def interval(draws):
    """EN: The central 95% of bootstrap draws. / TR: Bootstrap çekilişlerinin ortadaki %95'i."""
    lo, hi = np.quantile(draws, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return round(float(lo), 2), round(float(hi), 2)


def ape(price, pred_log):
    """EN: Absolute % error per listing. / TR: İlan başına mutlak % hata."""
    return np.abs(price - np.expm1(pred_log)) / price * 100


def forward_q(train_price, train_oof_log, test_log):
    """
    EN: The calibration of the two q arms, from training numbers only: q from the training set's own OOF log errors
        (finite-sample 90%), band edges from the quartiles of its OOF predictions, one q per band.
        Returns: (q, [q per band], [3 band edges], test predictions in ₺, band of each test listing).
    TR: İki q kolunun kalibrasyonu, yalnız eğitim sayılarından: q eğitim kümesinin kendi OOF log hatalarından (sonlu
        örneklem %90), bant sınırları OOF tahminlerinin çeyreklerinden, bant başına bir q.
        Döndürür: (q, [bant başına q], [3 bant sınırı], ₺ olarak test tahminleri, her test ilanının bandı).
    """
    tr_pred = np.clip(np.expm1(train_oof_log), 0, PRICE_CAP)
    tr_err = np.abs(np.log1p(train_price) - np.log1p(tr_pred))
    bounds = [float(x) for x in np.quantile(tr_pred, [.25, .5, .75])]
    tr_band = CF.band_of(tr_pred, bounds)
    q = CF.conformal_quantile(tr_err)
    q_band = [CF.conformal_quantile(tr_err[tr_band == b]) for b in range(len(BANDS))]
    te_pred = np.clip(np.expm1(test_log), 0, PRICE_CAP)
    return q, q_band, bounds, te_pred, CF.band_of(te_pred, bounds)


def forward_coverage(train_price, train_oof_log, test_price, test_log, no_comparable):
    """
    EN: The two q arms applied forward (forward_q): one q for all, or one q per band (Mondrian). Coverage on the
        test listings overall, per band and on those whose (model, year) never occurs in training. Only training
        numbers set q and the edges. Returns: {"q", "q_band", "bounds", "n", "coverage_global", "coverage_band",
        "bands": [[band, n, cov_global, cov_band]], "no_comparable": {"n", "coverage_global", "coverage_band"}}.
    TR: İki q kolu ileriye uygulanır (forward_q): herkese tek q ya da bant başına q (Mondrian). Kapsama test
        ilanlarında genel, bant başına ve (model, yıl)'ı eğitimde hiç geçmeyenlerde. q'yu ve sınırları yalnız eğitim
        sayıları belirler. Döndürür: {"q", "q_band", "bounds", "n", "coverage_global", "coverage_band", "bands":
        [[bant, n, kapsama_genel, kapsama_bant]], "no_comparable": {"n", "coverage_global", "coverage_band"}}.
    """
    q, q_band, bounds, te_pred, te_band = forward_q(train_price, train_oof_log, test_log)
    ins_g = CF.inside(test_price, te_pred, q)
    ins_b = CF.inside(test_price, te_pred, np.asarray(q_band)[te_band])
    pc = lambda m: round(float(m.mean() * 100), 2) if m.size else None       # noqa: E731
    return {"q": round(q, 4), "q_band": [round(x, 4) for x in q_band], "bounds": [round(b, 0) for b in bounds],
            "n": int(len(test_price)), "coverage_global": pc(ins_g), "coverage_band": pc(ins_b),
            "bands": [[name, int((te_band == b).sum()), pc(ins_g[te_band == b]), pc(ins_b[te_band == b])]
                      for b, name in enumerate(BANDS)],
            "no_comparable": {"n": int(no_comparable.sum()), "coverage_global": pc(ins_g[no_comparable]),
                              "coverage_band": pc(ins_b[no_comparable])}}


def forward_scale(train_X, train_price, train_oof_log, train_cells, train_fold, test_X, test_log, test_cells):
    """
    EN: The scale model applied forward; the test prices are not an input. Scores: each training listing's OOF log
        error over an s that never saw its inner fold (CF.nested_scores), its comparables counted without that fold.
        k is their finite-sample 90% quantile. The s that prices the test listings is trained on the whole training
        set; a test listing's comparables are the training listings of its (model, year) cell.
        Returns: {"k", "s", "r" (half-width k·s), "p" (hit probability), "comparables"}, arrays per test listing.
    TR: Ölçek modeli ileriye uygulanır; test fiyatları girdi değildir. Skorlar: her eğitim ilanının OOF log hatası
        bölü kendi iç katını görmemiş bir s (CF.nested_scores); emsali o kat sayılmadan sayılır. k bunların sonlu
        örneklem %90 yüzdeliği. Test ilanlarını fiyatlayan s bütün eğitim kümesiyle eğitilir; test ilanının emsali
        kendi (model, yıl) hücresindeki eğitim ilanlarıdır.
        Döndürür: {"k", "s", "r" (yarı genişlik k·s), "p" (isabet olasılığı), "comparables"}, test ilanı başına diziler.
    """
    n = len(train_price)
    tr_pred = np.clip(np.expm1(train_oof_log), 0, PRICE_CAP)
    tr_err = np.abs(np.log1p(train_price) - np.log1p(tr_pred))
    log_pred = np.concatenate([np.log1p(tr_pred), np.log1p(np.clip(np.expm1(test_log), 0, PRICE_CAP))])
    cells = pd.Series(np.asarray(train_cells, dtype=object)).astype(str)
    te_n = pd.Series(np.asarray(test_cells, dtype=object)).astype(str).map(cells.value_counts()).fillna(0).astype(int)
    counts = np.concatenate([CF.other_fold_counts(train_cells, train_fold), te_n.values])
    scores, _ = CF.nested_scores(train_X, tr_err, log_pred[:n], counts[:n], train_fold)
    both = pd.concat([train_X, test_X], ignore_index=True)
    Xtr, Xte, catc = CF.scale_matrix(both, np.arange(n), np.arange(n, len(both)), log_pred, counts)
    s = CF.predict_scale(CF.fit_scale(Xtr, tr_err, catc), Xte)
    k = CF.conformal_quantile(scores)
    return {"k": k, "s": s, "r": k * s, "p": CF.p_within(np.sort(scores), s), "comparables": te_n.values}


def arm_report(test_price, te_pred, arms, no_comparable, scale_r):
    """
    EN: The arms side by side on the test listings. Per arm: coverage, median and mean width (% of the prediction),
        interval score, coverage without a comparable, and the misses split into above and below the interval (% of
        the test listings). Per fifth of the scale width (narrowest first): listings and coverage of every arm.
        arms: {name: half-width per listing or one number}.
        Returns: {"arms": [[arm, coverage, width_median, width_mean, interval_score, coverage_no_comparable,
        above, below]], "by_width_fifth": [[fifth, n, coverage per arm...]]}.
    TR: Kollar test ilanlarında yan yana. Kol başına: kapsama, medyan ve ortalama genişlik (tahminin %'si), aralık
        skoru, emsalsizde kapsama ve kaçırmaların aralığın üstünde ve altında kalanlara bölünmesi (test ilanlarının
        %'si). Ölçek genişliğinin beşte biri başına (en dar önce): ilan sayısı ve her kolun kapsaması.
        arms: {ad: ilan başına ya da tek sayı yarı genişlik}.
        Döndürür: {"arms": [[kol, kapsama, genişlik_medyan, genişlik_ortalama, aralık_skoru, emsalsiz_kapsama,
        üstte, altta]], "by_width_fifth": [[beşte_bir, n, kol başına kapsama...]]}.
    """
    pc = lambda m: round(float(np.mean(m) * 100), 2)                         # noqa: E731
    fifth = pd.qcut(pd.Series(scale_r).rank(method="first"), 5, labels=False).values
    rows, ins = [], {}
    for name, r in arms.items():
        lo, hi = CF.interval(te_pred, r)
        ins[name] = (test_price >= lo) & (test_price <= hi)
        w = CF.width_pct(te_pred, r)
        rows.append([name, pc(ins[name]), round(float(np.median(w)), 1), round(float(np.mean(w)), 1),
                     round(CF.interval_score(test_price, te_pred, r), 4), pc(ins[name][no_comparable]),
                     pc(test_price > hi), pc(test_price < lo)])
    return {"arms": rows,
            "by_width_fifth": [[i + 1, int((fifth == i).sum())] + [pc(ins[a][fifth == i]) for a in arms]
                               for i in range(5)]}


def forward_test(rows, snaps):
    """
    EN: The single forward test: train on every listing seen before the last snapshot (latest price per ad_id), price
        the listings first seen in the last snapshot. Returns: {"train_until", "test", "n_train", "n", "MAPE",
        "ci_lo", "ci_hi", "trees", "signed_error", "coverage": forward_coverage(...) + the arms side by side and the
        scale model's k, "confidence": CF.reliability(...)}.
    TR: Tek ileri test: son taramadan önce görülen her ilanla eğit (ad_id başına son fiyat), son taramada ilk kez
        görülen ilanları fiyatla. Döndürür: {"train_until", "test", "n_train", "n", "MAPE", "ci_lo", "ci_hi", "trees",
        "signed_error", "coverage": forward_coverage(...) + kollar yan yana ve ölçek modelinin k'sı,
        "confidence": CF.reliability(...)}.
    """
    train = latest_per_ad(rows[rows["snap"].isin(snaps[:-1])])
    predict, trees, oof_log = fit_forward(train)
    te = test_rows(rows, set(train["ad_id"]), snaps[-1])
    price = te["price"].values.astype(float)
    pl = predict(te)
    lo, hi = interval(cluster_boot(te["model"].values, ape(price, pl))[:, 0])
    cells = set(zip(train["model"], train["gb_year"]))
    no_comp = np.array([c not in cells for c in zip(te["model"], te["gb_year"])])
    tr_price = train["price"].values.astype(float)
    cov = forward_coverage(tr_price, oof_log, price, pl, no_comp)
    # EN: scale arm — the inner folds are those of fit_forward's CV | TR: ölçek kolu — iç katlar fit_forward'ın CV'sindekiler
    tr_fold = np.zeros(len(train), dtype=int)
    for f, (_, va) in enumerate(make_folds(len(train))):
        tr_fold[va] = f
    cell = lambda d: (d["model"].astype(str) + "|" + d["gb_year"].astype(str)).values       # noqa: E731
    sc = forward_scale(train[FEATURES].reset_index(drop=True), tr_price, oof_log, cell(train), tr_fold,
                       te[FEATURES].reset_index(drop=True), pl, cell(te))
    assert np.array_equal(sc["comparables"] == 0, no_comp), "no-comparable listings differ | emsalsiz ilanlar farklı"
    q, q_band, _, te_pred, te_band = forward_q(tr_price, oof_log, pl)
    arms = arm_report(price, te_pred, {"global": q, "band": np.asarray(q_band)[te_band], "scale": sc["r"]},
                      no_comp, sc["r"])
    signed = (np.log1p(price) - np.log1p(te_pred)) * 100
    hit = np.abs(signed) / 100 <= np.log1p(CF.WITHIN)
    return {"train_until": snaps[-2], "test": snaps[-1], "n_train": int(len(train)), "n": int(len(te)),
            "MAPE": mape(price, pl), "ci_lo": lo, "ci_hi": hi, "trees": trees,
            "signed_error": {"mean": round(float(signed.mean()), 2), "median": round(float(np.median(signed)), 2)},
            "coverage": {**cov, **arms,
                         "scale": {"k": round(sc["k"], 4), "floor_hits": int((sc["s"] <= CF.S_FLOOR).sum())},
                         "arm_columns": ["arm", "coverage", "width_median", "width_mean", "interval_score",
                                         "coverage_no_comparable", "missed_above", "missed_below"]},
            "confidence": CF.reliability(sc["p"], hit)}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under methodology.forward_test (site tree and reports), with the protocol that produced it.
    TR: methodology.forward_test altında yayımlanır (site ağacı ve raporlar), onu üreten protokolle.
    """
    return {"methodology": {"forward_test": {
        **res,
        "protocol": {"train": "every listing seen before the last snapshot, latest price per ad_id",
                     "test": "listings first seen in the last snapshot",
                     "setup": "cv.fold_matrices + cv.LGB_PARAMS; trees = median of an inner 5-fold early-stopping CV "
                              "on the training set, then one fit without early stopping",
                     "scale": "s and k from the training set's nested OOF scores; the served s is trained on the "
                              "whole training set; signed_error = (log price − log prediction) × 100",
                     "interval": {"level": LEVEL, "n_boot": N_BOOT, "resample": "model"}}}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rows = load_clean(all_snapshots=True)              # every snapshot row, ORDER BY ad_id, search_date

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
snaps = sorted(rows["snap"].unique())
res = forward_test(rows, snaps)
print({k: res[k] for k in ("train_until", "test", "n_train", "n", "MAPE", "ci_lo", "ci_hi", "trees")},
      "coverage | kapsama", res["coverage"]["coverage_global"], res["coverage"]["coverage_band"],
      res["coverage"]["no_comparable"], "· arms | kollar", res["coverage"]["arms"], res["coverage"]["scale"],
      "· hit | isabet", res["confidence"]["mean_predicted"], res["confidence"]["mean_observed"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
if __name__ == "__main__":      # imported by notebooks/: no save | notebooks/ import edince kayıt yok
    print("written | yazıldı:", save_metrics("08_forward_test", to_metrics(res)))
