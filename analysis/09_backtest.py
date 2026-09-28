"""
09_backtest.py
EN: Technical report §9 — does a model trained on the past price the next snapshot? Four arms, all with the
    headline setup (cv.fold_matrices: category codes and model/series TF-IDF+SVD built on the training part only;
    cv.LGB_PARAMS; log1p price; MAPE as cv.price_metrics on expm1 of the log prediction):
      - single: train on one snapshot, test on the NEW ad_ids of every later snapshot (no leakage);
      - cumulative: train on all snapshots up to t, same test rule (its first block is the single arm's);
      - insample: 5-fold OOF on the listings accumulated up to t (no time dimension); its last block is every
        listing, and must equal the headline OOF of 07_model_comparison exactly;
      - per_snapshot: 5-fold OOF inside each snapshot on its own (all listings live that day).
    The forward arms train like the served model (07_final_model): trees = median of the early-stopping iterations
    of a 5-fold CV on the training set, then one fit on the whole training set without early stopping; the test
    snapshot is never used to stop. Every forward row carries a 95% interval from a bootstrap that resamples the
    test listings by model (listings of one model are not independent), and single vs cumulative are compared on
    the listings both arms tested, with the same paired resamples. 2026-09-28 (simplification list): the lighter
    setup (raw categories, fixed 800/500 trees) left with the "compare rows only" caveat.
TR: Teknik rapor §9 — geçmişle eğitilen model bir sonraki taramayı fiyatlıyor mu? Dört kol; hepsi manşet kurulumla
    (cv.fold_matrices: kategori kodları ve model/seri TF-IDF+SVD yalnız eğitim kısmında kurulur; cv.LGB_PARAMS;
    log1p fiyat; MAPE, log tahminin expm1'i üzerinde cv.price_metrics ile):
      - single: tek taramada eğit, sonraki her taramanın YENİ ad_id'lerinde test et (sızıntı yok);
      - cumulative: t'ye kadarki bütün taramalarda eğit, aynı test kuralı (ilk bloğu single'ınkiyle aynı);
      - insample: t'ye kadar biriken ilanlarda 5-fold OOF (zaman boyutu yok); son bloğu bütün ilanlar ve
        07_model_comparison'ın manşet OOF'una birebir eşit olmalı;
      - per_snapshot: her taramanın kendi içinde 5-fold OOF (o gün yayında olan bütün ilanlar).
    İleri kollar servis edilen model gibi eğitilir (07_final_model): ağaç = eğitim kümesinde 5 katlı CV'nin erken
    durdurma turlarının medyanı, sonra bütün eğitim kümesinde erken durdurmasız tek eğitim; test taraması durdurmak
    için hiç kullanılmaz. Her ileri satır, test ilanlarını modele göre yeniden örnekleyen bootstrap'ten %95 aralık
    taşır (bir modelin ilanları bağımsız değil); single ve cumulative, iki kolun da test ettiği ilanlarda aynı eşli
    yeniden örneklemelerle karşılaştırılır. 2026-09-28 (sadeleştirme listesi): hafif kurulum (ham kategori, sabit
    800/500 ağaç) "yalnız satırları karşılaştırın" uyarısıyla birlikte kalktı.
Output / Çıktı: metrics/09_backtest.json
"""

# %% [1] Setup | Kurulum
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib.common import FEATURES, load_clean, save_metrics
from lib.cv import LGB_PARAMS, N_JOBS, SEED, fold_matrices, lgb_oof, load_oof, make_folds, price_metrics

MIN_TEST, MIN_OOF = 30, 100
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
        trees).
    TR: Servis edilen model gibi eğitir: eğitim kümesinde erken durdurmalı 5 katlı CV ağaç sayısını seçer (katların
        medyanı), sonra bütün kümede tek eğitim. Döndürür: (predict(test_tablosu) → log tahminler, ağaç).
    """
    X, yl = train[FEATURES].reset_index(drop=True), np.log1p(train["price"].values.astype(float))
    trees = int(np.median(lgb_oof(X, yl, make_folds(len(X)))["iters"]))

    def predict(test):
        """EN: Log predictions for test rows. / TR: Test satırları için log tahminler."""
        both = pd.concat([X, test[FEATURES].reset_index(drop=True)], ignore_index=True)
        tr, te = np.arange(len(X)), np.arange(len(X), len(both))
        Xtr, Xte, catc = fold_matrices(both, tr, te)
        mdl = lgb.LGBMRegressor(n_jobs=N_JOBS, **{**LGB_PARAMS, "n_estimators": trees})
        mdl.fit(Xtr, yl, categorical_feature=catc)
        return mdl.predict(Xte)
    return predict, trees


def test_rows(rows, seen, snap):
    """
    EN: The listings of snap the model never saw (new ad_ids), one row each; None if fewer than 30.
    TR: snap'in modelin hiç görmediği ilanları (yeni ad_id), her biri tek satır; 30'dan azsa None.
    """
    te = rows[(rows["snap"] == snap) & (~rows["ad_id"].isin(seen))].drop_duplicates("ad_id", keep="last")
    return te.reset_index(drop=True) if len(te) >= MIN_TEST else None


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


def forward_arm(rows, snaps, cumulative):
    """
    EN: The single or cumulative arm. Returns: (rows [train, test, MAPE, n, lo, hi, trees], {(train, test): (ad_ids,
        APE, model)} for the paired comparison).
    TR: single ya da cumulative kol. Döndürür: (satırlar [eğitim, test, MAPE, n, alt, üst, ağaç], eşli karşılaştırma
        için {(eğitim, test): (ad_id'ler, APE, model)}).
    """
    out, keep = [], {}
    for i in range(1, len(snaps)) if cumulative else range(len(snaps) - 1):
        train_snaps = snaps[:i] if cumulative else [snaps[i]]
        train = latest_per_ad(rows[rows["snap"].isin(train_snaps)])
        predict, trees = fit_forward(train)
        seen = set(train["ad_id"])
        for t in snaps[(i if cumulative else i + 1):]:
            te = test_rows(rows, seen, t)
            if te is None:
                continue
            price = te["price"].values.astype(float)
            pl = predict(te)
            a = ape(price, pl)
            lo, hi = interval(cluster_boot(te["model"].values, a)[:, 0])
            label = ("→" + train_snaps[-1][5:]) if cumulative else train_snaps[0][5:]
            out.append([label, t[5:], mape(price, pl), int(len(te)), lo, hi, trees])
            keep[(label, t[5:])] = (te["ad_id"].values, a, te["model"].values)
    return out, keep


def paired(single_keep, cumulative_keep, first):
    """
    EN: Cumulative − single MAPE on the listings both arms tested in the same test snapshot, for each pair whose
        training differs (not the first snapshot, where the two arms are one experiment); 95% interval from the same model-resampled draws for both. Returns: [[test, single train,
        cumulative train, n, single MAPE, cumulative MAPE, difference, lo, hi], ...].
    TR: Aynı test taramasında iki kolun da test ettiği ilanlarda cumulative − single MAPE, eğitimi farklı her çift
        için (ilk tarama değil; orada iki kol tek deney); %95 aralık ikisi için aynı model-yeniden-örneklemeli çekilişlerden. Döndürür: [[test, single eğitim,
        cumulative eğitim, n, single MAPE, cumulative MAPE, fark, alt, üst], ...].
    """
    out = []
    for (c_train, test), (c_ids, c_ape, c_model) in cumulative_keep.items():
        s_train = c_train.lstrip("→")
        if s_train == first or (s_train, test) not in single_keep:
            continue
        s_ids, s_ape, _ = single_keep[(s_train, test)]
        common, ci, si = np.intersect1d(c_ids, s_ids, return_indices=True)
        d = cluster_boot(c_model[ci], s_ape[si], c_ape[ci])
        lo, hi = interval(d[:, 1] - d[:, 0])
        s_m, c_m = float(s_ape[si].mean()), float(c_ape[ci].mean())
        out.append([test, s_train, c_train, int(len(common)), round(s_m, 2), round(c_m, 2), round(c_m - s_m, 2), lo, hi])
    return out


def oof_arm(sub):
    """
    EN: 5-fold OOF MAPE inside a set of listings with the headline cv.lgb_oof. Returns: (MAPE or None, n, log
        predictions or None).
    TR: Bir ilan kümesinin içinde manşet cv.lgb_oof ile 5-fold OOF MAPE. Döndürür: (MAPE ya da None, n, log
        tahminler ya da None).
    """
    if len(sub) < MIN_OOF:
        return None, len(sub), None
    pl = lgb_oof(sub[FEATURES].reset_index(drop=True), np.log1p(sub["price"].values.astype(float)),
                 make_folds(len(sub)))["pred_log"]
    return mape(sub["price"].values.astype(float), pl), len(sub), pl


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology.backtest).
    TR: Site ağacında yayımlanır (methodology.backtest).
    """
    return {"methodology": {"backtest": {
        "single": res["single"], "cumulative": res["cumulative"], "paired": res["paired"],
        "insample": res["insample"], "per_snapshot": res["per_snapshot"],
        "columns": {"forward": ["train", "test", "MAPE", "n", "ci_lo", "ci_hi", "trees"],
                    "paired": ["test", "single_train", "cumulative_train", "n_common", "single_MAPE",
                               "cumulative_MAPE", "difference", "ci_lo", "ci_hi"]},
        "protocol": {"leak_free": ["single", "cumulative"], "plain_kfold": ["insample", "per_snapshot"],
                     "same_experiment": "cumulative first block (<= first snapshot) equals the single first block",
                     "setup": "headline: cv.fold_matrices + cv.LGB_PARAMS; forward trees = median of an inner 5-fold "
                              "early-stopping CV on the training set, then one fit without early stopping",
                     "interval": {"level": LEVEL, "n_boot": N_BOOT, "resample": "model"}}}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rows = load_clean(all_snapshots=True)              # every snapshot row, ORDER BY ad_id, search_date
listings = load_clean()
stored, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
snaps = sorted(rows["snap"].unique())
single, single_keep = forward_arm(rows, snaps, cumulative=False)
cumulative, cumulative_keep = forward_arm(rows, snaps, cumulative=True)
insample = []
for i in range(1, len(snaps) + 1):
    mp, n, pl = oof_arm(latest_per_ad(rows[rows["snap"].isin(snaps[:i])]))
    insample.append(["→" + snaps[i - 1][5:] if i > 1 else snaps[0][5:], mp, n])
# EN: the last block is every listing: it must be the headline OOF itself | TR: son blok bütün ilanlar: manşet OOF'un kendisi olmalı
assert n == len(listings) and np.array_equal(np.expm1(pl), stored["lgb"].values), \
    "last insample block ≠ headline OOF | son insample bloğu manşet OOF'a eşit değil"
per_snapshot = [[s[5:], *oof_arm(rows[rows["snap"] == s].reset_index(drop=True))[:2]] for s in snaps]
res = {"single": single, "cumulative": cumulative, "paired": paired(single_keep, cumulative_keep, snaps[0][5:]),
       "insample": insample, "per_snapshot": per_snapshot}
print("single:", single, "\npaired:", res["paired"], "\ninsample:", insample, "\nper_snapshot:", per_snapshot)

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("09_backtest", to_metrics(res), run_id=oof_info["run_id"]))
