"""
Aralık: tahmin etrafındaki %90 aralık, ölçek modeli ve isabet olasılığı (4. defter).
Aralık log ölçekte kurulur: expm1(log1p(tahmin) ± r), r yarı genişlik. Tek pay kolunda r herkes için aynı q. Ölçek
modelinde r = k · s(x): s, ilanın kat dışı mutlak log hatasını tahmin modelinin metin dışı kodlu girdilerinden, log
tahminden ve emsal sayısından tahmin eden küçük bir LightGBM'dir; emsal arttıkça artamaz (monotonluk kısıtı) ve
S_FLOOR'da alttan kırpılır. k, iç içe skorların (hata / s) sonlu örneklem %90 yüzdeliği. İsabet olasılığı aynı
skorlardan: ilan fiyatının tahminin ±%10 yakınında olma olasılığı. Ayarlar bir kez kondu, aranmadı (karar kaydı,
2026-11-23). Yöntem eski zincirin analysis/lib/conformal.py dosyasından; kod bu defter için yeniden yazıldı.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import data, folds, model
from lib.errors import PRICE_CAP

LEVEL = 0.90                    # aralığın hedef kapsaması
SCALE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, random_state=folds.SEED, verbose=-1, deterministic=True,
                    force_row_wise=True, n_jobs=model.N_JOBS)
S_FLOOR = 0.02                  # s'nin alt sınırı (log puan)
WITHIN = 0.10                   # isabet olasılığının toleransı: tahminin ±%10'u
SHOWN_MAX = 9                   # "10'da N" gösteriminde N'nin üst sınırı
LOG_PRED, COMPARABLES = "log_pred", "comparables"      # ölçek modeli matrisinin son iki kolonu


def interval(pred_price, r):
    """
    Tahmin etrafındaki aralık: expm1(log1p(tahmin) ± r); tahmin önce [0, PRICE_CAP]'e kırpılır, alt uç 0'ın, üst uç
    PRICE_CAP'in dışına çıkmaz. r tek sayı ya da ilan başına bir dizi. Döndürür: (alt, üst) ₺.
    """
    log_pred = np.log1p(np.clip(np.asarray(pred_price, dtype=float), 0, PRICE_CAP))
    return np.expm1(np.clip(log_pred - r, 0, None)), np.expm1(np.clip(log_pred + r, 0, np.log1p(PRICE_CAP)))


def inside(price, pred_price, r):
    """Gerçek fiyat aralığın içindeyse (uçlar dahil) True."""
    low, high = interval(pred_price, r)
    price = np.asarray(price, dtype=float)
    return (price >= low) & (price <= high)


def conformal_quantile(values, level=LEVEL):
    """
    Sonlu örneklem conformal yüzdeliği: en küçükten ceil((n + 1) · level)'inci değer. Kalibrasyon kısmı bu sıra için
    çok küçükse durur.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * level))
    if not 1 <= rank <= len(ordered):
        raise ValueError(f"kalibrasyon kısmı çok küçük: {len(ordered)} değer, istenen sıra {rank}")
    return float(ordered[rank - 1])


def cross_single_q(abs_err, fold, level=LEVEL):
    """
    Tek pay kolu, çapraz kalibreli: her katın ilanlarına öteki katların mutlak log hatalarından tek q. Hiçbir ilanın
    kendi hatası kendi q'suna girmez. Döndürür: ilan başına q.
    """
    abs_err, fold = np.asarray(abs_err, dtype=float), np.asarray(fold)
    q = np.empty(len(abs_err))
    for k in np.unique(fold):
        q[fold == k] = conformal_quantile(abs_err[fold != k], level)
    return q


def other_fold_counts(cells, fold):
    """
    Her ilanın emsali, kendi katı sayılmadan: aynı hücredeki (ör. araç modeli + model yılı) ilanlardan öteki katlarda
    olanların sayısı. Kat dışı hatayı kendi katını görmemiş bir tahmin modeli üretti; emsal de öyle sayılır.
    Döndürür: ilan başına tamsayı.
    """
    cell = pd.Series(np.asarray(cells, dtype=object)).astype(str)
    in_fold = pd.Series(np.asarray(fold)).astype(str) + "#" + cell
    return (cell.map(cell.value_counts()).to_numpy() - in_fold.map(in_fold.value_counts()).to_numpy()).astype(int)


def scale_matrices(table, train, valid, log_pred, comparables):
    """
    Ölçek modelinin matrisleri: tahmin modelinin kat matrisi metin kolonları olmadan (model.fold_matrices; kategori
    kodları yalnız eğitim kısmından, görülmemiş değer -1), ardından log tahmin ve emsal sayısı. log_pred ve comparables
    table'ın satırlarıyla hizalı. Döndürür: (eğitim matrisi, öteki matris, kategorik kolonlar).
    """
    X_train, X_valid, categorical = model.fold_matrices(table, train, valid, drop=data.TEXT)
    log_pred, comparables = np.asarray(log_pred, dtype=float), np.asarray(comparables)
    X_train[LOG_PRED], X_train[COMPARABLES] = log_pred[train], comparables[train]
    X_valid[LOG_PRED], X_valid[COMPARABLES] = log_pred[valid], comparables[valid]
    return X_train, X_valid, categorical


def fit_scale(X, abs_err, categorical):
    """
    Ölçek modeli: mutlak log hata üzerinde LightGBM (SCALE_PARAMS); yalnız emsal kolonunda artmayan (monotonluk
    kısıtı). Döndürür: eğitilmiş LGBMRegressor.
    """
    monotone = [-1 if column == COMPARABLES else 0 for column in X.columns]
    regressor = lgb.LGBMRegressor(**SCALE_PARAMS, monotone_constraints=monotone)
    return regressor.fit(X, np.asarray(abs_err, dtype=float), categorical_feature=categorical)


def predict_scale(regressor, X):
    """s(x): ölçek modelinin tahmini, S_FLOOR'da alttan kırpılmış."""
    return np.clip(regressor.predict(X), S_FLOOR, None)


def nested_scale(table, abs_err, log_pred, comparables, fold):
    """
    İç içe s: her katın ilanlarına, yalnız öteki katların hatalarıyla eğitilmiş bir ölçek modelinin s'si. Böylece s
    puanladığı katı hiç görmez. Döndürür: ilan başına s.
    """
    abs_err, fold = np.asarray(abs_err, dtype=float), np.asarray(fold)
    rows, s = np.arange(len(abs_err)), np.empty(len(abs_err))
    for k in np.unique(fold):
        X_train, X_valid, categorical = scale_matrices(table, rows[fold != k], rows[fold == k], log_pred, comparables)
        s[fold == k] = predict_scale(fit_scale(X_train, abs_err[fold != k], categorical), X_valid)
    return s


def cross_scale(table, abs_err, log_pred, comparables, fold, level=LEVEL):
    """
    Ölçek modeli kolu, çapraz kalibreli. s: nested_scale. Puanlanan her kat için k ve isabet olasılığının skorları
    yalnız öteki dört kattan gelir: o dört katın her biri, kalan üç katla eğitilmiş bir s ile puanlanır (20 küçük
    model). Puanlanan kat ne s'ye ne k'ya girer. Döndürür: {"s": ilan başına s, "scores": hata / s,
    "r": yarı genişlik k · s, "p": isabet olasılığı, "k_by_fold": puanlanan kat başına k}.
    """
    abs_err, fold = np.asarray(abs_err, dtype=float), np.asarray(fold)
    s = nested_scale(table, abs_err, log_pred, comparables, fold)
    rows, r, p, k_by_fold = np.arange(len(abs_err)), np.empty(len(abs_err)), np.empty(len(abs_err)), []
    for k in np.unique(fold):
        calibration, scored = fold != k, fold == k
        inner = []
        for j in np.unique(fold[calibration]):
            train = calibration & (fold != j)
            X_train, X_valid, categorical = scale_matrices(table, rows[train], rows[fold == j], log_pred, comparables)
            inner.append(abs_err[fold == j] / predict_scale(fit_scale(X_train, abs_err[train], categorical), X_valid))
        inner = np.sort(np.concatenate(inner))
        k_by_fold.append(conformal_quantile(inner, level))
        r[scored] = k_by_fold[-1] * s[scored]
        p[scored] = p_within(inner, s[scored])
    return {"s": s, "scores": abs_err / s, "r": r, "p": p, "k_by_fold": k_by_fold}


def fit_scale_all(table, abs_err, log_pred, comparables):
    """
    Servis edilen ölçek modeli: bütün ilanlarla, kategori kodları bütün ilanlardan (son tahmin modelinin kodlarıyla
    aynı). Döndürür: eğitilmiş LGBMRegressor.
    """
    rows = np.arange(len(table))
    X, _, categorical = scale_matrices(table, rows, rows, log_pred, comparables)
    return fit_scale(X, abs_err, categorical)


def p_within(sorted_scores, s, within=WITHIN):
    """
    İsabet olasılığı: sıralı skorlardan log(1 + within) / s'in altında ya da ona eşit olanların payı, yani ilan
    fiyatının tahminin ±within yakınında (log cinsinden) olma olasılığının kestirimi.
    """
    cut = np.log1p(within) / np.asarray(s, dtype=float)
    return np.searchsorted(np.asarray(sorted_scores, dtype=float), cut, side="right") / len(sorted_scores)


def shown(p):
    """İsabet olasılığının "10'da N" gösterimi: N = 10 · p aşağı yuvarlanır, en çok SHOWN_MAX."""
    return np.minimum(np.floor(np.asarray(p, dtype=float) * 10 + 1e-9).astype(int), SHOWN_MAX)


def interval_score(price, pred_price, r, level=LEVEL):
    """
    Log ölçekte ortalama aralık (Winkler) skoru: genişlik artı fiyatın aralığın dışında kaldığı uzaklığın 2 / (1 -
    level) katı; küçük olan iyi.
    """
    low, high = interval(pred_price, r)
    y, low, high, weight = np.log1p(np.asarray(price, dtype=float)), np.log1p(low), np.log1p(high), 2 / (1 - level)
    return float(((high - low) + weight * np.clip(low - y, 0, None) + weight * np.clip(y - high, 0, None)).mean())


def width_pct(pred_price, r):
    """Aralığın genişliği, tahminin yüzdesi olarak."""
    low, high = interval(pred_price, r)
    return (high - low) / np.clip(np.asarray(pred_price, dtype=float), 0, PRICE_CAP) * 100


def arm_summary(price, pred_price, r, no_comparable):
    """
    Bir aralık kolunun özeti (%): kapsama, genişliğin medyanı ve ortalaması, aralık skoru, emsalsiz ilanlarda kapsama,
    fiyatın aralığın üstünde ve altında kaldığı ilanların payı. Döndürür: sözlük.
    """
    price = np.asarray(price, dtype=float)
    low, high = interval(pred_price, r)
    covered, width = (price >= low) & (price <= high), width_pct(pred_price, r)
    return {"coverage": 100 * covered.mean(), "width_median": float(np.median(width)), "width_mean": width.mean(),
            "interval_score": interval_score(price, pred_price, r), "coverage_no_comparable":
            100 * covered[np.asarray(no_comparable, dtype=bool)].mean(), "missed_above": 100 * (price > high).mean(),
            "missed_below": 100 * (price < low).mean()}


def reliability(p_hit, hit):
    """
    İsabet olasılığının güvenilirliği (%): ortalama öngörülen ve gerçekleşen; öngörünün onda birlik dilimlerinde
    öngörülen ve gerçekleşen; Brier skoru ve herkese genel isabet payını veren sabit öngörünün Brier skoru, aradaki
    iyileşme (%); öngörünün 10., 50. ve 90. yüzdeliği; "10'da N" değeri başına ilan ve gerçekleşen pay.
    Döndürür: sözlük; "deciles" ve "by_shown" tablo.
    """
    p_hit, hit = np.asarray(p_hit, dtype=float), np.asarray(hit, dtype=bool).astype(float)
    decile = pd.qcut(pd.Series(p_hit).rank(method="first"), 10, labels=False).to_numpy()
    frame = pd.DataFrame({"decile": decile + 1, "shown": shown(p_hit), "predicted": 100 * p_hit, "observed": 100 * hit})
    brier, constant = float(np.mean((p_hit - hit) ** 2)), float(np.mean((hit.mean() - hit) ** 2))
    return {"mean_predicted": 100 * p_hit.mean(), "mean_observed": 100 * hit.mean(),
            "deciles": frame.groupby("decile")[["predicted", "observed"]].mean(),
            "brier": brier, "brier_constant": constant, "skill_pct": 100 * (1 - brier / constant),
            "quantiles": {q: 100 * float(np.quantile(p_hit, q / 100)) for q in (10, 50, 90)},
            "by_shown": frame.groupby("shown").agg(n=("observed", "size"), observed=("observed", "mean"))}


def served_examples(final, scale, k, sorted_scores, table, cars, counts):
    """
    Servis edilen tarifin işlenmiş örnekleri. final: son tahmin modeli (model.fit_final); scale: servis edilen ölçek
    modeli (fit_scale_all); table: son tahmin modelinin ilanları (data.FEATURES); cars: data.FEATURES kolonlu örnek
    araçlar; counts: {(araç modeli, yaş): ilan sayısı}. Her araç tahmin modelinin matrisine, kodlar ve TF-IDF + SVD
    ilanlardan öğrenilerek çevrilir (görülmemiş kategori -1); log tahmin [0, log1p(PRICE_CAP)]'e kırpılır; emsal
    counts'tan aranır (araç modeli adı birebir, yoksa 0). Döndürür: araç başına tahmin, emsal, s, alt ve üst pay (%),
    isabet olasılığı (%) ve "10'da N".
    """
    both = pd.concat([table[data.FEATURES], cars[data.FEATURES]], ignore_index=True)
    rows, examples = np.arange(len(table)), np.arange(len(table), len(both))
    _, X_cars, _ = model.fold_matrices(both, rows, examples)
    log_pred = np.clip(final.predict(X_cars[final.feature_name_]), 0, np.log1p(PRICE_CAP))
    comparables = np.array([counts.get((str(name).strip(), int(age)), 0)
                            for name, age in zip(cars["model"], cars["vehicle_age"])])
    _, X_scale, _ = scale_matrices(both, rows, examples, np.r_[np.zeros(len(table)), log_pred],
                                   np.r_[np.zeros(len(table), dtype=int), comparables])
    s = predict_scale(scale, X_scale[scale.feature_name_])
    r, p = k * s, p_within(sorted_scores, s)
    return pd.DataFrame({"price": np.expm1(log_pred), "comparables": comparables, "s": s,
                         "lower_pct": 100 * np.expm1(-r), "upper_pct": 100 * np.expm1(r), "hit_pct": 100 * p,
                         "shown": shown(p)}, index=cars.index)
