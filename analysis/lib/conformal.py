"""
conformal.py
EN: The conformal-interval pieces shared by 08_conformal_coverage (cross-calibrated coverage on the OOF),
    08_forward_test (forward coverage on the listings new in the last snapshot) and 07_final_model (the served
    bundle): the interval around a prediction, the finite-sample quantile, the predicted-price band of a listing, and
    the scale model of the per-car interval (2026-11-23, docs/decisions.md).
    Scale model: s(x) is a small LightGBM that predicts a listing's absolute OOF log error from the main model's coded
    inputs without the text columns, the log prediction and the number of comparables (training listings in the same
    (model, year) cell); it cannot rise with the comparables (monotone constraint) and is floored at S_FLOOR. The
    interval is expm1(log1p(prediction) ± k·s(x)), k the finite-sample 90% quantile of the nested scores e / s(x)
    (each listing scored by an s that never saw its fold). The hit probability is the share of those scores at or
    below log(1 + WITHIN) / s(x). The settings were fixed once and not searched.
TR: 08_conformal_coverage (OOF'ta çapraz kalibreli kapsama), 08_forward_test (son taramada yeni ilanlarda ileri
    kapsama) ve 07_final_model'in (servis paketi) ortak conformal aralık parçaları: tahmin etrafındaki aralık, sonlu
    örneklem yüzdeliği, bir ilanın tahmin fiyatı bandı ve araca göre aralığın ölçek modeli (2026-11-23,
    docs/decisions.md).
    Ölçek modeli: s(x), bir ilanın mutlak OOF log hatasını ana modelin metin sütunları dışındaki kodlu girdilerinden,
    log tahminden ve emsal sayısından (aynı (model, yıl) hücresindeki eğitim ilanları) tahmin eden küçük bir LightGBM;
    emsal arttıkça artamaz (monotonluk kısıtı) ve S_FLOOR'da alttan kırpılır. Aralık expm1(log1p(tahmin) ± k·s(x));
    k, iç içe skorların e / s(x) sonlu örneklem %90 yüzdeliği (her ilanı kendi katını görmemiş bir s puanlar). İsabet
    olasılığı bu skorlardan log(1 + WITHIN) / s(x)'in altında ya da ona eşit olanların payı. Ayarlar bir kez kondu,
    aranmadı.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import TEXT
from .cv import N_JOBS, PRICE_CAP, SEED, fold_matrices

LEVEL = 0.90
SCALE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, deterministic=True,
                    force_row_wise=True)
S_FLOOR = 0.02                  # lower bound of s (log points) | s'nin alt sınırı (log puan)
WITHIN = 0.10                   # the hit probability's tolerance: ±10% of the prediction | isabetin toleransı: ±%10
LOG_PRED, COMPARABLES = "log_pred", "comparables"
SHOWN_MAX = 9                   # the site shows "N in 10", never 10 | site "10'da N" gösterir, 10'u hiç


def interval(pred_price, q):
    """
    EN: The conformal interval around each prediction: expm1(log1p(prediction) ± q), the lower end clipped at
        0 and the upper at log1p(15M); q may be one number or one per listing. Returns: (lower, upper) in ₺.
    TR: Her tahmin etrafındaki conformal aralık: expm1(log1p(tahmin) ± q); alt uç 0'da, üst uç log1p(15M)'de
        kırpılır; q tek sayı ya da ilan başına olabilir. Döndürür: (alt, üst) ₺.
    """
    pl = np.log1p(np.clip(pred_price, 0, PRICE_CAP))
    return np.expm1(np.clip(pl - q, 0, None)), np.expm1(np.clip(pl + q, 0, np.log1p(PRICE_CAP)))


def inside(price, pred_price, q):
    """EN: True where the actual price is inside the interval. / TR: Gerçek fiyat aralığın içindeyse True."""
    lo, hi = interval(pred_price, q)
    return (price >= lo) & (price <= hi)


def conformal_quantile(abs_err, level=LEVEL):
    """
    EN: The finite-sample conformal quantile: the ceil((n+1)·level)-th smallest absolute error. Stops if the
        calibration part is too small for that rank.
    TR: Sonlu örneklem conformal yüzdeliği: en küçükten ceil((n+1)·level)'inci mutlak hata. Kalibrasyon kısmı o
        sıra için çok küçükse durur.
    """
    r = np.sort(np.asarray(abs_err, dtype=float))
    k = int(np.ceil((len(r) + 1) * level))
    assert 1 <= k <= len(r), f"calibration part too small | kalibrasyon kısmı çok küçük: n={len(r)}"
    return float(r[k - 1])


def band_of(pred_price, bounds):
    """
    EN: The predicted-price band of each listing, 0–3, on three cut points; a value on a cut point goes to the lower
        band (as pd.qcut does).
    TR: Her ilanın tahmin fiyatı bandı, 0–3, üç kesim noktasına göre; kesim noktasındaki değer alttaki banda gider
        (pd.qcut gibi).
    """
    return np.searchsorted(np.asarray(bounds, dtype=float), pred_price, side="left")


def other_fold_counts(cells, fold):
    """
    EN: Comparables of each listing counted without its own fold: the number of listings in the same cell (e.g.
        (model, year)) that sit in the other folds. A listing's OOF error was made by a model that never saw its
        fold, so its comparables are counted the same way. Returns: int array, one per listing.
    TR: Her ilanın emsali, kendi katı sayılmadan: aynı hücredeki (ör. (model, yıl)) ilanlardan öteki katlarda
        olanların sayısı. Bir ilanın OOF hatasını kendi katını görmemiş bir model üretti; emsali de öyle sayılır.
        Döndürür: ilan başına bir tamsayı dizisi.
    """
    cell = pd.Series(np.asarray(cells, dtype=object)).astype(str)
    in_fold = pd.Series(np.asarray(fold)).astype(str) + "#" + cell
    return (cell.map(cell.value_counts()).values - in_fold.map(in_fold.value_counts()).values).astype(int)


def scale_inputs(matrix, log_pred, comparables):
    """
    EN: The scale model's input: a coded matrix without text columns, then the log prediction and the comparables
        count, in that order. Returns: a new DataFrame.
    TR: Ölçek modelinin girdisi: metin sütunsuz kodlu matris, ardından log tahmin ve emsal sayısı, bu sırayla.
        Döndürür: yeni bir DataFrame.
    """
    out = matrix.reset_index(drop=True).copy()
    out[LOG_PRED], out[COMPARABLES] = np.asarray(log_pred, dtype=float), np.asarray(comparables)
    return out


def scale_matrix(X, tr, va, log_pred, comparables):
    """
    EN: The scale model's matrices for a (train, scored) split of X (listings[FEATURES]): cv.fold_matrices without
        the text columns (categories coded from the training part, unseen → −1) + log prediction + comparables.
        Returns: (train matrix, scored matrix, categorical column names).
    TR: X'in (listings[FEATURES]) bir (eğitim, puanlanan) bölmesi için ölçek modelinin matrisleri: metin sütunsuz
        cv.fold_matrices (kategoriler eğitim kısmından kodlanır, görülmemiş → −1) + log tahmin + emsal.
        Döndürür: (eğitim matrisi, puanlanan matris, kategorik kolon adları).
    """
    cols = [c for c in X.columns if c not in TEXT]
    Xtr, Xva, catc = fold_matrices(X[cols].reset_index(drop=True), tr, va, cols=cols, use_text=False)
    return scale_inputs(Xtr, log_pred[tr], comparables[tr]), scale_inputs(Xva, log_pred[va], comparables[va]), catc


def fit_scale(matrix, abs_err, catc):
    """
    EN: The scale model: LightGBM (SCALE_PARAMS) on the absolute log error, non-increasing in the comparables column
        only. Returns: the fitted LGBMRegressor.
    TR: Ölçek modeli: mutlak log hata üzerinde LightGBM (SCALE_PARAMS), yalnız emsal sütununda artmayan.
        Döndürür: eğitilmiş LGBMRegressor.
    """
    mono = [-1 if c == COMPARABLES else 0 for c in matrix.columns]
    return lgb.LGBMRegressor(n_jobs=N_JOBS, **SCALE_PARAMS, monotone_constraints=mono).fit(
        matrix, abs_err, categorical_feature=catc)


def predict_scale(model, matrix):
    """EN: s(x), floored at S_FLOOR. / TR: s(x), S_FLOOR'da alttan kırpılır."""
    return np.clip(model.predict(matrix), S_FLOOR, None)


def nested_scores(X, abs_err, log_pred, comparables, fold):
    """
    EN: The nested scores e / s(x): each fold's listings are scored by an s trained on the other folds' errors only.
        This shows that s never sees the scored fold; the errors themselves come from the main model's own
        cross-fitting. Returns: (scores, s), one value per listing.
    TR: İç içe skorlar e / s(x): her katın ilanlarını yalnız öteki katların hatalarıyla eğitilmiş bir s puanlar.
        Bu, s'nin puanlanan katı görmediğini gösterir; hataların kendisi ana modelin kendi çapraz uydurmasından gelir.
        Döndürür: (skorlar, s), ilan başına bir değer.
    """
    idx, s = np.arange(len(abs_err)), np.full(len(abs_err), np.nan)
    for k in np.unique(fold):
        Xtr, Xva, catc = scale_matrix(X, idx[fold != k], idx[fold == k], log_pred, comparables)
        s[fold == k] = predict_scale(fit_scale(Xtr, abs_err[fold != k], catc), Xva)
    assert not np.isnan(s).any(), "a listing got no s | s almayan ilan var"
    return abs_err / s, s


def p_within(sorted_scores, s, within=WITHIN):
    """
    EN: The hit probability: the share of the (sorted) nested scores at or below log(1 + within) / s(x), i.e. the
        estimated chance that the listing price is within ±within of the prediction (in log terms).
    TR: İsabet olasılığı: (sıralı) iç içe skorlardan log(1 + within) / s(x)'in altında ya da ona eşit olanların payı;
        yani ilan fiyatının tahminin ±within yakınında olma olasılığının kestirimi (log cinsinden).
    """
    cut = np.log1p(within) / np.asarray(s, dtype=float)
    return np.searchsorted(sorted_scores, cut, side="right") / len(sorted_scores)


def interval_score(price, pred_price, r, level=LEVEL):
    """
    EN: Mean interval (Winkler) score in log space: width plus 2/α times the distance by which the price falls
        outside; smaller is better. r: half-width per listing or one number.
    TR: Log uzayında ortalama aralık (Winkler) skoru: genişlik artı fiyatın dışarıda kaldığı uzaklığın 2/α katı;
        küçük daha iyi. r: ilan başına ya da tek sayı yarı genişlik.
    """
    lo, hi = interval(pred_price, r)
    y, lo_l, hi_l, w = np.log1p(price), np.log1p(lo), np.log1p(hi), 2 / (1 - level)
    return float(((hi_l - lo_l) + w * np.clip(lo_l - y, 0, None) + w * np.clip(y - hi_l, 0, None)).mean())


def width_pct(pred_price, r):
    """EN: Interval width as % of the prediction. / TR: Tahminin yüzdesi olarak aralık genişliği."""
    lo, hi = interval(pred_price, r)
    return (hi - lo) / pred_price * 100


def reliability(p_hit, hit):
    """
    EN: Reliability of the hit probability: mean predicted and observed (%), per decile of the prediction; Brier score
        against a constant predictor (the overall hit share); the 10th, 50th and 90th percentile of the prediction; and
        the observed share per value the site shows (floor(10·p), at most SHOWN_MAX).
        Returns: {"mean_predicted", "mean_observed", "deciles": [[predicted, observed] × 10], "brier",
        "brier_constant", "skill_pct", "p10", "p50", "p90", "by_shown": [[shown, n, observed %], ...]}.
    TR: İsabet olasılığının güvenilirliği: ortalama öngörülen ve gerçekleşen (%), öngörünün onda birlik dilimlerinde;
        sabit bir öngörüye (genel isabet payı) karşı Brier skoru; öngörünün 10., 50. ve 90. yüzdeliği; ve sitenin
        gösterdiği her değer için gerçekleşen pay (floor(10·p), en çok SHOWN_MAX).
        Döndürür: {"mean_predicted", "mean_observed", "deciles": [[öngörülen, gerçekleşen] × 10], "brier",
        "brier_constant", "skill_pct", "p10", "p50", "p90", "by_shown": [[gösterilen, n, gerçekleşen %], ...]}.
    """
    h = hit.astype(float)
    dec = pd.qcut(pd.Series(p_hit).rank(method="first"), 10, labels=False).values
    brier, const = float(np.mean((p_hit - h) ** 2)), float(np.mean((h.mean() - h) ** 2))
    shown = np.minimum(np.floor(p_hit * 10 + 1e-9).astype(int), SHOWN_MAX)
    return {"mean_predicted": round(float(p_hit.mean() * 100), 1), "mean_observed": round(float(h.mean() * 100), 1),
            "deciles": [[round(float(p_hit[dec == i].mean() * 100), 1), round(float(h[dec == i].mean() * 100), 1)]
                        for i in range(10)],
            "brier": round(brier, 4), "brier_constant": round(const, 4), "skill_pct": round((1 - brier / const) * 100, 1),
            **{f"p{q}": round(float(np.quantile(p_hit, q / 100) * 100), 1) for q in (10, 50, 90)},
            "by_shown": [[int(v), int((shown == v).sum()), round(float(h[shown == v].mean() * 100), 1)]
                         for v in np.unique(shown)]}
