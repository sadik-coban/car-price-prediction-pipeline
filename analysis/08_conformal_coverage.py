"""
08_conformal_coverage.py
EN: Technical report §8 — the 90% conformal interval expm1(log1p(prediction) ± q) and where it holds. Coverage is
    measured cross-calibrated: each of the model's 5 folds gets a q computed only from the other folds' OOF log
    errors (finite-sample quantile), so no listing's own error calibrates its interval. Two arms on the same folds
    (pre-registered, plans/08-mondrian-coverage): one q for the whole market, and one q per predicted-price
    quartile band (Mondrian). Bands are cut on the predicted price — what a pricing tool knows; 2026-09-28 the
    actual-price grouping left the report. Also: coverage and median interval width per band, the band bounds, the
    q of each band on all OOF errors, and the served q (cv.conformal_q, the one 07_final_model ships).
    Needs 07_model_comparison's OOF artefact.
TR: Teknik rapor §8 — %90 conformal aralık expm1(log1p(tahmin) ± q) ve nerede tuttuğu. Kapsama çapraz kalibreli
    ölçülür: modelin 5 katının her biri q'sunu yalnız öteki katların OOF log hatalarından alır (sonlu örneklem
    yüzdeliği); hiçbir ilanın kendi hatası kendi aralığını kalibre etmez. Aynı katlarda iki kol (ön kayıtlı,
    plans/08-mondrian-coverage): bütün piyasa için tek q ve tahmin edilen fiyatın her çeyrek bandı için ayrı q
    (Mondrian). Bantlar tahmin edilen fiyattan kesilir — bir fiyatlama aracının bildiği o; 2026-09-28'de gerçek
    fiyata göre gruplama rapordan çıktı. Ayrıca: bant başına kapsama ve medyan aralık genişliği, bant sınırları,
    her bandın bütün OOF hatalarındaki q'su ve servis edilen q (cv.conformal_q, 07_final_model'in gönderdiği).
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/08_conformal_coverage.json
"""

# %% [1] Setup | Kurulum
import numpy as np

from lib.common import load_clean, save_metrics
from lib.cv import PRICE_CAP, conformal_q, load_oof

LEVEL = 0.90
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
# EN: H1's acceptance band (%) as pre-registered in plans/08-mondrian-coverage (locked 2026-09-28)
# TR: H1'in kabul aralığı (%), plans/08-mondrian-coverage'daki ön kayıttaki gibi (2026-09-28'de kilitlendi)
ACCEPT_BAND = [88, 92]


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def interval(pred_price, q):
    """
    EN: The conformal interval around each prediction: expm1(log1p(prediction) ± q), the lower end clipped at
        0 and the upper at log1p(15M); q may be one number or one per listing. Returns: (lower, upper) in ₺.
    TR: Her tahmin etrafındaki conformal aralık: expm1(log1p(tahmin) ± q); alt uç 0'da, üst uç log1p(15M)'de
        kırpılır; q tek sayı ya da ilan başına olabilir. Döndürür: (alt, üst) ₺.
    """
    pl = np.log1p(np.clip(pred_price, 0, PRICE_CAP))
    return np.expm1(np.clip(pl - q, 0, None)), np.expm1(np.clip(pl + q, 0, np.log1p(PRICE_CAP)))


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


def cross_conformal(abs_err, fold, band, level=LEVEL):
    """
    EN: Cross-calibrated q per listing: for each fold, the global q from the other folds' errors and the band q
        from the other folds' errors in the same band. Returns: (q_global, q_band), one value per listing.
    TR: İlan başına çapraz kalibreli q: her kat için öteki katların hatalarından genel q ve öteki katların aynı
        banttaki hatalarından bant q'su. Döndürür: (q_genel, q_bant), ilan başına bir değer.
    """
    q_global, q_band = np.full(len(abs_err), np.nan), np.full(len(abs_err), np.nan)
    for k in np.unique(fold):
        cal, scored = fold != k, fold == k
        q_global[scored] = conformal_quantile(abs_err[cal], level)
        for b in np.unique(band):
            q_band[scored & (band == b)] = conformal_quantile(abs_err[cal & (band == b)], level)
    assert not (np.isnan(q_global).any() or np.isnan(q_band).any()), "a listing got no q | q almayan ilan var"
    return q_global, q_band


def band_rows(price, pred, band, q_global, q_band):
    """
    EN: Per band: listings, coverage % of both arms, median interval width as % of the prediction and in ₺ for
        both arms. Returns: [[band, n, cov_global, cov_band, width_pct_global, width_pct_band, width_tl_global,
        width_tl_band], ...].
    TR: Bant başına: ilan, iki kolun kapsama %'si, iki kolun medyan aralık genişliği tahminin %'si olarak ve ₺.
        Döndürür: [[bant, n, kapsama_genel, kapsama_bant, genişlik_%_genel, genişlik_%_bant, genişlik_₺_genel,
        genişlik_₺_bant], ...].
    """
    rows = []
    for b, name in enumerate(QUARTILES):
        m = band == b
        out = [name, int(m.sum())]
        cov, wid_pct, wid_tl = [], [], []
        for q in (q_global, q_band):
            lo, hi = interval(pred[m], q[m])
            cov.append(round(float(((price[m] >= lo) & (price[m] <= hi)).mean() * 100), 2))
            wid_pct.append(round(float(np.median((hi - lo) / pred[m] * 100)), 1))
            wid_tl.append(round(float(np.median(hi - lo)), 0))
        rows.append(out + cov + wid_pct + wid_tl)
    return rows


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.conformal: coverage per predicted-price band, both arms); the
        report-only numbers go to report.
    TR: Site ağacında yayımlanır (domain.conformal: tahmin fiyatı bandı başına kapsama, iki kol); yalnız raporun
        kullandığı sayılar report'a gider.
    """
    rows = res["rows"]
    return {"domain": {"conformal": {
                "by_quantile": [[r[0], r[2]] for r in rows], "mondrian_by_quantile": [[r[0], r[3]] for r in rows],
                "coverage_target": round(100 * LEVEL),
                "note": ("Bantlar tahmin edilen fiyatın çeyrekleri; kapsama çapraz kalibreli (her katın q'su öteki "
                         "katlardan). by_quantile = tek q, mondrian_by_quantile = bant başına q.")}},
            "report": {"conformal_q": res["q_served"], "conformal_all": res["all_global"], "q_bounds": res["bounds"],
                       "mondrian": {"by_band": rows, "all": res["all_band"], "q_by_band": res["q_by_band"],
                                    "accept_band": ACCEPT_BAND,
                                    "columns": ["band", "n", "coverage_global", "coverage_band", "width_pct_global",
                                                "width_pct_band", "width_tl_global", "width_tl_band"]}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
pred = np.clip(oof["lgb"].values, 0, PRICE_CAP)
fold = oof["fold"].values
abs_err = np.abs(np.log1p(price) - np.log1p(pred))
bounds = [float(x) for x in np.quantile(pred, [.25, .5, .75])]
band = band_of(pred, bounds)
q_global, q_band = cross_conformal(abs_err, fold, band)
rows = band_rows(price, pred, band, q_global, q_band)
inside = lambda q: (price >= interval(pred, q)[0]) & (price <= interval(pred, q)[1])   # noqa: E731
res = {"rows": rows, "bounds": bounds, "q_served": conformal_q(np.log1p(price), pred, LEVEL),
       "all_global": round(float(inside(q_global).mean() * 100), 2), "all_band": round(float(inside(q_band).mean() * 100), 2),
       "q_by_band": [[name, round(conformal_quantile(abs_err[band == b]), 4)] for b, name in enumerate(QUARTILES)]}
print(f"overall | genel: global {res['all_global']}% · band {res['all_band']}% ·",
      [(r[0], r[2], r[3]) for r in rows])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("08_conformal_coverage", to_metrics(res), run_id=oof_info["run_id"]))
