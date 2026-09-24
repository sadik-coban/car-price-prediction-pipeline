"""
08_residuals.py
EN: Technical report §8 — how the OOF errors are distributed. Median % error per price quartile; the same in
    lira (where the lira error sits, and the mean bias by actual vs predicted quartile — grouping by the actual
    price produces regression to the mean); the calibration slope (actual ~ a + b·predicted); error bands;
    the best and worst predictions; median error per model against its number of listings; and the points of
    the predicted-vs-actual and residual figures. Needs 07_model_comparison's OOF artefact.
TR: Teknik rapor §8 — OOF hataları nasıl dağılıyor. Fiyat çeyreği başına medyan % hata; aynısı lira olarak
    (lira hatası nerede toplanıyor, gerçek ve tahmin çeyreğine göre ortalama sapma — gerçek fiyata göre
    gruplamak ortalamaya dönüş üretir); kalibrasyon eğimi (gerçek ~ a + b·tahmin); hata bantları; en iyi ve en
    kötü tahminler; model başına medyan hatanın ilan sayısına karşı dağılımı; tahmin-gerçek ve artık
    figürlerinin noktaları. 07_model_comparison'ın OOF artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/08_residuals.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from lib.common import load_clean, save_metrics
from lib.cv import PRICE_CAP, load_oof, residual_pct

QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
APE_CAP = 500                        # % error cap against overflow at tiny prices | küçük fiyatta taşma tavanı
BAND_EDGES = [0, 5, 10, 20, None]
SYM_RATIO = 1.2                      # symmetric band: one side >1.2× the other | bir taraf ötekinin 1.2 katı
MIN_MODEL_N = 5


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def ape_values(price, pred):
    """
    EN: Absolute % error per listing, the denominator floored at ₺10k and the result capped at 500%.
    TR: İlan başına mutlak % hata; payda ₺10 bin altına inmez, sonuç %500'de kesilir.
    """
    ape = np.abs((price - pred) / np.maximum(price, 1e4)) * 100
    return np.minimum(np.where(np.isfinite(ape), ape, np.nan), APE_CAP)


def quartile_median_ape(price, ape):
    """
    EN: Median % error per actual-price quartile. Returns: [[quartile, median %], ...].
    TR: Gerçek fiyat çeyreği başına medyan % hata. Döndürür: [[çeyrek, medyan %], ...].
    """
    qb = pd.qcut(price, 4, labels=QUARTILES, duplicates="drop").astype(str)
    return [[b, float(np.nanmedian(ape[qb == b]))] for b in QUARTILES]


def listing_rows(listings, pred, ape, n, largest):
    """
    EN: The n listings with the largest (or smallest) % error: model, age, km, actual, prediction, % error.
    TR: % hatası en büyük (ya da en küçük) n ilan: model, yaş, km, gerçek, tahmin, % hata.
    """
    d = listings.reset_index(drop=True).assign(pred=pred, ape=ape).dropna(subset=["ape"])
    d = d.nlargest(n, "ape") if largest else d.nsmallest(n, "ape")
    return [[str(r["model"])[:35], float(r["vehicle_age"]), float(r["gb_mileage"]), float(r["price"]), float(r["pred"]),
             float(r["ape"])] for _, r in d.iterrows()]


def error_by_model(listings, ape, min_n):
    """
    EN: Per model with at least min_n listings: listings, median % error, median price.
    TR: En az min_n ilanı olan her model için: ilan, medyan % hata, medyan fiyat.
    """
    d = listings.reset_index(drop=True).assign(ape=ape).dropna(subset=["ape"])
    ms = d.groupby("model").agg(n=("ape", "size"), med=("ape", "median"), price=("price", "median")).reset_index()
    return [[int(r.n), float(r.med), float(r.price)] for _, r in ms[ms.n >= min_n].iterrows()]


def lira_by_quartile(price, pred, ape):
    """
    EN: Per actual-price quartile: listings, share of total lira error (%), mean |error| ₺, mean bias
        (prediction − actual) ₺, median % error.
    TR: Gerçek fiyat çeyreği başına: ilan, toplam lira hatasındaki pay (%), ortalama |hata| ₺, ortalama sapma
        (tahmin − gerçek) ₺, medyan % hata.
    """
    dev = pred - price
    qb = pd.qcut(price, 4, labels=QUARTILES).astype(str)
    tot = float(np.abs(dev).sum())
    return [[b, int((qb == b).sum()), 100 * float(np.abs(dev[qb == b]).sum()) / tot, float(np.abs(dev[qb == b]).mean()),
             float(dev[qb == b].mean()), float(np.median(ape[qb == b]))] for b in QUARTILES]


def bias_by_predicted_quartile(price, pred):
    """
    EN: Per predicted-price quartile (what a pricing tool knows): listings, mean and median bias ₺.
    TR: Tahmin fiyatı çeyreği başına (bir fiyatlama aracının bildiği): ilan, ortalama ve medyan sapma ₺.
    """
    dev = pred - price
    qp = pd.qcut(pred, 4, labels=QUARTILES).astype(str)
    return [[b, int((qp == b).sum()), float(dev[qp == b].mean()), float(np.median(dev[qp == b]))] for b in QUARTILES]


def error_bands(price, pred):
    """
    EN: The OOF error distribution: listings per |error| band (0–5, 5–10, 10–20, >20 %), median and mean
        error, over/under-predictions beyond 20%, the symmetric 1.2× count, extremes and the share within 10%.
    TR: OOF hata dağılımı: |hata| bandı başına ilan (0–5, 5–10, 10–20, >%20), medyan ve ortalama hata, %20'yi
        aşan fazla/düşük tahminler, simetrik 1.2× sayısı, uçlar ve %10 içindeki pay.
    """
    r = (price - pred) / price * 100
    a = np.abs(r)
    bands = []
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        m = (a > lo) if hi is None else ((a <= hi) if lo == 0 else ((a > lo) & (a <= hi)))
        bands.append([lo, hi, int(m.sum()), float(m.mean() * 100)])
    return {"err_n": int(len(r)), "err_bands": bands, "err_median": float(np.median(r)),
            "err_abs_median": float(np.median(a)), "err_abs_mean": float(a.mean()),
            "err_over20": int((r < -20).sum()), "err_under20": int((r > 20).sum()),
            "err_sym_over": int((pred / price > SYM_RATIO).sum()), "err_sym_under": int((price / pred > SYM_RATIO).sum()),
            "err_min": float(r.min()), "err_max": float(r.max()), "err_in10": float((a <= 10).mean() * 100)}


def log_r2(price, pred):
    """
    EN: OOF R² on log1p(price) — comparable with the hedonic model's R² (the headline R² is in ₺).
    TR: log1p(fiyat) üzerinde OOF R² — hedonik modelin R²'siyle karşılaştırılabilir (manşet R² ₺ cinsinden).
    """
    ly, lp = np.log1p(price), np.log1p(pred)
    return 1 - float(((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain: quantile_error, oof_outliers, oof_best, residual_vs_n,
        pred_vs_true, residual_scatter), the report inputs (error_drivers: lira_ceyrek, tahmin_ceyrek,
        kalibrasyon) and report-only numbers (report).
    TR: Site ağacında (domain: quantile_error, oof_outliers, oof_best, residual_vs_n, pred_vs_true,
        residual_scatter), rapor girdilerinde (error_drivers: lira_ceyrek, tahmin_ceyrek, kalibrasyon) ve
        yalnız rapor sayılarında (report) yayımlanır.
    """
    price, pred, resid = res["price"], res["pred"], res["resid"]
    row = lambda r: r[:5] + [round(r[5], 1)]                              # noqa: E731
    snap_med = res["snap_median"]
    return {
        "domain": {
            "quantile_error": [[b, round(v, 2)] for b, v in res["quartile_ape"]],
            "oof_outliers": [row(r) for r in res["worst"]], "oof_best": [row(r) for r in res["best"]],
            "residual_vs_n": [[n, round(m, 2), p] for n, m, p in res["by_model"]],
            "pred_vs_true": {
                "points": [[float(a), float(b)] for a, b in zip(price, pred)],
                "ideal_line": [float(min(price.min(), pred.min())), float(max(price.max(), pred.max()))],
                "r2": round(float(r2_score(price, pred)), 4), "n": int(len(price)),
                "not": ("OOF tahmin (sızıntısız), TÜM noktalar. Frontend: yoğunluk/hexbin veya "
                        "düşük-opacity ile render (30K nokta ham scatter'da okunmaz).")},
            "residual_scatter": {
                "points": [[float(b), round(float(r), 2)] for b, r in zip(pred, resid)],
                "mean_resid_pct": round(float(np.mean(resid)), 2), "std_resid_pct": round(float(np.std(resid)), 2),
                "n": int(len(price)),
                "not": (f"Artık% = (gerçek-tahmin)/gerçek, TÜM noktalar. Ortalama sıfıra yakın, ama medyan artık "
                        f"dönemlere göre %{min(snap_med):+.2f} ile %{max(snap_med):+.2f} arasında kayıyor (model "
                        f"zamanı görmüyor). Fiyata göre yanlılık tahmin edilen fiyata göre gruplanarak ölçülmeli; "
                        f"gerçek fiyata göre gruplamak ortalamaya dönüş üretir. Frontend: yoğunluk/hexbin render önerilir.")}},
        "error_drivers": {
            "lira_ceyrek": [[b, n, round(s, 1), round(m, 0), round(d, 0), round(a, 2)] for b, n, s, m, d, a in res["lira"]],
            "tahmin_ceyrek": [[b, n, round(m, 0), round(md, 0)] for b, n, m, md in res["bias_pred"]],
            "kalibrasyon": {"egim": round(res["slope"], 4), "kesisim": round(res["intercept"], 0)}},
        "report": {**res["bands"], "model_r2_log": round(res["r2_log"], 4)},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
pred = np.clip(oof["lgb"].values, 0, PRICE_CAP)
ape = ape_values(price, pred)
resid = residual_pct(price, oof["lgb"].values)
slope, intercept = np.polyfit(pred, price, 1)
res = {"price": price, "pred": pred, "resid": resid, "quartile_ape": quartile_median_ape(price, ape),
       "worst": listing_rows(listings, pred, ape, 10, largest=True), "best": listing_rows(listings, pred, ape, 5, largest=False),
       "by_model": error_by_model(listings, ape, MIN_MODEL_N),
       "snap_median": pd.Series(resid).groupby(listings["snap"].values).median().tolist(),
       "lira": lira_by_quartile(price, pred, np.minimum(np.abs(resid), APE_CAP)),
       "bias_pred": bias_by_predicted_quartile(price, pred), "slope": float(slope), "intercept": float(intercept),
       "bands": error_bands(price, pred), "r2_log": log_r2(price, pred)}
print("median % error by quartile | çeyreğe göre medyan % hata:", [(b, round(v, 2)) for b, v in res["quartile_ape"]])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("08_residuals", to_metrics(res), run_id=oof_info["run_id"]))
