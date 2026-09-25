"""
08_conformal_coverage.py
EN: Technical report §8 — the 90% conformal interval. q is the 90th percentile of the OOF log errors; the
    interval is expm1(prediction_log ± q). Coverage is measured overall and per price quartile — by the
    actual price and, as a check that the finding does not depend on the grouping, by the predicted price.
    Needs 07_model_comparison's OOF artefact.
TR: Teknik rapor §8 — %90 conformal aralık. q, OOF log hatalarının 90. yüzdeliği; aralık
    expm1(tahmin_log ± q). Kapsama genel ve fiyat çeyreği başına ölçülür — gerçek fiyata göre ve bulgunun
    gruplamaya bağlı olmadığını sınamak için tahmin edilen fiyata göre. 07_model_comparison'ın OOF
    artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/08_conformal_coverage.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd

from lib.common import load_clean, save_metrics
from lib.cv import PRICE_CAP, conformal_q, load_oof

LEVEL = 0.90
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def interval(pred_price, q):
    """
    EN: The conformal interval around each prediction: expm1(log1p(prediction) ± q), the lower end clipped at
        0 and the upper at log1p(15M). Returns: (lower, upper) in ₺.
    TR: Her tahmin etrafındaki conformal aralık: expm1(log1p(tahmin) ± q); alt uç 0'da, üst uç log1p(15M)'de
        kırpılır. Döndürür: (alt, üst) ₺.
    """
    pl = np.log1p(np.clip(pred_price, 0, PRICE_CAP))
    return np.expm1(np.clip(pl - q, 0, None)), np.expm1(np.clip(pl + q, 0, np.log1p(PRICE_CAP)))


def coverage_by_quartile(actual, inside, cut_on):
    """
    EN: Share (%) of listings whose actual price is inside the interval, per quartile of cut_on (the actual
        or the predicted price). Returns: {quartile: coverage %}.
    TR: Gerçek fiyatı aralığın içinde kalan ilanların payı (%), cut_on'un (gerçek ya da tahmin fiyatı)
        çeyreği başına. Döndürür: {çeyrek: kapsama %}.
    """
    qb = pd.qcut(cut_on, 4, labels=QUARTILES, duplicates="drop").astype(str)
    return {b: float(inside[qb == b].mean() * 100) for b in QUARTILES}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.conformal); the report-only numbers go to report.
    TR: Site ağacında yayımlanır (domain.conformal); yalnız raporun kullandığı sayılar report'a gider.
    """
    return {"domain": {"conformal": {
                "by_quantile": [[b, round(res["by_actual"][b], 1)] for b in QUARTILES],
                "coverage_target": round(100 * LEVEL),
                "note": "Q1 (ucuz) under-coverage: model ucuzlarda daha belirsiz (dürüst bulgu)."}},
            "report": {"conformal_q": res["q"], "conformal_all": res["all"],
                       "conformal_by_pred": [[b, res["by_pred"][b]] for b in QUARTILES],
                       "q_bounds": res["bounds"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
pred = np.clip(oof["lgb"].values, 0, PRICE_CAP)
q = conformal_q(np.log1p(price), pred, LEVEL)
lo, hi = interval(pred, q)
inside = (price >= lo) & (price <= hi)
res = {"q": q, "all": float(inside.mean() * 100), "by_actual": coverage_by_quartile(price, inside, price),
       "by_pred": coverage_by_quartile(price, inside, pred),
       "bounds": [float(x) for x in np.quantile(price, [.25, .5, .75])]}
print(f"q {q:.4f} · overall | genel {res['all']:.1f}% ·", {b: round(v, 1) for b, v in res["by_actual"].items()})

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("08_conformal_coverage", to_metrics(res), run_id=oof_info["run_id"]))
