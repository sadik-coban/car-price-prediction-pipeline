"""
07_text_flag.py
EN: Technical report §7 — do listings whose own text talks about a conversion or a modification go wrong
    more often? The raw rate difference is confounded with the car (almost half of the performance families
    carry such wording), so a logistic regression controls for age band, log mileage, price quartile,
    performance family, log number of comparables and brand; the odds ratio and its 95% CI are published.
    Detectors: analysis/lib/text_flags.py (pattern matching, no LLM). Needs 07_model_comparison's OOF artefact.
TR: Teknik rapor §7 — kendi metninde dönüşüm ya da modifiye geçen ilanlarda büyük hata daha sık mı? Ham oran
    farkı araçla karışık (performans ailelerinin neredeyse yarısında bu ifade geçiyor); bu yüzden lojistik
    regresyon yaş bandı, log km, fiyat çeyreği, performans ailesi, log emsal sayısı ve markayı kontrol eder;
    olasılık oranı ve %95 GA yayımlanır. Dedektörler: analysis/lib/text_flags.py (desen eşleme, LLM yok).
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/07_text_flag.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from lib import segment_rule as SR
from lib import text_flags as TF
from lib.common import load_clean, save_metrics
from lib.cv import large_errors, load_oof, residual_pct

AGE_BANDS = [-1, 5, 10, 15, 18, 100]
FORMULA = "big ~ flag + C(age_band) + log_km + C(price_q) + perf + log_comps + C(brand)"


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def text_flag(listings):
    """
    EN: Conversion or modification wording in the listing's own text. Returns: bool array.
    TR: İlanın kendi metninde dönüşüm ya da modifiye ifadesi. Döndürür: bool dizi.
    """
    desc = TF.descriptions(listings)
    return TF.conversion_flag(desc) | TF.modification_flag(desc)


def performance_family(listings):
    """
    EN: Listings whose segment was resolved outside the plain series map (M/S/RS lines, i8, Z4 M).
    TR: Segmenti düz seri haritası dışında çözülen ilanlar (M/S/RS serileri, i8, Z4 M).
    """
    return np.array([SR.resolve(s, m)[1] != "harita" for s, m in zip(listings["series"], listings["model"])])


def controlled_odds(listings, big, flag, perf):
    """
    EN: Logistic regression of large error on the text flag with controls. Returns: odds ratio, 95% CI, p.
    TR: Büyük hatanın metin bayrağına kontrollerle lojistik regresyonu. Döndürür: olasılık oranı, %95 GA, p.
    """
    age = pd.to_numeric(listings["vehicle_age"], errors="coerce").values.astype(float)
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    price = listings["price"].astype(float).values
    d = pd.DataFrame({"big": big.astype(int), "flag": flag.astype(int),
                      "age_band": pd.cut(age, AGE_BANDS).astype(str),
                      "log_km": np.log1p(np.nan_to_num(km, nan=np.nanmedian(km))),
                      "price_q": pd.qcut(price, 4, labels=False).astype(str), "perf": perf.astype(int),
                      "log_comps": np.log(listings.groupby("model")["model"].transform("size").values),
                      "brand": listings["brand"].astype(str).values})
    fit = smf.logit(FORMULA, data=d).fit(disp=0)
    ci = fit.conf_int().loc["flag"]
    return {"or": float(np.exp(fit.params["flag"])), "ci_lo": float(np.exp(ci[0])), "ci_hi": float(np.exp(ci[1])),
            "p": float(fit.pvalues["flag"])}


def flag_rates(resid, big, flag, perf):
    """
    EN: Size of the flagged group, its large-error rate and median |error| against the rest, and the flag's
        share inside the performance families.
    TR: Bayraklı grubun büyüklüğü, büyük hata oranı ve medyan |hata|'sı ile geri kalanınki, ve bayrağın
        performans aileleri içindeki payı.
    """
    return {"n": int(flag.sum()), "share": float(flag.mean()), "perf_share": float(flag[perf].mean()),
            "big": float(big[flag].mean()) if flag.any() else None, "other_big": float(big[~flag].mean()),
            "median_err": float(np.median(np.abs(resid[flag]))) if flag.any() else None,
            "other_median_err": float(np.median(np.abs(resid[~flag])))}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the report inputs (error_drivers.text_flag).
    TR: Rapor girdilerinde yayımlanır (error_drivers.text_flag).
    """
    o, r = res["odds"], res["rates"]
    pct = lambda v, d: round(100 * v, d) if v is not None else None      # noqa: E731
    return {"error_drivers": {"text_flag": {
        "controlled": {"or": round(o["or"], 2), "ci_lo": round(o["ci_lo"], 2), "ci_hi": round(o["ci_hi"], 2),
                       "p": round(o["p"], 3), "perf_share_pct": pct(r["perf_share"], 1),
                       "controls": "yas bandi, log km, fiyat ceyregi, performans ailesi, log emsal, marka"},
        "n": r["n"], "pct": pct(r["share"], 2), "big_pct": pct(r["big"], 1), "other_big_pct": pct(r["other_big"], 1),
        "median_error_pct": round(r["median_err"], 1) if r["median_err"] is not None else None,
        "other_median_error_pct": round(r["other_median_err"], 1)}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
resid = residual_pct(listings["price"].values.astype(float), oof["lgb"].values)
over, under = large_errors(resid)
big = over | under
flag, perf = text_flag(listings), performance_family(listings)
res = {"odds": controlled_odds(listings, big, flag, perf), "rates": flag_rates(resid, big, flag, perf)}
print("flagged | bayraklı:", res["rates"]["n"], "· OR", round(res["odds"]["or"], 2))

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("07_text_flag", to_metrics(res), run_id=oof_info["run_id"]))
