"""
06_hedonic.py
EN: Technical report §6 — the hedonic model: controlled price effects (log price ~ age, mileage, damage,
    power, size + segment/brand/fuel/transmission). Age and mileage are centred on the median car, so the
    linear coefficients are the marginal effects there. Confidence intervals come from a 1000-replicate row
    bootstrap. Also: VIF (fitted, uncentred, dummies), hp–cc correlation by fuel, assumption tests, the same
    model with model identity added, and how many listings the OLS drops for missing hp/cc.
    No period effect in the model (the owner's decision).
TR: Teknik rapor §6 — hedonik model: kontrollü fiyat etkileri (log fiyat ~ yaş, km, hasar, güç, hacim +
    segment/marka/yakıt/vites). Yaş ve km medyan araca ortalanır; doğrusal katsayılar o noktadaki marjinal
    etkidir. Güven aralıkları 1000 tekrarlı satır bootstrap'inden. Ayrıca: VIF (kurulan, ortalanmamış, kukla),
    yakıta göre hp–cc korelasyonu, varsayım testleri, model kimliği eklenmiş aynı model ve OLS'in eksik hp/cc
    yüzünden attığı ilan sayısı. Modelde dönem etkisi yok (kullanıcının kararı).
Output / Çıktı: metrics/06_hedonic.json
"""

# %% [1] Setup | Kurulum
import os
import time

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from lib.common import load_clean, save_metrics

SEED = 42
N_BOOT = 1000
N_JOBS = int(os.environ.get("N_JOBS") or (os.cpu_count() or 4))
FORMULA = ("log_price~age+I(age**2)+km10+I(km10**2)+age:km10+dmg+painted+changed"
           "+hp100+cc_L+C(segment)+C(brand)+C(kb_fuel)+C(kb_transmission)")
TRACKED = ["age", "I(age ** 2)", "km10", "I(km10 ** 2)", "age:km10", "dmg", "painted", "changed", "hp100", "cc_L"]
LABELS = {"age": "yaş", "I(age ** 2)": "yaş²", "km10": "km(100K)", "I(km10 ** 2)": "km²", "age:km10": "yaş×km",
          "dmg": "ağır hasar", "painted": "boyalı", "changed": "değişen", "hp100": "+100 HP", "cc_L": "+1 litre"}
MODEL_ID_TERMS = {"age": "age", "km10": "km100k", "hp100": "hp100", "cc_L": "cc_litre", "dmg": "heavy_damage",
                  "painted": "painted", "changed": "changed"}


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def pct(coef):
    """
    EN: A log coefficient as a percentage effect on price: 100·(e^c − 1).
    TR: Bir log katsayısının fiyattaki yüzde etkisi: 100·(e^c − 1).
    """
    return 100 * (np.exp(coef) - 1)


def hedonic_frame(listings):
    """
    EN: The regression table: log price, age, mileage in 100k km, damage counts, hp/100, litres; rows with
        missing or non-positive hp/cc dropped; age and mileage centred on their medians.
        Returns: (frame, age centre, mileage centre in 100k km).
    TR: Regresyon tablosu: log fiyat, yaş, 100 bin km cinsinden km, hasar sayıları, hp/100, litre; hp/cc'si
        eksik ya da pozitif olmayan satırlar atılır; yaş ve km medyanlarına ortalanır.
        Döndürür: (tablo, yaş merkezi, 100 bin km cinsinden km merkezi).
    """
    h = listings.copy()
    h["log_price"] = np.log1p(h["price"])
    h["age"], h["km10"], h["dmg"] = h["vehicle_age"], h["gb_mileage"] / 1e5, h["is_heavy_damaged"]
    h["painted"] = pd.to_numeric(h["count_painted"], errors="coerce").fillna(0)
    h["changed"] = pd.to_numeric(h["count_changed"], errors="coerce").fillna(0)
    h["hp"] = pd.to_numeric(h["power_hp_val"], errors="coerce")
    h["cc"] = pd.to_numeric(h["engine_cc_val"], errors="coerce")
    h["hp100"], h["cc_L"] = h["hp"] / 100, h["cc"] / 1000
    h = h.dropna(subset=["log_price", "age", "km10", "segment", "hp", "cc"])
    h = h[(h["hp"] > 0) & (h["cc"] > 0)].reset_index(drop=True)
    age_c, km_c = float(h["age"].median()), float(h["km10"].median())
    h["age"], h["km10"] = h["age"] - age_c, h["km10"] - km_c
    return h, age_c, km_c


def dropped_rows(listings):
    """
    EN: How many listings the OLS drops, and why (hp missing/≤0, cc missing/≤0, both, total).
    TR: OLS'in kaç ilanı attığı ve neden (hp eksik/≤0, cc eksik/≤0, ikisi birden, toplam).
    """
    hp = listings[["power_hp_low", "power_hp_up"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    cc = pd.to_numeric(listings["engine_cc_up"], errors="coerce")
    bad_hp, bad_cc = hp.isna() | (hp <= 0), cc.isna() | (cc <= 0)
    return {"hp": int(bad_hp.sum()), "cc": int(bad_cc.sum()), "both": int((bad_hp & bad_cc).sum()),
            "total": int((bad_hp | bad_cc).sum())}


def one_bootstrap(h, seed):
    """
    EN: One bootstrap replicate: OLS on a resample of rows. Returns: {term: coefficient} or None if it fails.
    TR: Tek bir bootstrap tekrarı: yeniden örneklenmiş satırlarda OLS. Döndürür: {terim: katsayı}, başarısızsa None.
    """
    idx = np.random.default_rng(seed).integers(0, len(h), len(h))
    try:
        fit = smf.ols(FORMULA, data=h.iloc[idx]).fit()
        return {k: fit.params.get(k, np.nan) for k in TRACKED}
    except Exception:
        return None


def bootstrap_table(h, fit, n_boot, n_jobs, seed):
    """
    EN: 1000 row-bootstrap replicates in parallel; stops if any replicate fails. For each tracked term:
        point estimate, bootstrap mean, 2.5–97.5 percentiles, std, % effect, whether the CI contains zero.
        Returns: (table, number of replicates, seconds).
    TR: Paralel 1000 satır-bootstrap tekrarı; biri başarısız olursa durur. Her izlenen terim için: nokta
        tahmin, bootstrap ortalaması, %2.5–97.5 yüzdelikleri, std, yüzde etki, GA'nın sıfırı içerip içermediği.
        Döndürür: (tablo, tekrar sayısı, saniye).
    """
    seeds = np.random.default_rng(seed).integers(0, 2**31, n_boot)
    t0 = time.time()
    reps = Parallel(n_jobs=n_jobs)(delayed(one_bootstrap)(h, s) for s in seeds)
    seconds = time.time() - t0
    reps = [r for r in reps if r is not None]
    assert len(reps) == n_boot, f"bootstrap: {n_boot - len(reps)} replicates failed"
    table = []
    for k in TRACKED:
        arr = np.array([r[k] for r in reps])
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        table.append({"term": LABELS[k], "point": round(float(fit.params[k]), 4), "boot_mean": round(float(arr.mean()), 4),
                      "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "std": round(float(arr.std()), 4),
                      "pct_effect": round(pct(fit.params[k]), 2), "contains_zero": bool(lo <= 0 <= hi)})
    return table, len(reps), seconds


def vif_tables(h, fit, age_c, km_c):
    """
    EN: VIF from the fitted (centred) design, from the same design uncentred, and the largest dummy VIF.
        Stops if the VIF input is not the fitted model's own design.
    TR: VIF: kurulan (ortalanmış) tasarımdan, aynı tasarımın ortalanmamış hâlinden ve en büyük kukla VIF'i.
        VIF girdisi kurulan modelin kendi tasarımı değilse durur.
    """
    def vif_of(x):
        """EN: VIF of every column except the intercept. / TR: Sabit terim hariç her kolonun VIF'i."""
        return {c: float(variance_inflation_factor(x.values, i)) for i, c in enumerate(x.columns) if c != "Intercept"}
    x_fit = pd.DataFrame(fit.model.exog, columns=fit.model.exog_names)
    assert x_fit.shape[1] == len(fit.params), "VIF input is not the fitted design"
    raw = h.copy()
    raw["age"], raw["km10"] = raw["age"] + age_c, raw["km10"] + km_c
    x_raw = patsy.dmatrix(FORMULA.split("~", 1)[1], raw, return_type="dataframe")
    assert list(x_raw.columns) == list(x_fit.columns), "uncentred design has different columns"
    v_fit, v_raw = vif_of(x_fit), vif_of(x_raw)
    dummies = {k: v for k, v in v_fit.items() if k.startswith("C(")}
    worst = max(dummies, key=dummies.get)
    return ([[t, round(v_fit[t], 2)] for t in TRACKED], [[t, round(v_raw[t], 2)] for t in TRACKED],
            [worst, round(dummies[worst], 2)])


def fuel_correlation(h, min_n=30):
    """
    EN: hp–cc correlation within each fuel type (Pearson, Pearson on logs, Spearman, median cc/hp).
    TR: Her yakıt türü içinde hp–cc korelasyonu (Pearson, log'larda Pearson, Spearman, medyan cc/hp).
    """
    out = []
    for fuel in h["kb_fuel"].value_counts().index:
        sub = h[h["kb_fuel"] == fuel]
        if len(sub) >= min_n:
            out.append({"fuel": str(fuel)[:20], "pearson": round(float(stats.pearsonr(sub.hp, sub.cc)[0]), 3),
                        "pearson_log": round(float(stats.pearsonr(np.log(sub.hp), np.log(sub.cc))[0]), 3),
                        "spearman": round(float(stats.spearmanr(sub.hp, sub.cc)[0]), 3),
                        "cc_hp_ratio": round(float((sub.cc / sub.hp).median()), 1), "n": int(len(sub))})
    return out


def model_identity_fit(h, fit):
    """
    EN: The same OLS with C(model) added (fitted once, no bootstrap): R² and the linear coefficients with
        and without model identity. The design is singular (model fixes brand/segment); statsmodels solves it
        with a pseudo-inverse and the linear terms stay identified.
    TR: C(model) eklenmiş aynı OLS (bir kez, bootstrap yok): R² ve model kimliği olmadan/ile doğrusal katsayılar.
        Tasarım tekil (model markayı/segmenti belirliyor); statsmodels pseudo-inverse ile çözer, doğrusal
        terimler tanımlı kalır.
    """
    fit_m = smf.ols(FORMULA + "+C(model)", data=h).fit()
    return {"r2": round(float(fit_m.rsquared), 4), "n_model": int(h["model"].nunique()),
            "coef_pct": {name: [round(pct(fit.params[k]), 2), round(pct(fit_m.params[k]), 2)]
                            for k, name in MODEL_ID_TERMS.items()}}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.hedonic, domain.hedonic_reliability) and the report inputs
        (error_drivers.hedonic_dropped).
    TR: Site ağacında (domain.hedonic, domain.hedonic_reliability) ve rapor girdilerinde
        (error_drivers.hedonic_dropped) yayımlanır.
    """
    fit, p = res["fit"], res["fit"].params
    return {
        "domain": {
            "hedonic": {"r2": round(float(fit.rsquared), 4), "age_pct": round(pct(p["age"]), 2),
                        "damage_pct": round(pct(p["dmg"]), 2), "km100k_pct": round(pct(p["km10"]), 2),
                        "hp100_pct": round(pct(p["hp100"]), 2), "cc_litre_pct": round(pct(p["cc_L"]), 2),
                        "note": "Dönem etkisi modelde yok (kullanıcı kararı): dönemler havuzlanarak kestirildi."},
            "hedonic_reliability": {
                "model_r2": round(float(fit.rsquared), 4), "n": int(fit.nobs), "unit": {"hp": "100 HP", "cc": "1 litre"},
                "engine_effect": {"hp100_pct": round(pct(p["hp100"]), 1), "cc_litre_pct": round(pct(p["cc_L"]), 1)},
                "bootstrap": res["boot"], "vif": res["vif"], "vif_raw": res["vif_raw"], "vif_dummy": res["vif_dummy"],
                "center": {"age": res["age_c"], "km": res["km_c"] * 1e5},
                "fuel_correlation": res["fuel"], "overall_correlation": res["hp_cc_r"],
                "assumptions": {"homoskedasticity_p": res["bp_p"], "normality_p": res["jb_p"],
                             "note": ("Breusch-Pagan ve Jarque-Bera ihlal (p<0.05). Bu yüzden çıkarım çıplak OLS p-değerine "
                                     "değil, satır bootstrap'inin %2.5–97.5 yüzdeliklerine dayanıyor.")},
                "bootstrap_setup": {"n_boot": res["n_boot"], "n_requested": N_BOOT, "n_jobs": N_JOBS,
                                    "seconds": round(res["seconds"], 1)},
                "note": ("Ham (log yok) cc+HP: birim başına yorum. Yaş ve km medyan araca ortalandı; doğrusal "
                        "katsayılar o noktadaki marjinal etki. Güven aralıkları satır bootstrap'inden."),
                "with_model": {**res["model_id"],
                               "note": "C(model) eklenmiş tek OLS (bootstrap yok); katsayi_pct = [model yok, model var], yüzde etki."}}},
        "error_drivers": {"hedonic_dropped": res["dropped"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
h, age_c, km_c = hedonic_frame(listings)
fit = smf.ols(FORMULA, data=h).fit(cov_type="HC3")
dropped = dropped_rows(listings)
assert len(listings) - dropped["total"] == int(fit.nobs), "OLS row count does not match the dropped count"
# EN: one BLAS thread per bootstrap worker (avoids oversubscription) | TR: işçi başına tek BLAS iş parçacığı
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
boot, n_boot, seconds = bootstrap_table(h, fit, N_BOOT, N_JOBS, SEED)
vif, vif_raw, vif_dummy = vif_tables(h, fit, age_c, km_c)
bp, jb = het_breuschpagan(fit.resid, fit.model.exog), stats.jarque_bera(fit.resid)
res = {"fit": fit, "boot": boot, "n_boot": n_boot, "seconds": seconds, "vif": vif, "vif_raw": vif_raw,
       "vif_dummy": vif_dummy, "age_c": age_c, "km_c": km_c, "fuel": fuel_correlation(h),
       "hp_cc_r": round(float(stats.pearsonr(h.hp, h.cc)[0]), 3), "bp_p": float(bp[1]), "jb_p": float(jb[1]),
       "model_id": model_identity_fit(h, fit), "dropped": dropped}
print(f"R² {fit.rsquared:.4f} | with model | model ile {res['model_id']['r2']} | bootstrap {n_boot} in {seconds:.0f}s")

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("06_hedonic", to_metrics(res)))
