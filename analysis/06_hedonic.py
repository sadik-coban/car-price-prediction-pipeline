"""
06_hedonic.py
EN: Technical report §6 — the hedonic model: controlled price effects (log price ~ age, mileage, damage, power,
    size), in two columns side by side. Segment control: segment, brand, fuel and transmission dummies. Model
    control: the same plus C(model), so each effect is measured within one model name. Age and mileage are
    centred on the median car, so the linear coefficients are the marginal effects there. Listings of the same
    model are not independent and the error variance is not equal (Breusch-Pagan), so the 95% intervals come from
    standard errors clustered by model (735 models; the 22 series are too few and too uneven to cluster on).
    2026-09-28 (simplification list): this replaces the 1000-replicate row bootstrap, which treated listings as
    independent; the uncentred-VIF table, the dummy VIF, the Jarque-Bera test and the hp–cc table by fuel left the
    report with it. Also: the largest VIF of the segment column's design (centred and uncentred), the overall hp–cc
    correlation, and how many listings the OLS drops for missing hp/cc. No period effect in the model (the owner's
    decision, 2026-09-23).
TR: Teknik rapor §6 — hedonik model: kontrollü fiyat etkileri (log fiyat ~ yaş, km, hasar, güç, hacim), yan yana
    iki sütun. Segment kontrolü: segment, marka, yakıt ve vites kuklaları. Model kontrolü: aynısı ve C(model);
    her etki tek bir model adının içinde ölçülür. Yaş ve km medyan araca ortalanır; doğrusal katsayılar o noktadaki
    marjinal etkidir. Aynı modelin ilanları bağımsız değil ve hata varyansı eşit değil (Breusch-Pagan); bu yüzden
    %95 aralıklar modele göre kümelenmiş standart hatalardan (735 model; 22 seri kümelemek için az ve dengesiz).
    2026-09-28 (sadeleştirme listesi): bu, ilanları bağımsız sayan 1000 tekrarlı satır bootstrap'inin yerini alır;
    ortalanmamış VIF tablosu, kukla VIF'i, Jarque-Bera testi ve yakıta göre hp–cc tablosu onunla rapordan çıktı.
    Ayrıca: segment sütunu tasarımının en yüksek VIF'i (ortalanmış ve ortalanmamış), genel hp–cc korelasyonu ve
    OLS'in eksik hp/cc yüzünden attığı ilan sayısı. Modelde dönem etkisi yok (kullanıcının kararı, 2026-09-23).
Output / Çıktı: metrics/06_hedonic.json
"""

# %% [1] Setup | Kurulum
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from lib.common import load_clean, save_metrics

TERMS = "age+I(age**2)+km10+I(km10**2)+age:km10+dmg+painted+changed+hp100+cc_L"
CONTROLS = "C(segment)+C(brand)+C(kb_fuel)+C(kb_transmission)"
# EN: column id → formula; the model column adds C(model) (it pins brand and almost always segment, so the design
#     is singular; statsmodels solves it with a pseudo-inverse and the linear terms stay identified)
# TR: sütun kimliği → formül; model sütunu C(model) ekler (markayı ve neredeyse her zaman segmenti belirler, tasarım
#     tekil; statsmodels pseudo-inverse ile çözer, doğrusal terimler tanımlı kalır)
COLUMNS = {"segment": f"log_price~{TERMS}+{CONTROLS}", "model": f"log_price~{TERMS}+{CONTROLS}+C(model)"}
# EN: patsy term name → English id the reports index by | TR: patsy terim adı → raporların indekslediği İngilizce kimlik
TERM_ID = {"age": "age", "I(age ** 2)": "age_sq", "km10": "km100k", "I(km10 ** 2)": "km_sq", "age:km10": "age_x_km",
           "dmg": "heavy_damage", "painted": "painted", "changed": "changed", "hp100": "hp100", "cc_L": "litre"}
CLUSTER = "model"


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


def clustered_fit(h, formula, groups):
    """
    EN: OLS with standard errors clustered by `groups` (small-sample corrected, t with G − 1 df). The model
        column's design is singular by construction (see COLUMNS): that one warning is expected and silenced here;
        coefficient_rows stops if a tracked term is not identified.
    TR: Standart hataları `groups`'a göre kümelenmiş OLS (küçük örneklem düzeltmeli, G − 1 serbestlik dereceli t).
        Model sütununun tasarımı yapısı gereği tekil (bkz. COLUMNS): o tek uyarı beklenen, burada susturulur;
        izlenen bir terim tanımsızsa coefficient_rows durur.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularMatrixWarning)
        return smf.ols(formula, data=h).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)


def coefficient_rows(fit):
    """
    EN: For every tracked term: point, clustered SE, 95% CI (log scale) and the same as a % effect, and whether the
        CI contains zero. Stops if a term is missing or not finite (the pseudo-inverse must keep it identified).
    TR: İzlenen her terim için: nokta, kümeli SH, %95 GA (log ölçek), aynısının % etki hâli ve GA'nın sıfırı içerip
        içermediği. Bir terim eksik ya da sonlu değilse durur (pseudo-inverse onu tanımlı tutmalı).
    """
    ci = fit.conf_int(0.05)
    rows = []
    for k, tid in TERM_ID.items():
        b, se, lo, hi = fit.params[k], fit.bse[k], ci.loc[k, 0], ci.loc[k, 1]
        assert np.isfinite([b, se, lo, hi]).all() and se > 0, f"{tid}: not identified | tanımsız"
        rows.append({"term": tid, "point": round(float(b), 4), "se": round(float(se), 4), "ci_lo": round(float(lo), 4),
                     "ci_hi": round(float(hi), 4), "pct_effect": round(pct(b), 2), "pct_lo": round(pct(lo), 2),
                     "pct_hi": round(pct(hi), 2), "contains_zero": bool(lo <= 0 <= hi)})
    return rows


def vif_max(h, fit, age_c, km_c):
    """
    EN: The largest VIF among the tracked terms, on the fitted (centred) design and on the same design uncentred.
        Stops if the VIF input is not the fitted model's own design. Returns: {term, value, raw_term, raw_value}.
    TR: İzlenen terimler arasında en yüksek VIF; kurulan (ortalanmış) tasarımda ve aynı tasarımın ortalanmamış
        hâlinde. VIF girdisi kurulan modelin kendi tasarımı değilse durur. Döndürür: {term, value, raw_term, raw_value}.
    """
    x_fit = pd.DataFrame(fit.model.exog, columns=fit.model.exog_names)
    assert x_fit.shape[1] == len(fit.params), "VIF input is not the fitted design"
    raw = h.copy()
    raw["age"], raw["km10"] = raw["age"] + age_c, raw["km10"] + km_c
    x_raw = patsy.dmatrix(COLUMNS["segment"].split("~", 1)[1], raw, return_type="dataframe")
    assert list(x_raw.columns) == list(x_fit.columns), "uncentred design has different columns"
    v_fit = {k: float(variance_inflation_factor(x_fit.values, x_fit.columns.get_loc(k))) for k in TERM_ID}
    v_raw = {k: float(variance_inflation_factor(x_raw.values, x_raw.columns.get_loc(k))) for k in TERM_ID}
    top, top_raw = max(v_fit, key=v_fit.get), max(v_raw, key=v_raw.get)
    return {"term": TERM_ID[top], "value": round(v_fit[top], 2), "raw_term": TERM_ID[top_raw],
            "raw_value": round(v_raw[top_raw], 2)}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.hedonic: the model column's headline effects, which the decision note
        prints; domain.hedonic_reliability: both columns) and the report inputs (error_drivers.hedonic_dropped).
    TR: Site ağacında (domain.hedonic: model sütununun manşet etkileri, karar notunun bastıkları;
        domain.hedonic_reliability: iki sütun) ve rapor girdilerinde (error_drivers.hedonic_dropped) yayımlanır.
    """
    fits, rows = res["fits"], res["rows"]
    head = {r["term"]: r["pct_effect"] for r in rows["model"]}
    return {
        "domain": {
            "hedonic": {"r2": round(float(fits["model"].rsquared), 4), "control": "model", "age_pct": head["age"],
                        "damage_pct": head["heavy_damage"], "km100k_pct": head["km100k"], "hp100_pct": head["hp100"],
                        "cc_litre_pct": head["litre"],
                        "note": ("Model kontrolü sütunu: etkiler aynı model adı içinde. Dönem etkisi modelde yok "
                                 "(kullanıcı kararı): dönemler havuzlanarak kestirildi.")},
            "hedonic_reliability": {
                "n": int(fits["segment"].nobs), "cluster": CLUSTER, "n_clusters": res["n_clusters"],
                "center": {"age": res["age_c"], "km": res["km_c"] * 1e5},
                "columns": {c: {"r2": round(float(fits[c].rsquared), 4), "coefficients": rows[c]} for c in COLUMNS},
                "vif": res["vif"], "hp_cc_correlation": res["hp_cc_r"], "homoskedasticity_p": res["bp_p"]}},
        "error_drivers": {"hedonic_dropped": res["dropped"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
h, age_c, km_c = hedonic_frame(listings)
dropped = dropped_rows(listings)
groups = pd.factorize(h[CLUSTER])[0]
fits = {c: clustered_fit(h, f, groups) for c, f in COLUMNS.items()}
assert all(len(listings) - dropped["total"] == int(f.nobs) for f in fits.values()), "OLS row count ≠ dropped count"
bp = het_breuschpagan(fits["segment"].resid, fits["segment"].model.exog)
res = {"fits": fits, "rows": {c: coefficient_rows(f) for c, f in fits.items()}, "n_clusters": int(groups.max() + 1),
       "age_c": age_c, "km_c": km_c, "vif": vif_max(h, fits["segment"], age_c, km_c),
       "hp_cc_r": round(float(stats.pearsonr(h.hp, h.cc)[0]), 3), "bp_p": float(bp[1]), "dropped": dropped}
for c in COLUMNS:
    print(c, f"R² {fits[c].rsquared:.4f}", [(r["term"], r["pct_effect"], r["pct_lo"], r["pct_hi"]) for r in res["rows"][c]])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("06_hedonic", to_metrics(res)))
