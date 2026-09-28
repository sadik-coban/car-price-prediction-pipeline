"""
06_hedonic.py
EN: Technical report §6 — the hedonic model: controlled price effects (log price ~ age, mileage, damage, power,
    size), in two columns side by side. Segment control: segment, brand, fuel and transmission dummies. Model
    control: the same with the model name absorbed (within-model transform, the fixed-effects estimator), so each
    effect is measured within one model name. Age and mileage are centred on the median car, so the linear
    coefficients are the marginal effects there. Listings of the same model are not independent and the error
    variance is not equal (Breusch-Pagan), so the 95% intervals come from standard errors clustered by model (735
    models). Also: a cluster-robust test of the difference between the two columns per term (stacked influence
    functions), two sensitivities (clustering by series; the model column without the §8 listings whose engine value
    breaks from their model), how much power varies within a model name, the largest VIF of the segment column's
    design (centred and uncentred), the overall hp–cc Pearson correlation, and how many listings the OLS drops.
    2026-09-28 (simplification list): this replaces the 1000-replicate row bootstrap, which treated listings as
    independent. 2026-09-28 (audit): the model column was a dummy-variable OLS solved by a pseudo-inverse on a
    singular design, whose clustered SEs blew up on small perturbations; the within transform gives the same
    estimates on a full-rank design. No period effect in the model (the owner's decision, 2026-09-23).
TR: Teknik rapor §6 — hedonik model: kontrollü fiyat etkileri (log fiyat ~ yaş, km, hasar, güç, hacim), yan yana
    iki sütun. Segment kontrolü: segment, marka, yakıt ve vites kuklaları. Model kontrolü: aynısı, model adı emilerek
    (model içi dönüşüm, sabit etkiler kestiricisi); her etki tek bir model adının içinde ölçülür. Yaş ve km medyan
    araca ortalanır; doğrusal katsayılar o noktadaki marjinal etkidir. Aynı modelin ilanları bağımsız değil ve hata
    varyansı eşit değil (Breusch-Pagan); bu yüzden %95 aralıklar modele göre kümelenmiş standart hatalardan (735
    model). Ayrıca: iki sütun arasındaki farkın terim başına kümeli testi (yığılmış etki fonksiyonları), iki
    duyarlılık (seriye göre kümeleme; §8'in motor değeri modelinden kopan ilanları olmadan model sütunu), gücün model
    adı içinde ne kadar değiştiği, segment sütunu tasarımının en yüksek VIF'i (ortalanmış ve ortalanmamış), genel
    hp–cc Pearson korelasyonu ve OLS'in attığı ilan sayısı. 2026-09-28 (sadeleştirme listesi): bu, ilanları bağımsız
    sayan 1000 tekrarlı satır bootstrap'inin yerini alır. 2026-09-28 (denetim): model sütunu tekil tasarımda
    pseudo-inverse ile çözülen kukla OLS'ti ve kümeli SH'leri küçük değişikliklerde patlıyordu; model içi dönüşüm aynı
    tahminleri tam ranklı tasarımda verir. Modelde dönem etkisi yok (kullanıcının kararı, 2026-09-23).
Output / Çıktı: metrics/06_hedonic.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from lib import spec_rule as SPEC
from lib.common import load_clean, save_metrics

TERMS = "age+I(age**2)+km10+I(km10**2)+age:km10+dmg+painted+changed+hp100+cc_L"
CONTROLS = "C(segment)+C(brand)+C(kb_fuel)+C(kb_transmission)"
FORMULA = f"log_price~{TERMS}+{CONTROLS}"
# EN: column id → the group whose fixed effect is absorbed (None = plain OLS on FORMULA)
# TR: sütun kimliği → sabit etkisi emilen grup (None = FORMULA üzerinde düz OLS)
COLUMNS = {"segment": None, "model": "model"}
# EN: patsy term name → English id the reports index by | TR: patsy terim adı → raporların indekslediği İngilizce kimlik
TERM_ID = {"age": "age", "I(age ** 2)": "age_sq", "km10": "km100k", "I(km10 ** 2)": "km_sq", "age:km10": "age_x_km",
           "dmg": "heavy_damage", "painted": "painted", "changed": "changed", "hp100": "hp100", "cc_L": "litre"}
CLUSTER, SENS_CLUSTER = "model", "series"
ALPHA = 0.05


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


def fit_column(h, absorb, cluster):
    """
    EN: One column: FORMULA by OLS, or with the fixed effect of `absorb` taken out by demeaning y and X within its
        groups (Frisch–Waugh: the same estimates as its dummies). Columns constant inside every group (intercept,
        brand, nested segment dummies) drop out; stops if the rest is not full rank. SEs clustered by `cluster`
        (t with G − 1 df). Returns: (fit, R² of the full model = 1 − SSR / total SS around the mean).
    TR: Tek sütun: FORMULA düz OLS ile ya da `absorb`'un sabit etkisi gruplarının içinde y ve X ortalamadan
        çıkarılarak alınmış hâliyle (Frisch–Waugh: kuklalarıyla aynı tahminler). Her grubun içinde sabit kolonlar
        (sabit terim, marka, iç içe segment kuklaları) düşer; kalan tam ranklı değilse durur. SH'ler `cluster`'a göre
        kümeli (G − 1 serbestlik dereceli t). Döndürür: (fit, tam modelin R²'si = 1 − SSR / ortalama etrafı toplam KT).
    """
    groups = pd.factorize(h[cluster])[0]
    y, X = patsy.dmatrices(FORMULA, h, return_type="dataframe")
    y = y.iloc[:, 0]
    if absorb:
        g = pd.factorize(h[absorb])[0]
        X = X - X.groupby(g).transform("mean")
        y_fit = y - y.groupby(g).transform("mean")
        # EN: keep a column only if it adds rank (a redundant control dummy leaves the column space, so the fit and
        #     the tracked terms, unchanged); a tracked term must never be the one dropped
        # TR: bir kolon ancak rankı artırıyorsa kalır (gereksiz bir kontrol kuklası sütun uzayını, dolayısıyla uyumu
        #     ve izlenen terimleri değiştirmez); düşen kolon asla izlenen bir terim olmamalı
        keep = []
        for c in X.columns:
            if float(np.abs(X[c]).max()) > 1e-9 and np.linalg.matrix_rank(X[keep + [c]].values) == len(keep) + 1:
                keep.append(c)
        assert set(TERM_ID) <= set(keep), f"a tracked term is not identified | izlenen terim tanımsız: {set(TERM_ID) - set(keep)}"
        X = X[keep]
    else:
        y_fit = y
    assert np.linalg.matrix_rank(X.values) == X.shape[1], "design not full rank | tasarım tam ranklı değil"
    fit = sm.OLS(y_fit, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    return fit, 1 - float((fit.resid ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def coefficient_rows(fit):
    """
    EN: For every tracked term: point, clustered SE, 95% CI (log scale) and the same as a % effect, and whether the
        CI contains zero. Stops if a term is missing or not finite.
    TR: İzlenen her terim için: nokta, kümeli SH, %95 GA (log ölçek), aynısının % etki hâli ve GA'nın sıfırı içerip
        içermediği. Bir terim eksik ya da sonlu değilse durur.
    """
    ci = fit.conf_int(ALPHA)
    rows = []
    for k, tid in TERM_ID.items():
        b, se, lo, hi = fit.params[k], fit.bse[k], ci.loc[k, 0], ci.loc[k, 1]
        assert np.isfinite([b, se, lo, hi]).all() and se > 0, f"{tid}: not identified | tanımsız"
        rows.append({"term": tid, "point": round(float(b), 4), "se": round(float(se), 4), "ci_lo": round(float(lo), 4),
                     "ci_hi": round(float(hi), 4), "pct_effect": round(pct(b), 2), "pct_lo": round(pct(lo), 2),
                     "pct_hi": round(pct(hi), 2), "contains_zero": bool(lo <= 0 <= hi)})
    return rows


def influence(fit):
    """
    EN: Per-listing influence of every coefficient, (X'X)⁻¹ x_i e_i (rows = listings). Returns: DataFrame.
    TR: Her katsayının ilan başına etkisi, (X'X)⁻¹ x_i e_i (satırlar = ilanlar). Döndürür: DataFrame.
    """
    X = np.asarray(fit.model.exog)
    return pd.DataFrame((X * np.asarray(fit.resid)[:, None]) @ np.linalg.inv(X.T @ X), columns=fit.model.exog_names)


def difference_test(fit_a, fit_b, groups):
    """
    EN: Cluster-robust test of (column a − column b) per tracked term, both fitted on the same listings: the variance
        of the difference is the clustered sum of the stacked influences, G/(G−1)·Σ_g (Σ_{i∈g} ψa_i − ψb_i)²; t with
        G − 1 df. Returns: [{term, diff (log), se, t, significant}].
    TR: Aynı ilanlarda kurulmuş iki sütun için terim başına (a sütunu − b sütunu) kümeli testi: farkın varyansı yığılmış
        etkilerin kümeli toplamı, G/(G−1)·Σ_g (Σ_{i∈g} ψa_i − ψb_i)²; G − 1 serbestlik dereceli t.
        Döndürür: [{term, diff (log), se, t, significant}].
    """
    ia, ib = influence(fit_a), influence(fit_b)
    n_g = int(groups.max() + 1)
    crit = stats.t.ppf(1 - ALPHA / 2, n_g - 1)
    out = []
    for k, tid in TERM_ID.items():
        d = np.bincount(groups, weights=(ia[k] - ib[k]).values, minlength=n_g)
        se = float(np.sqrt(n_g / (n_g - 1) * (d ** 2).sum()))
        diff = float(fit_a.params[k] - fit_b.params[k])
        out.append({"term": tid, "diff": round(diff, 6), "se": round(se, 6), "t": round(diff / se, 2),
                    "significant": bool(abs(diff / se) > crit)})
    return out


def zero_in_ci(fit):
    """EN: Tracked terms whose 95% CI contains zero. / TR: %95 GA'sı sıfırı içeren izlenen terimler."""
    ci = fit.conf_int(ALPHA)
    return [tid for k, tid in TERM_ID.items() if ci.loc[k, 0] <= 0 <= ci.loc[k, 1]]


def hp_within(h, group):
    """
    EN: How much power varies inside a model name: models with more than one power value, and the within-model vs
        overall standard deviation of hp.
    TR: Güç bir model adının içinde ne kadar değişiyor: birden çok güç değeri olan model sayısı ve hp'nin model içi ile
        genel standart sapması.
    """
    g = h.groupby(group)["hp"]
    return {"models": int(g.ngroups), "models_varying": int((g.nunique() > 1).sum()),
            "sd_within": round(float((h["hp"] - g.transform("mean")).std()), 1), "sd_overall": round(float(h["hp"].std()), 1)}


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
    x_raw = patsy.dmatrix(FORMULA.split("~", 1)[1], raw, return_type="dataframe")
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
        prints; domain.hedonic_reliability: both columns, the difference test, the sensitivities) and the report
        inputs (error_drivers.hedonic_dropped).
    TR: Site ağacında (domain.hedonic: model sütununun manşet etkileri, karar notunun bastıkları;
        domain.hedonic_reliability: iki sütun, fark testi, duyarlılıklar) ve rapor girdilerinde
        (error_drivers.hedonic_dropped) yayımlanır.
    """
    fits, rows, r2 = res["fits"], res["rows"], res["r2"]
    head = {r["term"]: r["pct_effect"] for r in rows["model"]}
    spec = {r["term"]: r for r in res["spec_rows"]}
    return {
        "domain": {
            "hedonic": {"r2": round(r2["model"], 4), "control": "model", "age_pct": head["age"],
                        "damage_pct": head["heavy_damage"], "km100k_pct": head["km100k"], "hp100_pct": head["hp100"],
                        "cc_litre_pct": head["litre"],
                        "note": ("Model kontrolü sütunu: etkiler aynı model adı içinde. Dönem etkisi modelde yok "
                                 "(kullanıcı kararı): dönemler havuzlanarak kestirildi.")},
            "hedonic_reliability": {
                "n": int(fits["segment"].nobs), "cluster": CLUSTER, "n_clusters": res["n_clusters"],
                "center": {"age": res["age_c"], "km": res["km_c"] * 1e5},
                "columns": {c: {"r2": round(r2[c], 4), "coefficients": rows[c]} for c in COLUMNS},
                "difference": res["difference"],
                "series_cluster": {"n_clusters": res["n_series"], "zero_in_ci": res["series_zero"]},
                "without_spec_outliers": {"n_dropped": res["spec_n"],
                                          "coefficients": [spec["hp100"], spec["litre"]]},
                "hp_within_model": res["hp_within"],
                "vif": res["vif"], "hp_cc_pearson": res["hp_cc_r"], "homoskedasticity_p": res["bp_p"]}},
        "error_drivers": {"hedonic_dropped": res["dropped"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
listings["spec_off"] = SPEC.spec_outlier_mask(listings)[0]
h, age_c, km_c = hedonic_frame(listings)
dropped = dropped_rows(listings)
fitted = {c: fit_column(h, absorb, CLUSTER) for c, absorb in COLUMNS.items()}
fits, r2 = {c: f for c, (f, _) in fitted.items()}, {c: r for c, (_, r) in fitted.items()}
assert all(len(listings) - dropped["total"] == int(f.nobs) for f in fits.values()), "OLS row count ≠ dropped count"
groups = pd.factorize(h[CLUSTER])[0]
bp = het_breuschpagan(fits["segment"].resid, fits["segment"].model.exog)
series_fits = {c: fit_column(h, absorb, SENS_CLUSTER)[0] for c, absorb in COLUMNS.items()}
h_spec = h[~h["spec_off"]].reset_index(drop=True)
res = {"fits": fits, "r2": r2, "rows": {c: coefficient_rows(f) for c, f in fits.items()},
       "n_clusters": int(groups.max() + 1), "difference": difference_test(fits["segment"], fits["model"], groups),
       "n_series": int(h[SENS_CLUSTER].nunique()), "series_zero": {c: zero_in_ci(f) for c, f in series_fits.items()},
       "spec_n": int(h["spec_off"].sum()), "spec_rows": coefficient_rows(fit_column(h_spec, "model", CLUSTER)[0]),
       "hp_within": hp_within(h, CLUSTER), "age_c": age_c, "km_c": km_c, "vif": vif_max(h, fits["segment"], age_c, km_c),
       "hp_cc_r": round(float(stats.pearsonr(h.hp, h.cc)[0]), 3), "bp_p": float(bp[1]), "dropped": dropped}
for c in COLUMNS:
    print(c, f"R² {r2[c]:.4f}", [(r["term"], r["pct_effect"], r["pct_lo"], r["pct_hi"]) for r in res["rows"][c]])
print("difference | fark:", [(d["term"], d["t"], d["significant"]) for d in res["difference"]])
print("series clusters | seri kümeleri:", res["series_zero"], "· without spec outliers | tutarsızlar olmadan:",
      [(r["term"], r["pct_effect"]) for r in res["spec_rows"] if r["term"] in ("hp100", "litre")], res["spec_n"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("06_hedonic", to_metrics(res)))
