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
    2026-11-23 (owner's request, docs/decisions.md; added after the results were seen, no pre-registration):
      - the ladder: how much of log price a regression explains with only the car's identity (segment controls, or
        the car model name), with only its condition (age, mileage, damage, paint, changed parts), and with both —
        in-sample R² and 5-fold out-of-fold R² (cv.make_folds; the validation fold never enters the fit);
      - a third column without any identity control (the ten terms alone), next to the segment and car-model columns,
        to show how each effect moves when the identity is held fixed;
      - age as yearly steps: age dummies and a mileage spline within the car model; the average change per year
        between two ages (the quadratic's single number depends on the functional form);
      - painted and changed parts by count (1, 2, 3, 4, 5 and more) within the car model, because the effect of a
        painted part is not the same for the first and the fifth.
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
    2026-11-23 (kullanıcının isteği, docs/decisions.md; sonuçlar görüldükten sonra eklendi, ön kayıt yok):
      - merdiven: bir regresyon log fiyatın ne kadarını yalnız aracın kimliğiyle (segment kontrolleri ya da araç
        modelinin adı), yalnız durumuyla (yaş, km, hasar, boya, değişen parça) ve ikisiyle birlikte açıklıyor —
        örneklem içi R² ve 5 katlı kat dışı R² (cv.make_folds; doğrulama katı uyuma hiç girmez);
      - hiçbir kimlik kontrolü olmayan üçüncü sütun (yalnız on terim), segment ve araç modeli sütunlarının yanında;
        kimlik sabitlenince her etkinin nasıl değiştiğini gösterir;
      - yıllık basamaklarla yaş: araç modeli içinde yaş kuklaları ve km eğrisi (spline); iki yaş arasında yıl başına
        ortalama değişim (ikinci dereceden terimin tek sayısı fonksiyon biçimine bağlı);
      - sayıya göre boyalı ve değişen parça (1, 2, 3, 4, 5 ve fazlası), araç modeli içinde; çünkü boyalı bir parçanın
        etkisi ilkinde ve beşincisinde aynı değil.
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
from lib.cv import make_folds

CONDITION = "age+I(age**2)+km10+I(km10**2)+age:km10+dmg+painted+changed"
ENGINE = "hp100+cc_L"
TERMS = f"{CONDITION}+{ENGINE}"
CONTROLS = "C(segment)+C(brand)+C(kb_fuel)+C(kb_transmission)"
FORMULA = f"log_price~{TERMS}+{CONTROLS}"
# EN: the ladder's regressions, id → right-hand side: identity only, condition only, then both
# TR: merdivenin regresyonları, kimlik → sağ taraf: yalnız kimlik, yalnız durum, sonra ikisi
LADDER = {"identity_segment": CONTROLS, "identity_model": "C(model)", "condition": CONDITION,
          "condition_engine": TERMS, "segment_control": f"{TERMS}+{CONTROLS}",
          "model_control": f"{TERMS}+{CONTROLS}+C(model)"}
AGE_STEPS = [(1, 5), (5, 10), (10, 15), (15, 20)]      # average change per year between these ages | bu yaşlar arası
KM_SPLINE_DF = 6
COUNT_TOP = 5                                           # 5 parts and more are one bucket | 5 ve fazlası tek kova
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


def fit_column(h, absorb, cluster, formula=FORMULA, tracked=TERM_ID):
    """
    EN: One column: `formula` by OLS, or with the fixed effect of `absorb` taken out by demeaning y and X within its
        groups (Frisch–Waugh: the same estimates as its dummies). Columns constant inside every group (intercept,
        brand, nested segment dummies) drop out; stops if the rest is not full rank. SEs clustered by `cluster`
        (t with G − 1 df). tracked: the terms that must stay identified.
        Returns: (fit, R² of the full model = 1 − SSR / total SS around the mean).
    TR: Tek sütun: `formula` düz OLS ile ya da `absorb`'un sabit etkisi gruplarının içinde y ve X ortalamadan
        çıkarılarak alınmış hâliyle (Frisch–Waugh: kuklalarıyla aynı tahminler). Her grubun içinde sabit kolonlar
        (sabit terim, marka, iç içe segment kuklaları) düşer; kalan tam ranklı değilse durur. SH'ler `cluster`'a göre
        kümeli (G − 1 serbestlik dereceli t). tracked: tanımlı kalması gereken terimler.
        Döndürür: (fit, tam modelin R²'si = 1 − SSR / ortalama etrafı toplam KT).
    """
    groups = pd.factorize(h[cluster])[0]
    y, X = patsy.dmatrices(formula, h, return_type="dataframe")
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
        assert set(tracked) <= set(keep), f"a tracked term is not identified | izlenen terim tanımsız: {set(tracked) - set(keep)}"
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


def oof_predictions(X, y, folds):
    """
    EN: Out-of-fold least-squares predictions: each fold's rows are predicted by a fit on the other folds only. A
        dummy level absent from the training folds gets a zero coefficient (minimum-norm solution), i.e. the listing
        is predicted from the remaining columns. Returns: one prediction per row.
    TR: Kat dışı en küçük kareler tahminleri: her katın satırları yalnız öteki katlardaki uyumla tahmin edilir.
        Eğitim katlarında hiç olmayan bir kukla düzeyi sıfır katsayı alır (en küçük normlu çözüm); yani ilan kalan
        sütunlardan tahmin edilir. Döndürür: satır başına bir tahmin.
    """
    pred = np.zeros(len(y))
    for tr, va in folds:
        pred[va] = X[va] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    return pred


def ladder_rows(h, folds, specs=LADDER):
    """
    EN: For each regression of the ladder: number of columns, in-sample R² and out-of-fold R² of log price (both
        against the total sum of squares around the overall mean).
        Returns: [[id, columns, r2_in_sample, r2_out_of_fold], ...].
    TR: Merdivenin her regresyonu için: sütun sayısı, log fiyatın örneklem içi ve kat dışı R²'si (ikisi de genel
        ortalama etrafındaki toplam kareler toplamına göre).
        Döndürür: [[kimlik, sütun, r2_örneklem_içi, r2_kat_dışı], ...].
    """
    y = h["log_price"].values
    tss = float(((y - y.mean()) ** 2).sum())
    rows = []
    for name, rhs in specs.items():
        X = patsy.dmatrix(rhs, h, return_type="dataframe").values
        fit_in = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append([name, int(X.shape[1]), round(1 - float(((y - fit_in) ** 2).sum()) / tss, 4),
                     round(1 - float(((y - oof_predictions(X, y, folds)) ** 2).sum()) / tss, 4)])
    return rows


def age_steps(h, age_c, km_c, cluster, steps=AGE_STEPS):
    """
    EN: Age as yearly steps, within the car model: one dummy per age (years) and a B-spline of mileage instead of
        the quadratic terms; the other terms and controls as in FORMULA. For each (from, to): the average change of
        log price per year between the two ages, with its clustered 95% interval, as % effects.
        Returns: [[age_from, age_to, listings between them, pct per year, pct_lo, pct_hi], ...].
    TR: Yıllık basamaklarla yaş, araç modeli içinde: ikinci dereceden terimler yerine yaş (yıl) başına bir kukla ve
        km'nin B-spline'ı; öteki terimler ve kontroller FORMULA'daki gibi. Her (başlangıç, bitiş) için: iki yaş
        arasında log fiyatın yıl başına ortalama değişimi, kümeli %95 aralığıyla, % etki olarak.
        Döndürür: [[yaş_başlangıç, yaş_bitiş, aradaki ilan, yıl başına %, %_alt, %_üst], ...].
    """
    d = h.copy()
    d["age_years"], d["km_raw"] = (d["age"] + age_c).round().astype(int), d["km10"] + km_c
    level = lambda a: f"C(age_years)[T.{a}]"                                         # noqa: E731
    need = [level(a) for pair in steps for a in pair]
    formula = f"log_price~C(age_years)+bs(km_raw, df={KM_SPLINE_DF})+dmg+painted+changed+{ENGINE}+{CONTROLS}"
    fit, _ = fit_column(d, cluster, cluster, formula=formula, tracked=need)
    names, rows = list(fit.params.index), []
    for a, b in steps:
        w = np.zeros(len(names))
        w[names.index(level(b))], w[names.index(level(a))] = 1 / (b - a), -1 / (b - a)
        t = fit.t_test(w)
        lo, hi = t.conf_int(ALPHA)[0]
        rows.append([a, b, int(((d["age_years"] >= a) & (d["age_years"] <= b)).sum()), round(pct(float(t.effect[0])), 2),
                     round(pct(float(lo)), 2), round(pct(float(hi)), 2)])
    return rows


def count_effects(h, column, cluster, top=COUNT_TOP):
    """
    EN: The effect of 1, 2, … painted (or changed) parts against none, within the car model: the linear count is
        replaced by one dummy per count, `top` and more in one bucket; the rest of FORMULA stays.
        Returns: [[count (the last one means `top` and more), listings, pct effect, pct_lo, pct_hi], ...].
    TR: Hiç yokken 1, 2, … boyalı (ya da değişen) parçanın etkisi, araç modeli içinde: doğrusal sayı yerine sayı
        başına bir kukla, `top` ve fazlası tek kovada; FORMULA'nın geri kalanı aynı.
        Döndürür: [[sayı (sonuncusu `top` ve fazlası demek), ilan, % etki, %_alt, %_üst], ...].
    """
    d = h.copy()
    d["bucket"] = np.minimum(d[column].astype(int), top)
    level = lambda b: f"C(bucket)[T.{b}]"                                            # noqa: E731
    terms = TERMS.replace(f"+{column}", "")
    assert column not in terms.split("+"), f"{column} still linear | hâlâ doğrusal"
    fit, _ = fit_column(d, cluster, cluster, formula=f"log_price~C(bucket)+{terms}+{CONTROLS}",
                        tracked=[level(b) for b in range(1, top + 1)])
    ci = fit.conf_int(ALPHA)
    return [[b, int((d["bucket"] == b).sum()), round(pct(fit.params[level(b)]), 2), round(pct(ci.loc[level(b), 0]), 2),
             round(pct(ci.loc[level(b), 1]), 2)] for b in range(1, top + 1)]


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.hedonic: the model column's headline effects, which the decision note
        prints; domain.hedonic_reliability: both columns, the difference test, the sensitivities) and the report
        inputs (error_drivers.hedonic_dropped). Report-only (report.hedonic): the ladder, the column without
        identity controls, age as yearly steps and the panel effects by count.
    TR: Site ağacında (domain.hedonic: model sütununun manşet etkileri, karar notunun bastıkları;
        domain.hedonic_reliability: iki sütun, fark testi, duyarlılıklar) ve rapor girdilerinde
        (error_drivers.hedonic_dropped) yayımlanır. Yalnız rapor (report.hedonic): merdiven, kimlik kontrolsüz
        sütun, yıllık basamaklarla yaş ve sayıya göre parça etkileri.
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
        "report": {"hedonic": {
            "ladder": {"columns": ["spec", "n_columns", "r2_in_sample", "r2_out_of_fold"], "rows": res["ladder"]},
            "no_identity": {"r2": round(r2["none"], 4), "coefficients": rows["none"]},
            "age_steps": {"columns": ["age_from", "age_to", "n", "pct_per_year", "pct_lo", "pct_hi"],
                          "rows": res["age_steps"], "km_spline_df": KM_SPLINE_DF},
            "by_count": {"columns": ["count", "n", "pct_effect", "pct_lo", "pct_hi"], "top": COUNT_TOP,
                         "painted": res["painted_by_count"], "changed": res["changed_by_count"]}}},
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
# EN: the column without identity controls, and the ladder | TR: kimlik kontrolsüz sütun ve merdiven
fit_none, r2["none"] = fit_column(h, None, CLUSTER, formula=f"log_price~{TERMS}")
ladder = ladder_rows(h, make_folds(len(h)))
ladder_r2 = {r[0]: r[2] for r in ladder}
assert abs(ladder_r2["segment_control"] - r2["segment"]) < 1e-4 and abs(ladder_r2["model_control"] - r2["model"]) < 1e-4 \
    and abs(ladder_r2["condition_engine"] - r2["none"]) < 1e-4, "ladder ≠ columns | merdiven sütunlarla uyuşmuyor"
res = {"fits": fits, "r2": r2, "rows": {**{c: coefficient_rows(f) for c, f in fits.items()},
                                         "none": coefficient_rows(fit_none)},
       "ladder": ladder, "age_steps": age_steps(h, age_c, km_c, CLUSTER),
       "painted_by_count": count_effects(h, "painted", CLUSTER), "changed_by_count": count_effects(h, "changed", CLUSTER),
       "n_clusters": int(groups.max() + 1), "difference": difference_test(fits["segment"], fits["model"], groups),
       "n_series": int(h[SENS_CLUSTER].nunique()), "series_zero": {c: zero_in_ci(f) for c, f in series_fits.items()},
       "spec_n": int(h["spec_off"].sum()), "spec_rows": coefficient_rows(fit_column(h_spec, "model", CLUSTER)[0]),
       "hp_within": hp_within(h, CLUSTER), "age_c": age_c, "km_c": km_c, "vif": vif_max(h, fits["segment"], age_c, km_c),
       "hp_cc_r": round(float(stats.pearsonr(h.hp, h.cc)[0]), 3), "bp_p": float(bp[1]), "dropped": dropped}
for c in COLUMNS:
    print(c, f"R² {r2[c]:.4f}", [(r["term"], r["pct_effect"], r["pct_lo"], r["pct_hi"]) for r in res["rows"][c]])
print("ladder | merdiven:", ladder)
print("no identity | kimlik yok:", [(r["term"], r["pct_effect"]) for r in res["rows"]["none"]])
print("age steps | yaş basamakları:", res["age_steps"])
print("painted:", res["painted_by_count"], "· changed:", res["changed_by_count"])
print("difference | fark:", [(d["term"], d["t"], d["significant"]) for d in res["difference"]])
print("series clusters | seri kümeleri:", res["series_zero"], "· without spec outliers | tutarsızlar olmadan:",
      [(r["term"], r["pct_effect"]) for r in res["spec_rows"] if r["term"] in ("hp100", "litre")], res["spec_n"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
if __name__ == "__main__":      # imported by notebooks/: no save | notebooks/ import edince kayıt yok
    print("written | yazıldı:", save_metrics("06_hedonic", to_metrics(res)))
