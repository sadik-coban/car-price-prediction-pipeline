"""
Hedonik regresyon: kümeli standart hatalı OLS (bir grubun sabit etkisi istenirse grup içi dönüşümle alınır), katsayı
tablosu, aynı ilanlarda kurulmuş iki regresyonun katsayı farkının kümeli testi ve kat dışı tahminler.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

ALPHA = 0.05      # %95 aralıklar


def pct_effect(coef):
    """Log fiyattaki bir katsayının (ya da aralık sınırının) fiyata yüzde etkisi: 100 · (e^katsayı - 1)."""
    return 100 * (np.exp(coef) - 1)


def fit_clustered(table, formula, cluster, absorb=None, tracked=()):
    """
    formula'yı (patsy) OLS ile kurar. Standart hatalar cluster kolonuna göre kümeli; aralıklar küme sayısı eksi bir
    serbestlik dereceli t'den. absorb verilirse o kolonun sabit etkisi grup içi dönüşümle alınır: hedef ve her tasarım
    kolonu kendi grubunun ortalamasından çıkarılır (Frisch-Waugh; grup kuklalarıyla aynı katsayılar). Grubun içinde
    sabit kalan (sabit terim, marka gibi) ya da rank artırmayan kolonlar düşer.
    Tabloda boş değer varsa, tracked'daki bir terim tasarımda kalmazsa ya da tasarım tam ranklı değilse durur.
    Döndürür: (fit, R²); R² = 1 - artıkların kareler toplamı / hedefin ortalama etrafındaki kareler toplamı.
    """
    y, X = patsy.dmatrices(formula, table, return_type="dataframe", NA_action="raise")
    y = y.iloc[:, 0]
    y_fit = y
    if absorb:
        groups = pd.factorize(table[absorb])[0]
        X = X - X.groupby(groups).transform("mean")
        y_fit = y - y.groupby(groups).transform("mean")
        keep = []
        for column in X.columns:
            if np.abs(X[column]).max() > 1e-9 and np.linalg.matrix_rank(X[keep + [column]].values) == len(keep) + 1:
                keep.append(column)
        X = X[keep]
    lost = [term for term in tracked if term not in X.columns]
    if lost:
        raise ValueError(f"tasarımda kalmayan terim: {lost}")
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        raise ValueError("tasarım tam ranklı değil")
    clusters = pd.factorize(table[cluster])[0]
    fit = sm.OLS(y_fit, X).fit(cov_type="cluster", cov_kwds={"groups": clusters}, use_t=True)
    return fit, float(1 - (fit.resid ** 2).sum() / ((y - y.mean()) ** 2).sum())


def coefficient_table(fit, terms):
    """
    terms'teki her terim için katsayı (log ölçek), kümeli standart hata ve %95 aralık; bunların fiyata yüzde etkisi
    (pct, pct_lo, pct_hi) ve aralığın sıfırı içerip içermediği. Döndürür: terim başına bir satır.
    """
    ci = fit.conf_int(ALPHA).loc[terms]
    table = pd.DataFrame({"coef": fit.params[terms], "se": fit.bse[terms], "ci_lo": ci[0], "ci_hi": ci[1]})
    table["pct"] = pct_effect(table["coef"])
    table["pct_lo"] = pct_effect(table["ci_lo"])
    table["pct_hi"] = pct_effect(table["ci_hi"])
    table["contains_zero"] = (table["ci_lo"] <= 0) & (table["ci_hi"] >= 0)
    return table


def influence(fit):
    """Her satırın her katsayıya etkisi: (X'X)^-1 x_i e_i. Satırlar gözlemler, kolonlar katsayılar."""
    X, resid = np.asarray(fit.model.exog), np.asarray(fit.resid)
    return pd.DataFrame((X * resid[:, None]) @ np.linalg.inv(X.T @ X), columns=fit.model.exog_names)


def difference_test(fit_a, fit_b, clusters, terms):
    """
    Aynı satırlarda (aynı sırada) kurulmuş iki regresyonun terms'teki her katsayısı için a - b farkının kümeli testi.
    Farkın varyansı iki etkinin farkının küme toplamlarından: G / (G - 1) · Σ_g (Σ_{i∈g} ψa_i - ψb_i)²; t, G - 1
    serbestlik dereceli (G küme sayısı). clusters: her satırın kümesi (ör. araç modeli).
    Döndürür: terim başına fark (log ölçek), standart hata, t ve %5 düzeyinde anlamlı mı.
    """
    psi_a, psi_b = influence(fit_a), influence(fit_b)
    groups = pd.factorize(clusters)[0]
    n_groups = groups.max() + 1
    critical = stats.t.ppf(1 - ALPHA / 2, n_groups - 1)
    rows = []
    for term in terms:
        sums = np.bincount(groups, weights=(psi_a[term] - psi_b[term]).to_numpy(), minlength=n_groups)
        se = np.sqrt(n_groups / (n_groups - 1) * (sums ** 2).sum())
        diff = fit_a.params[term] - fit_b.params[term]
        rows.append({"diff": diff, "se": se, "t": diff / se, "significant": bool(abs(diff / se) > critical)})
    return pd.DataFrame(rows, index=terms)


def oof_predictions(X, y, folds):
    """
    Kat dışı en küçük kareler tahminleri: her katın satırları yalnız öteki katlarda kurulan regresyonla tahmin edilir.
    Eğitim katlarında hiç görülmeyen bir kukla düzeyi sıfır katsayı alır (en küçük normlu çözüm), yani o satır kalan
    kolonlardan tahmin edilir. X: tasarım matrisi (numpy dizisi), y: hedef. Döndürür: satır başına bir tahmin.
    """
    pred = np.zeros(len(y))
    for train, valid in folds:
        pred[valid] = X[valid] @ np.linalg.lstsq(X[train], y[train], rcond=None)[0]
    return pred
