"""
03_association.py
EN: Technical report §3 — in which direction the categorical features depend on each other (Theil's U,
    asymmetric: model pins down series and brand, not the other way round), plus the Spearman correlation among
    the numeric features and the pairs above |ρ| 0.5. 2026-09-28: Cramér's V, its permutation floors and the
    Pearson matrix left the report (simplification list); nothing reads them, so they are not computed.
TR: Teknik rapor §3 — kategorik öznitelikler birbirine hangi yönde bağlı (Theil's U, asimetrik: model seriyi ve
    markayı belirler, tersi değil); ayrıca sayısal öznitelikler arasında Spearman korelasyonu ve |ρ| 0,5'in
    üzerindeki çiftler. 2026-09-28: Cramér's V, permütasyon tabanları ve Pearson matrisi rapordan çıktı
    (sadeleştirme listesi); onları okuyan yok, hesaplanmıyor.
Output / Çıktı: metrics/03_association.json
"""

# %% [1] Setup | Kurulum
import itertools

import numpy as np
import pandas as pd

from lib.common import CAT, NUM, TEXT, load_clean, save_metrics

HIGH_CORR = 0.5


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def theils_u(a, b):
    """
    EN: Theil's U(a | b): the share of a's uncertainty removed by knowing b (asymmetric, 0–1).
    TR: Theil's U(a | b): b'yi bilmenin a'nın belirsizliğinden giderdiği pay (asimetrik, 0–1).
    """
    def entropy(s):
        """EN: Shannon entropy of a series. / TR: Bir serinin Shannon entropisi."""
        p = s.value_counts(normalize=True)
        return -np.sum(p * np.log(p + 1e-12))
    hx = entropy(a)
    pxy = pd.crosstab(a, b)
    pxy = pxy / pxy.sum().sum()
    hxy = -np.sum(pxy.values * np.log(pxy.values + 1e-12))
    return (hx - (hxy - entropy(b))) / hx if hx > 0 else 0


def association_matrix(df, cols, fn):
    """
    EN: fn(df[i], df[j]) for every pair of columns, rounded to 3 decimals. Returns: list of rows.
    TR: Her kolon çifti için fn(df[i], df[j]), 3 haneye yuvarlı. Döndürür: satır listesi.
    """
    return [[round(fn(df[i], df[j]), 3) for j in cols] for i in cols]


def numeric_correlation(df, cols, threshold):
    """
    EN: Spearman matrix of the numeric features and the pairs with |ρ| > threshold, largest first.
    TR: Sayısal özniteliklerin Spearman matrisi ve |ρ| > eşik olan çiftler, büyükten küçüğe.
    """
    spearman = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(3)
    high = [[a, b, round(float(spearman.loc[a, b]), 3)] for a, b in itertools.combinations(cols, 2)
            if abs(float(spearman.loc[a, b])) > threshold]
    return {"spearman": spearman.values.tolist(), "high_pairs": sorted(high, key=lambda x: -abs(x[2]))}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology and domain) under the names the site and reports read.
    TR: Site ağacında (methodology ve domain), sitenin ve raporların okuduğu adlarla yayımlanır.
    """
    nc = res["numeric"]
    return {"methodology": {"theils_matrix": {"labels": res["cols"], "matrix": res["theils"]}},
            "domain": {"numeric_correlation": {"labels": NUM, "spearman": nc["spearman"],
                                               "high_pairs": nc["high_pairs"]}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
cols = CAT + TEXT                            # model and series included | model ve seri dahil
res = {"cols": cols, "theils": association_matrix(listings, cols, theils_u),
       "numeric": numeric_correlation(listings, NUM, HIGH_CORR)}
print("high numeric pairs | yüksek sayısal çiftler:", res["numeric"]["high_pairs"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("03_association", to_metrics(res)))
