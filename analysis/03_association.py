"""
03_association.py
EN: Technical report §3 — how strongly the categorical features depend on each other (Cramér's V,
    symmetric) and in which direction (Theil's U, asymmetric), each with a permutation floor (both are
    biased upward at high cardinality), plus Pearson/Spearman among the numeric features.
TR: Teknik rapor §3 — kategorik öznitelikler birbirine ne kadar bağlı (Cramér's V, simetrik) ve hangi yönde
    (Theil's U, asimetrik), her biri permütasyon tabanıyla (yüksek kardinalitede ikisi de yukarı yanlı);
    ayrıca sayısal öznitelikler arasında Pearson/Spearman.
Output / Çıktı: metrics/03_association.json
"""

# %% [1] Setup | Kurulum
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from lib.common import CAT, NUM, TEXT, load_clean, save_metrics

SEED = 42
N_PERMUTATIONS = 5
HIGH_CORR = 0.5


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def cramers_v(a, b):
    """
    EN: Cramér's V between two categorical series (symmetric strength of association, 0–1).
    TR: İki kategorik seri arasında Cramér's V (simetrik bağıntı gücü, 0–1).
    """
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1))


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


def permutation_floor(df, cols, fn, rng, n_perm):
    """
    EN: The value fn gives when column j is shuffled (no real association): mean of n_perm shuffles per cell.
        rng is consumed in cell order, so calling order matters for reproducibility.
    TR: j kolonu karıştırıldığında (gerçek bağıntı yokken) fn'in verdiği değer: hücre başına n_perm
        karıştırmanın ortalaması. rng hücre sırasıyla tüketilir; tekrarlanabilirlik için çağrı sırası önemli.
    """
    out = []
    for i in cols:
        row = []
        for j in cols:
            if i == j:
                row.append(1.0)
                continue
            vals = [fn(df[i], pd.Series(rng.permutation(df[j].values), index=df.index)) for _ in range(n_perm)]
            row.append(round(float(np.mean(vals)), 3))
        out.append(row)
    return out


def numeric_correlation(df, cols, threshold):
    """
    EN: Pearson and Spearman matrices of the numeric features and the pairs with |Pearson r| > threshold.
    TR: Sayısal özniteliklerin Pearson ve Spearman matrisleri ve |Pearson r| > eşik olan çiftler.
    """
    nc = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson, spearman = nc.corr(method="pearson").round(3), nc.corr(method="spearman").round(3)
    high = [[a, b, round(float(pearson.loc[a, b]), 3)] for a, b in itertools.combinations(cols, 2)
            if abs(float(pearson.loc[a, b])) > threshold]
    return {"pearson": pearson.values.tolist(), "spearman": spearman.values.tolist(),
            "high_pairs": sorted(high, key=lambda x: -abs(x[2]))}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology and domain) under the names the site and reports read.
    TR: Site ağacında (methodology ve domain), sitenin ve raporların okuduğu adlarla yayımlanır.
    """
    cols, nc = res["cols"], res["numeric"]
    return {
        "methodology": {
            "cramers_matrix": {"labels": cols, "matrix": res["cramers"]},
            "theils_matrix": {"labels": cols, "matrix": res["theils"]},
            "cramers_null": {"labels": cols, "matrix": res["cramers_null"], "P": N_PERMUTATIONS},
            "theils_null": {"labels": cols, "matrix": res["theils_null"], "P": N_PERMUTATIONS},
            "g_mpv": ("Segment seri ve model adından türetildi (segment_of, deterministik; çözülemeyen kalırsa durur). "
                      "Ham gb_segment kirli olduğu için kullanılmadı."),
            "assoc_model": ("Kategorik ilişki Cramér's V (simetrik) + Theil's U (asimetrik) ile ölçüldü. "
                            "model/series yüksek-kardinaliteli (Cramér's V üste-yanlı olabilir); Theil's U yönlü — "
                            "model markayı/segmenti ~belirler, tersi değil. Etiketler column_labels ile TR/EN.")},
        "domain": {"numeric_correlation": {
            "labels": NUM, "pearson": nc["pearson"], "spearman": nc["spearman"], "high_pairs": nc["high_pairs"],
            "note": ("Sayısal feature korelasyonu (kategorik için Cramér/Theil ayrı). "
                    "Pearson=lineer, Spearman=monotonik ilişki. |r|>0.5 çiftler dikkat çeker; "
                    "çoklu-bağlantı VIF ile ayrıca kontrol edildi (değerler: hedonic_reliability.vif).")}},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
cols = CAT + TEXT                            # model and series included | model ve seri dahil
rng = np.random.default_rng(SEED)            # one stream: Cramér first, then Theil | tek akış: önce Cramér, sonra Theil
res = {"cols": cols,
       "cramers": association_matrix(listings, cols, cramers_v),
       "theils": association_matrix(listings, cols, theils_u)}
res["cramers_null"] = permutation_floor(listings, cols, cramers_v, rng, N_PERMUTATIONS)
res["theils_null"] = permutation_floor(listings, cols, theils_u, rng, N_PERMUTATIONS)
res["numeric"] = numeric_correlation(listings, NUM, HIGH_CORR)
print("high numeric pairs | yüksek sayısal çiftler:", len(res["numeric"]["high_pairs"]))

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("03_association", to_metrics(res)))
