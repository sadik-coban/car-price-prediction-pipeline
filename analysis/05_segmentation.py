"""
05_segmentation.py
EN: Market structure: price by segment and body type, depreciation curves by age and mileage (decision note
    figures), the BMW/Audi price comparison, and whether the listings form natural clusters (technical report §5:
    the KMeans silhouette for k = 2..8). 2026-09-29 (second simplification list): §5 is one paragraph — there are no
    clear clusters — so the fixed-k clusters, their profiles, the elbow curve and the PCA are not computed.
TR: Piyasa yapısı: segment ve kasa tipine göre fiyat, yaş ve kilometreye göre değer kaybı eğrileri (karar notu
    figürleri), BMW/Audi fiyat karşılaştırması ve ilanların doğal kümeler oluşturup oluşturmadığı (teknik rapor §5:
    k = 2..8 için KMeans silhouette). 2026-09-29 (ikinci sadeleştirme listesi): §5 tek paragraf — belirgin küme
    yok — bu yüzden sabit k'li kümeler, profilleri, dirsek eğrisi ve PCA hesaplanmıyor.
Output / Çıktı: metrics/05_segmentation.json
"""

# %% [1] Setup | Kurulum
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.common import NUM, load_clean, save_metrics

SEED = 42
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE = 3000
MIN_BODY_N, MIN_AGE_N, MAX_AGE = 80, 10, 25
KM_BIN, KM_LIMIT = 25000, 400000


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def median_by(listings, col, min_n=0):
    """
    EN: Median price and count per value of col (values with fewer than min_n listings dropped), cheapest first.
        Returns: [[value, median, n], ...].
    TR: col'un her değeri için medyan fiyat ve sayı (min_n'den az ilanlı değerler atılır), en ucuz önce.
        Döndürür: [[değer, medyan, n], ...].
    """
    t = listings.groupby(col).agg(median=("price", "median"), n=("price", "size")).reset_index()
    return t[t["n"] >= min_n].sort_values("median").values.tolist()


def age_curve(listings, max_age, min_n):
    """
    EN: Median, count and mean price per vehicle age (age ≤ max_age, at least min_n listings).
        Returns: [[age, median, n, mean], ...].
    TR: Araç yaşı başına medyan, sayı ve ortalama fiyat (yaş ≤ max_age, en az min_n ilan).
        Döndürür: [[yaş, medyan, n, ortalama], ...].
    """
    a = (listings[listings["vehicle_age"] <= max_age].groupby("vehicle_age")
         .agg(median=("price", "median"), n=("price", "size"), mean=("price", "mean")).reset_index())
    a = a[a["n"] >= min_n]
    a["mean"] = a["mean"].round(0)
    return a.values.tolist()


def mileage_curve(listings, bin_width, limit):
    """
    EN: Median, count and mean price per mileage bin below limit (bins aligned to the limit), and how many
        listings are at or above it. Returns: (rows [[bin, median, n, mean], ...], {"upper_limit", "outside"}).
    TR: limit altındaki kilometre kovası başına medyan, sayı ve ortalama fiyat (kovalar sınıra hizalı) ve
        sınırda ya da üstünde kaç ilan olduğu. Döndürür: (satırlar, {"upper_limit", "outside"}).
    """
    d = listings.assign(km_bin=listings["gb_mileage"] // bin_width * bin_width)
    k = (d[d["gb_mileage"] < limit].groupby("km_bin")
         .agg(median=("price", "median"), n=("price", "size"), mean=("price", "mean")).reset_index())
    k["mean"] = k["mean"].round(0)
    return k.values.tolist(), {"upper_limit": limit, "outside": int((listings["gb_mileage"] >= limit).sum())}


def cliffs_delta(a, b, chunk=2000):
    """
    EN: Cliff's delta = P(a > b) − P(a < b), over ALL pairs (no sampling; chunked only to save memory).
    TR: Cliff's delta = P(a > b) − P(a < b), BÜTÜN çiftler üzerinden (örnekleme yok; parçalama yalnız bellek için).
    """
    a, b = np.asarray(a), np.asarray(b)
    gt = lt = 0
    for i in range(0, len(a), chunk):
        ac = a[i:i + chunk]
        gt += int((ac[:, None] > b[None, :]).sum())
        lt += int((ac[:, None] < b[None, :]).sum())
    return (gt - lt) / (len(a) * len(b))


def brand_compare(listings):
    """
    EN: BMW vs Audi price: medians, means, counts, Mann–Whitney p and Cliff's delta (kept for the site;
        the reports do not print the raw brand test — the owner's decision).
    TR: BMW ve Audi fiyatı: medyan, ortalama, sayı, Mann–Whitney p ve Cliff's delta (site için tutulur;
        raporlar ham marka testini basmaz — kullanıcının kararı).
    """
    bmw, audi = listings.loc[listings["brand"] == "bmw", "price"].values, listings.loc[listings["brand"] == "audi", "price"].values
    _stat, p = stats.mannwhitneyu(bmw, audi, alternative="two-sided")
    return {"bmw_median": float(np.median(bmw)), "audi_median": float(np.median(audi)),
            "bmw_n": int(len(bmw)), "audi_n": int(len(audi)),
            "bmw_mean": float(np.mean(bmw)), "audi_mean": float(np.mean(audi)),
            "mwu_p": float(p), "cliffs_delta": float(cliffs_delta(bmw, audi))}


def silhouette_by_k(listings, k_range, sample, seed):
    """
    EN: Standardised numeric features (median-filled); KMeans for every k in k_range and its silhouette on a random
        sample. Returns: [[k, silhouette], ...].
    TR: Standartlaştırılmış sayısal öznitelikler (medyanla doldurulmuş); k_range'deki her k için KMeans ve rastgele
        örneklemde silhouette'i. Döndürür: [[k, silhouette], ...].
    """
    nc = listings[NUM].fillna(listings[NUM].median())
    xs = StandardScaler().fit_transform(nc)
    idx = np.random.default_rng(seed).choice(len(xs), sample, replace=False)
    sil = []
    for kk in k_range:
        labels = KMeans(kk, random_state=seed, n_init=10).fit(xs).labels_
        sil.append([kk, round(float(silhouette_score(xs[idx], labels[idx])), 3)])
    return sil


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain and methodology) under the names the site and reports read.
    TR: Site ağacında (domain ve methodology), sitenin ve raporların okuduğu adlarla yayımlanır.
    """
    bc = res["brand"]
    return {
        "domain": {
            "segment_ladder": res["segment"], "body_median": res["body"], "age_depreciation": res["age"],
            "km_price_scope": res["km_scope"], "km_price": res["km_rows"],
            "brand_compare": {**bc, "cliffs_delta": round(bc["cliffs_delta"], 4),
                              "note": ("Mann-Whitney U: iki markanın fiyat dağılımı farkı (p<0.05 anlamlı). "
                                      "Cliff's δ: etki büyüklüğü (|δ|>0.33 orta, >0.47 büyük fark).")},
            "age_km_note": "Medyan tipik fiyat, ortalama aykırı-etkili. Açıklık = fiyat çarpıklığı."},
        "methodology": {"kmeans_selection": {"silhouette": res["silhouette"]}},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
km_rows, km_scope = mileage_curve(listings, KM_BIN, KM_LIMIT)
res = {"segment": median_by(listings, "segment"), "body": median_by(listings, "kb_body_type", MIN_BODY_N),
       "age": age_curve(listings, MAX_AGE, MIN_AGE_N), "km_rows": km_rows, "km_scope": km_scope,
       "brand": brand_compare(listings), "silhouette": silhouette_by_k(listings, K_RANGE, SILHOUETTE_SAMPLE, SEED)}
print("silhouette by k | k'ye göre silhouette:", res["silhouette"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("05_segmentation", to_metrics(res)))
