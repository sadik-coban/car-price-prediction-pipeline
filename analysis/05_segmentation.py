"""
05_segmentation.py
EN: Technical report §5 — market structure: price by segment and body type, depreciation curves by age
    and mileage, the BMW/Audi price comparison, and KMeans (k=3, fixed for interpretability) with PCA.
TR: Teknik rapor §5 — piyasa yapısı: segment ve kasa tipine göre fiyat, yaş ve kilometreye göre değer kaybı
    eğrileri, BMW/Audi fiyat karşılaştırması, KMeans (k=3, yorumlanabilirlik için sabit) ve PCA.
Output / Çıktı: metrics/05_segmentation.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.common import NUM, load_clean, save_metrics

SEED = 42
K = 3                      # fixed for interpretability, not chosen by silhouette | silhouette ile değil, sabit
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE = 3000
MIN_BODY_N, MIN_AGE_N, MAX_AGE = 80, 10, 25
KM_BIN, KM_LIMIT = 25000, 400000
DAMAGE_AXES = ("door_", "fender_", "bumper_")


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


def cluster_name(m):
    """
    EN: A short name for a cluster from its heavy-damage share, age and mileage.
    TR: Ağır hasar payı, yaş ve kilometresinden bir küme için kısa ad.
    """
    if m["is_heavy_damaged"].mean() > 0.4:
        return "Hasarlı"
    if m["vehicle_age"].median() <= 10 and m["gb_mileage"].median() < 160000:
        return "Genç & temiz premium"
    if m["vehicle_age"].median() >= 13:
        return "Yaşlı & yüksek-km ekonomik"
    return "Orta segment"


def kmeans_and_pca(listings, k, k_range, sample, seed):
    """
    EN: Standardised numeric features (median-filled), elbow + silhouette for k in k_range (silhouette on a
        random sample), KMeans with the fixed k, cluster profiles, and a 3-component PCA.
        Returns: dict with elbow, silhouette, profiles, pca axes and the 2-D/3-D scatter points.
    TR: Standartlaştırılmış sayısal öznitelikler (medyanla doldurulmuş), k_range için dirsek + silhouette
        (silhouette rastgele örneklemde), sabit k ile KMeans, küme profilleri ve 3 bileşenli PCA.
        Döndürür: dirsek, silhouette, profiller, PCA eksenleri ve 2-B/3-B nokta bulutu içeren sözlük.
    """
    nc = listings[NUM].fillna(listings[NUM].median())
    xs = StandardScaler().fit_transform(nc)
    idx = np.random.default_rng(seed).choice(len(xs), sample, replace=False)
    elbow, sil = [], []
    for kk in k_range:
        km_k = KMeans(kk, random_state=seed, n_init=10).fit(xs)
        elbow.append([kk, round(float(km_k.inertia_), 0)])
        sil.append([kk, round(float(silhouette_score(xs[idx], km_k.labels_[idx])), 3)])
    labels = KMeans(k, random_state=seed, n_init=10).fit(xs).labels_
    pca_m = PCA(3, random_state=seed).fit(xs)
    pca = pca_m.transform(xs)
    profiles = []
    for c in range(k):
        mask = labels == c
        m = listings[mask]
        z = (nc[mask].mean() - nc.mean()) / nc.std()
        top = z.abs().sort_values(ascending=False).head(3)
        damage_axis = str(top.index[0]).startswith(DAMAGE_AXES) or str(top.index[0]).endswith("_state")
        profiles.append({"cluster": int(c), "name": "Hasar yoğun" if damage_axis else cluster_name(m), "n": int(mask.sum()),
                         "median": float(m["price"].median()), "age": float(m["vehicle_age"].median()),
                         "km": float(m["gb_mileage"].median()), "hp": float(m["power_hp_val"].median()),
                         "heavy_damage_pct": round(100 * float(m["is_heavy_damaged"].mean()), 0),
                         "distinguishing": [[kk, "+" if z[kk] > 0 else "-"] for kk in top.index]})
    names = pd.Series([p["name"] for p in profiles]).value_counts()
    for p in profiles:                        # repeated names get numbered | tekrar eden adlar numaralanır
        if names[p["name"]] > 1:
            p["name"] = f"{p['name']} ({p['cluster'] + 1})"
    axes = [{"pc": f"PC{i + 1}", "var_pct": round(pca_m.explained_variance_ratio_[i] * 100, 1),
             "top": [[n, round(w, 2)] for n, w in sorted(zip(NUM, pca_m.components_[i]), key=lambda x: -abs(x[1]))[:4]]}
            for i in range(3)]
    return {"elbow": elbow, "silhouette": sil, "profiles": profiles, "axes": axes,
            "pca12": [[round(float(pca[i, 0]), 2), round(float(pca[i, 1]), 2), int(labels[i])] for i in range(len(xs))],
            "pca13": [[round(float(pca[i, 0]), 2), round(float(pca[i, 2]), 2), int(labels[i])] for i in range(len(xs))]}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain and methodology) under the names the site and reports read.
    TR: Site ağacında (domain ve methodology), sitenin ve raporların okuduğu adlarla yayımlanır.
    """
    km, bc = res["km"], res["brand"]
    best = max(km["silhouette"], key=lambda r: r[1])
    return {
        "domain": {
            "segment_ladder": res["segment"], "body_median": res["body"], "age_depreciation": res["age"],
            "km_price_scope": res["km_scope"], "km_price": res["km_rows"],
            "brand_compare": {**bc, "cliffs_delta": round(bc["cliffs_delta"], 4),
                              "note": ("Mann-Whitney U: iki markanın fiyat dağılımı farkı (p<0.05 anlamlı). "
                                      "Cliff's δ: etki büyüklüğü (|δ|>0.33 orta, >0.47 büyük fark).")},
            "age_km_note": "Medyan tipik fiyat, ortalama aykırı-etkili. Açıklık = fiyat çarpıklığı.",
            "kmeans": km["profiles"], "pca_scatter": km["pca12"], "pca_scatter_13": km["pca13"]},
        "methodology": {
            "kmeans_selection": {"elbow": km["elbow"], "silhouette": km["silhouette"], "chosen_k": K,
                                 "note": (f"Silhouette en yüksek k={best[0]} ({best[1]}); k={K} için "
                                         f"{dict(km['silhouette']).get(K)}. k={K} silhouette ile değil, "
                                         f"yorumlanabilirlik için sabit seçildi.")},
            "pca_axes": km["axes"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
km_rows, km_scope = mileage_curve(listings, KM_BIN, KM_LIMIT)
res = {"segment": median_by(listings, "segment"), "body": median_by(listings, "kb_body_type", MIN_BODY_N),
       "age": age_curve(listings, MAX_AGE, MIN_AGE_N), "km_rows": km_rows, "km_scope": km_scope,
       "brand": brand_compare(listings), "km": kmeans_and_pca(listings, K, K_RANGE, SILHOUETTE_SAMPLE, SEED)}
print("clusters | kümeler:", [(p["name"], p["n"]) for p in res["km"]["profiles"]])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("05_segmentation", to_metrics(res)))
