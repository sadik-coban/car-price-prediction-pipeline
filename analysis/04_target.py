"""
04_target.py
EN: Technical report §4 — the target: how skewed the raw price is, what the log transform does, and the
    price histogram (the collection cap is included, so no listing falls outside the bins).
TR: Teknik rapor §4 — hedef: ham fiyat ne kadar çarpık, log dönüşümü ne yapıyor ve fiyat histogramı
    (toplama tavanı dahil; hiçbir ilan kovaların dışında kalmaz).
Output / Çıktı: metrics/04_target.json
"""

# %% [1] Setup | Kurulum
import numpy as np
from scipy import stats

from lib.common import load_clean, save_metrics

HIST_EDGES = np.linspace(0, 6.5e6, 27)       # 250K-wide bins up to the 6.5M cap | tavana kadar 250K'lık kovalar


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def price_distribution(price):
    """
    EN: Skewness of the raw and log1p price, the median and the P10/P90 of the price.
    TR: Ham ve log1p fiyatın çarpıklığı, fiyatın medyanı ve P10/P90'ı.
    """
    return {"skew_raw": float(stats.skew(price)), "skew_log": float(stats.skew(np.log1p(price))),
            "median": float(np.median(price)),
            "p10": float(np.percentile(price, 10)), "p90": float(np.percentile(price, 90))}


def price_histogram(price, edges):
    """
    EN: Listing count per price bin; stops if any listing falls outside the edges.
        Returns: [[bin left edge, count], ...].
    TR: Fiyat kovası başına ilan sayısı; kenarların dışına düşen ilan varsa durur.
        Döndürür: [[kova sol kenarı, sayı], ...].
    """
    counts, _ = np.histogram(price, bins=edges)
    assert int(counts.sum()) == len(price), f"histogram leaves {len(price) - int(counts.sum())} listings out"
    return [[float(edges[i]), int(counts[i])] for i in range(len(counts))]


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain) under the names the site and reports read.
    TR: Site ağacında (domain), sitenin ve raporların okuduğu adlarla yayımlanır.
    """
    return {"domain": {"price_dist": res["dist"], "price_histogram": res["hist"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
res = {"dist": price_distribution(price), "hist": price_histogram(price, HIST_EDGES)}
print(res["dist"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("04_target", to_metrics(res)))
