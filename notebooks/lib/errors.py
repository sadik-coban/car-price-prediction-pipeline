"""
Hata tanımları: kat dışı tahminin gerçek fiyata göre yüzde hatası (artık) ve büyük hata; en büyük ₺ hatalarının
yönü için simetrik karşılaştırma.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np

PRICE_CAP = 1.5e7            # tahmin 0 ile 15 milyon ₺ arasına kırpılır; servis edilen tahminin sınırı
LARGE_ERROR_PCT = 20.0       # |artık| bunun üstündeyse büyük hata


def residual_pct(price, pred):
    """
    Artık, gerçek fiyatın yüzdesi olarak: (gerçek - tahmin) / gerçek · 100; tahmin önce [0, PRICE_CAP]'e kırpılır.
    Eksi artık: tahmin ilan fiyatının üstünde.
    """
    price = np.asarray(price, dtype=float)
    return (price - np.clip(np.asarray(pred, dtype=float), 0, PRICE_CAP)) / price * 100


def large_error(resid):
    """Büyük hata: |artık| > %20; artık önce iki haneye yuvarlanır (tablolarda gösterildiği gibi)."""
    return np.abs(np.round(np.asarray(resid, dtype=float), 2)) > LARGE_ERROR_PCT


def top_lira(price, pred, top_n):
    """En büyük top_n ₺ hatasının (|tahmin - gerçek|) konumları, büyükten küçüğe; eşitlikte satır sırası."""
    gap = np.abs(np.asarray(pred, dtype=float) - np.asarray(price, dtype=float))
    return np.argsort(-gap, kind="stable")[:top_n]


def symmetric_under_counts(price, pred, top_n, n_draws, seed):
    """
    Fiyat tahminin etrafında log ölçekte simetrik dağılsaydı en büyük top_n ₺ hatasının kaçı eksik tahmin (tahmin
    fiyatın altında) olurdu: her ilan kendi |log hatasını| kendi tahmini etrafında rastgele bir işaretle taşır, en büyük
    top_n ₺ hatası yeniden seçilir. Aynı log hata, fiyat tahminin üstündeyken ₺ olarak daha büyüktür; sonuç bu yüzden
    top_n / 2'nin üstünde çıkar. Döndürür: çekiliş başına sayı (n_draws uzunluğunda).
    """
    pred = np.asarray(pred, dtype=float)
    log_gap = np.abs(np.log1p(pred) - np.log1p(np.asarray(price, dtype=float)))
    rng = np.random.default_rng(seed)
    counts = np.empty(n_draws, dtype=int)
    for i in range(n_draws):
        drawn = np.expm1(np.log1p(pred) + rng.choice([-1.0, 1.0], size=len(pred)) * log_gap)
        top = top_lira(drawn, pred, top_n)
        counts[i] = int((drawn[top] > pred[top]).sum())
    return counts
