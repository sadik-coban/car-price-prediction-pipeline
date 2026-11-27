"""
Hata tanımları: kat dışı tahminin gerçek fiyata göre yüzde hatası (artık) ve büyük hata.
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
