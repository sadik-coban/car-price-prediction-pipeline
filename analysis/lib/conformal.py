"""
conformal.py
EN: The conformal-interval pieces shared by 08_conformal_coverage (cross-calibrated coverage on the OOF) and
    09_backtest (forward coverage on later snapshots): the interval around a prediction, the finite-sample
    quantile, and the predicted-price band of a listing.
TR: 08_conformal_coverage (OOF'ta çapraz kalibreli kapsama) ve 09_backtest (sonraki taramalarda ileri kapsama) ortak
    conformal aralık parçaları: tahmin etrafındaki aralık, sonlu örneklem yüzdeliği ve bir ilanın tahmin fiyatı bandı.
"""
import numpy as np

from .cv import PRICE_CAP

LEVEL = 0.90


def interval(pred_price, q):
    """
    EN: The conformal interval around each prediction: expm1(log1p(prediction) ± q), the lower end clipped at
        0 and the upper at log1p(15M); q may be one number or one per listing. Returns: (lower, upper) in ₺.
    TR: Her tahmin etrafındaki conformal aralık: expm1(log1p(tahmin) ± q); alt uç 0'da, üst uç log1p(15M)'de
        kırpılır; q tek sayı ya da ilan başına olabilir. Döndürür: (alt, üst) ₺.
    """
    pl = np.log1p(np.clip(pred_price, 0, PRICE_CAP))
    return np.expm1(np.clip(pl - q, 0, None)), np.expm1(np.clip(pl + q, 0, np.log1p(PRICE_CAP)))


def inside(price, pred_price, q):
    """EN: True where the actual price is inside the interval. / TR: Gerçek fiyat aralığın içindeyse True."""
    lo, hi = interval(pred_price, q)
    return (price >= lo) & (price <= hi)


def conformal_quantile(abs_err, level=LEVEL):
    """
    EN: The finite-sample conformal quantile: the ceil((n+1)·level)-th smallest absolute error. Stops if the
        calibration part is too small for that rank.
    TR: Sonlu örneklem conformal yüzdeliği: en küçükten ceil((n+1)·level)'inci mutlak hata. Kalibrasyon kısmı o
        sıra için çok küçükse durur.
    """
    r = np.sort(np.asarray(abs_err, dtype=float))
    k = int(np.ceil((len(r) + 1) * level))
    assert 1 <= k <= len(r), f"calibration part too small | kalibrasyon kısmı çok küçük: n={len(r)}"
    return float(r[k - 1])


def band_of(pred_price, bounds):
    """
    EN: The predicted-price band of each listing, 0–3, on three cut points; a value on a cut point goes to the lower
        band (as pd.qcut does).
    TR: Her ilanın tahmin fiyatı bandı, 0–3, üç kesim noktasına göre; kesim noktasındaki değer alttaki banda gider
        (pd.qcut gibi).
    """
    return np.searchsorted(np.asarray(bounds, dtype=float), pred_price, side="left")
