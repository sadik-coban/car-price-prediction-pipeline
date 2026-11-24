"""
Piyasa: iki grubun fiyat dağılımlarını karşılaştıran etki büyüklüğü (Cliff's delta).
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np


def cliffs_delta(a, b):
    """
    Cliff's delta: a'dan rastgele bir değerin b'den rastgele bir değerden büyük olma olasılığı eksi küçük olma
    olasılığı, bütün (a, b) çiftleri üzerinden (örnekleme yok). -1 ile 1 arasında; 0 iki dağılım iç içe demek.
    """
    a, b = np.asarray(a, dtype=float), np.sort(np.asarray(b, dtype=float))
    smaller = np.searchsorted(b, a, side="left").sum()           # her a için ondan küçük b sayısı, toplam
    larger = (len(b) - np.searchsorted(b, a, side="right")).sum()  # her a için ondan büyük b sayısı, toplam
    return float((smaller - larger) / (len(a) * len(b)))
