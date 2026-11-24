"""
Katlar: kat dışı sayıların ortak 5 katı (hedonik regresyon ve tahmin modeli aynı fonksiyonu kullanır).
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 5


def make_folds(n):
    """
    n satırın 5 (eğitim, doğrulama) konum çifti: karıştırılmış KFold, tohum 42. Katlar satır sırasına bağlıdır;
    data.load_rows sabit bir sıra (ad_id) verir.
    """
    return list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))
