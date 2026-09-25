"""
07_lofo.py
EN: Technical report §7 — leave-one-feature-out: how much the OOF RMSE rises when a feature (or a group of
    features) is removed and the model refit with the same 5-fold CV and settings. Removing a group and its
    members one by one shows whether features stand in for each other (hp alone vs hp + cc together).
    The trees early stopping chose are kept for every refit.
TR: Teknik rapor §7 — bir-özniteliği-çıkar (LOFO): bir öznitelik (ya da öznitelik grubu) çıkarılıp model aynı
    5-fold CV ve ayarla yeniden kurulunca OOF RMSE ne kadar artıyor. Grubu ve üyelerini tek tek çıkarmak,
    özniteliklerin birbirinin yerini tutup tutmadığını gösterir (yalnız hp ile hp + cc birlikte).
    Her yeniden kurulumda erken durdurmanın seçtiği ağaç sayısı saklanır.
Output / Çıktı: metrics/07_lofo.json
"""

# %% [1] Setup | Kurulum
import numpy as np

from lib.common import FEATURES, NUM, load_clean, save_metrics
from lib.cv import lgb_oof, make_folds, price_metrics

GROUPS = {
    "ENGINE": ["power_hp_val", "engine_cc_val"],
    "DAMAGE_COLS": ["roof_state", "hood_state", "trunk_state", "door_changed", "door_painted", "door_local",
                    "fender_changed", "fender_painted", "fender_local", "bumper_changed", "bumper_painted",
                    "bumper_local", "is_heavy_damaged"],
    "MODEL_SERIES": ["model", "series"],
}


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def removal_sets(num, groups):
    """
    EN: What to remove, in order: every numeric feature alone, every other group member alone (panel states,
        model, series), then each group whole. Returns: {name: [columns]}.
    TR: Neyin çıkarılacağı, sırayla: her sayısal öznitelik tek başına, grupların öteki üyeleri tek başına
        (panel durumları, model, seri), sonra her grup bütün hâlinde. Döndürür: {ad: [kolonlar]}.
    """
    singles = {c: [c] for c in num}
    for cols in groups.values():
        for c in cols:
            singles.setdefault(c, [c])
    return {**singles, **groups}


def rmse_without(X, yl, price, folds, drop):
    """
    EN: OOF RMSE (₺) and the per-fold trees after removing the columns in drop.
    TR: drop'taki kolonlar çıkarıldıktan sonra OOF RMSE (₺) ve fold başına ağaç sayısı.
    """
    r = lgb_oof(X, yl, folds, cols=[c for c in FEATURES if c not in drop])
    return price_metrics(price, np.expm1(r["pred_log"]))["RMSE"], r["iters"]


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology.lofo = [name, ΔRMSE ₺, single/group], largest first;
        methodology.lofo_trees = trees per fold for the base and every removal).
    TR: Site ağacında yayımlanır (methodology.lofo = [ad, ΔRMSE ₺, tekil/grup], büyükten küçüğe;
        methodology.lofo_trees = taban ve her çıkarma için fold başına ağaç).
    """
    rows = [[name, round(rmse - res["base_rmse"], 1), "group" if name in GROUPS else "single"]
            for name, (rmse, _it) in res["removed"].items()]
    return {"methodology": {"lofo": sorted(rows, key=lambda x: -x[1]),
                            "lofo_trees": {"baseline": res["base_iters"],
                                          **{name: it for name, (_r, it) in res["removed"].items()}}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak (~15 min | ~15 dk)
X = listings[FEATURES].copy()
price = listings["price"].values.astype(float)
yl = np.log1p(price)
folds = make_folds(len(X))
base_rmse, base_iters = rmse_without(X, yl, price, folds, [])
removed = {}
for name, drop in removal_sets(NUM, GROUPS).items():
    removed[name] = rmse_without(X, yl, price, folds, drop)
    print(f"  {name:16} ΔRMSE ₺{removed[name][0] - base_rmse:,.0f}")
res = {"base_rmse": base_rmse, "base_iters": base_iters, "removed": removed}

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("07_lofo", to_metrics(res)))
