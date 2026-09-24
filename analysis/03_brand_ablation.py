"""
03_brand_ablation.py
EN: Technical report §3 — does the brand carry information once series and model are known? The full model
    is refit with only the identity columns changed: brand only (series, model and the segment derived from
    them removed), series + model (brand removed), and brand + series + model (the headline model). Same
    5-fold OOF and settings as the headline. Gate: if brand wins splits in the CV models but removing it
    leaves the OOF identical, the column subset is not reaching the model — the run stops.
TR: Teknik rapor §3 — seri ve model bilinince marka bilgi taşıyor mu? Tam model yalnız kimlik kolonları
    değiştirilerek yeniden kurulur: yalnız marka (seri, model ve onlardan türetilen segment çıkarılır),
    seri + model (marka çıkarılır) ve marka + seri + model (manşet model). Manşetle aynı 5-fold OOF ve ayar.
    Kapı: marka CV modellerinde bölme kazanıyor ama çıkarılınca OOF birebir aynı kalıyorsa kolon alt kümesi
    modele ulaşmıyor demektir — koşum durur.
Output / Çıktı: metrics/03_brand_ablation.json
"""

# %% [1] Setup | Kurulum
import numpy as np

from lib.common import FEATURES, load_clean, save_metrics
from lib.cv import lgb_oof, make_folds, price_metrics, round_metrics

ARMS = {"sadece_brand": [c for c in FEATURES if c not in ("model", "series", "segment")],
        "seri_model": [c for c in FEATURES if c != "brand"],
        "brand_seri_model": FEATURES}


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def run_arms(X, yl, folds, arms):
    """
    EN: 5-fold OOF LightGBM for every arm (feature subset). Returns: {arm: lgb_oof result}.
    TR: Her kol (öznitelik alt kümesi) için 5-fold OOF LightGBM. Döndürür: {kol: lgb_oof sonucu}.
    """
    return {name: lgb_oof(X, yl, folds, cols=cols) for name, cols in arms.items()}


def plumbing_check(full, without_brand):
    """
    EN: How often brand splits in the headline CV models, and whether removing brand changed the OOF at all.
        Identical OOF is legitimate only if brand never splits; otherwise the run stops.
        Returns: {"brand_split_cv", "oof_ayni", "max_fark_tl"}.
    TR: Marka manşet CV modellerinde kaç kez bölme yapıyor ve markayı çıkarmak OOF'u hiç değiştirdi mi.
        Birebir aynı OOF yalnız marka hiç bölme yapmıyorsa meşru; değilse koşum durur.
        Döndürür: {"brand_split_cv", "oof_ayni", "max_fark_tl"}.
    """
    splits = int(sum(d.get("brand", 0) for d in full["splits"]))
    a, b = np.expm1(without_brand["pred_log"]), np.expm1(full["pred_log"])
    same = bool(np.array_equal(a, b))
    if same and splits > 0:
        raise SystemExit(f"BRAND ABLATION PLUMBING ERROR | MARKA ABLASYONU TESİSAT HATASI: brand splits {splits} "
                         f"times in the CV models but removing it leaves the OOF identical")
    return {"brand_split_cv": splits, "oof_ayni": same, "max_fark_tl": float(np.abs(a - b).max())}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.brand_ablation).
    TR: Site ağacında yayımlanır (domain.brand_ablation).
    """
    chk = res["check"]
    return {"domain": {"brand_ablation": {
        **{arm: round_metrics(m) for arm, m in res["metrics"].items()},
        "dogrulama": {"brand_split_cv": chk["brand_split_cv"], "oof_ayni": chk["oof_ayni"],
                      "max_fark_tl": round(chk["max_fark_tl"], 2)},
        "not": ('TAM MODELDE kimlik ablasyonu: marka/seri/model kolonları değişir; "yalnız marka" kolu seri ve model '
                'adından türetilen segmenti de çıkarır, diğer öznitelikler sabit. Manşetle aynı 5-fold OOF ve parametreler; '
                'model/seri fold-içi TF-IDF+SVD ile girer, brand kategorik. brand+seri+model = manşet model. '
                "dogrulama: marka CV fold modellerinde kaç bölme kazandı ve iki kolun OOF'u aynı mı.")}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
X = listings[FEATURES].copy()
price = listings["price"].values.astype(float)
yl = np.log1p(price)
arms = run_arms(X, yl, make_folds(len(X)), ARMS)
res = {"metrics": {k: price_metrics(price, np.expm1(v["pred_log"])) for k, v in arms.items()},
       "check": plumbing_check(arms["brand_seri_model"], arms["seri_model"])}
print({k: round(v["MAPE"], 2) for k, v in res["metrics"].items()}, res["check"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("03_brand_ablation", to_metrics(res)))
