"""
shap/02_oof_shap.py
EN: SHAP report §2 — out-of-fold (leak-free) SHAP. The model the report explains must be the one that made the
    error: the final model saw every listing in training, so here every listing is explained by the fold model
    that never saw it. The five folds are refit exactly as in 07_model_comparison (analysis/lib/cv.py) and each fold
    explains its own validation rows (shap.TreeExplainer, exact, log-price space). Gate: the refit OOF
    predictions must match 07's stored OOF row by row (≤ ₺1), otherwise nothing is written. The 170 SVD
    dimensions of the model/series name are summed per row into MODEL_SERIES (signed sum); the raw dimensions
    are kept as well, for the "members summed separately" comparison.
TR: SHAP raporu §2 — fold dışı (sızıntısız) SHAP. Raporun açıkladığı model hatayı yapan model olmalı: final
    model her ilanı eğitimde gördü, bu yüzden burada her ilan onu hiç görmemiş fold modeliyle açıklanır. Beş
    fold 07_model_comparison'daki gibi yeniden kurulur (analysis/lib/cv.py) ve her fold kendi doğrulama satırlarını
    açıklar (shap.TreeExplainer, exact, log fiyat uzayı). Kapı: yeniden kurulan OOF tahminleri 07'nin sakladığı
    OOF ile satır satır eşleşmeli (≤ ₺1), yoksa hiçbir şey yazılmaz. Model/seri adının 170 SVD boyutu satır
    başına MODEL_SERIES'te toplanır (işaretli toplam); ham boyutlar da "üyeler ayrı toplanınca" karşılaştırması
    için saklanır.
Output / Çıktı: metrics/shap/02_oof_shap.json · data/analysis/oof_shap.npz
"""

# %% [1] Setup | Kurulum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent.parent))

import numpy as np                                                             # noqa: E402
import shap                                                                    # noqa: E402

from lib.common import ANALYSIS_DIR, FEATURES, load_clean, save_metrics       # noqa: E402
from lib.cv import N_JOBS, PRICE_CAP, fit_lgb_fold, fold_matrices, load_oof, make_folds, price_metrics  # noqa: E402

MS = "MODEL_SERIES (text)"
TOL_MAX_TL, TOL_MAPE = 1.0, 0.01
NPZ = ANALYSIS_DIR / "oof_shap.npz"


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def text_only_key(col):
    """
    EN: Group key: the SVD dimensions of the name (model_*, series_*) → MODEL_SERIES; every other column
        keeps its own name.
    TR: Grup anahtarı: adın SVD boyutları (model_*, series_*) → MODEL_SERIES; öteki her kolon kendi adıyla.
    """
    return MS if col.startswith(("model_", "series_")) else col


def oof_shap(X, yl, folds):
    """
    EN: Refit the five fold models and explain each fold's validation rows. Returns: OOF log predictions, the
        grouped SHAP matrix (n × 24, float64), per-row base value, fold number, group names, and the raw
        SHAP of the SVD dimensions with their names.
    TR: Beş fold modelini yeniden kurar ve her fold'un doğrulama satırlarını açıklar. Döndürür: OOF log
        tahminleri, gruplu SHAP matrisi (n × 24, float64), satır başına taban değer, fold numarası, grup adları
        ve SVD boyutlarının ham SHAP'i ile adları.
    """
    n = len(X)
    pred, base, fold_of = np.zeros(n), np.zeros(n), np.full(n, -1, dtype=int)
    grouped = raw = groups = raw_names = None
    for k, (tr, va) in enumerate(folds, 1):
        Xtr, Xva, catc = fold_matrices(X, tr, va)
        mdl = fit_lgb_fold(Xtr, yl[tr], Xva, yl[va], catc)
        pred[va] = mdl.predict(Xva)
        expl = shap.TreeExplainer(mdl)
        sv = np.asarray(expl.shap_values(Xva))
        keys = [text_only_key(c) for c in Xva.columns]
        uniq = list(dict.fromkeys(keys))
        multi = [i for i, kk in enumerate(keys) if keys.count(kk) > 1]
        if grouped is None:
            grouped, raw = np.zeros((n, len(uniq))), np.zeros((n, len(multi)))
            groups, raw_names = uniq, [str(Xva.columns[i]) for i in multi]
        assert uniq == groups, f"fold {k}: group order changed | grup sırası değişti"
        assert [str(Xva.columns[i]) for i in multi] == raw_names, f"fold {k}: raw column order changed"
        raw[va] = sv[:, multi]
        for j, u in enumerate(uniq):
            grouped[va, j] = sv[:, [i for i, kk in enumerate(keys) if kk == u]].sum(axis=1)
        base[va] = float(np.ravel(expl.expected_value)[0])
        fold_of[va] = k
    return {"pred_log": pred, "sv": grouped, "base": base, "fold": fold_of, "groups": groups,
            "raw": raw, "raw_names": raw_names}


def gate(price, new_price, published_price, published_mape):
    """
    EN: Row-by-row difference between the refit OOF and 07's stored OOF, and the MAPE against the published
        one. Stops (nothing written) above ₺1 or 0.01 MAPE points.
    TR: Yeniden kurulan OOF ile 07'nin sakladığı OOF arasındaki satır satır fark ve yayımlanan MAPE'ye göre
        MAPE. ₺1'in ya da 0.01 MAPE puanının üstünde durur (hiçbir şey yazılmaz).
    """
    d = np.abs(new_price - published_price)
    mape_new = float(np.mean(np.abs((price - new_price) / price)) * 100)
    if d.max() > TOL_MAX_TL or abs(mape_new - published_mape) > TOL_MAPE:
        raise SystemExit(f"G-OOF GATE FAILED | G-OOF KAPISI DÜŞTÜ: max ₺{d.max():,.2f} · MAPE {mape_new:.4f} vs "
                         f"{published_mape:.4f} — nothing written | hiçbir dosya yazılmadı")
    return {"max_gap_tl": float(d.max()), "mean_gap_tl": float(d.mean()), "mape_refit": round(mape_new, 4),
            "mape_published": round(published_mape, 4), "n_threads": N_JOBS, "threshold_max_tl": TOL_MAX_TL,
            "threshold_mape": TOL_MAPE}


def global_tables(sv, groups, raw, raw_names):
    """
    EN: Global importance two ways: mean |signed group sum| (the rule the report uses) and the sum of the
        members' mean |SHAP| (no cancellation; the same for single-member groups).
        Returns: (table, table_separate) as [[group, mean |SHAP|, share %], ...], largest first.
    TR: Küresel önem iki yolla: ortalama |işaretli grup toplamı| (raporun kuralı) ve üyelerin ortalama
        |SHAP|'lerinin toplamı (iptal yok; tek üyeli gruplarda aynı).
        Döndürür: (tablo, tablo_ayri) — [[grup, ortalama |SHAP|, pay %], ...], büyükten küçüğe.
    """
    mab = np.abs(sv).mean(axis=0)
    tot = float(mab.sum()) or 1.0
    table = [[g, round(float(v), 4), round(float(v) / tot * 100, 1)] for g, v in sorted(zip(groups, mab), key=lambda kv: -kv[1])]
    sep = {g: float(v) for g, v in zip(groups, mab)}
    raw_groups = [text_only_key(c) for c in raw_names]
    raw_mab = np.abs(raw).mean(axis=0)
    for g in set(raw_groups):
        sep[g] = float(sum(v for v, gg in zip(raw_mab, raw_groups) if gg == g))
    tot2 = sum(sep.values()) or 1.0
    return table, [[g, round(v, 4), round(v / tot2 * 100, 1)] for g, v in sorted(sep.items(), key=lambda kv: -kv[1])]


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under oof_shap (the SHAP report reads the gate and the global tables).
    TR: oof_shap altında yayımlanır (SHAP raporu kapıyı ve küresel tabloları okur).
    """
    return {"oof_shap": {
        "gate": res["gate"], "n": res["n"], "groups": res["groups"], "global": res["table"],
        "global_separate": res["table_sep"],
        "grouping_rule": ("kuresel = mean|sum phi| (grup ici isaretli toplam, sonra mutlak ortalama); "
                            "kuresel_ayri = sum mean|phi| (her uye ayri, iptal yok). Tek uyeli gruplarda ayni."),
        "note": ("Her ilan, onu egitimde hic gormemis fold modeliyle aciklandi (5-fold, "
                "ureticinin kurulumunun aynisi). SHAP degerleri log1p(fiyat) uzayinda ve "
                "text_only_key ile 24 kaleme gruplandi.")}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
X = listings[FEATURES].copy()
price = listings["price"].values.astype(float)
r = oof_shap(X, np.log1p(price), make_folds(len(X)))
published = np.clip(oof["lgb"].values, 0, PRICE_CAP)
res = {"n": len(X), "groups": r["groups"],
       "gate": gate(price, np.expm1(r["pred_log"]), published, round(price_metrics(price, published)["MAPE"], 2))}
res["table"], res["table_sep"] = global_tables(r["sv"], r["groups"], r["raw"], r["raw_names"])
print(f"G-OOF ✓ max ₺{res['gate']['max_gap_tl']:.2f} ·", res["table"][:3])

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
NPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(NPZ, shap=r["sv"], base=r["base"], fold=r["fold"].astype(np.int8), groups=np.array(r["groups"]),
                    oof=np.expm1(r["pred_log"]), raw_shap=r["raw"], raw_names=np.array(r["raw_names"]),
                    raw_groups=np.array([text_only_key(c) for c in r["raw_names"]]), run_id=np.array(oof_info["run_id"]))
print("written | yazıldı:", save_metrics("shap/02_oof_shap", to_metrics(res), run_id=oof_info["run_id"]))
