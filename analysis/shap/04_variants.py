"""
shap/04_variants.py
EN: SHAP report §4 — three variants, the same price, different reasons. CatBoost's SHAP can only be computed on
    the FINAL models (shap.TreeExplainer does not support CatBoost text features), so here all three final
    models are explained on all listings: LightGBM (shap.TreeExplainer, exact; additivity gate), CatBoost on
    the same TF-IDF + SVD matrix and CatBoost with its native text engine (CatBoost's own ShapValues, one
    thread — several threads moved the decimals between runs). Same rule as §3: a group's value is the signed
    per-row sum, then mean |·|. The matrix is rebuilt from the served encoders, as serving builds it. Also the
    site's domain.shap and its PNGs (data/serving/shap_plots/). Needs 07_final_model's serving files.
TR: SHAP raporu §4 — üç varyant, aynı fiyat, farklı gerekçe. CatBoost'un SHAP'i yalnız FİNAL modellerde
    hesaplanabiliyor (shap.TreeExplainer CatBoost metin özniteliklerini desteklemiyor); bu yüzden burada üç
    final model bütün ilanlarda açıklanır: LightGBM (shap.TreeExplainer, exact; toplamsallık kapısı), aynı
    TF-IDF + SVD matrisinde CatBoost ve kendi metin motoruyla CatBoost (CatBoost'un kendi ShapValues'ı, tek iş
    parçacığı — çok iş parçacığında ondalıklar koşudan koşuya oynuyordu). §3'le aynı kural: grup değeri satır
    başına işaretli toplam, sonra ortalama |·|. Matris servis edilen encoder'lardan, servisin kurduğu gibi
    yeniden kurulur. Ayrıca sitenin domain.shap'i ve PNG'leri (data/serving/shap_plots/).
    07_final_model'in servis dosyalarına ihtiyaç duyar.
Output / Çıktı: metrics/shap/04_variants.json · data/serving/shap_plots/*.png
"""

# %% [1] Setup | Kurulum
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent.parent))

import matplotlib                                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402
import shap                                                                    # noqa: E402
from catboost import CatBoostRegressor, Pool                                   # noqa: E402

from lib.common import ROOT, load_clean, save_metrics                          # noqa: E402

SEED = 42
MS = "MODEL_SERIES (text)"
SERVING_DIR = ROOT / "data" / "serving"
PLOT_DIR = SERVING_DIR / "shap_plots"
DAMAGE_COLS = ("roof_state", "hood_state", "trunk_state", "door_changed", "door_painted", "door_local",
               "fender_changed", "fender_painted", "fender_local", "bumper_changed", "bumper_painted",
               "bumper_local", "is_heavy_damaged")


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def group_key(col):
    """
    EN: Coarse group: SVD dimensions → MODEL_SERIES, hp + cc → ENGINE, damage features → DAMAGE.
    TR: Kaba grup: SVD boyutları → MODEL_SERIES, hp + cc → ENGINE, hasar öznitelikleri → DAMAGE.
    """
    if col.startswith("model_") or col.startswith("series_"):
        return MS
    if col in ("power_hp_val", "engine_cc_val"):
        return "ENGINE"
    return "DAMAGE" if col in DAMAGE_COLS else col


def native_key(col):
    """
    EN: As group_key, but the native model's raw model and series names also count as MODEL_SERIES.
    TR: group_key gibi, ama native modelin ham model ve seri adı da MODEL_SERIES sayılır.
    """
    return MS if col in ("model", "series") else group_key(col)


def grouped(sv, cols, keyfn=group_key):
    """
    EN: Per group: mean |signed per-row sum of its members' SHAP| and its share. Returns: [[group, value, %], ...].
    TR: Grup başına: ortalama |üyelerin SHAP'inin satır başına işaretli toplamı| ve payı. Döndürür: [[grup, değer, %], ...].
    """
    keys = [keyfn(c) for c in cols]
    g = {u: float(np.abs(sv[:, [i for i, k in enumerate(keys) if k == u]].sum(axis=1)).mean()) for u in dict.fromkeys(keys)}
    tot = sum(g.values()) or 1.0
    return [[k, round(v, 4), round(v / tot * 100, 1)] for k, v in sorted(g.items(), key=lambda x: -x[1])]


def separate_sum(sv, cols, keyfn=group_key):
    """
    EN: Per group: the sum of its members' mean |SHAP| (no cancellation) and its share.
    TR: Grup başına: üyelerinin ortalama |SHAP|'lerinin toplamı (iptal yok) ve payı.
    """
    g = {}
    for k, v in zip([keyfn(c) for c in cols], np.abs(sv).mean(axis=0)):
        g[k] = g.get(k, 0.0) + float(v)
    tot = sum(g.values()) or 1.0
    return [[k, round(v, 4), round(v / tot * 100, 1)] for k, v in sorted(g.items(), key=lambda x: -x[1])]


def serving_matrix(listings, enc):
    """
    EN: The model matrix as serving builds it from encoders: category codes (unseen → stop), numeric columns,
        stored TF-IDF + SVD of model/series, in feat_cols order. Returns: (TF-IDF/SVD matrix, native matrix).
    TR: Servisin encoder'lardan kurduğu model matrisi: kategori kodları (görülmemiş → dur), sayısal kolonlar,
        saklanan TF-IDF + SVD ile model/seri, feat_cols sırasında. Döndürür: (TF-IDF/SVD matrisi, native matris).
    """
    parts = [pd.DataFrame({c: listings[c].map(enc["cat_maps"][c]) for c in enc["CAT"]}, index=listings.index),
             pd.DataFrame({c: listings[c].values for c in enc["NUM"]}, index=listings.index)]
    for txt in enc["TEXT"]:
        t = enc["tfidf"][txt]
        emb = t["svd"].transform(t["vec"].transform(listings[txt]))
        parts.append(pd.DataFrame(emb, columns=[f"{txt}_{i}" for i in range(t["n"])], index=listings.index))
    Xf = pd.concat(parts, axis=1)
    assert Xf[enc["CAT"]].notna().all().all(), "a category is missing from the encoders | kategori haritasında yok"
    return Xf[enc["feat_cols"]], listings[enc["TEXT"] + enc["CAT"] + enc["NUM"]].copy()


def lgb_shap(model, Xf):
    """
    EN: Exact TreeSHAP of the final LightGBM; stops if SHAP + base does not rebuild the prediction (< 1e-4).
        Returns: (SHAP matrix, base value, additivity error).
    TR: Final LightGBM'in exact TreeSHAP'i; SHAP + taban tahmini vermezse (< 1e-4) durur.
        Döndürür: (SHAP matrisi, taban değer, toplamsallık hatası).
    """
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(Xf))
    base = float(np.ravel(expl.expected_value)[0])
    err = float(np.abs(sv.sum(axis=1) + base - model.predict(Xf)).max())
    assert err < 1e-4, f"G2 additivity failed | toplamsallık bozuk {err:.2e}"
    return sv, base, err


def catboost_shap(model, pool):
    """
    EN: CatBoost's own ShapValues on one thread (the last column, the expected value, dropped).
    TR: CatBoost'un kendi ShapValues'ı, tek iş parçacığında (son kolon, beklenen değer, atılır).
    """
    return np.asarray(model.get_feature_importance(pool, type="ShapValues", thread_count=1))[:, :-1]


def fig_bar(table, title):
    """
    EN: Site PNG: grouped SHAP shares as horizontal bars. / TR: Site PNG'si: gruplu SHAP payları yatay bar.
    """
    names, vals = [r[0] for r in table], [r[2] for r in table]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.45)))
    ax.barh(range(len(names)), vals, color=["#1B5FAE" if n in (MS, "ENGINE", "DAMAGE") else "#2E7CD6" for n in names])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("katkı (%)")
    ax.set_title(title, fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    return fig


def fig_site_beeswarm(sv, data, native, title):
    """
    EN: Site PNG: SHAP beeswarm; only the SVD dimensions are summed into MODEL_SERIES (native: no grouping).
        Colour = feature value (text/categories grey). Seeded so the jitter is stable.
    TR: Site PNG'si: SHAP beeswarm; yalnız SVD boyutları MODEL_SERIES'te toplanır (native: gruplama yok).
        Renk = öznitelik değeri (metin/kategori gri). Yayılma kararlı olsun diye tohumlanır.
    """
    cols = list(data.columns)
    keys = [MS if c.startswith(("model_", "series_")) else c for c in cols]
    uniq = list(dict.fromkeys(keys))
    sv_g, val_g = np.zeros((len(data), len(uniq))), np.zeros((len(data), len(uniq)))
    for gi, g in enumerate(uniq):
        idx = [i for i, k in enumerate(keys) if k == g]
        sv_g[:, gi] = sv[:, idx].sum(axis=1)
        val_g[:, gi] = data.iloc[:, idx].apply(pd.to_numeric, errors="coerce").mean(axis=1).values
    exp = shap.Explanation(values=sv_g, data=val_g, feature_names=uniq,
                           base_values=np.full(len(sv_g), float(sv.sum(axis=1).mean())))
    np.random.seed(SEED)
    plt.figure()
    shap.plots.beeswarm(exp, max_display=len(uniq), show=False)
    grouping = "native: ham feature (gruplama yok)" if native else "yalnız SVD→MODEL_SERIES gruplu"
    plt.title(f"{title} — SHAP beeswarm ({grouping})", fontsize=10)
    plt.tight_layout()
    return plt.gcf()


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.shap) and under shap_final (the SHAP report's §2/§4).
    TR: Site ağacında (domain.shap) ve shap_final altında (SHAP raporunun §2/§4'ü) yayımlanır.
    """
    t = res["tables"]
    return {"domain": {"shap": {
                "n_used": res["n"],
                "method": ("TÜM VERİ (örnekleme YOK). Grup SHAP = satır bazında üye SHAP toplamı (işaretli net), sonra "
                           "ortalama-mutlak. LGB→shap.TreeExplainer, CatBoost→kendi ShapValues (text-güvenli, EXACT)."),
                "note": ("SVD boyutları (model_0..) → MODEL_SERIES gruplandı. ENGINE (hp+cc), DAMAGE (boya/değişen/hasar) "
                        "gruplu. Native ham text (SVD yok)."),
                "png": sorted(f"shap_plots/{p}" for p in res["pngs"]),
                "lightgbm_tfidf_svd": t["lgb"], "catboost_tfidf_svd": t["cb"], "catboost_native": t["native"]}},
            "shap_final": {"add_err": res["add_err"], "base_log": res["base"], "final_model_table": t["lgb"],
                           "final_model_table_separate": t["lgb_sep"], "catboost_tfidf_svd": t["cb"],
                           "catboost_tfidf_svd_separate": t["cb_sep"], "catboost_native": t["native"],
                           "catboost_native_separate": t["native_sep"], "catboost_native_combined": t["native_joined"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
with open(SERVING_DIR / "final_model.pkl", "rb") as fh:
    enc = pickle.load(fh)
native = CatBoostRegressor()
native.load_model(str(SERVING_DIR / "serve" / "catboost_native.cbm"))

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
Xf, Xn = serving_matrix(listings, enc)
sv_lgb, base, add_err = lgb_shap(enc["final_lgb"], Xf)
sv_cb = catboost_shap(enc["final_cb"], Pool(Xf, cat_features=enc["cat_idx"]))
native_pool = Pool(Xn, cat_features=[Xn.columns.get_loc(c) for c in enc["CAT"]],
                   text_features=[Xn.columns.get_loc(c) for c in enc["TEXT"]])
sv_nat = catboost_shap(native, native_pool)
fcols, ncols = list(Xf.columns), list(Xn.columns)
tables = {"lgb": grouped(sv_lgb, fcols), "lgb_sep": separate_sum(sv_lgb, fcols),
          "cb": grouped(sv_cb, fcols), "cb_sep": separate_sum(sv_cb, fcols),
          "native": grouped(sv_nat, ncols), "native_sep": separate_sum(sv_nat, ncols),
          "native_joined": grouped(sv_nat, ncols, keyfn=native_key)}
plots = {"shap_lightgbm.png": lambda: fig_bar(tables["lgb"], "SHAP — LightGBM (TF-IDF+SVD) — tüm veri"),
         "beeswarm_lightgbm.png": lambda: fig_site_beeswarm(sv_lgb, Xf, False, "LightGBM (TF-IDF+SVD)"),
         "shap_catboost_svd.png": lambda: fig_bar(tables["cb"], "SHAP — CatBoost (TF-IDF+SVD) — tüm veri"),
         "beeswarm_catboost_svd.png": lambda: fig_site_beeswarm(sv_cb, Xf, False, "CatBoost (TF-IDF+SVD)"),
         "shap_catboost_native.png": lambda: fig_bar(tables["native"], "SHAP — CatBoost (native text) — tüm veri"),
         "beeswarm_catboost_native.png": lambda: fig_site_beeswarm(sv_nat, Xn, True, "CatBoost (native text)")}
res = {"n": len(Xf), "add_err": add_err, "base": base, "tables": tables, "pngs": list(plots)}
print(f"additivity | toplamsallık {add_err:.1e} ·", {k: v[0] for k, v in tables.items() if not k.endswith(("_sep", "_joined"))})

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
PLOT_DIR.mkdir(parents=True, exist_ok=True)
for name, draw in plots.items():
    fig = draw()
    fig.savefig(PLOT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
print("written | yazıldı:", save_metrics("shap/04_variants", to_metrics(res)))
