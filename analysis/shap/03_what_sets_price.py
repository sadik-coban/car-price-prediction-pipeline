"""
shap/03_what_sets_price.py
EN: SHAP report §3 — what sets the price, from the out-of-fold SHAP of shap/02 (every listing explained by the
    model that never saw it). The global table groups hp + cc into ENGINE and the 13 damage features into
    DAMAGE (signed per-row sum, then mean |·|); the typical price effect is the median of |e^s − 1|. Also: is
    the ranking stable across folds; do age, mileage and power push the price the expected way (Spearman — a
    gate: a flipped sign stops the run); the dependence numbers behind the curves (age, what mileage costs per
    100k km, power bands); shap's own two-cohort split; and age × mileage in the same mileage band.
    Draws figures sh-01 … sh-05 in TR and EN (the shap library plots need the full SHAP matrix).
TR: SHAP raporu §3 — fiyatı ne belirliyor; shap/02'nin fold dışı SHAP'inden (her ilan onu hiç görmemiş modelle
    açıklanmış). Küresel tablo hp + cc'yi ENGINE, 13 hasar özniteliğini DAMAGE altında toplar (satır başına
    işaretli toplam, sonra ortalama |·|); fiyatta tipik etki |e^s − 1|'in medyanı. Ayrıca: sıralama fold'dan
    fold'a sabit mi; yaş, km ve güç fiyatı beklenen yöne mi itiyor (Spearman — kapı: işaret ters dönerse koşum
    durur); eğrilerin arkasındaki bağımlılık sayıları (yaş, 100 bin km başına kilometrenin bedeli, güç
    bantları); shap'in kendi iki gruplu bölmesi; aynı km bandında yaş × km.
    sh-01 … sh-05 figürlerini TR ve EN çizer (shap kütüphanesinin grafikleri tam SHAP matrisini ister).
Output / Çıktı: metrics/shap/03_what_sets_price.json · reports/figures/{tr,en}-sh-01..05-*.png
"""

# %% [1] Setup | Kurulum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent.parent))

import matplotlib                                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402
import shap                                                                    # noqa: E402
from matplotlib.ticker import FuncFormatter                                    # noqa: E402
from scipy.stats import spearmanr                                              # noqa: E402

from lib.common import ANALYSIS_DIR, CAT, NUM, ROOT, load_clean, save_metrics # noqa: E402
from lib.cv import load_oof                                                    # noqa: E402
from lib.labels import KM_SUF, lb, other_features_tr                           # noqa: E402

SEED = 42
MS = "MODEL_SERIES (text)"
NPZ = ANALYSIS_DIR / "oof_shap.npz"
FIGDIR = ROOT / "reports" / "figures"
C1, C2, C3, GRID = "#2563eb", "#64748b", "#dc2626", "#e5e7eb"
DAMAGE_COLS = ("roof_state", "hood_state", "trunk_state", "door_changed", "door_painted", "door_local",
               "fender_changed", "fender_painted", "fender_local", "bumper_changed", "bumper_painted",
               "bumper_local", "is_heavy_damaged")
AGE_YOUNG, AGE_OLD = (0, 3), (18, 100)
HP_BANDS = [(0, 150), (150, 200), (200, 250), (250, 10**9)]
AGE_ABS_BANDS = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 100)]
KM_WINDOWS = [(0, 50_000), (100_000, 150_000), (200_000, 250_000), (350_000, 10**9)]
KM_BAND, AGE_SPLIT = (150_000, 250_000), 10
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def group_key(col):
    """
    EN: Coarse group: the name's SVD dimensions → MODEL_SERIES, hp + cc → ENGINE, damage features → DAMAGE.
    TR: Kaba grup: adın SVD boyutları → MODEL_SERIES, hp + cc → ENGINE, hasar öznitelikleri → DAMAGE.
    """
    if col.startswith("model_") or col.startswith("series_"):
        return MS
    if col in ("power_hp_val", "engine_cc_val"):
        return "ENGINE"
    return "DAMAGE" if col in DAMAGE_COLS else col


def regroup(sv, names, keyfn=group_key):
    """
    EN: Coarse table: per group the mean |signed per-row sum| and its share. Returns: [[group, value, %], ...].
    TR: Kaba tablo: grup başına ortalama |satır başına işaretli toplam| ve payı. Döndürür: [[grup, değer, %], ...].
    """
    keys = [keyfn(n) for n in names]
    g = {u: float(np.abs(sv[:, [i for i, k in enumerate(keys) if k == u]].sum(axis=1)).mean()) for u in dict.fromkeys(keys)}
    tot = sum(g.values()) or 1.0
    return [[k, round(v, 4), round(v / tot * 100, 1)] for k, v in sorted(g.items(), key=lambda kv: -kv[1])]


def separate_sum(sv, cols, keyfn=group_key):
    """
    EN: A group as the SUM of its members' mean |SHAP| (no cancellation). Not the report's rule: it grows with
        the number of pieces the input is split into; used only for §7's "how much the split changes it".
    TR: Grup = üyelerinin ortalama |SHAP|'lerinin TOPLAMI (iptal yok). Raporun kuralı değil: girdinin kaç
        parçaya bölündüğüne göre büyür; yalnız §7'nin "bölme inceliği sonucu ne kadar değiştirir" ölçüsü için.
    """
    g = {}
    for k, v in zip([keyfn(c) for c in cols], np.abs(sv).mean(axis=0)):
        g[k] = g.get(k, 0.0) + float(v)
    tot = sum(g.values()) or 1.0
    return [[k, round(v, 4), round(v / tot * 100, 1)] for k, v in sorted(g.items(), key=lambda kv: -kv[1])]


def typical_effect(sv, names):
    """
    EN: Per coarse group: median over listings of |e^s − 1| (%) and the share of listings pushed down (%).
    TR: Kaba grup başına: ilanlar üzerinde |e^s − 1|'in medyanı (%) ve aşağı itilen ilanların payı (%).
    """
    keys = [group_key(c) for c in names]
    out = {}
    for u in dict.fromkeys(keys):
        s = sv[:, [i for i, k in enumerate(keys) if k == u]].sum(axis=1)
        out[u] = [round(float(np.median(np.abs(np.expm1(s)))) * 100, 2), round(float(100 * (s < 0).mean()), 1)]
    return out


def direction(x, s):
    """
    EN: Spearman correlation between a feature's value and its SHAP (missing values excluded).
    TR: Bir özniteliğin değeri ile SHAP'i arasındaki Spearman korelasyonu (eksik değerler dışarıda).
    """
    m = np.isfinite(x)
    r = spearmanr(x[m], s[m]).statistic
    assert np.isfinite(r), "direction could not be computed | yön hesaplanamadı"
    return float(r)


def band_median(x, s, lo, hi):
    """
    EN: Median SHAP of listings with lo ≤ x < hi, and their count. / TR: lo ≤ x < hi ilanların medyan SHAP'i ve sayısı.
    """
    m = np.isfinite(x) & (x >= lo) & (x < hi)
    return (float(np.median(s[m])), int(m.sum())) if m.sum() else (float("nan"), 0)


def dependence_facts(listings, sv, names, price):
    """
    EN: The numbers behind the dependence curves: age (0–2 vs 18+, as attribution ratio and as actual median
        price ratio, and which groups take the rest of the gap), power bands, mean |SHAP| of age by age band,
        and what mileage costs per 100k km between window medians (centre = median mileage in the window).
    TR: Bağımlılık eğrilerinin arkasındaki sayılar: yaş (0–2 ile 18+, atıf oranı ve gerçek medyan fiyat oranı
        olarak, ve farkın kalanını hangi grupların aldığı), güç bantları, yaş bandına göre yaşın ortalama
        |SHAP|'i ve pencere medyanları arasında 100 bin km başına kilometrenin bedeli (merkez = penceredeki
        medyan km).
    """
    col = {c: names.index(c) for c in ("vehicle_age", "gb_mileage", "power_hp_val")}
    age = pd.to_numeric(listings["vehicle_age"], errors="coerce").values
    hp = pd.to_numeric(listings["power_hp_val"], errors="coerce").values
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    s_age, s_hp, s_km = sv[:, col["vehicle_age"]], sv[:, col["power_hp_val"]], sv[:, col["gb_mileage"]]
    f = {}
    f["age_young"], f["age_young_n"] = band_median(age, s_age, *AGE_YOUNG)
    f["age_old"], f["age_old_n"] = band_median(age, s_age, *AGE_OLD)
    f["hp_low"], _ = band_median(hp, s_hp, *HP_BANDS[0])
    f["hp_high"], _ = band_median(hp, s_hp, *HP_BANDS[-1])
    f["age_ratio"] = float(np.exp(f["age_young"] - f["age_old"]))
    f["hp_ratio"] = float(np.exp(f["hp_high"] - f["hp_low"]))
    young, old = (age >= AGE_YOUNG[0]) & (age < AGE_YOUNG[1]), age >= AGE_OLD[0]
    f["age_price_ratio"] = float(np.median(price[young]) / np.median(price[old]))
    groups = {}
    for j, nm in enumerate(names):
        groups.setdefault(group_key(nm), []).append(j)
    gap = {g: float(sv[young][:, ix].sum(axis=1).mean() - sv[old][:, ix].sum(axis=1).mean())
           for g, ix in groups.items() if g != "vehicle_age"}
    f["age_gap_top"] = [g for g, v in sorted(gap.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    f["hp_bands"] = [(lo, hi, band_median(hp, s_hp, lo, hi)[0]) for lo, hi in HP_BANDS]
    f["age_abs"] = [(lo, hi, float(np.abs(s_age[(age >= lo) & (age < hi)]).mean())) for lo, hi in AGE_ABS_BANDS]
    mids, ns, centers = [], [], []
    for lo, hi in KM_WINDOWS:
        m_, n_ = band_median(km, s_km, lo, hi)
        mids.append(m_)
        ns.append(n_)
        centers.append(float(np.median(km[np.isfinite(km) & (km >= lo) & (km < hi)])))
    f["km_centers"] = centers
    f["km_rates"] = [((1 - np.exp(mids[i + 1] - mids[i])) * 100) / ((centers[i + 1] - centers[i]) / 1e5) for i in range(3)]
    f["km_n_last"] = ns[-1]
    f["km_flat"] = max(f["km_rates"]) / max(min(f["km_rates"]), 1e-9) >= 1.35
    f["km_son_min"] = f["km_rates"][-1] == min(f["km_rates"])
    return f


def km_age_facts(listings, s_km):
    """
    EN: In the same 150–250k km band: median mileage SHAP under 10 years and at 10+, with counts.
    TR: Aynı 150–250 bin km bandında: 10 yaş altı ve 10+ için medyan km SHAP'i ve sayıları.
    """
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    age = pd.to_numeric(listings["vehicle_age"], errors="coerce").values
    b = (km >= KM_BAND[0]) & (km < KM_BAND[1])
    return {"young": float(np.median(s_km[b & (age < AGE_SPLIT)])), "old": float(np.median(s_km[b & (age >= AGE_SPLIT)])),
            "n_young": int((b & (age < AGE_SPLIT)).sum()), "n_old": int((b & (age >= AGE_SPLIT)).sum())}


def grouped_explanation(listings, G, lang):
    """
    EN: One shap.Explanation over the 24 OOF items (beeswarm and cohorts use it). Colour data: numeric
        features only; categories carry codes whose size means nothing → NaN (grey).
    TR: 24 OOF kalemi üzerinde tek shap.Explanation (beeswarm ve grup grafikleri kullanır). Renk verisi: yalnız
        sayısal öznitelikler; kategoriler anlamsız kod taşır → NaN (gri).
    """
    data = np.full(G["sv"].shape, np.nan)
    for j, u in enumerate(G["names"]):
        if u not in CAT and u != MS:
            data[:, j] = pd.to_numeric(listings[u], errors="coerce").values
    return shap.Explanation(values=G["sv"], base_values=G["base"], data=data, feature_names=[lb(u, lang) for u in G["names"]])


def cohort_facts(listings, G, lang):
    """
    EN: shap's own decision-tree split into two cohorts, and the feature with the largest mean |SHAP| after it.
        Returns: (cohort bar data, {"split": [name0, name1], "name", "v0", "v1"}).
    TR: shap'in kendi karar ağacıyla iki gruba bölmesi ve bölmeden sonra ortalama |SHAP|'i en büyük öznitelik.
        Döndürür: (grup bar verisi, {"split": [ad0, ad1], "name", "v0", "v1"}).
    """
    coh = grouped_explanation(listings, G, lang).cohorts(2).abs.mean(0)
    k = list(coh.cohorts)
    c0, c1 = coh.cohorts[k[0]], coh.cohorts[k[1]]
    idx = int(np.argmax(c0.values + c1.values))
    return coh, {"split": [str(k[0]), str(k[1])], "name": c0.feature_names[idx],
                 "v0": float(c0.values[idx]), "v1": float(c1.values[idx])}


def _title(lang, title, xlabel=None, size=(7.4, 4.6)):
    """
    EN: Resizes the library figure to the report and sets its title/x label. Returns: the figure.
    TR: Kütüphane figürünü rapora göre boyutlar, başlığını/x etiketini koyar. Döndürür: figür.
    """
    fig = plt.gcf()
    fig.set_size_inches(*size)
    if xlabel:
        plt.gca().set_xlabel(xlabel)
    plt.gca().set_title(title, fontsize=10)
    return fig


def _tr_other(lang):
    """
    EN: Turkish text for shap's "Sum of N other features" row. / TR: shap'in "Sum of N other features" satırının Türkçesi.
    """
    if lang == "tr":
        ax = plt.gca()
        ax.set_yticklabels([other_features_tr(t.get_text()) for t in ax.get_yticklabels()])


def fig_importance(table, lang):
    """
    EN: sh-01: share of mean |SHAP| per coarse group (LightGBM, OOF). / TR: sh-01: kaba grup başına ortalama |SHAP| payı.
    """
    names, vals = [lb(r[0], lang) for r in table][::-1], [r[2] for r in table][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(range(len(names)), vals, color=C1)
    ax.set_yticks(range(len(names)), names)
    for i, x in enumerate(vals):
        ax.text(x + .4, i, f"%{x:.1f}" if lang == "tr" else f"{x:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("katkı payı (%)" if lang == "tr" else "share of total attribution (%)")
    ax.set_title("Fiyatı en çok ne belirliyor — LightGBM, ortalama |SHAP| payı" if lang == "tr"
                 else "What drives price — LightGBM, share of mean |SHAP|")
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    return fig


def fig_beeswarm(listings, G, lang):
    """
    EN: sh-02: beeswarm of the 24 items (12 shown); the seed fixes the random jitter so the PNG is stable.
    TR: sh-02: 24 kalemin beeswarm'ı (12'si görünür); tohum rastgele yayılmayı sabitler, PNG kararlı kalır.
    """
    np.random.seed(SEED)
    plt.figure()
    shap.plots.beeswarm(grouped_explanation(listings, G, lang), max_display=12, show=False, color_bar_label="")
    _tr_other(lang)
    if lang == "tr":
        for a_ in plt.gcf().axes:
            if [t_.get_text() for t_ in a_.get_yticklabels()] == ["Low", "High"]:
                a_.set_yticklabels(["Düşük", "Yüksek"])
    return _title(lang, "Her nokta bir ilan: öznitelik hangi yöne itiyor" if lang == "tr"
                  else "Each dot is a listing: which way the feature pushes",
                  "SHAP değeri (log fiyat) — sağ = fiyatı yukarı iter" if lang == "tr"
                  else "SHAP value (log price) — right = pushes price up")


def fig_dependence(listings, G, lang):
    """
    EN: sh-03: effect on price (%, e^s − 1) against age, mileage and power, with the median of 20 quantile bins
        (placed at the bin's median x).
    TR: sh-03: yaş, km ve güce karşı fiyata etki (%, e^s − 1), 20 yüzdelik dilimin medyanıyla (dilimin
        medyan x'ine konur).
    """
    cols = ["vehicle_age", "gb_mileage", "power_hp_val"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    n_bins = []
    for ax, c in zip(axes, cols):
        x = pd.to_numeric(listings[c], errors="coerce").values
        xx = x / 1000 if c == "gb_mileage" else x
        pct = (np.exp(G["sv"][:, G["names"].index(c)]) - 1) * 100
        ax.scatter(xx, pct, s=3, alpha=.10, color=C1, linewidths=0)
        ax.axhline(0, color=C2, lw=.8)
        q = pd.qcut(pd.Series(xx), 20, duplicates="drop")
        med = pd.Series(pct).groupby(q, observed=True).median()
        n_bins.append(len(med))
        xm = pd.Series(xx).groupby(q, observed=True).median()
        ax.plot(xm.values, med.values, color=C3, lw=1.6)
        ax.set_xlabel(lb(c, lang) + (KM_SUF[lang] if c == "gb_mileage" else ""))
        ax.grid(color=GRID, lw=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("fiyata etkisi (%)" if lang == "tr" else "effect on price (%)")
    dl = " · ".join(f"{lb(c, lang)} {k}" for c, k in zip(cols, n_bins))
    fig.suptitle(f"Öznitelik değeri ve fiyata etkisi — kırmızı çizgi 20'lik yüzdelik dilimlerin medyanı "
                 f"(aynı değerler birleşince: {dl} dilim)" if lang == "tr" else
                 f"Feature value and its effect on price — red line is the median of 20 quantile bins "
                 f"(tied values merge: {dl} bins)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_cohorts(coh, lang):
    """
    EN: sh-04: mean |SHAP| in shap's two cohorts. / TR: sh-04: shap'in iki grubunda ortalama |SHAP|.
    """
    np.random.seed(SEED)
    plt.figure()
    shap.plots.bar(coh, max_display=12, show=False)
    _tr_other(lang)
    return _title(lang, "shap'in bulduğu iki grupta ortalama |SHAP|" if lang == "tr" else "Mean |SHAP| in the two groups shap found",
                  "ortalama |SHAP| (log fiyat)" if lang == "tr" else "mean |SHAP| (log price)", (7.6, 4.6))


def fig_km_age(listings, G, lang):
    """
    EN: sh-05: mileage SHAP against mileage, coloured by age (shap.plots.scatter).
    TR: sh-05: km'ye karşı km SHAP'i, yaşa göre renkli (shap.plots.scatter).
    """
    cols = [c for c in CAT + NUM if c in G["names"]]
    data = np.column_stack([pd.to_numeric(listings[c], errors="coerce").values if c in NUM else
                            listings[c].map({v: i for i, v in enumerate(listings[c].unique())}).values.astype(float)
                            for c in cols])
    ei = shap.Explanation(values=G["sv"][:, [G["names"].index(c) for c in cols]], base_values=G["base"],
                          data=data, feature_names=cols)
    np.random.seed(SEED)
    plt.figure()
    shap.plots.scatter(ei[:, "gb_mileage"], color=ei[:, "vehicle_age"], show=False)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1000:g}"))
    ax.set_xlabel(lb("gb_mileage", lang) + KM_SUF[lang])
    ax.set_ylabel(("SHAP katkısı — " if lang == "tr" else "SHAP contribution — ") + lb("gb_mileage", lang))
    cb = [a for a in plt.gcf().axes if a.get_ylabel() == "vehicle_age"]
    assert len(cb) == 1, f"colour bar not found ({len(cb)}) — shap version changed | renk çubuğu bulunamadı"
    cb[0].set_ylabel(lb("vehicle_age", lang))
    return _title(lang, "Aynı kilometre, farklı yaş: iki öznitelik birlikte çalışıyor" if lang == "tr"
                  else "Same mileage, different age: the two features act together", None, (7.0, 4.0))


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under shap (the SHAP report reads these; figure names included).
    TR: shap altında yayımlanır (SHAP raporu bunları okur; figür adları dahil).
    """
    return {"shap": {"n": res["n"], "lightgbm_tfidf_svd": res["table"], "oof_ayri": res["separate"],
                     "n_svd": res["n_svd"], "fold_sira": res["fold_rank"], "tipik": res["typical"],
                     "direction": res["direction"], "dep_facts": res["dep"], "km_age": res["km_age"],
                     "cohorts": res["cohorts"], "figures": res["figures"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)
z = np.load(NPZ, allow_pickle=False)
if str(z["run_id"]) != oof_info["run_id"]:
    raise SystemExit("oof_shap.npz is from another run | başka koşumdan — run first | önce: python analysis/shap/02_oof_shap.py")
G = {"sv": z["shap"].astype(float), "names": [str(g) for g in z["groups"]], "base": z["base"].astype(float),
     "fold": z["fold"], "raw": z["ham_shap"].astype(float)}

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
table = regroup(G["sv"], G["names"])
assert np.allclose(G["raw"].sum(axis=1), G["sv"][:, G["names"].index(MS)], atol=1e-8), "raw SVD SHAP ≠ grouped column"
cols = [c for c in G["names"] if c != MS]
svc = np.column_stack([G["sv"][:, G["names"].index(c)] for c in cols] + [G["raw"]])
directions = {c: direction(pd.to_numeric(listings[c], errors="coerce").values, G["sv"][:, G["names"].index(c)])
              for c in ("vehicle_age", "gb_mileage", "power_hp_val")}
assert directions["vehicle_age"] < 0 and directions["gb_mileage"] < 0 and directions["power_hp_val"] > 0, \
    f"direction gate failed | yön kapısı tutmadı: {directions}"
coh, cohorts = {}, {}
for lang in ("tr", "en"):
    coh[lang], cohorts[lang] = cohort_facts(listings, G, lang)
res = {"n": len(listings), "table": table, "separate": separate_sum(svc, cols + ["model_svd"] * G["raw"].shape[1]),
       "n_svd": int(G["raw"].shape[1]),
       "fold_rank": [[r[0] for r in regroup(G["sv"][G["fold"] == k], G["names"])] for k in sorted(set(G["fold"].tolist()))],
       "typical": typical_effect(G["sv"], G["names"]), "direction": directions,
       "dep": dependence_facts(listings, G["sv"], G["names"], price),
       "km_age": km_age_facts(listings, G["sv"][:, G["names"].index("gb_mileage")]), "cohorts": cohorts,
       "figures": [f"{lang}-sh-{no:02d}-{slug}.png" for lang in ("tr", "en")
                   for no, slug in ((1, "importance"), (2, "beeswarm"), (3, "dependence"), (4, "cohorts"), (5, "km-age"))]}
print("top | en büyük:", table[:3], "· directions | yönler:", {k: round(v, 2) for k, v in directions.items()})

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
FIGDIR.mkdir(parents=True, exist_ok=True)
for lang in ("tr", "en"):
    for no, slug, draw in ((1, "importance", lambda: fig_importance(table, lang)),
                           (2, "beeswarm", lambda: fig_beeswarm(listings, G, lang)),
                           (3, "dependence", lambda: fig_dependence(listings, G, lang)),
                           (4, "cohorts", lambda: fig_cohorts(coh[lang], lang)),
                           (5, "km-age", lambda: fig_km_age(listings, G, lang))):
        fig = draw()
        fig.savefig(FIGDIR / f"{lang}-sh-{no:02d}-{slug}.png")
        plt.close(fig)
print("written | yazıldı:", save_metrics("shap/03_what_sets_price", to_metrics(res), run_id=oof_info["run_id"]))
