"""
shap/06_one_listing.py
EN: SHAP report §6 — how one prediction is built. The example is not hand-picked: it is the listing whose
    |OOF error| is closest to the median — neither an extreme case nor a perfect prediction, a listing where the
    model makes its typical error. Its out-of-fold SHAP (shap/02) is drawn as a waterfall (figure sh-06, TR and
    EN; gate: the parts plus the fold's base value give that fold model's OOF prediction). The numbers the
    text prints (the three largest contributions with their value labels, the rest, which rows the waterfall
    shows) are published too.
TR: SHAP raporu §6 — tek bir tahmin nasıl kuruluyor. Örnek elle seçilmez: |OOF hata|'sı medyana en yakın ilan —
    ne uç vaka ne kusursuz tahmin, modelin tipik hatasını yaptığı bir ilan. Fold dışı SHAP'i (shap/02) waterfall
    olarak çizilir (sh-06 figürü, TR ve EN; kapı: parçalar + fold'un taban değeri o fold modelinin OOF
    tahminini verir). Metnin bastığı sayılar da (değer etiketleriyle en büyük üç katkı, kalan, waterfall'da
    hangi satırların göründüğü) yayımlanır.
Output / Çıktı: metrics/shap/06_one_listing.json · reports/figures/{tr,en}-sh-06-waterfall.png
"""

# %% [1] Setup | Kurulum
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent.parent))

import matplotlib                                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402
import shap                                                                    # noqa: E402

from lib.common import ANALYSIS_DIR, ROOT, load_clean, save_metrics           # noqa: E402
from lib.cv import load_oof                                                    # noqa: E402
from lib.labels import lb, other_features_tr, value_label                     # noqa: E402

SEED = 42
NPZ = ANALYSIS_DIR / "oof_shap.npz"
FIGDIR = ROOT / "reports" / "figures"
WF_MAX = 10              # waterfall: WF_MAX−1 features on their own row + "N other features" | kendi satırında
TOP = 3
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def typical_listing(price, oof_price):
    """
    EN: Row index of the listing whose |OOF error| (₺) is closest to the median |OOF error|.
    TR: |OOF hata|'sı (₺) medyan |OOF hata|'ya en yakın ilanın satır indeksi.
    """
    err = np.abs(oof_price - price)
    return int(np.argmin(np.abs(err - np.median(err))))


def breakdown(names, sv_row, row):
    """
    EN: The listing's contributions ordered by |SHAP|: the TOP largest with value labels (TR, EN), the signed
        sum of the rest, and the items the waterfall shows on their own row.
    TR: İlanın katkıları |SHAP|'e göre sıralı: değer etiketleriyle (TR, EN) en büyük TOP tanesi, kalanın
        işaretli toplamı ve waterfall'un kendi satırında gösterdiği kalemler.
    """
    order = sorted(zip(names, sv_row), key=lambda kv: -abs(kv[1]))
    top = [[k, float(v), {lang: value_label(row, k, lang) for lang in ("tr", "en")}] for k, v in order[:TOP]]
    return {"top": top, "rest": float(sum(v for _k, v in order[TOP:])), "n_rest": len(names) - TOP,
            "visible": [k for k, _v in order[:WF_MAX - 1]], "n_other": len(names) - (WF_MAX - 1)}


def millions(x):
    """
    EN: ₺ in millions, 2 decimals, half-up (595000 → 0.60). / TR: Milyon ₺, 2 hane, yarım yukarı (595000 → 0.60).
    """
    return Decimal(str(float(x) / 1e6)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fig_waterfall(names, sv_row, base, row, case, lang):
    """
    EN: sh-06: shap.plots.waterfall of the listing. Value labels are baked into the feature names and data is
        None, so category names and TR/EN number formats show instead of codes. Gate: parts + base = the
        fold model's OOF prediction (relative 1e-5).
    TR: sh-06: ilanın shap.plots.waterfall'u. Değer etiketleri öznitelik adına gömülür ve data None geçilir;
        böylece kod yerine kategori adı ve TR/EN sayı biçimi görünür. Kapı: parçalar + taban = fold modelinin
        OOF tahmini (göreli 1e-5).
    """
    got = float(np.expm1(sv_row.sum() + base))
    assert abs(got - case["pred"]) / case["pred"] < 1e-5, f"waterfall sum ≠ prediction | tahmini vermiyor: {got:,.2f}"
    labels = [f"{lb(u, lang)} = {value_label(row, u, lang)}" for u in names]
    np.random.seed(SEED)
    plt.figure()
    shap.plots.waterfall(shap.Explanation(values=sv_row, base_values=base, data=None, feature_names=labels),
                         max_display=WF_MAX, show=False)
    if lang == "tr":
        ax = plt.gca()
        ax.set_yticklabels([other_features_tr(t.get_text()) for t in ax.get_yticklabels()])
    fig = plt.gcf()
    fig.set_size_inches(7.6, 4.4)
    plt.gca().set_title(f"{case['name']} · {case['year']} — " + ("gerçek " if lang == "tr" else "actual ")
                        + f"₺{millions(case['price'])}M · "
                        + ("ilanı görmemiş modelin tahmini " if lang == "tr" else "prediction of the model that never saw it ")
                        + f"₺{millions(case['pred'])}M", fontsize=10)
    return fig


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under shap_case (the SHAP report's §2 and §6).
    TR: shap_case altında yayımlanır (SHAP raporunun §2 ve §6'sı).
    """
    return {"shap_case": {**res["case"], **res["parts"], "wf_max": WF_MAX, "n_items": res["n_items"],
                          "km_label": res["km_label"], "figures": res["figures"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)
z = np.load(NPZ, allow_pickle=False)
if str(z["run_id"]) != oof_info["run_id"]:
    raise SystemExit("oof_shap.npz is from another run | başka koşumdan — run first | önce: python analysis/shap/02_oof_shap.py")
names, sv, base_all, oof_shap_price = [str(g) for g in z["groups"]], z["shap"].astype(float), z["base"].astype(float), z["oof"]

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
i = typical_listing(price, oof_shap_price)
row = listings.iloc[i]
case = {"name": str(row["model"]), "year": int(pd.to_numeric(listings["gb_year"], errors="coerce").iat[i]),
        "price": float(price[i]), "pred": float(oof_shap_price[i])}
res = {"case": case, "parts": breakdown(names, sv[i], row), "n_items": len(names),
       "km_label": {lang: value_label(row, "gb_mileage", lang) for lang in ("tr", "en")},
       "figures": [f"{lang}-sh-06-waterfall.png" for lang in ("tr", "en")]}
print(case["name"], case["year"], "·", [(k, round(v, 3)) for k, v, _l in res["parts"]["top"]])

# %% [6] Save | Kaydet — the only cell that writes files | dosya yazan tek hücre
for lang in ("tr", "en"):
    fig = fig_waterfall(names, sv[i], float(base_all[i]), row, case, lang)
    fig.savefig(FIGDIR / f"{lang}-sh-06-waterfall.png")
    plt.close(fig)
print("written | yazıldı:", save_metrics("shap/06_one_listing", to_metrics(res), run_id=oof_info["run_id"]))
