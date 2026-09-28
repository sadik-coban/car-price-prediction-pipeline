"""
report_common.py
EN: What the technical and business report builders share: the metrics view (report_lib/metrics_view.py), the
    report numbers derived once for both reports (derive — formatting-level arithmetic on published metrics
    only; every computation on data lives in analysis/), the figure templates and figures (drawn from metrics
    values), number/label formatters and the TR/EN label dictionaries. No analysis here.
TR: Teknik rapor ve karar notu derleyicilerinin paylaştıkları: metrik görünümü (report_lib/metrics_view.py), iki
    rapor için bir kez türetilen rapor sayıları (derive — yalnız yayımlanmış metrikler üzerinde biçim düzeyinde
    aritmetik; veri üzerindeki her hesap analysis/'te), figür şablonları ve figürler (metrik değerlerinden
    çizilir), sayı/etiket biçimleyicileri ve TR/EN etiket sözlükleri. Burada analiz yok.
"""
import os
import pathlib
import re
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402

from . import metrics_view as MV                   # noqa: E402  the single metrics reader | tek metrik okuyucu
from .column_labels import COLUMN_LABELS           # noqa: E402

# --- Paths: from __file__, independent of cwd | Yollar: __file__'a göre, cwd'ye bağlı değil ---------------
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root | depo kökü
# EN: where the outputs go; CARDATASYS_OUT redirects them (tests rebuild the reports into a temp folder)
# TR: çıktıların gittiği yer; CARDATASYS_OUT yönlendirir (testler raporları geçici klasöre yeniden üretir)
OUT_ROOT = pathlib.Path(os.environ.get("CARDATASYS_OUT") or ROOT)
REPORTS_DIR = OUT_ROOT / "reports"                 # the six report md files | altı rapor md'si
FIGDIR = REPORTS_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Hata histogrami x-ekseni gorunumu (±%). Veri degil, cizim siniri: disarida kalan ilan sayisi
# grafikte ve metinde veriden yazilir. Uc kuyruk (-%178) tum ekseni ezmesin diye.
RESID_VIEW = 40

# EN: every path the report text reads; a missing one stops the build instead of silently dropping a paragraph
# TR: rapor metninin okuduğu her yol; eksik olan paragrafı sessizce düşürmek yerine derlemeyi durdurur
REQUIRED = [
    "domain.drift.all_pairs", "domain.drift.note",
    "domain.hedonic_reliability.center", "domain.brand_ablation.validation",
    "methodology.column_accounting", "methodology.kb_gb_twins", "domain.hedonic_reliability.columns",
    "domain.segment_ladder", "domain.model_year_median.ladder", "domain.price_dist.p10", "domain.price_dist.p90",
    "domain.final_results.training.target", "domain.shap.lightgbm_tfidf_svd", "domain.kmeans",
    "methodology.theils_matrix", "methodology.column_missing",
    "methodology.backtest.per_snapshot", "methodology.backtest.insample", "methodology.backtest.protocol", "methodology.backtest.paired",
    "methodology.backtest.horizon", "methodology.backtest.forward_coverage",
    "methodology.backtest.columns",
    "methodology.systematic_missing.systematic_groups", "methodology.systematic_missing.note",
    "methodology.pca_axes", "meta.repro", "meta.brands", "column_labels"] + [f"error_drivers.{p_}" for p_ in [
    "plate_scope", "segment_quality", "hedonic_dropped", "per_model_error", "per_model_buckets", "lira_quartile",
    "lira_scaled", "scope", "price_changes", "unspecified", "baseline_equal_terms", "text_flag",
    "age_sensitivity", "age_cuts", "spec_outliers.blind_spot", "period_shift",
    "by_model_year_n", "by_segment_FS", "by_age", "by_snapshot", "raw_columns", "examples",
    "engine_rule.engine_cc", "engine_rule.power_hp", "unspecified.structure", "live_vs_gone"]] + [
    f"report.{p_}" for p_ in ["q_bounds", "model_r2_log", "err_bands", "conformal_q", "conformal_all",
                              "mondrian", "text_ablation"]]


def write_md(path, text):
    """
    EN: Writes with LF line ends on every platform (write_text would give CRLF on Windows).
    TR: Her platformda LF satır sonuyla yazar (write_text Windows'ta CRLF üretirdi).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

# LOFO grafiginde cizilen CAKISMAYAN anahtarlar. Ham methodology.lofo hem tekil hem grup
# cikarmalarini tasir; ikisini ayni eksene basmak cift sayim olur (DAMAGE_COLS kendi 13 uyesiyle
# yarisir). Hem figur hem de altindaki kapsam tablosu bu listeyi kullanir.
LOFO_FLAT_KEYS = ["gb_mileage", "vehicle_age", "DAMAGE_COLS", "MODEL_SERIES", "ENGINE"]
# Grup anahtarlarinin okunur adi + SHAP tablosundaki karsiligi (site_data.shap ayni gruplari baska adla tutuyor).
LOFO_GROUPS = {"DAMAGE_COLS": ("hasar grubu", "damage group", "DAMAGE"),
             "MODEL_SERIES": ("model/seri adı", "model/series name", "MODEL_SERIES (text)"),
             "ENGINE": ("motor (hp + cc)", "engine (hp + cc)", "ENGINE")}


def lofo_name(d, k, lang):
    """
    EN: Display name of a LOFO key: group keys get their own label, single features the column label.
    TR: Bir LOFO anahtarının görünen adı: grup anahtarları kendi etiketini, tekil öznitelikler kolon etiketini
        alır.
    """
    g = LOFO_GROUPS.get(k)
    return (g[0] if lang == "tr" else g[1]) if g else col(d, k, lang)

# ============================================================================
# ORNEK NOTLARI — metin ilan metinleri okunarak elle yazildi (2026-09-17, kullanici karari); SAYILAR canli
# (2026-09-27, arsivden canliya hicbir sey kurali): 08_large_errors her koşuda examples[].compare'i hesaplar
# (seri sayisi, ayni adin oteki ilanlari, notun karsilastirdigi grubun n'i ve medyani). Her notun iddiasi veriyle
# kapili; tutmazsa uretec DURUR, not sessizce bayatlamaz. Anahtar (model, yil, fiyat) 08'deki bir ornekle eslesmeli.
# ============================================================================
def _note_640i(e, lang):
    """
    EN: The 640i converted to an M6 (read from the ad text; no number in it).
    TR: M6'ya dönüştürülmüş 640i (ilan metninden okundu; içinde sayı yok).
    """
    return ("İlan metnine göre araç komple M6 dönüşümü: M6 motoru ve M6 kasa parçaları takılmış. Form hâlâ 640i "
            "dediği için model onu sıradan bir 640i gibi fiyatlıyor; alıcı ise bir M6'ya bakıyor." if lang == "tr" else
            "Per the ad text the car is a full M6 conversion: M6 engine and M6 body parts. The form still says 640i, "
            "so the model prices an ordinary 640i while the buyer is looking at an M6.")


def _note_r8(e, lang):
    """
    EN: The only R8, priced like the S5 4.2 FSI Quattros; stops unless it is alone in its series and the model's
        estimate is within 15% of that group's median.
    TR: Tek R8, S5 4.2 FSI Quattro'lar gibi fiyatlanmış; serisinde tek değilse ya da model tahmini o grubun
        medyanının %15'i içinde değilse durur.
    """
    c, peer = e["compare"], e["compare"]["peer"]
    assert c["series_n"] == 1 and peer and abs(e["pred"] / peer["median"] - 1) < .15, f"R8 notu bayat: {c}, {e['pred']}"
    return (f"Veride tek R8. Formdaki model adı yalnız \"4.2 FSI Quattro R-tronic\"; aynı motor adını taşıyan "
            f"{peer['model']}'ların medyanı {tlm(peer['median'])} ({num(peer['n'], lang)} ilan) ve model tahmini buna "
            f"yakın. Emsali olmayan bir süper otomobili model, adı benzeyen S5 gibi fiyatlamış." if lang == "tr" else
            f"The only R8 in the data. Its model name on the form is just \"4.2 FSI Quattro R-tronic\"; the "
            f"{peer['model']}s sharing that engine name have a median of {tlm(peer['median'])} "
            f"({num(peer['n'], lang)} listings), and the model's estimate is close to that. With no comparable, the "
            f"model priced a supercar like the similarly named S5.")


def _note_750i(e, lang):
    """
    EN: The 750i Long priced right, pulled up by its name's other, expensive listing; stops unless the name has
        exactly one other listing and the price is within 10% of the same-year 730d median.
    TR: Piyasaya uygun fiyatlı 750i Long, adın öteki pahalı ilanından etkilenmiş; adın tam bir öteki ilanı yoksa ya
        da fiyat aynı yılın 730d medyanının %10'u içinde değilse durur.
    """
    c, peer = e["compare"], e["compare"]["peer"]
    assert len(c["others"]) == 1 and peer and abs(e["price"] / peer["median"] - 1) < .10, f"750i notu bayat: {c}"
    (o_year, o_price), = c["others"]
    return (f"Veride bu addan {num(e['n_model'], lang)} ilan var; diğeri {tlm(o_price)}'lik dönüşümlü bir {o_year} araç. "
            f"Bu ilan ise aynı yılın {peer['model_prefix']}'leriyle ({num(peer['n'], lang)} ilan, medyan "
            f"{tlm(peer['median'])}) uyumlu ve metni bakımlı, masrafsız diyor. İlan piyasaya uygun, yanılan model: "
            f"emsali olmadığı için muhtemelen adın diğer, pahalı ilanından etkileniyor." if lang == "tr" else
            f"There are {num(e['n_model'], lang)} listings under this name; the other is a {tlm(o_price)} converted "
            f"{o_year} car. This listing is in line with same-year {peer['model_prefix']}s ({num(peer['n'], lang)} "
            f"listings, median {tlm(peer['median'])}) and its text says well-maintained with no pending costs. The "
            f"listing is priced right and the model is wrong: lacking a comparable, it is probably pulled up by the "
            f"name's other, expensive listing.")


EXAMPLE_NOTES = {("640i", 2011, 5_600_000): _note_640i,
                 ("4.2 FSI Quattro R-tronic", 2008, 4_690_000): _note_r8,
                 ("750i Long", 2007, 1_190_000): _note_750i}


# --- Palet: tek renk ailesi, susleme yok ------------------------------------
C1, C2, C3, GRID = "#2563eb", "#64748b", "#dc2626", "#e5e7eb"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10})


# ============================================================================
#  VERI
# ============================================================================
def load_report_view():
    """
    EN: The merged metrics view (metrics_view.load_view) plus the static column labels; stops if a required
        path is missing, so no paragraph is silently dropped.
    TR: Birleşik metrik görünümü (metrics_view.load_view) ve durağan kolon etiketleri; zorunlu bir yol eksikse
        durur, böylece hiçbir paragraf sessizce düşmez.
    """
    d = MV.load_view(ROOT)
    d["column_labels"] = COLUMN_LABELS
    missing = [p_ for p_ in REQUIRED if not MV.has_path(d, p_)]
    if missing:
        raise SystemExit(f"metrics missing/stale | metrik eksik/bayat — run | koşun: python analysis/run_all.py · {missing}")
    return d


def derive(d):
    """
    EN: The report numbers, derived ONCE and shared by both reports, so a number cannot differ between them.
        Only formatting-level arithmetic on published metrics (differences, ratios, picking rows); numbers that
        need the data come ready from analysis/ (d["report"]).
    TR: Rapor sayıları, BİR KEZ türetilir ve iki rapor paylaşır; böylece bir sayı ikisinde farklı çıkamaz.
        Yalnız yayımlanmış metrikler üzerinde biçim düzeyinde aritmetik (fark, oran, satır seçme); veri
        gerektiren sayılar analysis/'ten hazır gelir (d["report"]).
    """
    dom, met, meta, rep = d["domain"], d["methodology"], d["meta"], d["report"]
    lgb = dom["model_compare"]["lightgbm"]
    year_med = dom["model_year_median"]
    ba = dom["brand_ablation"]
    pd_ = dom["price_dist"]
    hr = dom["hedonic_reliability"]

    model_mae, base_mae = lgb["MAE"], year_med["baseline"]["MAE"]
    # EN: 2026-09-28 (owner's decision): the decision note prints the model-control column (effects within one
    #     model name); an unknown term id stops (no silent None)
    # TR: 2026-09-28 (kullanıcı kararı): karar notu model kontrolü sütununu basar (etkiler tek model adı içinde);
    #     bilinmeyen terim kimliği durdurur (sessiz None yok)
    hed_terms, hed_ci = {}, {}
    for r in hr["columns"]["model"]["coefficients"]:
        assert r["term"] in HED_TERM, f"unknown hedonic term | bilinmeyen hedonik terim: {r['term']}"
        hed_terms[r["term"]], hed_ci[r["term"]] = r["pct_effect"], (r["pct_lo"], r["pct_hi"])

    v = {
        # olcek
        "n_raw": meta["n_raw"], "n_dedup": meta["n_dedup"],
        "n_dup_rows": meta["n_raw"] - meta["n_dedup"],
        "n_features": meta["n_features"],
        "snapshots": meta["snapshots"],
        "n_snapshots": len(meta["snapshots"]),
        # fiyat dagilimi
        "median": pd_["median"], "p10": pd_["p10"], "p90": pd_["p90"],
        "skew_raw": pd_["skew_raw"], "skew_log": pd_["skew_log"],
        # model vs taban
        "model_mae": model_mae, "model_mape": lgb["MAPE"], "model_r2": lgb["R2"],
        "base_mae": base_mae, "base_mape": year_med["baseline"]["MAPE"],
        "better_pct": (base_mae - model_mae) / base_mae * 100,
        "gap_tl": base_mae - model_mae,
        "ladder": year_med["coverage"], "tiers": year_med["metric_breakdown"],
        # kalibrasyon
        "oof_r2": dom["pred_vs_true"]["r2"],
        "resid_mean": dom["residual_scatter"]["mean_resid_pct"],
        "resid_std": dom["residual_scatter"]["std_resid_pct"],
        "cov_target": dom["conformal"]["coverage_target"],
        "cov_q1": dom["conformal"]["by_quantile"][0][1],
        # EN: predicted-price quartile bounds and the per-band (Mondrian) interval arm (08_conformal_coverage)
        # TR: tahmin fiyatı çeyrek sınırları ve bant başına (Mondrian) aralık kolu (08_conformal_coverage)
        "q_bounds": rep["q_bounds"], "mondrian": rep["mondrian"],
        # hedonik: segment sutununun R²'si ve n'i; karar notunun etkileri model sutunundan
        "hed_r2": hr["columns"]["segment"]["r2"], "hed_n": hr["n"], "hed_center": hr["center"],
        "hed_terms": hed_terms, "hed_ci": hed_ci,
        # marka
        "brand_mae_delta": abs(ba["brand_series_model"]["MAE"] - ba["series_model"]["MAE"]),
        "brand_mape_delta": abs(ba["brand_series_model"]["MAPE"] - ba["series_model"]["MAPE"]),
        # veri saglami
        "dup_strict_n": met["content_duplicates"]["strict_extra"],
        "dup_strict_pct": met["content_duplicates"]["strict_pct"],
        "dup_loose_n": met["content_duplicates"]["loose_extra"],
        "dup_loose_pct": met["content_duplicates"]["loose_pct"],
        "n_missing_cols": len(met["systematic_missing"]["column_missing_all"]),
        # kumeler
        "k": met["kmeans_selection"]["chosen_k"],
        "clusters": dom["kmeans"],
        "pca": met["pca_axes"],
    }
    # EN: the model's OOF R² on log price, comparable with the hedonic R² (08_residuals)
    # TR: modelin log fiyattaki OOF R²'si, hedonik R² ile karşılaştırılabilir (08_residuals)
    v["model_r2_log"] = rep["model_r2_log"]
    v["cov_q4"] = dom["conformal"]["by_quantile"][-1][1]
    # PSI ozeti — karar notunun "bugun kayma kucuk" cumlesi buna kapili. Esik ureticinin notundan;
    # not bicimi degisirse sessizce varsayilana dusmez, durur.
    _th = re.search(r"PSI<([\d.]+).*?>([\d.]+)", dom["drift"]["note"])
    assert _th, "drift notunda PSI esikleri bulunamadi"
    v["psi_max"] = max(r[2] for r in dom["drift"]["all_pairs"])    # [pair, KS, PSI, EMD, shared %]
    v["psi_safe"], v["psi_retrain"] = float(_th.group(1)), float(_th.group(2))
    v["bt_paired"], v["bt_columns"] = met["backtest"]["paired"], met["backtest"]["columns"]
    v["bt_forward"] = met["backtest"]["forward_coverage"]
    v["lofo"] = met["lofo"]                      # karar notu "farki kapatan" siralamasi (2026-09-23)
    # EN: OOF error distribution: bands, median/mean error, extremes (08_residuals)
    # TR: OOF hata dağılımı: bantlar, medyan/ortalama hata, uçlar (08_residuals)
    v.update({k: rep[k] for k in ("err_n", "err_bands", "err_median", "err_abs_median", "err_abs_mean", "err_over20",
                                  "err_under20", "err_sym_over", "err_sym_under", "err_min", "err_max", "err_in10")})
    return v


# ============================================================================
#  FIGUR SABLONLARI  (5 tane, 26 figuru karsilar)
# ============================================================================
def _save(fig, name):
    """
    EN: Saves a figure as reports/figures/<name>.png and closes it. Returns: the file name.
    TR: Figürü reports/figures/<name>.png olarak kaydeder ve kapatır. Döndürür: dosya adı.
    """
    p = FIGDIR / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def bar(name, labels, values, title, xlabel="", horizontal=False, color=C1, fmt=None):
    """
    EN: Bar chart template (vertical or horizontal) saved under name. Returns: the file name.
    TR: Çubuk grafik şablonu (dikey ya da yatay), name adıyla kaydedilir. Döndürür: dosya adı.
    """
    h = max(2.4, 0.32 * len(labels)) if horizontal else 3.2
    fig, ax = plt.subplots(figsize=(7, h))
    if horizontal:
        ax.barh(range(len(labels)), values, color=color)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        ax.bar(range(len(labels)), values, color=color)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    if fmt:
        (ax.xaxis if horizontal else ax.yaxis).set_major_formatter(fmt)
    return _save(fig, name)


def line(name, x, series, title, xlabel="", ylabel=""):
    """
    EN: Line chart template; series = [(label, values, colour, style), ...]. Returns: the file name.
    TR: Çizgi grafik şablonu; series = [(etiket, değerler, renk, stil), ...]. Döndürür: dosya adı.
    """
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for lbl, ys, col, st in series:
        ax.plot(x, ys, st, label=lbl, color=col, lw=1.6, ms=3)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False)
    return _save(fig, name)


def scatter(name, xs, ys, title, xlabel="", ylabel="", ref=None, logx=False, c=None):
    """
    EN: Scatter template with an optional reference line ref = (xs, ys). Returns: the file name.
    TR: İsteğe bağlı referans çizgili (ref = (xs, ys)) nokta bulutu şablonu. Döndürür: dosya adı.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(xs, ys, s=3, alpha=.18, c=c if c is not None else C1, lw=0)
    if ref is not None:
        ax.plot(ref[0], ref[1], "-", color=C3, lw=1.2)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
    return _save(fig, name)


def heatmap(name, labels_y, labels_x, matrix, title, vmin=None, vmax=None, cmap="RdYlGn", marks=()):
    """
    EN: Annotated heatmap template; marks = cells to mark with a dot. Returns: the file name.
    TR: Değerleri yazılı ısı haritası şablonu; marks = noktayla işaretlenecek hücreler. Döndürür: dosya adı.
    """
    m = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, .5 * len(labels_x) + 2.6),
                                    max(4, .42 * len(labels_y) + 1.8)))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_y)), labels_y)
    if m.shape[0] * m.shape[1] <= 100:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
    for i, j in marks:                                   # or. tek ilanlik hucreler (fig 19)
        ax.plot(j + .32, i - .3, "o", ms=3.5, color="#111")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=.8)
    return _save(fig, name)


def errorbar(name, labels, series, title, xlabel=""):
    """
    EN: Point + interval chart template; series = [(legend, points, lows, highs), ...], drawn side by side on each
        row. Returns: the file name.
    TR: Nokta + aralık grafik şablonu; series = [(lejant, noktalar, alt, üst), ...], her satırda yan yana çizilir.
        Döndürür: dosya adı.
    """
    fig, ax = plt.subplots(figsize=(7, max(2.6, .42 * len(labels))))
    y = np.arange(len(labels))
    step = .32 / max(len(series) - 1, 1)
    for i, (leg, point, lo, hi) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * step
        ax.errorbar(point, y + off, xerr=[np.array(point) - np.array(lo), np.array(hi) - np.array(point)],
                    fmt="o", color=(C1, C3, C2)[i % 3], capsize=3, ms=4, lw=1, label=leg)
    ax.axvline(0, color="#444", lw=.8, ls="--")
    ax.set_yticks(list(y), labels); ax.invert_yaxis()
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=.7); ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7)
    return _save(fig, name)


# ============================================================================
#  FIGURLER — numara reg() ile; hangi rapora girdigi BUSINESS_FIGS / TECHNICAL_FIGS
# ============================================================================
def build_figures(d, v, lang, only=None):
    """
    EN: Draws the report's figures from metrics values (no analysis) and returns {no: (file, title)}.
        only: the figure numbers to draw (None = all); each builder draws only its own report's figures.
    TR: Raporun figürlerini metrik değerlerinden çizer (analiz yok) ve {no: (dosya, başlık)} döndürür.
        only: çizilecek figür numaraları (None = hepsi); her derleyici yalnız kendi raporunun figürlerini çizer.
    """
    want = (lambda no: True) if only is None else (lambda no: no in only)
    dom, met = d["domain"], d["methodology"]
    L = (lambda tr, en: tr if lang == "tr" else en)
    p = lang  # dosya adi oneki
    F = {}

    def reg(no, fname, title):
        """
        EN: Registers figure no with its file and title.
        TR: no numaralı figürü dosyası ve başlığıyla kaydeder.
        """
        F[no] = (fname, title)

    # 00 taban vs model  [IS]
    if want(0):
        t = L("Ortalama hata: emsal medyanı vs model", "Mean error: comparable median vs model")
        reg(0, bar(f"{p}-00-base-vs-model", [L("emsal medyanı", "comparable median"), "model"],
                   [v["base_mae"] / 1e3, v["model_mae"] / 1e3], t,
                   L("ortalama mutlak hata (₺bin)", "mean absolute error (₺k)")), t)

    # 01 kasa tipine gore medyan  [TEKNIK]
    if want(1):
        # Ureticide en az 80 ilanli kasa tipleri tutuluyor; disarida kalan sayi ve her cubugun n'i yazilir.
        # 2026-09-26 (kullanici): sitenin kasa tipi vermedigi ilanlar (missing) cubuk degil, basliktaki sayi:
        # gercek bir kasa tipi degiller, karisik modellerden gelip "en pahali tip" gibi okunuyorlardi.
        _unk01 = [r for r in dom["body_median"] if r[0] == "missing"]
        assert len(_unk01) == 1, f"baslik kasa tipi verilmeyen ilanlari sayiyor: {len(_unk01)} satir"
        rows = [r for r in dom["body_median"] if r[0] != "missing"]
        _nu01 = int(_unk01[0][2])
        _n01 = v["n_dedup"] - _nu01 - sum(int(r[2]) for r in rows)
        _head01 = L("Kasa tipine göre medyan fiyat", "Median price by body style")
        _note01 = L(f"en az 80 ilanlı tipler; kasa tipi verilmeyen {num(_nu01, lang)} ve daha az ilanlı tiplerdeki "
                    f"{num(_n01, lang)} ilan dışarıda",
                    f"types with 80+ listings; the {num(_nu01, lang)} listings with no body style and the "
                    f"{num(_n01, lang)} in smaller types left out")
        t = f"{_head01} ({_note01})"
        # Grafigin icinde baslik iki satir (uzun not gorseli genisletiyordu); rapordaki gorsel adi tek satir.
        reg(1, bar(f"{p}-01-body-median", [r[0] + f" · {num(int(r[2]), lang)}" for r in rows],
                   [r[1] / 1e6 for r in rows],
                   f"{_head01}\n({_note01})", L("medyan fiyat (₺M)", "median price (₺M)"), horizontal=True), t)

    # 02 (segmente gore medyan) 2026-09-22'de cikti: karar notundaki kume bolumuyle birlikte
    # gitti, teknik rapor onu hic kullanmiyordu. Segment kirilimi teknik §5'te duruyor.

    # 03 hedonik etkiler, iki sutun  [TEKNIK] — 2026-09-28: bootstrap yerine modele gore kumeli %95 GA.
    if want(3):
        cols = dom["hedonic_reliability"]["columns"]
        t = L("Hedonik etkiler (nokta + modele göre kümeli %95 GA)",
              "Hedonic effects (point + 95% CI clustered by model)")
        terms = [r["term"] for r in cols["segment"]["coefficients"]]
        assert terms == [r["term"] for r in cols["model"]["coefficients"]], "hedonik sutunlarin terimleri farkli"
        series = [(leg, [r["pct_effect"] for r in cols[k]["coefficients"]], [r["pct_lo"] for r in cols[k]["coefficients"]],
                   [r["pct_hi"] for r in cols[k]["coefficients"]])
                  for k, leg in (("segment", L("segment kontrolü", "segment control")),
                                 ("model", L("model kontrolü", "model control")))]
        reg(3, errorbar(f"{p}-03-hedonic-ci", [HED_TERM[k][0 if lang == "tr" else 1] for k in terms], series,
                        t, L("fiyata etki %", "effect on price %")), t)

    # 04 LOFO - DUZ surum  [TEKNIK]
    if want(4):
        lofo = {r[0]: r[1] for r in met["lofo"]}
        names = [k for k in LOFO_FLAT_KEYS if k in lofo]
        t = L("LOFO — öznitelik çıkınca ΔRMSE (çakışmayan gruplar)",
              "LOFO — ΔRMSE when a feature is removed (non-overlapping groups)")
        reg(4, bar(f"{p}-04-lofo-flat", [lofo_name(d, k, lang) for k in names], [lofo[k] / 1000 for k in names], t,
                   L("ΔRMSE (₺bin)", "ΔRMSE (₺k)"), horizontal=True), t)

    # 05 yasa gore fiyat  [IS]
    if want(5):
        rows = dom["age_depreciation"]
        t = L("Yaşa göre ham fiyat — düzeltilmemiş (medyan + ort.)", "Raw price by age — unadjusted (median + mean)")
        reg(5, line(f"{p}-05-age-price", [r[0] for r in rows],
                    [(L("medyan", "median"), [r[1] / 1e6 for r in rows], C1, "o-"),
                     (L("ortalama", "mean"), [r[3] / 1e6 for r in rows], C2, "s--")],
                    t, L("yaş (yıl)", "age (years)"), L("fiyat (₺M)", "price (₺M)")), t)

    # 06 km'ye gore fiyat  [IS]
    if want(6):
        rows = dom["km_price"]
        _kk = dom["km_price_scope"]
        _w06 = rows[1][0] - rows[0][0]                         # kova genisligi; nokta kovanin ortasina
        t = L(f"Kilometreye göre ham fiyat — düzeltilmemiş (medyan + ort.; {num(_kk['upper_limit'] // 1000, lang)} bin km "
              f"altı, {num(_kk['outside'], lang)} ilan dışarıda)",
              f"Raw price by mileage — unadjusted (median + mean; below {num(_kk['upper_limit'] // 1000, lang)}k km, "
              f"{num(_kk['outside'], lang)} listings left out)")
        reg(6, line(f"{p}-06-km-price", [(r[0] + _w06 / 2) / 1000 for r in rows],
                    [(L("medyan", "median"), [r[1] / 1e6 for r in rows], C1, "o-"),
                     (L("ortalama", "mean"), [r[3] / 1e6 for r in rows], C2, "s--")],
                    t, L("kilometre (bin km)", "mileage (k km)"), L("fiyat (₺M)", "price (₺M)")), t)

    # 07 marka karsilastirmasi  [IS]
    if want(7):
        bc = dom["brand_compare"]
        t = L("Ham medyan fiyat: BMW vs Audi — model karmasını yansıtır",
              "Raw median price: BMW vs Audi — reflects the model mix")
        reg(7, bar(f"{p}-07-brand", ["BMW", "Audi"],
                   [bc["bmw_median"] / 1e6, bc["audi_median"] / 1e6], t,
                   L("medyan fiyat (₺M)", "median price (₺M)")), t)

    # 08 tahmin vs gercek  [TEKNIK]
    if want(8):
        pv = dom["pred_vs_true"]
        pts = np.array(pv["points"], dtype=float)
        t = L(f"Tahmin vs Gerçek (R² {pv['r2']:.3f})", f"Predicted vs Actual (R² {pv['r2']:.3f})")
        lim = [0, float(np.nanpercentile(pts[:, 0], 99.5))]
        reg(8, scatter(f"{p}-08-pred-vs-true", pts[:, 0] / 1e6, pts[:, 1] / 1e6, t,
                       L("gerçek (₺M)", "actual (₺M)"), L("tahmin (₺M)", "predicted (₺M)"),
                       ref=([lim[0] / 1e6, lim[1] / 1e6], [lim[0] / 1e6, lim[1] / 1e6])), t)

    # 09 artik vs tahmin  [TEKNIK]
    if want(9):
        rs = dom["residual_scatter"]
        pts = np.array(rs["points"], dtype=float)
        t = L("Artık% vs Tahmin", "Residual% vs Predicted")
        reg(9, scatter(f"{p}-09-residual", pts[:, 0] / 1e6, pts[:, 1], t,
                       L("tahmin (₺M)", "predicted (₺M)"),
                       L("artık % = (gerçek − tahmin)/gerçek; eksi = fazla tahmin",
                         "residual % = (actual − pred.)/actual; negative = over-predicted"),
                       ref=([0, float(np.nanmax(pts[:, 0])) / 1e6], [0, 0])), t)

    # 10 ceyrege gore hata  [IS + TEKNIK]
    if want(10):
        rows = dom["quantile_error"]
        # quantile_error = ceyrek basina MEDYAN APE (uretici np.nanmedian) - MAPE degil
        # 2026-09-28: ceyrekler tahmin edilen fiyattan (08_residuals); gercek fiyata gore gruplama cikti.
        t = L("Tahmin edilen fiyat çeyreğine göre medyan hata (%)", "Median error by predicted-price quartile (%)")
        reg(10, bar(f"{p}-10-quartile-error", [r[0] for r in rows], [r[1] for r in rows],
                    t, L("medyan mutlak hata %", "median absolute error %")), t)

    # 11 ilan adedi vs hata  [TEKNIK]
    if want(11):
        # Kaynak error_drivers.json → TUM modeller. Eski surum site_data'daki residual_vs_n'i kullaniyordu;
        # uretici n<5 modelleri atiyor (build_site_data.py ms[ms.n>=5]) ve hikayenin en guclu kismi
        # (745 modelin ~307'si) gorunmuyordu. Ayrica s=3/alpha=.18 ile noktalar neredeyse gorunmuyordu
        # ve egilim cizgisi yoktu. Simdi: 1-4 ilanli modeller ICI BOS isaretle ayrilir (gizlenmez),
        # %CAP ustu noktalar ust kenarda sayisiyla gosterilir (yoksa y ekseni 100'e uzayip huniyi ezer),
        # kova medyani kalin cizgi — noktalar arka plan olur.
        ed_ = v["ed"]
        pme, bk = ed_["per_model_error"], ed_["per_model_buckets"]
        t = L("Emsali az olan modelde hata büyük — model başına medyan hata",
              "Fewer comparables, larger error — median error per model")
        CAP = 40
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        many = [(n, m) for n, m in pme if n >= 5 and m <= CAP]
        few = [(n, m) for n, m in pme if n < 5 and m <= CAP]
        over = [(n, m) for n, m in pme if m > CAP]
        ax.scatter([n for n, _ in many], [m for _, m in many], s=14, alpha=.45, color=C1, lw=0,
                   label=L("model (5+ ilan)", "model (5+ listings)"))
        ax.scatter([n for n, _ in few], [m for _, m in few], s=18, facecolors="none", edgecolors=C1,
                   lw=.8, alpha=.75,
                   label=L("model (1–4 ilan: medyanı birkaç ilandan, gürültülü)",
                           "model (1–4 listings: median from a few ads, noisy)"))
        if over:
            ax.scatter([n for n, _ in over], [CAP + 1.5] * len(over), marker="^", s=24, color=C2, lw=0,
                       label=L(f"%{CAP} üstü: {len(over)} model (kenarda)",
                               f"above {CAP}%: {len(over)} models (at edge)"))
        # kova medyani: y = error_drivers.json'daki kova degeri (tek kaynak), x = kovadaki medyan ilan sayisi
        bx, by = [], []
        for b in bk:
            ns = [n for n, _ in pme if b["lo"] <= n <= b["hi"]]
            if ns:
                bx.append(float(np.median(ns))); by.append(b["median_of_medians"])
        ax.plot(bx, by, "o-", color=C3, lw=2.4, ms=6, zorder=5,
                label=L("kova medyanı (1 · 2–4 · 5–19 · 20–99 · 100+ ilan)",
                        "bucket median (1 · 2–4 · 5–19 · 20–99 · 100+ listings)"))
        for x_, y_ in zip(bx, by):
            ax.annotate(f"%{y_:.1f}" if lang == "tr" else f"{y_:.1f}%", (x_, y_),
                        textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8,
                        color=C3, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.9))  # daireler ustune binmesin
        ax.set_xscale("log"); ax.set_ylim(0, CAP + 4)
        _xt = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]                  # 10^0 yerine düz sayı
        ax.set_xticks(_xt, [num(x, lang) for x in _xt]); ax.minorticks_off()
        ax.set_xlabel(L("modeldeki ilan sayısı (log)", "listings per model (log)"))
        ax.set_ylabel(L("model başına medyan hata %", "median error per model %"))
        ax.set_title(t); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(11, _save(fig, f"{p}-11-n-vs-error"), t)

    # 12 kapsama  [IS + TEKNIK]
    if want(12):
        # 2026-09-28: tahmin fiyati bandi basina iki kol (tek oran, banda gore / Mondrian), capraz kalibreli.
        rows, rows_m = dom["conformal"]["by_quantile"], dom["conformal"]["mondrian_by_quantile"]
        assert [r[0] for r in rows] == [r[0] for r in rows_m], "kapsama kollarinin bantlari farkli"
        tgt = dom["conformal"]["coverage_target"]
        t = L(f"%{tgt} aralık kaç ilanda tuttu (hedef %{tgt})",
              f"How often the {tgt}% range held (target {tgt}%)")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(rows))
        ax.bar(x - .2, [r[1] for r in rows], width=.4, color=C2, label=L("tek oran", "one margin"))
        ax.bar(x + .2, [r[1] for r in rows_m], width=.4, color=C1, label=L("banda göre", "per band"))
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_xlabel(L("tahmin edilen fiyat çeyreği", "predicted-price quartile"))
        ax.set_ylim(min(r[1] for r in rows + rows_m) - 5, 100)
        ax.axhline(tgt, color=C3, ls="--", lw=1.2)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_ylabel(L("kapsama %", "coverage %")); ax.set_title(t)
        ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(12, _save(fig, f"{p}-12-coverage"), t)

    # 13 drift histogram  [TEKNIK]
    if want(13):
        hist = dom["drift"]["hist"]
        edges = hist["edges"]
        t = L("Fiyat dağılımı — dönemlere göre", "Price distribution by snapshot")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for snap, ys in hist.items():
            if snap == "edges":
                continue
            # 2026-09-23: eskiden step(sol kenarlar, where="mid") — her cubuk yarim bin sola kayiyordu.
            assert edges and len(edges) == len(ys) + 1, "drift histogrami: kenar sayisi yukseklik+1 degil"
            ax.stairs(np.array(ys) * 1e6, np.array(edges) / 1e6, lw=1.3, label=snap)   # yogunluk ₺M basina
        ax.set_xlabel(L("fiyat (₺M)", "price (₺M)")); ax.set_ylabel(L("yoğunluk (₺M başına)", "density (per ₺M)"))
        ax.set_title(t); ax.legend(frameon=False, fontsize=7)
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(13, _save(fig, f"{p}-13-drift-hist"), t)

    # 14 drift KDE (log)  [TEKNIK]
    if want(14):
        kde = dom["drift"]["kde_log"]
        t = L("Log-fiyat yoğunluğu — dönemlere göre", "Log-price density by snapshot")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        xs = kde["x"]
        for snap, ys in kde.items():
            if snap == "x":
                continue
            ax.plot(xs, ys, lw=1.3, label=snap)
        ax.set_xlabel(L("log fiyat", "log price")); ax.set_ylabel(L("yoğunluk", "density"))
        ax.set_title(t); ax.legend(frameon=False, fontsize=7)
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(14, _save(fig, f"{p}-14-drift-kde"), t)

    # 15 veri miktari ve hata  [IS + TEKNIK]
    if want(15):
        # Iki cizgi AYNI 4 donemde hizalanir (indeksle). Eski surum x'e yazi etiketi veriyordu;
        # insample etiketleri '→01-27' gibi okla basladigi icin matplotlib 7 ayri kategori kurup
        # yesil cizgiyi turuncunun bittigi yerden devam ettiriyordu — iki cizgi hic hizalanmiyordu.
        # Farki yaratan sey ILAN SAYISI (turuncu ~11 bin sabit, yesil 11→30 bin buyuyor); o yuzden
        # yesil noktalarin altina n yazilir, turuncunun aciklamasina tek donemin n araligi.
        bt = met["backtest"]
        ps, ins = bt["per_snapshot"], bt["insample"]
        # 2026-09-28: baslik iddia tasimiyor (iki cizgi farkli ilan kumeleri); birikimin etkisi §9'un eşli
        # karsilastirmasinda, ayni test ilanlarinda olculuyor.
        t = L("Tek dönem ve biriken dönemler — ortalama yüzde hata",
              "Single period and pooled periods — mean percentage error")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        xs = list(range(len(ps)))
        _nps = [r[2] for r in ps]
        ax.plot(xs, [r[1] for r in ps], "o-", color=C1, lw=1.6, ms=4,
                label=L(f"yalnız o dönemin ilanları ({num(min(_nps), lang)}–{num(max(_nps), lang)} ilan)",
                        f"that period's listings only ({num(min(_nps), lang)}–{num(max(_nps), lang)})"))
        ax.plot(xs[:len(ins)], [r[1] for r in ins], "o-", color=C2, lw=1.6, ms=4,
                label=L("o tarihe kadarki tüm dönemler birlikte", "all periods up to that date, pooled"))
        for x_, r in zip(xs, ins):
            ax.annotate(f"{num(r[2], lang)} " + L("ilan", "listings"), (x_, r[1]),
                        textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7, color=C2)
        ax.set_xticks(xs, [r[0] for r in ps])
        ax.set_xlabel(L("dönem", "period")); ax.set_ylabel("MAPE %"); ax.set_title(t)
        ax.margins(x=0.08, y=0.25)          # kenardaki n etiketleri kesilmesin
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(15, _save(fig, f"{p}-15-backtest"), t)

    # 16 eksiklik orani  [TEKNIK]
    if want(16):
        rows = met["systematic_missing"]["column_missing_all"]
        # 2026-09-23: baslik "ayni oran = birlikte eksik blok" diyordu; transmission_brand %74.1 ile gb_drivetrain %74.0
        # ayni oranda ama yalniz %48 birlikte eksik. Renk artik GERCEK blok uyeligi (systematic_groups[].columns).
        # gb_drivetrain de "Cekis" diye yaziliyordu — modelin %1.5 eksik kb_drivetrain'iyle ayni ad; ayni adli
        # ikizi olan kolona sekme adi eklenir.
        _group_colors = ["#2563eb", "#7c3aed", "#0891b2", "#d97706", "#059669", "#db2777"]
        _member_color = {c_: _group_colors[gi % len(_group_colors)]
                for gi, g in enumerate(met["systematic_missing"]["systematic_groups"]) for c_ in g["columns"]}
        _lab = d["column_labels"]

        def _col16(k):
            """
            EN: Column label for the missing-rate chart; adds the tab name when a gb_/kb_ twin has the same
                label.
            TR: Eksiklik grafiği için kolon etiketi; aynı etiketli gb_/kb_ ikizi varsa sekme adını ekler.
            """
            t_ = col(d, k, lang)
            twin = ("kb_" + k[3:]) if k.startswith("gb_") else (("gb_" + k[3:]) if k.startswith("kb_") else None)
            if twin and twin in _lab and col(d, twin, lang) == t_:
                t_ += L(" (Genel Bakış)", " (Overview)") if k.startswith("gb_") else L(" (KısaBilgi)", " (Quick info)")
            return t_
        t = L("Eksiklik oranı (%) — renk = birlikte eksik blok (aşağıdaki tablo) · gri = blok dışı\n"
              "(etiketsiz olanlar ham kolon adı)",
              "Missing rate (%) — colour = co-missing block (table below) · grey = outside any block\n"
              "(unlabelled ones are raw column names)")
        reg(16, bar(f"{p}-16-missing", [_col16(r[0]) for r in rows], [r[1] for r in rows], t,
                    L("eksik %", "missing %"), horizontal=True, color=[_member_color.get(r[0], "#b8bec8") for r in rows]), t)

    # 17 Theil's U ve 18 Cramer's V matrisleri 2026-09-28'de cikti (sadelestirme listesi); numaralar bos kalir.
    # 19 seri x segment  [TEKNIK]
    if want(19):
        rows = dom["series_segment_matrix"]
        series = sorted({r[0] for r in rows}); segs = sorted({r[1] for r in rows})
        M = np.full((len(series), len(segs)), np.nan)
        for s_, g_, med, _n in rows:
            M[series.index(s_), segs.index(g_)] = med / 1e6
        # Baslik KOSULLU (2026-09-23): eskiden sabit "Her seri tek bir segmente dusuyor" yaziyordu.
        # Yeni kuralda performans aileleri model adindan cozuldugu icin birden fazla segmente yayilabilir.
        _repeated = sorted({r[0] for r in rows if sum(1 for q in rows if q[0] == r[0]) > 1})
        t = (L("Her seri tek bir segmente düşüyor — medyan fiyat (₺M)",
               "Every series lands in exactly one segment — median price (₺M)") if not _repeated else
             L(f"Seri × segment — medyan fiyat (₺M); {len(_repeated)} seri birden fazla segmente düşüyor",
               f"Series × segment — median price (₺M); {len(_repeated)} series span more than one segment"))
        # Tek ilanlik hucreler isaretlenir (son denetim: koyu hucrelerin bazisi n=1).
        _n19 = {(s_, g_): int(n_) for s_, g_, _m, n_ in rows}
        _single_cells = [(series.index(s_), segs.index(g_)) for (s_, g_), n_ in _n19.items() if n_ == 1]
        t = t + L(" · • = tek ilan", " · • = single listing")
        _f19 = heatmap(f"{p}-19-series-segment", series, segs, M, t, cmap="YlGnBu", marks=_single_cells)
        reg(19, _f19, t)

    # 21 Spearman  [TEKNIK] — 20 Pearson 2026-09-28'de cikti (sadelestirme listesi).
    if want(21):
        nc = dom["numeric_correlation"]
        _nl = [col(d, k, lang) for k in nc["labels"]]
        nm = L("Spearman korelasyonu", "Spearman correlation")
        reg(21, heatmap(f"{p}-21-spearman", _nl, _nl, nc["spearman"], nm, -1, 1, cmap="RdYlGn"), nm)

    # 22/23 PCA  [TEKNIK]
    if want(22) or want(23):
        pca = {a["pc"]: a for a in met["pca_axes"]}
        # 2026-09-23: lejant uretilen ADLARI kullaniyordu; K=3'te iki kume ayni adi aldi. Tablodaki
        # numarali etiketlerle ayni (cluster_labels) — figur ve tablo birbirine baglanir.
        clusters = {c["cluster"]: lab for c, lab in zip(dom["kmeans"], cluster_labels(dom["kmeans"], lang))}
        for no, key, pcy in [(22, "pca_scatter", "PC2"), (23, "pca_scatter_13", "PC3")]:
            pts = np.array(dom[key], dtype=float)
            v1 = pca["PC1"]["var_pct"]; v2 = pca[pcy]["var_pct"]
            t = L(f"PCA — PC1 %{v1} × {pcy} %{v2}", f"PCA — PC1 {v1}% × {pcy} {v2}%")
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            for cid, nmc in clusters.items():
                m = pts[:, 2] == cid
                ax.scatter(pts[m, 0], pts[m, 1], s=3, alpha=.25, lw=0, label=nmc)
            ax.set_xlabel("PC1"); ax.set_ylabel(pcy); ax.set_title(t)
            ax.legend(frameon=False, fontsize=7, markerscale=3)
            ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
            reg(no, _save(fig, f"{p}-{no}-{key.replace('_', '-')}"), t)

    # 24 k secimi  [TEKNIK]
    if want(24):
        ks = met["kmeans_selection"]
        t = L("k seçimi — dirsek + siluet", "k selection — Elbow + Silhouette")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot([r[0] for r in ks["elbow"]], [r[1] / 1e3 for r in ks["elbow"]], "o-", color=C1,
                label=L("dirsek (atalet, bin)", "elbow (inertia, thousands)"))
        ax2 = ax.twinx()
        ax2.plot([r[0] for r in ks["silhouette"]], [r[1] for r in ks["silhouette"]], "s--",
                 color=C3, label=L("siluet", "silhouette"))
        ax.axvline(ks["chosen_k"], color=C2, ls=":", lw=1.2)
        ax.set_ylabel(L("atalet (bin)", "inertia (thousands)")); ax2.set_ylabel(L("siluet", "silhouette"))
        ax.set_xlabel(L("k (küme sayısı) · noktalı çizgi = seçilen k", "k (number of clusters) · dotted = chosen k"))
        ax.set_title(t)
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        fig.legend(frameon=False, loc="upper right", bbox_to_anchor=(.98, .95), fontsize=8)
        reg(24, _save(fig, f"{p}-24-k-selection"), t)

    # 25 fiyat histogrami  [TEKNIK]
    if want(25):
        rows = dom["price_histogram"]
        t = L("Fiyat histogramı — tüm veri (kesikli çizgi = medyan)",
              "Price histogram — all data (dashed line = median)")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar([r[0] / 1e6 for r in rows], [r[1] for r in rows],
               width=(rows[1][0] - rows[0][0]) / 1e6 * .9, color=C1, align="edge")
        ax.axvline(v["median"] / 1e6, color=C3, ls="--", lw=1.2)
        ax.set_xlabel(L("fiyat (₺M)", "price (₺M)")); ax.set_ylabel(L("ilan", "listings"))
        ax.set_title(t); ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(25, _save(fig, f"{p}-25-price-hist"), t)

    # 26 hata dagilimi — OOF artik % histogrami  [TEKNIK]
    if want(26):
        r = np.array(dom["residual_scatter"]["points"], dtype=float)[:, 1]
        lim = RESID_VIEW
        t = L("OOF hata dağılımı — artık % = (gerçek − tahmin) / gerçek",
              "OOF error distribution — residual % = (actual − predicted) / actual")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(r[(r >= -lim) & (r <= lim)], bins=np.arange(-lim, lim + 1, 1), color=C1)
        ax.axvspan(-10, 10, color=C2, alpha=.08, lw=0)
        ax.axvline(0, color=C3, ls="--", lw=1.2)
        ax.set_xlabel(L("artık % (eksi = model fazla tahmin etti) · gri bant = ±%10",
                        "residual % (negative = model over-predicted) · grey band = ±10%"))
        ax.set_ylabel(L("ilan", "listings")); ax.set_title(t)
        n_lo, n_hi = int((r < -lim).sum()), int((r > lim).sum())
        ax.text(.01, .97, L(f"görünüm dışı: {num(n_lo, lang)} ilan < -%{lim} · {num(n_hi, lang)} ilan > +%{lim}",
                            f"outside view: {num(n_lo, lang)} listings < -{lim}% · {num(n_hi, lang)} > +{lim}%"),
                transform=ax.transAxes, va="top", fontsize=7, color=C2)
        ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
        reg(26, _save(fig, f"{p}-26-error-hist"), t)

    # 27 fiyat ceyregine gore LIRA hatasi  [IS + TEKNIK] — figur 10'un eslikcisi (2026-09-23).
    if want(27):
        # Yuzde hata ucuz ceyrege isaret ediyor; lira hatasi pahaliya. Kaynak error_drivers.lira_quartile.
        lc = v["ed"]["lira_quartile"]
        t = L("Tahmin edilen fiyat çeyreğine göre lira hatası", "Lira error by predicted-price quartile")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        qs = [r[0] for r in lc]
        a1.bar(qs, [r[2] for r in lc], color=C1)
        for x_, r_ in zip(qs, lc):
            a1.annotate(P(r_[2], lang), (x_, r_[2]), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
        a1.set_ylabel(L("toplam lira hatasındaki pay %", "share of total lira error %"))
        a1.set_title(L("hata nerede birikiyor", "where the error adds up"), fontsize=9)
        # Son denetim (2026-09-23): sapma GERCEK fiyat ceyregine gore cizilince ortalamaya donus "ucuzda fazla,
        # pahalida dusuk" deseni uretiyordu. Bir fiyatlama araci yalniz tahmini bilir -> tahmin ceyregi.
        tc = v["ed"]["pred_quartile"]
        # Ortalama tek basina kuyruk hatalarina bagli (medyan her ceyrekte pozitif) ve otomatik eksen kucuk farki
        # buyuk gosteriyordu: ortalama + medyan yan yana, eksen genel MAE'nin yarisinda sabit.
        _x = np.arange(len(tc))
        a2.bar(_x - .2, [r[2] / 1e3 for r in tc], width=.4, color=C1, label=L("ortalama", "mean"))
        a2.bar(_x + .2, [r[3] / 1e3 for r in tc], width=.4, color=C2, label=L("medyan", "median"))
        a2.set_xticks(_x, [r[0] for r in tc])
        a2.axhline(0, color="#444", lw=.8)
        _lim = .5 * v["model_mae"] / 1e3
        a2.set_ylim(-_lim, _lim)
        a2.legend(frameon=False, fontsize=7)
        a2.set_ylabel(L("sapma: tahmin − gerçek (₺bin; eksen ±MAE/2)", "bias: predicted − actual (₺k; axis ±MAE/2)"))
        a1.set_xlabel(L("tahmin edilen fiyat çeyreği", "predicted-price quartile"))
        a2.set_xlabel(L("tahmin edilen fiyat çeyreği", "predicted-price quartile"))
        a2.set_title(L("tahmine göre gruplanınca sapma", "bias when grouped by prediction"), fontsize=9)
        for a_ in (a1, a2):
            a_.grid(axis="y", color=GRID, lw=.7); a_.set_axisbelow(True)
        fig.suptitle(t)
        reg(27, _save(fig, f"{p}-27-quartile-lira"), t)

    # 28 lira olceginde artik  [TEKNIK] — yuzde artik (figur 09) pahali ucu ezer.
    if want(28):
        _pp = np.array(dom["pred_vs_true"]["points"], dtype=float)
        t = L("Lira ölçeğinde hata — tahmin − gerçek", "Error in lira — predicted − actual")
        # Son denetim (2026-09-23): x ekseni GERCEK fiyattaydi -> ortalamaya donus yelpazesi (+ 6.5M tavan).
        # Fiyatlama aracinin bildigi tek sey tahmin; x = tahmin.
        reg(28, scatter(f"{p}-28-residual-lira", _pp[:, 1] / 1e6, (_pp[:, 1] - _pp[:, 0]) / 1e6, t,
                        L("tahmin edilen fiyat (₺M)", "predicted price (₺M)"),
                        L("tahmin − gerçek (₺M; eksi = düşük tahmin)", "predicted − actual (₺M; negative = under)"),
                        ref=([0, float(_pp[:, 1].max()) / 1e6], [0, 0])), t)

    # 29 motor gucu/hacmi: araliktan tek sayiya  [TEKNIK] — kullanicinin kurali (2026-09-23).
    if want(29):
        # Kutu = ceyrekler, cizgi = medyan, biyik = %5–%95 (ham dizi JSON'da yok; error_drivers.engine_rule).
        _er = v["ed"]["engine_rule"]
        t = L("Aralıktan tek sayıya: adayın aynı modelin kesin değerinden uzaklığı",
              "From a range to one number: each candidate's distance from the model's exact value")
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
        _cand_label = {"lower": L("alt sınır", "lower bound"), "mid": L("orta nokta", "midpoint"),
               "upper": L("üst sınır", "upper bound")}
        for ax, (p_, heading) in zip(axs, (("engine_cc", L("motor hacmi", "engine size")),
                                       ("power_hp", L("motor gücü", "engine power")))):
            h_ = _er[p_]
            ks = list(h_["candidates"])
            st = []
            for k in ks:
                q_ = h_["candidates"][k]["percentiles"]
                st.append({"whislo": q_[0], "q1": q_[1], "med": q_[2], "q3": q_[3], "whishi": q_[4],
                           "fliers": [], "label": _cand_label[k] + (L("\n(seçilen)", "\n(chosen)") if k == h_["chosen"] else "")})
            bp = ax.bxp(st, showfliers=False, patch_artist=True, widths=.55)
            for k, box, med in zip(ks, bp["boxes"], bp["medians"]):
                box.set_facecolor(C1 if k == h_["chosen"] else "#d9d9d9")
                box.set_edgecolor("#444"); med.set_color("#111")
            for i_, k in enumerate(ks, 1):
                m_ = h_["candidates"][k]["median"]
                ax.annotate(f"{m_:g}", (i_ + .3, m_), textcoords="offset points", xytext=(3, 0), va="center",
                            fontsize=8, fontweight="bold" if k == h_["chosen"] else "normal")
            ax.set_title(f"{heading} · " + L(f"referanslı {num(h_['with_reference'], lang)} aralıklı ilan",
                                           f"{num(h_['with_reference'], lang)} range listings with a reference"),
                         fontsize=9)
            ax.set_ylabel(L(f"|aday − kesin değer| ({h_['unit']})", f"|candidate − exact value| ({h_['unit']})"))
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
        fig.suptitle(t)
        fig.text(.5, .005, L("kutu = çeyrekler · çizgi ve sayı = medyan · bıyık = %5–%95 · kesin değer = aynı modelin "
                             "kesin değerli ilanlarının medyanı",
                             "box = quartiles · line and number = median · whiskers = 5th–95th percentile · exact "
                             "value = median of the same model's exact-value listings"),
                 ha="center", fontsize=7, color=C2)
        fig.tight_layout(rect=(0, .04, 1, 1))
        reg(29, _save(fig, f"{p}-29-hp-cc-rule"), t)

    # 30 sitenin verdigi alt x ust sinir  [TEKNIK] — deneysel scatter'dan rapora (2026-09-25).
    if want(30):
        # Kesin deger kosegende (alt = ust), aralik kosegenin ustunde; nokta = (alt, ust) cifti, buyuklugu ilan sayisi.
        _er = v["ed"]["engine_rule"]
        t = L("Sitenin verdiği değerler: alt × üst sınır", "What the site gives: lower × upper bound")
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2))
        for ax, (p_, heading) in zip(axs, (("engine_cc", L("motor hacmi", "engine size")),
                                       ("power_hp", L("motor gücü", "engine power")))):
            h_ = _er[p_]
            c_ = np.array([[lo_, up_, r_, n_] for lo_, up_, r_, n_ in h_["pairs"]], dtype=float)
            nmax = c_[:, 3].max()
            for is_r, col_, label_ in ((0, C1, L("kesin değer", "exact value")), (1, C3, L("aralık", "range"))):
                s_ = c_[c_[:, 2] == is_r]
                ax.scatter(s_[:, 0], s_[:, 1], s=6 + 260 * s_[:, 3] / nmax, color=col_, alpha=.55, lw=0,
                           label=L(f"{label_} ({num(s_[:, 3].sum(), lang)} ilan)",
                                   f"{label_} ({num(s_[:, 3].sum(), lang)} listings)"))
            lim = [c_[:, :2].min() * .95, c_[:, :2].max() * 1.02]
            ax.plot(lim, lim, "--", color=C2, lw=.8, label="y = x")
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
            ax.set_title(heading, fontsize=9)
            ax.set_xlabel(L(f"alt sınır ({h_['unit']})", f"lower bound ({h_['unit']})"))
            ax.set_ylabel(L(f"üst sınır ({h_['unit']})", f"upper bound ({h_['unit']})"))
            ax.legend(loc="upper left", frameon=False, fontsize=7, markerscale=.5)
            ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        fig.suptitle(t)
        _c, _h = _er["engine_cc"], _er["power_hp"]
        fig.text(.5, .005, L(f"nokta büyüklüğü = ilan sayısı · kesin değer köşegende (alt = üst) · çizilemeyen: açık "
                             f"uçlu aralık {num(_c['open_ended'], lang)}/{num(_h['open_ended'], lang)}, değersiz "
                             f"{num(_c['no_value'], lang)}/{num(_h['no_value'], lang)} ilan (hacim/güç)",
                             f"point size = listings · exact values on the diagonal (lower = upper) · not drawn: "
                             f"open-ended range {num(_c['open_ended'], lang)}/{num(_h['open_ended'], lang)}, no value "
                             f"{num(_c['no_value'], lang)}/{num(_h['no_value'], lang)} listings (size/power)"),
                 ha="center", fontsize=7, color=C2)
        fig.tight_layout(rect=(0, .04, 1, 1))
        reg(30, _save(fig, f"{p}-30-engine-bounds"), t)

    return F


# ============================================================================
#  METIN  (tek sozluk, iki dil - eksik anahtar = hata)
# ============================================================================
BUSINESS_FIGS = [0, 5, 6, 7, 10, 12, 15, 27]        # 2 (segment medyani) 2026-09-22'de cikti
# 10/12/15 iki raporda da: karar notunda karar icin, teknik raporda kanit olarak (bilincli tekrar).
# 2026-09-28 (sadelestirme listesi): 17/18/20 cikti; 27 yalniz karar notunda.
TECHNICAL_FIGS = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30]


def num(x, lang="tr"):
    """
    EN: Thousands separator only on the number itself: tr 1.234.567, en 1,234,567 (never on sentence commas).
    TR: Binlik ayracı yalnız sayının kendisine: tr 1.234.567, en 1,234,567 (cümle virgüllerine asla).
    """
    t = f"{int(round(float(x))):,}"
    return t.replace(",", ".") if lang == "tr" else t


def _m2(x):
    """
    EN: Millions rounded to 2 decimals half-up (binary floats rounded .xx5 down: 1.545 → 1.54).
    TR: Milyon cinsinden 2 haneye yarım yukarı yuvarlar (ikili kayan nokta .xx5'i aşağı atıyordu: 1.545 → 1.54).
    """
    return Decimal(str(float(x) / 1e6)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def tlm(x):
    """
    EN: ₺ always in millions, for ranges with one unit: 845000 → ₺0.85M.
    TR: Aralıkta tek birim için ₺ her zaman milyon: 845000 → ₺0.85M.
    """
    return f"₺{_m2(x)}M"


def tl(x):
    """
    EN: ₺ in report format: 110072 → ₺110K, 1545000 → ₺1.55M.
    TR: ₺ rapor biçiminde: 110072 → ₺110K, 1545000 → ₺1.55M.
    """
    x = float(x)
    if abs(x) >= 1e6:
        return f"₺{_m2(x)}M"
    if abs(x) >= 1e3:
        return f"₺{x / 1e3:.0f}K"
    return f"₺{x:.0f}"


# ---- EN etiket sozlukleri (kaynak: sadik-portfolio/lib/labels.ts; eksik anahtar -> ham deger) ----
# EN: the hedonic model's term ids (06_hedonic columns[].coefficients[].term) -> (TR, EN) display label
# TR: hedonik modelin terim kimlikleri (06_hedonic columns[].coefficients[].term) -> (TR, EN) görünen etiket
HED_TERM = {"age": ("yaş", "age"), "age_sq": ("yaş²", "age²"), "km100k": ("km (100 bin)", "km (100k)"),
            "km_sq": ("km²", "km²"), "age_x_km": ("yaş×km", "age×km"), "heavy_damage": ("ağır hasar", "heavy damage"),
            "painted": ("boyalı", "painted"), "changed": ("değişen", "changed"), "hp100": ("+100 hp", "+100 hp"),
            "litre": ("+1 litre", "+1 litre")}
# EN: ids the metrics publish -> (TR, EN) display; id_label stops on an unknown id
# TR: metriklerin yayımladığı kimlikler -> (TR, EN) görünen ad; id_label bilinmeyen kimlikte durur
TIER_LABEL = {"model_year": ("model+yıl", "model + year"), "model": ("model", "model"), "global": ("global", "global")}
BAND_LABEL = {"economy": ("ekonomik", "economy"), "mid": ("orta", "mid"), "premium": ("premium", "premium")}
FUEL_EN = {"Benzin": "Petrol", "Dizel": "Diesel", "LPG & Benzin": "LPG & Petrol", "Hibrit": "Hybrid"}
CLUSTER_EN = {"Yaşlı & yüksek-km ekonomik": "Older, high-km economy", "Genç & temiz premium": "Newer, clean premium",
              "Hasarlı": "Damaged", "Orta segment": "Mid segment"}
LADDER_EN = {"(model, yıl) medyanı": "(model, year) median", "(model) medyanı — tüm yıllar": "(model) median — all years",
             "global medyan": "global median"}
# final_results.model_comparison anahtari -> (kazanan kodu, gorunen ad)
# Etiketlerde "model/seri adi" var cunku TF-IDF+SVD SERBEST METNE degil, yalniz model/series ad
# dizgilerine uygulaniyor (kullanici duzeltmesi 2026-09-20 — okur bunu ilan metni saniyordu).
VARIANTS = [("lightgbm_tfidf_svd", "lightgbm", "LightGBM (model/seri adı TF-IDF+SVD)"),
            ("catboost_tfidf_svd", "catboost", "CatBoost (model/seri adı TF-IDF+SVD)"),
            ("catboost_native", None, "CatBoost (model/seri adı native text)")]
VARIANTS_EN = {"LightGBM (model/seri adı TF-IDF+SVD)": "LightGBM (model/series name TF-IDF+SVD)",
               "CatBoost (model/seri adı TF-IDF+SVD)": "CatBoost (model/series name TF-IDF+SVD)",
               "CatBoost (model/seri adı native text)": "CatBoost (model/series name native text)"}


# Kucuk sayilar metinde rakam degil SOZCUK yazilir (kullanicinin uslubu): "iki cift", "dokuz gun".
# Turkce buyuk harf: 'iki' -> 'Iki' DEGIL 'İki' (str.capitalize() bunu bozar).
_NUMBER_WORDS_TR = {1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz",
            9: "dokuz", 10: "on"}
_NUMBER_WORDS_EN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten"}
MONTHS_TR = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül",
         "Ekim", "Kasım", "Aralık"]
MONTHS_EN = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]


# "yirmide biri" gibi kesirler: Turkce locative eki unlu/unsuz uyumuna gore degisiyor (yirmi->yirmide
# ama kirk->kirkta), kural yazmak yerine yaygin degerler burada; disinda kalan rakamla basilir.
_FRACTIONS_TR = {10: "onda", 20: "yirmide", 25: "yirmi beşte", 30: "otuzda", 40: "kırkta", 50: "ellide",
             60: "altmışta", 75: "yetmiş beşte", 100: "yüzde"}


def fraction_words(n, lang):
    """
    EN: The fraction 1/n in Turkish words ('yirmide biri'); falls back to digits.
    TR: 1/n kesri Türkçe sözcükle ('yirmide biri'); yoksa rakamla.
    """
    n = int(round(n))
    return f"{_FRACTIONS_TR.get(n, str(n) + chr(39) + 'de')} biri"


def number_word(n, lang, cap=False):
    """
    EN: Numbers up to ten as words (the owner's style), otherwise digits; cap=True capitalises (TR İ).
    TR: Ona kadar sayılar sözcükle (kullanıcının üslubu), üstü rakamla; cap=True ilk harfi büyütür (TR İ).
    """
    w = (_NUMBER_WORDS_TR if lang == "tr" else _NUMBER_WORDS_EN).get(int(n), str(int(n)))
    if cap and w[:1].isalpha():
        w = ("İ" + w[1:]) if w[:1] == "i" else w[:1].upper() + w[1:]
    return w


def id_label(m, key, lang):
    """
    EN: The display label of an id the metrics publish, from an {id: (tr, en)} map; an unknown id stops.
    TR: Metriklerin yayımladığı bir kimliğin görünen adı, {kimlik: (tr, en)} sözlüğünden; bilinmeyen kimlik durdurur.
    """
    return m[key][0 if lang == "tr" else 1]


def tx(m, key, lang):
    """
    EN: The raw value in TR, its dictionary translation in EN (raw if missing).
    TR: TR'de ham değer, EN'de sözlükteki karşılığı (yoksa ham).
    """
    return key if lang == "tr" else m.get(key, key)


def table(head, rows, align):
    """
    EN: Markdown table lines; align is a string of 'l'/'r' per column.
    TR: Markdown tablo satırları; align sütun başına 'l'/'r' dizisi.
    """
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def P(x, lang, nd=1, sign=False):
    """
    EN: Percent in report format: tr %6.5 / +%5.3, en 6.5% / +5.3%; a value that rounds to zero loses its sign.
    TR: Rapor biçiminde yüzde: tr %6.5 / +%5.3, en 6.5% / +5.3%; sıfıra yuvarlanan değer işaretsiz basılır.
    """
    if x is None:
        return "—"
    x = float(x)
    if abs(x) < 0.5 * 10 ** -nd:      # yuvarlaninca sifir olan deger isaretsiz basilir ("-%0.0" degil)
        x = 0.0
    s = f"{abs(x):.{nd}f}"
    sg = ("+" if x > 0 else "-" if x < 0 else "") if sign else ("-" if x < 0 else "")
    return f"{sg}%{s}" if lang == "tr" else f"{sg}{s}%"


def fp(p):
    """
    EN: A p-value: <0.001 or three decimals.
    TR: Bir p-değeri: <0.001 ya da üç hane.
    """
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tlx(x, lang):
    """
    EN: Whole ₺ with the language's thousands separator: ₺2.867.000 / ₺2,867,000.
    TR: Dilin binlik ayracıyla tam ₺: ₺2.867.000 / ₺2,867,000.
    """
    return f"₺{num(x, lang)}"


def col(d, key, lang):
    """
    EN: Display label of a raw column in lang (the raw name if unlabelled).
    TR: Ham kolonun lang dilindeki görünen etiketi (etiketsizse ham ad).
    """
    return (d.get("column_labels", {}).get(key) or {}).get(lang) or key


def cluster_labels(clusters, lang):
    """
    EN: Clusters are numbered, not named (generated names clashed and contradicted the axes); the label keeps
        the measured heavy-damage share.
    TR: Kümeler adlandırılmaz, numaralanır (üretilen adlar çakışıyor ve eksenlerle çelişiyordu); etikette
        ölçülen ağır hasar payı kalır.
    """
    return [(f"Küme {i} · ağır hasar %{c['heavy_damage_pct']:.0f}" if lang == "tr"
             else f"Cluster {i} · {c['heavy_damage_pct']:.0f}% heavy damage")
            for i, c in enumerate(clusters, 1)]


# ============================================================================
#  TEKNIK RAPOR — bolumler ayri fonksiyonlarda, SIRA tek yerde (SECTIONS).
#  Bolum numaralari elle yazilmaz: baslik enumerate ile, capraz referanslar section_no() ile uretilir.
# ============================================================================
class _Ctx:
    """
    EN: State shared by the section functions (numbers, figures, language, view) and line-writing helpers.
    TR: Bölüm fonksiyonlarının paylaştığı durum (sayılar, figürler, dil, görünüm) ve satır yazma yardımcıları.
    """

    def __init__(self, v, F, lang, d):
        """
        EN: Keeps the numbers v, figures F, language and the metrics view d.
        TR: v sayılarını, F figürlerini, dili ve d metrik görünümünü tutar.
        """
        self.v, self.F, self.lang, self.d = v, F, lang, d
        self.dom, self.met, self.meta = d["domain"], d["methodology"], d["meta"]
        self.hr = self.dom["hedonic_reliability"]
        self.out = []

    def L(self, tr, en):
        """
        EN: The TR or EN text, by the context's language.
        TR: Bağlamın diline göre TR ya da EN metin.
        """
        return tr if self.lang == "tr" else en

    def A(self, s=""):
        """
        EN: Appends a line to the output.
        TR: Çıktıya bir satır ekler.
        """
        self.out.append(s)

    def T(self, head, rows, align):
        """
        EN: Appends a markdown table and a blank line.
        TR: Bir markdown tablo ve boş satır ekler.
        """
        self.out.extend(table(head, rows, align))
        self.A("")

    def figs(self, *nos):
        """
        EN: Appends the figure image lines for the given numbers.
        TR: Verilen numaraların figür satırlarını ekler.
        """
        for n in nos:
            self.A(f"![{self.F[n][1]}](figures/{self.F[n][0]})")
            self.A("")
