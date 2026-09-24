"""
08_large_errors.py
EN: Technical report §8 — where the large errors (|OOF residual| > 20%) come from. Rates by the number of
    comparables (same model + year), by model, by age (every age and every cut point — 18 is not a hard
    break), by segment F/S, by snapshot and for listings still live vs gone; the worst listings ranked by lira
    (a percentage ranking can never show an under-prediction beyond 100%); engine specs that break away from
    their own model's median (a catalogue mismatch prices an engine the car does not have); the series that
    were silently counted as segment D before 2026-09-23; and three example listings with reasons read from
    the data and their own text (analysis/lib/text_flags.py). Needs 07_model_comparison's OOF artefact.
TR: Teknik rapor §8 — büyük hatalar (|OOF artık| > %20) nereden geliyor. Emsal sayısına (aynı model + yıl),
    modele, yaşa (her yaş ve her kesim noktası — 18 sert bir kırılma değil), F/S segmentine, taramaya ve hâlâ
    yayında olan/kalkan ilana göre oranlar; lira ile sıralanan en kötü ilanlar (yüzde sıralaması %100'ü aşan
    düşük tahmini hiç gösteremez); kendi modelinin medyanından kopan motor değerleri (katalog uyuşmazlığı
    aracın sahip olmadığı bir motoru fiyatlar); 2026-09-23 öncesi sessizce D segmenti sayılan seriler; ve
    gerekçeleri veriden ve ilanın kendi metninden (analysis/lib/text_flags.py) okunan üç örnek ilan.
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/08_large_errors.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd

from lib import segment_rule as SR
from lib import text_flags as TF
from lib.common import load_clean, save_metrics
from lib.cv import LARGE_ERROR_PCT, PRICE_CAP, large_errors, load_oof, residual_pct

# EN: the owner's choice (2026-09-17): a modified car, an extreme car, an old luxury car with no comparables.
#     (model, model year, price); each must match exactly one listing, otherwise the run stops.
# TR: kullanıcının seçimi (2026-09-17): modifiye · uç araba · emsalsiz yaşlı lüks. (model, model yılı, fiyat);
#     her biri tam bir ilana eşleşmeli, yoksa koşum durur.
EXAMPLE_KEYS = [("640i", 2011, 5_600_000), ("4.2 FSI Quattro R-tronic", 2008, 4_690_000), ("750i Long", 2007, 1_190_000)]
BRAND = {"bmw": "BMW", "audi": "Audi"}
COUNT_BINS = [("1", 1, 1), ("2–4", 2, 4), ("5–19", 5, 19), ("20–99", 20, 99), ("100+", 100, 10**9)]
OLD_AGE, FIRST_CUT = 18, 8
TOP_N_LIRA, APE_CAP = 100, 500
# EN: the six series outside the old segment map, silently counted as D before 2026-09-23 (a fixed, historical group)
# TR: 2026-09-23 öncesi haritada olmayan, sessizce D sayılan altı seri (sabit, tarihsel grup)
OLD_D_SERIES = ("S", "M Serisi", "i Serisi", "RS", "TTS", "Z Serisi")
SPEC_MIN_GROUP, SPEC_RATIO = 5, 1.5


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def rate(mask, big):
    """
    EN: Listings in mask and their large-error rate (%). / TR: Maskedeki ilan sayısı ve büyük hata oranı (%).
    """
    n = int(mask.sum())
    return {"n": n, "big_pct": round(100 * float(big[mask].mean()), 2) if n else None}


def by_comparables(n_model_year, big, over, under):
    """
    EN: Large-error rate per bin of (model, year) listing count, split into over- and under-predictions.
    TR: (model, yıl) ilan sayısı kovası başına büyük hata oranı; fazla ve düşük tahmine ayrılmış.
    """
    out = []
    for lab, lo, hi in COUNT_BINS:
        m = (n_model_year >= lo) & (n_model_year <= hi)
        out.append({"bin": lab, **rate(m, big), "over_pct": round(100 * float(over[m].mean()), 2),
                    "under_pct": round(100 * float(under[m].mean()), 2)})
    return out


def error_per_model(model, resid):
    """
    EN: Every model's listing count and median |% error| (models with <5 listings included), and the median
        of those medians per listing-count bin. Returns: (per model rows, buckets).
    TR: Her modelin ilan sayısı ve medyan |% hata|'sı (<5 ilanlı modeller dahil) ve ilan sayısı kovası başına
        bu medyanların medyanı. Döndürür: (model satırları, kovalar).
    """
    pm = pd.DataFrame({"model": model, "ape": np.abs(resid)}).groupby("model")["ape"].agg(["size", "median"])
    rows = [[int(n), round(float(m), 2)] for n, m in zip(pm["size"], pm["median"])]
    buckets = []
    for lab, lo, hi in COUNT_BINS:
        s = pm[(pm["size"] >= lo) & (pm["size"] <= hi)]
        buckets.append({"bin": lab, "lo": lo, "hi": hi, "n_models": int(len(s)), "n_listings": int(s["size"].sum()),
                        "median_of_medians": round(float(s["median"].median()), 2)})
    return rows, buckets


def by_snapshot(snap, resid):
    """
    EN: Median residual per snapshot the listing was LAST seen in (survival, not period), and live vs gone:
        listings still in the last snapshot against those that disappeared earlier.
    TR: İlanın SON görüldüğü tarama başına medyan artık (hayatta kalma, dönem değil) ve canlı/kaybolan: son
        taramada hâlâ yayında olanlar ile daha önce kalkanlar.
    """
    rows = [{"snapshot": s, "n": int(len(g)), "median_resid_pct": round(float(g.median()), 2)}
            for s, g in pd.Series(resid).groupby(snap.values)]
    live = (snap == snap.max()).values
    return rows, {"son_tarama": str(snap.max()), "canli_n": int(live.sum()),
                  "canli_medyan_artik": round(float(np.median(resid[live])), 2), "kaybolan_n": int((~live).sum()),
                  "kaybolan_medyan_artik": round(float(np.median(resid[~live])), 2)}


def by_age(age, big):
    """
    EN: Large-error rate at every age, and for every cut point t ≥ 8: rate at ≥t, rate below t, their ratio.
    TR: Her yaşta büyük hata oranı ve her t ≥ 8 kesim noktası için: ≥t oranı, t altı oranı, oranları.
    """
    each = [[int(a), int((age == a).sum()), round(100 * float(big[age == a].mean()), 2)]
            for a in sorted(np.unique(age[~np.isnan(age)]))]
    cuts = []
    for t in range(FIRST_CUT, int(np.nanmax(age)) + 1):
        ge, lt = age >= t, age < t
        if ge.sum() and lt.sum():
            a, b = 100 * float(big[ge].mean()), 100 * float(big[lt].mean())
            cuts.append([t, int(ge.sum()), round(a, 2), round(b, 2), round(a / b, 2)])
    return each, cuts


def lira_ranking(listings, price, pred, resid, path, text_flag):
    """
    EN: The worst listings by lira error (prediction − actual) next to the % ranking: the six worst rows, and
        in the top 100 how many are under/over-predicted, in the top price quartile, in a performance family
        or carry conversion/modification wording. Returns: (dict, lira order, top-10 % error rows).
    TR: Lira hatasına (tahmin − gerçek) göre en kötü ilanlar ve yüzde sıralaması: en kötü altı satır; ilk
        100'de kaçı düşük/fazla tahmin, en pahalı çeyrekte, performans ailesinde ya da dönüşüm/modifiye ifadeli.
        Döndürür: (sözlük, lira sırası, % hatası en büyük 10 satır).
    """
    age = pd.to_numeric(listings["vehicle_age"], errors="coerce").values.astype(float)
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    dev = pred - price
    ape = np.minimum(np.abs(resid), APE_CAP)
    order = np.argsort(-np.abs(dev), kind="stable")
    ape_top = pd.Series(ape).nlargest(10).index.values
    top = order[:TOP_N_LIRA]
    q75 = float(np.quantile(price, .75))
    row = lambda i: [str(listings["model"].iat[i]), int(age[i]), None if np.isnan(km[i]) else int(km[i]),   # noqa: E731
                     float(price[i]), round(float(pred[i]), 0), round(float(-resid[i]), 1), round(float(dev[i]), 0),
                     str(path[i])]
    perf = path != "harita"
    return {"en_kotu": [row(i) for i in order[:6]], "ilk_n": TOP_N_LIRA,
            "ilk_n_dusuk": int((dev[top] < 0).sum()), "ilk_n_fazla": int((dev[top] > 0).sum()),
            "ilk_n_q4": int((price[top] >= q75).sum()), "ilk_n_perf": int(perf[top].sum()),
            "perf_genel_pct": round(100 * float(perf.mean()), 2),
            "ape_ilk10_dusuk": int((dev[ape_top] < 0).sum()), "ape_10_esik": round(float(ape[ape_top[-1]]), 1),
            "dusuk_ape_max": round(float(np.abs(resid[dev < 0]).max()), 1),
            "ilk_n_metin": int(text_flag[top].sum()),
            "perf_seriler": sorted({str(x) for x in listings.loc[perf, "series"]})}, order, ape_top


def old_d_group(series, dev, resid, order):
    """
    EN: The series once silently counted as D: their mean bias and MAE in lira, MAPE, and how many are among
        the 50 worst lira errors — did the new segment rule shrink their error?
    TR: Bir zamanlar sessizce D sayılan seriler: lira olarak ortalama sapma ve MAE, MAPE ve en kötü 50 lira
        hatasında kaç tane oldukları — yeni segment kuralı hatalarını küçülttü mü?
    """
    g = series.isin(OLD_D_SERIES).values
    return {"seriler": list(OLD_D_SERIES), "n": int(g.sum()), "ort_sapma_tl": round(float(dev[g].mean()), 0),
            "diger_ort_sapma_tl": round(float(dev[~g].mean()), 0), "mae_tl": round(float(np.abs(dev[g]).mean()), 0),
            "mape": round(float(np.abs(resid[g]).mean()), 2), "tl_ilk50_icinde": int(g[order[:50]].sum())}


def spec_outliers(listings, resid, big):
    """
    EN: Listings whose power or size is more than 1.5× off the median of their own model (models with ≥5
        listings); their error against the rest. Models with <5 listings cannot be flagged at all — that blind
        spot is counted. Returns: (dict, mask).
    TR: Gücü ya da hacmi kendi modelinin medyanından 1.5 kattan fazla sapan ilanlar (≥5 ilanlı modeller);
        hataları geri kalanla. <5 ilanlı modellerde bayrak hiç kalkamaz — bu kör nokta sayılır.
        Döndürür: (sözlük, maske).
    """
    hp = listings[["power_hp_low", "power_hp_up"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    cc = pd.to_numeric(listings["engine_cc_up"], errors="coerce")
    g = listings.groupby("model")
    n_model = g["model"].transform("size")
    hp_med, cc_med = hp.groupby(listings["model"]).transform("median"), g["engine_cc_up"].transform("median")
    off = ((n_model >= SPEC_MIN_GROUP) & ((hp > SPEC_RATIO * hp_med) | (hp < hp_med / SPEC_RATIO)
                                          | (cc > SPEC_RATIO * cc_med) | (cc < cc_med / SPEC_RATIO))).fillna(False).values
    err = np.abs(resid)
    blind = (n_model < SPEC_MIN_GROUP).values
    return {"esik": SPEC_RATIO, "min_grup": SPEC_MIN_GROUP,
            "kor_nokta": {"ilan": int(blind.sum()), "pct": round(100 * float(blind.mean()), 2),
                          "model": int(listings.loc[blind, "model"].nunique()),
                          "medyan_hata_pct": round(float(np.median(err[blind])), 1) if blind.any() else None},
            "n": int(off.sum()), "pct": round(100 * float(off.mean()), 2),
            "medyan_hata_pct": round(float(np.median(err[off])), 1), "buyuk_hata_pct": round(100 * float(big[off].mean()), 1),
            "diger_medyan_hata_pct": round(float(np.median(err[~off])), 1),
            "diger_buyuk_hata_pct": round(100 * float(big[~off].mean()), 1)}, off


def example_reasons(listings, i, age, flags, n_model, n_model_year):
    """
    EN: Reasons for one example listing, read from the data and its own text (TR and EN lists).
    TR: Bir örnek ilanın veriden ve kendi metninden okunan gerekçeleri (TR ve EN listeleri).
    """
    tr, en = [], []
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    heavy = pd.to_numeric(listings["is_heavy_damaged"], errors="coerce").fillna(0).values == 1
    if flags["hp"][0][i]:
        tr.append(f"metinde {flags['hp'][1][i]} hp, formda {flags['hp'][2][i]} hp")
        en.append(f"text says {flags['hp'][1][i]} hp, form says {flags['hp'][2][i]} hp")
    if flags["model"][0][i]:
        tok = str(flags["model"][1][i]).upper()
        tr.append(f"metinde farklı M/RS modeli ({tok})")
        en.append(f"text names a different M/RS model ({tok})")
    if flags["conversion"][i]:
        tr.append("metinde dönüşüm ifadesi"); en.append("conversion wording in text")
    if flags["modification"][i]:
        tr.append("metinde modifiye ifadesi"); en.append("modification wording in text")
    if n_model[i] < 20:
        tr.append(f"veride bu modelden {int(n_model[i])} ilan"); en.append(f"{int(n_model[i])} listing(s) of this model in the data")
    if n_model_year[i] == 1:
        tr.append("aynı model+yılda başka ilan yok"); en.append("no other listing of the same model+year")
    seg = listings["segment"].iat[i]
    if seg in ("F", "S"):
        tr.append(f"segment {seg}"); en.append(f"segment {seg}")
    if age[i] >= OLD_AGE:
        tr.append(f"yaş {int(age[i])}"); en.append(f"age {int(age[i])}")
    if km[i] >= 300_000:
        tr.append("km ≥ 300 bin"); en.append("km ≥ 300k")
    if heavy[i]:
        tr.append("formda ağır hasar kaydı"); en.append("heavy-damage record on the form")
    return tr, en


def examples(listings, price, pred, resid, age, flags, n_model, n_model_year):
    """
    EN: The three example listings (EXAMPLE_KEYS); stops if a key does not match exactly one listing, so the
        report's text never goes stale silently. No ad_id or listing text is written.
    TR: Üç örnek ilan (EXAMPLE_KEYS); bir anahtar tam bir ilana eşleşmezse durur, böylece rapor metni sessizce
        bayatlamaz. ad_id ya da ilan metni yazılmaz.
    """
    year = pd.to_numeric(listings["gb_year"]).values
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    out = []
    for model, yr, pr in EXAMPLE_KEYS:
        idx = np.where((listings["model"].values == model) & (year == yr) & (price == pr))[0]
        if len(idx) != 1:
            raise SystemExit(f"EXAMPLE NOT FOUND / NOT UNIQUE | ÖRNEK BULUNAMADI/BİRDEN FAZLA: {(model, yr, pr)} → "
                             f"{len(idx)} rows; review EXAMPLE_KEYS and the report text")
        i = int(idx[0])
        tr, en = example_reasons(listings, i, age, flags, n_model, n_model_year)
        series, brand = listings["series"].iat[i], str(listings["brand"].iat[i])
        name = f"{BRAND.get(brand.lower(), brand)} {model}"
        if "Serisi" not in series and series not in model:
            name += f" ({series})"
        out.append({"key": [model, yr, pr], "name": name, "year": yr, "km": None if np.isnan(km[i]) else int(km[i]),
                    "price": float(pr), "pred": round(float(pred[i]), 0), "resid_pct": round(float(resid[i]), 1),
                    "n_model": int(n_model[i]), "n_model_year": int(n_model_year[i]), "reasons_tr": tr, "reasons_en": en})
    return out


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the report inputs (error_drivers).
    TR: Rapor girdilerinde yayımlanır (error_drivers).
    """
    return {"error_drivers": {
        "threshold_pct": LARGE_ERROR_PCT, "overall": res["overall"], "by_model_year_n": res["by_comparables"],
        "per_model_error": res["per_model"], "per_model_buckets": res["buckets"],
        "by_age": res["by_age"], "by_segment_FS": res["by_segment"], "by_snapshot": res["by_snapshot"],
        "kaybolan_canli": res["live"], "tl_olcekli": res["lira"], "yas_duyarlilik": res["age_each"],
        "yas_kesim": res["age_cuts"], "eski_d_grubu": res["old_d"], "spec_outliers": res["spec"],
        "examples": res["examples"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
pred = np.clip(oof["lgb"].values, 0, PRICE_CAP)
resid = residual_pct(price, oof["lgb"].values)
over, under = large_errors(resid)
big = over | under
age = pd.to_numeric(listings["vehicle_age"], errors="coerce").values.astype(float)
n_model = listings.groupby("model")["model"].transform("size").values
n_model_year = listings.groupby(["model", "gb_year"])["model"].transform("size").values
path = np.array([SR.resolve(s, m)[1] for s, m in zip(listings["series"], listings["model"])])
desc = TF.descriptions(listings)
flags = {"hp": TF.hp_conflict(desc, listings), "model": TF.model_conflict(desc, listings),
         "conversion": TF.conversion_flag(desc), "modification": TF.modification_flag(desc)}
per_model, buckets = error_per_model(listings["model"].values, resid)
snaps, live = by_snapshot(listings["snap"], resid)
age_each, age_cuts = by_age(age, big)
lira, lira_order, ape_top = lira_ranking(listings, price, pred, resid, path, flags["conversion"] | flags["modification"])
spec, off = spec_outliers(listings, resid, big)
spec["en_kotu_6_icinde"] = int(off[ape_top[:6]].sum())
spec["en_kotu_6_emsal"] = [[str(listings["model"].iat[i])[:35], int(n_model[i])] for i in ape_top[:6]]
fs = np.isin(listings["segment"].values, ["F", "S"])
res = {"overall": {"n": len(listings), "n_big": int(big.sum()), "n_over": int(over.sum()), "n_under": int(under.sum()),
                   "big_pct": round(100 * float(big.mean()), 2)},
       "by_comparables": by_comparables(n_model_year, big, over, under), "per_model": per_model, "buckets": buckets,
       "by_age": {"age_18plus": rate(age >= OLD_AGE, big), "age_under18": rate(age < OLD_AGE, big)},
       "by_segment": {"F_or_S": rate(fs, big), "other": rate(~fs, big)}, "by_snapshot": snaps, "live": live,
       "lira": lira, "age_each": age_each, "age_cuts": age_cuts,
       "old_d": old_d_group(listings["series"], pred - price, resid, lira_order), "spec": spec,
       "examples": examples(listings, price, pred, resid, age, flags, n_model, n_model_year)}
print(f"large errors | büyük hata {res['overall']['n_big']} ({res['overall']['n_over']} over / {res['overall']['n_under']} under)")

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("08_large_errors", to_metrics(res), run_id=oof_info["run_id"]))
