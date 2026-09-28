"""
08_large_errors.py
EN: Technical report §8 — where the large errors (|OOF residual| > 20%) come from. Rates by the number of
    comparables (same model + year), by model, by age (every age and every cut point — 18 is not a hard
    break), by segment F/S, by snapshot and for listings still live vs gone; where the 100 largest lira errors
    sit (under/over, top price quartile, performance families); engine specs that break away from
    their own model's median (a catalogue mismatch prices an engine the car does not have); and three example
    listings with the comparisons their report notes rest on, computed on every run (EXAMPLE_PEERS). Needs
    07_model_comparison's OOF artefact.
TR: Teknik rapor §8 — büyük hatalar (|OOF artık| > %20) nereden geliyor. Emsal sayısına (aynı model + yıl),
    modele, yaşa (her yaş ve her kesim noktası — 18 sert bir kırılma değil), F/S segmentine, taramaya ve hâlâ
    yayında olan/kalkan ilana göre oranlar; en büyük 100 lira hatasının nerede toplandığı (düşük/fazla, en
    pahalı çeyrek, performans aileleri); kendi modelinin medyanından kopan motor değerleri (katalog uyuşmazlığı
    aracın sahip olmadığı bir motoru fiyatlar); ve rapor notlarının dayandığı karşılaştırmalarıyla üç örnek ilan,
    her koşuda hesaplanır (EXAMPLE_PEERS). 07_model_comparison'ın OOF artefaktına ihtiyaç duyar.
Output / Çıktı: metrics/08_large_errors.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd

from lib import segment_rule as SR
from lib import spec_rule as SPEC
from lib.common import load_clean, save_metrics
from lib.cv import LARGE_ERROR_PCT, PRICE_CAP, large_errors, load_oof, residual_pct

# EN: the owner's choice (2026-09-17): a modified car, an extreme car, an old luxury car with no comparables.
#     (model, model year, price); each must match exactly one listing, otherwise the run stops.
# TR: kullanıcının seçimi (2026-09-17): modifiye · uç araba · emsalsiz yaşlı lüks. (model, model yılı, fiyat);
#     her biri tam bir ilana eşleşmeli, yoksa koşum durur.
EXAMPLE_KEYS = [("640i", 2011, 5_600_000), ("4.2 FSI Quattro R-tronic", 2008, 4_690_000), ("750i Long", 2007, 1_190_000)]
# EN: 2026-09-27 (nothing from the archive; the notes' numbers were hand-typed): the group each report note compares
#     its example with — an exact model name, or a model-name prefix in the example's own model year. Its size and
#     median are computed on every run; an empty group stops the run.
# TR: 2026-09-27 (arşivden hiçbir şey; notların sayıları elle yazılmıştı): her rapor notunun örneğini karşılaştırdığı
#     grup — tam bir model adı ya da örneğin kendi model yılında bir model adı öneki. Boyutu ve medyanı her koşuda
#     hesaplanır; boş grup koşumu durdurur.
EXAMPLE_PEERS = {("4.2 FSI Quattro R-tronic", 2008, 4_690_000): {"model": "S5 4.2 FSI Quattro"},
                 ("750i Long", 2007, 1_190_000): {"model_prefix": "730d", "same_year": True}}
BRAND = {"bmw": "BMW", "audi": "Audi"}
COUNT_BINS = [("1", 1, 1), ("2–4", 2, 4), ("5–19", 5, 19), ("20–99", 20, 99), ("100+", 100, 10**9)]
OLD_AGE, FIRST_CUT = 18, 8
TOP_N_LIRA, N_NULL, SEED = 100, 200, 42
SPEC_MIN_GROUP, SPEC_RATIO = SPEC.MIN_GROUP, SPEC.RATIO


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
    return rows, {"last_snapshot": str(snap.max()), "live_n": int(live.sum()),
                  "live_median_resid": round(float(np.median(resid[live])), 2), "gone_n": int((~live).sum()),
                  "gone_median_resid": round(float(np.median(resid[~live])), 2)}


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


def lira_ranking(listings, price, pred, path):
    """
    EN: Where the largest lira errors (|prediction − actual|) sit: of the top 100, how many are under- or
        over-predicted, in the top predicted-price quartile (2026-09-28: the predicted price, as every price-level
        grouping in §8; by the actual price an under-prediction is pulled into the top quartile by construction) or
        in a performance family (segment from the model name).
    TR: En büyük lira hataları (|tahmin − gerçek|) nerede: ilk 100'ün kaçı düşük ya da fazla tahmin, en pahalı
        tahmin fiyatı çeyreğinde (2026-09-28: §8'deki her fiyat düzeyi gruplaması gibi tahmin fiyatı; gerçek fiyata
        göre düşük tahmin yapısı gereği en pahalı çeyreğe çekilir) ya da bir performans ailesinde (segmenti model
        adından).
    """
    dev = pred - price
    top = np.argsort(-np.abs(dev), kind="stable")[:TOP_N_LIRA]
    q75 = float(np.quantile(pred, .75))
    perf = path != SR.PATH_SERIES_MAP
    return {"top_n": TOP_N_LIRA, "top_n_under": int((dev[top] < 0).sum()), "top_n_over": int((dev[top] > 0).sum()),
            "top_n_q4": int((pred[top] > q75).sum()), "top_n_perf": int(perf[top].sum()),
            "perf_overall_pct": round(100 * float(perf.mean()), 2),
            "perf_series": sorted({str(x) for x in listings.loc[perf, "series"]})}


def under_if_symmetric(price, pred, n_draws=N_NULL, seed=SEED):
    """
    EN: How many of the top-N lira errors would be under-predictions if the actual price scattered symmetrically
        around the prediction on the log scale: each listing keeps its |log error| around its own prediction with a
        random sign. The same % error is bigger in lira when the actual price is above the prediction, so this is well
        above N/2. Returns: {"mean", "lo", "hi", "draws"} (2.5–97.5 percentiles).
    TR: Gerçek fiyat tahminin etrafında log ölçekte simetrik dağılsaydı en büyük N lira hatasının kaçı düşük tahmin
        olurdu: her ilan |log hatasını| kendi tahmini etrafında rastgele işaretle taşır. Aynı % hata, gerçek fiyat
        tahminin üstündeyken liraca daha büyüktür; bu yüzden sonuç N/2'nin belirgin üstünde. Döndürür: {"mean", "lo",
        "hi", "draws"} (%2.5–97.5 yüzdelikleri).
    """
    r = np.abs(np.log1p(pred) - np.log1p(price))
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_draws):
        dev = pred - np.expm1(np.log1p(pred) + rng.choice([-1.0, 1.0], size=len(r)) * r)
        top = np.argsort(-np.abs(dev), kind="stable")[:TOP_N_LIRA]
        counts.append(int((dev[top] < 0).sum()))
    lo, hi = np.percentile(counts, [2.5, 97.5])
    return {"mean": round(float(np.mean(counts)), 1), "lo": int(round(lo)), "hi": int(round(hi)), "draws": n_draws}


def spec_outliers(listings, resid):
    """
    EN: Listings whose power or size is more than 1.5× off the median of their own model (models with ≥5
        listings); their median error against the rest. Models with <5 listings cannot be flagged at all — that
        blind spot is counted.
    TR: Gücü ya da hacmi kendi modelinin medyanından 1.5 kattan fazla sapan ilanlar (≥5 ilanlı modeller);
        medyan hataları geri kalanla. <5 ilanlı modellerde bayrak hiç kalkamaz — bu kör nokta sayılır.
    """
    off, blind = SPEC.spec_outlier_mask(listings, SPEC_MIN_GROUP, SPEC_RATIO)
    err = np.abs(resid)
    return {"threshold": SPEC_RATIO, "min_group": SPEC_MIN_GROUP,
            "blind_spot": {"listings": int(blind.sum()), "pct": round(100 * float(blind.mean()), 2),
                           "model": int(listings.loc[blind, "model"].nunique())},
            "n": int(off.sum()), "pct": round(100 * float(off.mean()), 2),
            "median_error_pct": round(float(np.median(err[off])), 1),
            "other_median_error_pct": round(float(np.median(err[~off])), 1)}


def example_compare(listings, price, i, peer):
    """
    EN: What an example's report note compares it with, computed on this run: how many listings share its
        series, the other listings of its model name ([year, price]) and, when the note names one (EXAMPLE_PEERS),
        the size and median price of the comparison group. Stops if a named group is empty.
    TR: Bir örneğin rapor notunun onu neyle karşılaştırdığı, bu koşuda hesaplanır: serisini kaç ilan paylaşıyor,
        model adının öteki ilanları ([yıl, fiyat]) ve not bir grup anıyorsa (EXAMPLE_PEERS) grubun boyutu ve medyan
        fiyatı. Adı geçen grup boşsa durur.
    """
    model = listings["model"].astype(str).values
    year = pd.to_numeric(listings["gb_year"]).values
    same_model = np.where(model == model[i])[0]
    out = {"series_n": int((listings["series"].values == listings["series"].iat[i]).sum()),
           "others": sorted([int(year[j]), float(price[j])] for j in same_model if j != i), "peer": None}
    if peer:
        names = listings["model"].astype(str)
        g = (names == peer["model"]).values if "model" in peer else names.str.startswith(peer["model_prefix"]).values
        if peer.get("same_year"):
            g = g & (year == year[i])
        if not g.any():
            raise SystemExit(f"EXAMPLE PEER GROUP EMPTY | ÖRNEK KARŞILAŞTIRMA GRUBU BOŞ: {peer}")
        out["peer"] = {**peer, "n": int(g.sum()), "median": float(np.median(price[g]))}
    return out


def examples(listings, price, pred, resid, n_model, n_model_year):
    """
    EN: The three example listings (EXAMPLE_KEYS) with the comparisons their notes rest on (example_compare);
        stops if a key does not match exactly one listing, so the report's text never goes stale silently.
        No ad_id or listing text is written.
    TR: Üç örnek ilan (EXAMPLE_KEYS) ve notlarının dayandığı karşılaştırmalar (example_compare); bir anahtar tam
        bir ilana eşleşmezse durur, böylece rapor metni sessizce bayatlamaz. ad_id ya da ilan metni yazılmaz.
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
        series, brand = listings["series"].iat[i], str(listings["brand"].iat[i])
        name = f"{BRAND.get(brand.lower(), brand)} {model}"
        if "Serisi" not in series and series not in model:
            name += f" ({series})"
        out.append({"key": [model, yr, pr], "name": name, "year": yr, "km": None if np.isnan(km[i]) else int(km[i]),
                    "price": float(pr), "pred": round(float(pred[i]), 0), "resid_pct": round(float(resid[i]), 1),
                    "n_model": int(n_model[i]), "n_model_year": int(n_model_year[i]),
                    "compare": example_compare(listings, price, i, EXAMPLE_PEERS.get((model, yr, pr)))})
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
        "live_vs_gone": res["live"], "lira_scaled": res["lira"], "age_sensitivity": res["age_each"],
        "age_cuts": res["age_cuts"], "spec_outliers": res["spec"],
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
per_model, buckets = error_per_model(listings["model"].values, resid)
snaps, live = by_snapshot(listings["snap"], resid)
age_each, age_cuts = by_age(age, big)
lira = {**lira_ranking(listings, price, pred, path), "under_if_symmetric": under_if_symmetric(price, pred)}
spec = spec_outliers(listings, resid)
fs =np.isin(listings["segment"].values, ["F", "S"])
res = {"overall": {"n": len(listings), "n_big": int(big.sum()), "n_over": int(over.sum()), "n_under": int(under.sum()),
                   "big_pct": round(100 * float(big.mean()), 2)},
       "by_comparables": by_comparables(n_model_year, big, over, under), "per_model": per_model, "buckets": buckets,
       "by_age": {"age_18plus": rate(age >= OLD_AGE, big), "age_under18": rate(age < OLD_AGE, big)},
       "by_segment": {"F_or_S": rate(fs, big), "other": rate(~fs, big)}, "by_snapshot": snaps, "live": live,
       "lira": lira, "age_each": age_each, "age_cuts": age_cuts,
       "spec": spec, "examples": examples(listings, price, pred, resid, n_model, n_model_year)}
print(f"large errors | büyük hata {res['overall']['n_big']} ({res['overall']['n_over']} over / {res['overall']['n_under']} under)")

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("08_large_errors", to_metrics(res), run_id=oof_info["run_id"]))
