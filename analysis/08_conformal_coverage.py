"""
08_conformal_coverage.py
EN: Technical report §8 — the 90% conformal interval and where it holds, and the hit probability. Coverage is
    measured cross-calibrated: each of the model's 5 folds is calibrated only from the other folds' OOF log errors
    (finite-sample quantile), so no listing's own error calibrates its interval. Three arms on the same folds:
      - one q for the whole market (the interval served until 2026-11) and one q per predicted-price quartile band
        (Mondrian); these two are pre-registered (plans/08-mondrian-coverage). Bands are cut on the predicted price —
        what a pricing tool knows;
      - the scale model (lib.conformal, decision note 2026-11-23, chosen after results were seen, no pre-registration):
        half-width k·s(x). For a scored fold, s comes from the other four folds and k from scores nested inside those
        four (each of them scored by an s trained on the remaining three), so neither sees the scored fold.
    Reported for the scale arm: coverage per band, per comparables bucket, on the axes s takes as inputs and on axes
    it does not see; width distribution and interval score of every arm; a table per kind of car; s against the
    comparables count; coverage on the latest snapshot's listings; and what coverage would be if prices were 2 / 5 /
    10% higher (a what-if on today's data, level only — not a time analysis).
    Hit probability: P(|log price − log prediction| ≤ log 1.1) from the same nested scores; its reliability (predicted
    against observed by decile), Brier score against a constant, and the observed share per value shown on the site
    ("N in 10", rounded down, at most 9).
    Also: the band bounds, and the single q (cv.conformal_q) 07_final_model still ships for the old backend.
    Needs 07_model_comparison's OOF artefact. Heavy: it trains 26 small LightGBM models (lib/records.py HEAVY).
TR: Teknik rapor §8 — %90 conformal aralık, nerede tuttuğu ve isabet olasılığı. Kapsama çapraz kalibreli ölçülür:
    modelin 5 katının her biri yalnız öteki katların OOF log hatalarıyla kalibre edilir (sonlu örneklem yüzdeliği);
    hiçbir ilanın kendi hatası kendi aralığını kalibre etmez. Aynı katlarda üç kol:
      - bütün piyasa için tek q (2026-11'e kadar servis edilen aralık) ve tahmin edilen fiyatın her çeyrek bandı için
        ayrı q (Mondrian); bu ikisi ön kayıtlı (plans/08-mondrian-coverage). Bantlar tahmin edilen fiyattan kesilir —
        bir fiyatlama aracının bildiği o;
      - ölçek modeli (lib.conformal, karar notu 2026-11-23, sonuçlar görüldükten sonra seçildi, ön kayıt yok): yarı
        genişlik k·s(x). Puanlanan kat için s öteki dört kattan, k o dört katın içinde iç içe kurulan skorlardan
        gelir (her biri kalan üç katla eğitilmiş bir s ile puanlanır); ikisi de puanlanan katı görmez.
    Ölçek kolu için raporlanan: bant başına, emsal kovası başına, s'nin girdisi olan ve görmediği eksenlerde kapsama;
    her kolun genişlik dağılımı ve aralık skoru; araç türü tablosu; emsal sayısına göre s; son taramanın ilanlarında
    kapsama; ve fiyatlar %2 / %5 / %10 yüksek olsaydı kapsama (bugünkü veride bir varsayım hesabı, yalnız seviye —
    zaman analizi değil).
    İsabet olasılığı: aynı iç içe skorlardan P(|log fiyat − log tahmin| ≤ log 1,1); güvenilirliği (onda birlik
    dilimlerde öngörülen ve gerçekleşen), sabit skora karşı Brier skoru ve sitede gösterilen her değer için
    gerçekleşen pay ("10'da N", aşağı yuvarlanır, en çok 9).
    Ayrıca: bant sınırları ve 07_final_model'in eski backend için hâlâ gönderdiği tek q (cv.conformal_q).
    07_model_comparison'ın OOF artefaktına ihtiyaç duyar. Ağır: 26 küçük LightGBM eğitir (lib/records.py HEAVY).
Output / Çıktı: metrics/08_conformal_coverage.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd

from lib import conformal as CF
from lib.common import FEATURES, load_clean, save_metrics
from lib.conformal import LEVEL, band_of, conformal_quantile, inside, interval
from lib.cv import PRICE_CAP, conformal_q, load_oof

QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
# EN: H1's acceptance band (%) as pre-registered in plans/08-mondrian-coverage (locked 2026-09-28)
# TR: H1'in kabul aralığı (%), plans/08-mondrian-coverage'daki ön kayıttaki gibi (2026-09-28'de kilitlendi)
ACCEPT_BAND = [88, 92]
MIN_GROUP = 200                                   # smallest group whose coverage is read | kapsaması okunan en küçük grup
COMPARABLE_BUCKETS = [("0", 0, 0), ("1_4", 1, 4), ("5_19", 5, 19), ("20_plus", 20, None)]
PD_GRID = [0, 1, 2, 5, 10, 20, 50, 100, 200]       # comparables values of the partial-dependence curve
WIDTH_PCTS = [5, 25, 50, 75, 95, 99]
LEVEL_SHIFTS = [2, 5, 10]                          # what-if price level shifts, % | varsayım fiyat seviyesi kaymaları, %


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def cross_conformal(abs_err, fold, band, level=LEVEL):
    """
    EN: Cross-calibrated q per listing: for each fold, the global q from the other folds' errors and the band q
        from the other folds' errors in the same band. Returns: (q_global, q_band), one value per listing.
    TR: İlan başına çapraz kalibreli q: her kat için öteki katların hatalarından genel q ve öteki katların aynı
        banttaki hatalarından bant q'su. Döndürür: (q_genel, q_bant), ilan başına bir değer.
    """
    q_global, q_band = np.full(len(abs_err), np.nan), np.full(len(abs_err), np.nan)
    for k in np.unique(fold):
        cal, scored = fold != k, fold == k
        q_global[scored] = conformal_quantile(abs_err[cal], level)
        for b in np.unique(band):
            q_band[scored & (band == b)] = conformal_quantile(abs_err[cal & (band == b)], level)
    assert not (np.isnan(q_global).any() or np.isnan(q_band).any()), "a listing got no q | q almayan ilan var"
    return q_global, q_band


def cross_scale(X, abs_err, log_pred, comparables, fold, s, level=LEVEL):
    """
    EN: The scale arm, cross-calibrated. s: each listing's s from the other four folds (CF.nested_scores). For a
        scored fold, k and the hit-probability scores come from the other four folds only: each of them is scored by
        an s trained on the remaining three. Returns: (half-width r = k·s, hit probability), one value per listing.
    TR: Ölçek kolu, çapraz kalibreli. s: her ilanın öteki dört kattan gelen s'si (CF.nested_scores). Puanlanan kat
        için k ve isabet olasılığının skorları yalnız öteki dört kattan gelir: her biri kalan üç katla eğitilmiş bir
        s ile puanlanır. Döndürür: (yarı genişlik r = k·s, isabet olasılığı), ilan başına bir değer.
    """
    idx, r, p = np.arange(len(abs_err)), np.full(len(abs_err), np.nan), np.full(len(abs_err), np.nan)
    for k in np.unique(fold):
        cal, scored = fold != k, fold == k
        inner = []
        for j in np.unique(fold[cal]):
            Xtr, Xva, catc = CF.scale_matrix(X, idx[cal & (fold != j)], idx[fold == j], log_pred, comparables)
            inner.append(abs_err[fold == j] / CF.predict_scale(CF.fit_scale(Xtr, abs_err[cal & (fold != j)], catc), Xva))
        inner = np.sort(np.concatenate(inner))
        r[scored] = conformal_quantile(inner, level) * s[scored]
        p[scored] = CF.p_within(inner, s[scored])
    assert not (np.isnan(r).any() or np.isnan(p).any()), "a listing got no interval | aralık almayan ilan var"
    return r, p


def pct(mask):
    """EN: Share of True in %, 2 decimals. / TR: True payı %, 2 hane."""
    return round(float(np.mean(mask) * 100), 2)


def band_rows(price, pred, band, q_global, q_band):
    """
    EN: Per band: listings, coverage % of both arms, median interval width as % of the prediction for both arms.
        Returns: [[band, n, cov_global, cov_band, width_pct_global, width_pct_band], ...].
    TR: Bant başına: ilan, iki kolun kapsama %'si, iki kolun medyan aralık genişliği tahminin %'si olarak.
        Döndürür: [[bant, n, kapsama_genel, kapsama_bant, genişlik_%_genel, genişlik_%_bant], ...].
    """
    rows = []
    for b, name in enumerate(QUARTILES):
        m = band == b
        out = [name, int(m.sum())]
        cov, wid_pct = [], []
        for q in (q_global, q_band):
            lo, hi = interval(pred[m], q[m])
            cov.append(round(float(((price[m] >= lo) & (price[m] <= hi)).mean() * 100), 2))
            wid_pct.append(round(float(np.median((hi - lo) / pred[m] * 100)), 1))
        rows.append(out + cov + wid_pct)
    return rows


def arm_rows(price, pred, arms, no_comparable):
    """
    EN: One row per arm: overall coverage %, median and mean width (% of the prediction), interval score (log
        space), coverage on the listings without a comparable. arms: {name: half-width per listing}.
        Returns: [[arm, coverage, width_median, width_mean, interval_score, coverage_no_comparable], ...].
    TR: Kol başına bir satır: genel kapsama %, medyan ve ortalama genişlik (tahminin %'si), aralık skoru (log uzayı),
        emsalsiz ilanlarda kapsama. arms: {ad: ilan başına yarı genişlik}.
        Döndürür: [[kol, kapsama, genişlik_medyan, genişlik_ortalama, aralık_skoru, emsalsiz_kapsama], ...].
    """
    rows = []
    for name, r in arms.items():
        ins, w = inside(price, pred, r), CF.width_pct(pred, r)
        rows.append([name, pct(ins), round(float(np.median(w)), 1), round(float(w.mean()), 1),
                     round(CF.interval_score(price, pred, r), 4), pct(ins[no_comparable])])
    return rows


def input_axes(df):
    """
    EN: Groupings on what s takes as inputs: age, mileage and power fifths, segment, fuel, transmission, body type,
        heavy damage, changed parts, how common the series is. Returns: {axis: group label per listing}.
    TR: s'nin girdisi olan şeylere göre gruplar: yaş, km ve beygir beşte birleri, segment, yakıt, vites, kasa, ağır
        hasar, değişen parça, serinin ne kadar yaygın olduğu. Döndürür: {eksen: ilan başına grup etiketi}.
    """
    fifth = lambda s: pd.qcut(s.fillna(-1).rank(method="first"), 5, labels=False).astype(str).values  # noqa: E731
    series_n = df["series"].map(df["series"].value_counts())
    return {"age": pd.cut(df["vehicle_age"], [-1, 3, 7, 12, 18, 25, 99]).astype(str).values,
            "mileage": fifth(df["gb_mileage"]), "power": fifth(df["power_hp_val"]),
            "segment": df["segment"].astype(str).values, "fuel": df["kb_fuel"].astype(str).values,
            "transmission": df["kb_transmission"].astype(str).values, "body_type": df["kb_body_type"].astype(str).values,
            "heavy_damage": df["is_heavy_damaged"].astype(str).values,
            "changed_parts": pd.cut(df["count_changed"].fillna(0), [-1, 0, 1, 3, 99]).astype(str).values,
            "series_frequency": pd.cut(series_n, [0, 20, 100, 500, 1e9]).astype(str).values}


def unseen_axes(df):
    """
    EN: Groupings on what s does not see: seller type, city (the 8 largest, the rest together), colour, warranty,
        first owner, trade-in, damage record amount. Returns: {axis: group label per listing}.
    TR: s'nin görmediği şeylere göre gruplar: satıcı tipi, il (en büyük 8, kalanı birlikte), renk, garanti, ilk sahip,
        takas, hasar kaydı tutarı. Döndürür: {eksen: ilan başına grup etiketi}.
    """
    city = df["location"].astype(str).str.split(",").str[-1].str.strip()
    return {"seller_type": df["kb_seller_type"].astype(str).values,
            "city": np.where(city.isin(city.value_counts().index[:8]), city, "rest"),
            "colour": df["kb_color"].astype(str).values, "warranty": df["gb_warranty_status"].astype(str).values,
            "first_owner": df["gb_is_first_owner"].astype(str).values,
            "trade_in": df["kb_trade_available"].astype(str).values,
            "damage_record": pd.cut(df["tramer_fee"].fillna(-1), [-2, -0.5, 0, 20000, 100000, 1e12]).astype(str).values}


def group_coverages(ins, axes, min_n=MIN_GROUP):
    """
    EN: Coverage of every group with at least min_n listings. Returns: [[axis, group, n, coverage %], ...].
    TR: En az min_n ilanlı her grubun kapsaması. Döndürür: [[eksen, grup, n, kapsama %], ...].
    """
    out = []
    for axis, g in axes.items():
        for v in pd.unique(g):
            m = g == v
            if m.sum() >= min_n:
                out.append([axis, str(v), int(m.sum()), float(ins[m].mean() * 100)])
    return out


def axis_summary(ins, in_axes, out_axes, target):
    """
    EN: How far groups sit from the target. Input axes: number of groups, how many are 5 points or more away from
        the target, the largest distance. Unseen axes: number of groups, how many are 3 points or more below the
        arm's own overall coverage, and the worst group. Returns: {"input": {...}, "unseen": {...}}.
    TR: Grupların hedeften uzaklığı. Girdi eksenleri: grup sayısı, hedeften 5 puan ya da daha fazla sapanların
        sayısı, en büyük sapma. Görülmeyen eksenler: grup sayısı, kolun kendi genel kapsamasının 3 puan ya da daha
        fazla altında olanların sayısı ve en kötü grup. Döndürür: {"input": {...}, "unseen": {...}}.
    """
    gi, go, own = group_coverages(ins, in_axes), group_coverages(ins, out_axes), float(ins.mean() * 100)
    dev = np.array([abs(g[3] - target) for g in gi])
    worst = min(go, key=lambda g: g[3])
    return {"input": {"n_groups": len(gi), "off_5pp": int((dev >= 5).sum()), "max_deviation": round(float(dev.max()), 1)},
            "unseen": {"n_groups": len(go), "below_own_3pp": int(sum(g[3] - own <= -3 for g in go)),
                       "worst": [worst[0], worst[1], worst[2], round(worst[3] - own, 1)]}}


def profile_rows(groups, price, pred, q_global, r_scale, p_hit, hit):
    """
    EN: The table per kind of car. For each group: listings, coverage with one q and with the scale model, the scale
        margin in % (from the group's median half-width), the interval around the group's median prediction today and
        with the scale model (₺), hit probability predicted and observed (%).
        Returns: [[group, n, cov_global, cov_scale, margin_low_pct, margin_high_pct, today_low, today_high, new_low,
        new_high, hit_predicted, hit_observed], ...].
    TR: Araç türü tablosu. Her grup için: ilan, tek q ve ölçek modeliyle kapsama, ölçek payı % (grubun medyan yarı
        genişliğinden), grubun medyan tahmini etrafında bugünkü ve ölçek modeliyle aralık (₺), öngörülen ve
        gerçekleşen isabet olasılığı (%).
        Döndürür: [[grup, n, kapsama_genel, kapsama_ölçek, pay_alt_%, pay_üst_%, bugün_alt, bugün_üst, yeni_alt,
        yeni_üst, isabet_öngörülen, isabet_gerçekleşen], ...].
    """
    in_g, in_s, rows = inside(price, pred, q_global), inside(price, pred, r_scale), []
    for name, m in groups.items():
        pm, rm, qm = float(np.median(pred[m])), float(np.median(r_scale[m])), float(np.median(q_global[m]))
        rows.append([name, int(m.sum()), round(float(in_g[m].mean() * 100), 1), round(float(in_s[m].mean() * 100), 1),
                     round(float((1 - np.exp(-rm)) * 100), 1), round(float((np.exp(rm) - 1) * 100), 1),
                     *(int(round(float(pm * np.exp(x)), -3)) for x in (-qm, qm, -rm, rm)),
                     round(float(p_hit[m].mean() * 100), 1), round(float(hit[m].mean() * 100), 1)])
    return rows


def profiles(df, band, comparables):
    """
    EN: The groups of the per-car table, by id. Returns: {id: boolean mask}.
    TR: Araç türü tablosunun grupları, kimlikleriyle. Döndürür: {kimlik: mantıksal maske}.
    """
    age = df["vehicle_age"].values
    top_km = pd.qcut(df["gb_mileage"].rank(method="first"), 5, labels=False).values == 4
    return {"all": np.ones(len(df), bool), "age_0_3": age <= 3, "age_4_7": (age >= 4) & (age <= 7),
            "age_8_12": (age >= 8) & (age <= 12), "age_13_18": (age >= 13) & (age <= 18), "age_19_plus": age >= 19,
            "no_comparable": comparables == 0, "comparables_20_plus": comparables >= 20,
            "cheapest_quarter": band == 0, "dearest_quarter": band == len(QUARTILES) - 1, "top_mileage_fifth": top_km,
            "heavy_damaged": df["is_heavy_damaged"].astype(int).values == 1,
            "segment_f": df["segment"].astype(str).values == "F"}


def level_shift_rows(price, pred, q_global, r_scale, shifts=LEVEL_SHIFTS):
    """
    EN: A what-if on today's data: coverage if every listing price were x% higher, overall and on the fifth of
        listings with the narrowest scale interval, for one q and the scale model; and the observed ±WITHIN hit share.
        Returns: [[shift %, cov_global, cov_scale, narrow_fifth_global, narrow_fifth_scale, hit_observed], ...],
        the first row being no shift.
    TR: Bugünkü veride bir varsayım hesabı: her ilan fiyatı %x yüksek olsaydı kapsama; genel ve ölçek aralığı en dar
        beşte bir ilanda, tek q ve ölçek modeli için; ayrıca gerçekleşen ±WITHIN isabet payı.
        Döndürür: [[kayma %, kapsama_genel, kapsama_ölçek, dar_beşte_bir_genel, dar_beşte_bir_ölçek,
        isabet_gerçekleşen], ...], ilk satır kaymasız.
    """
    narrow = pd.qcut(pd.Series(r_scale).rank(method="first"), 5, labels=False).values == 0
    rows = []
    for x in [0] + list(shifts):
        moved = price * (1 + x / 100)
        in_g, in_s = inside(moved, pred, q_global), inside(moved, pred, r_scale)
        hit = np.abs(np.log1p(moved) - np.log1p(pred)) <= np.log1p(CF.WITHIN)
        rows.append([x, pct(in_g), pct(in_s), pct(in_g[narrow]), pct(in_s[narrow]), pct(hit)])
    return rows


def partial_dependence(X, abs_err, log_pred, comparables, grid=PD_GRID):
    """
    EN: Mean s over all listings when the comparables count is set to each grid value, from an s trained on all
        listings (as the served one). Returns: [[comparables, mean s], ...].
    TR: Emsal sayısı ızgaradaki her değere konduğunda bütün ilanlardaki ortalama s; bütün ilanlarla eğitilmiş bir
        s'den (servis edilen gibi). Döndürür: [[emsal, ortalama s], ...].
    """
    idx = np.arange(len(abs_err))
    Xall, _, catc = CF.scale_matrix(X, idx, idx, log_pred, comparables)
    model = CF.fit_scale(Xall, abs_err, catc)
    return [[g, round(float(model.predict(Xall.assign(**{CF.COMPARABLES: g})).mean()), 4)] for g in grid]


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.conformal: coverage per predicted-price band for the three arms and the
        reliability points of the hit probability — the data of the report's figures); the report-only numbers go to
        report.
    TR: Site ağacında yayımlanır (domain.conformal: üç kol için tahmin fiyatı bandı başına kapsama ve isabet
        olasılığının güvenilirlik noktaları — rapor figürlerinin verisi); yalnız raporun kullandığı sayılar report'a
        gider.
    """
    rows, sc, conf = res["rows"], res["scale"], dict(res["confidence"])
    deciles = conf.pop("deciles")
    return {"domain": {"conformal": {
                "by_quantile": [[r[0], r[2]] for r in rows], "mondrian_by_quantile": [[r[0], r[3]] for r in rows],
                "scale_by_quantile": [[r[0], r[2]] for r in sc["by_band"]],
                "hit_reliability": deciles,
                "coverage_target": round(100 * LEVEL),
                "note": ("Bantlar tahmin edilen fiyatın çeyrekleri; kapsama çapraz kalibreli (her kat öteki katlardan "
                         "kalibre edilir). by_quantile = tek q, mondrian_by_quantile = bant başına q, "
                         "scale_by_quantile = ölçek modeli. hit_reliability = isabet olasılığının onda birlik "
                         "dilimlerinde [öngörülen %, gerçekleşen %].")}},
            "report": {"conformal_q": res["q_served"], "conformal_all": res["all_global"], "q_bounds": res["bounds"],
                       "mondrian": {"by_band": rows, "accept_band": ACCEPT_BAND,
                                    "columns": ["band", "n", "coverage_global", "coverage_band", "width_pct_global",
                                                "width_pct_band"]},
                       "scale": {**sc, "s_floor": CF.S_FLOOR, "min_group_n": MIN_GROUP,
                                 "arm_columns": ["arm", "coverage", "width_median", "width_mean", "interval_score",
                                                 "coverage_no_comparable"],
                                 "profile_columns": ["group", "n", "coverage_global", "coverage_scale",
                                                     "margin_low_pct", "margin_high_pct", "today_low", "today_high",
                                                     "new_low", "new_high", "hit_predicted", "hit_observed"],
                                 "level_shift_columns": ["shift_pct", "coverage_global", "coverage_scale",
                                                         "narrow_fifth_global", "narrow_fifth_scale", "hit_observed"]},
                       "confidence": {**conf, "within_pct": round(CF.WITHIN * 100), "shown_max": CF.SHOWN_MAX}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
oof, oof_info = load_oof(listings)

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
price = listings["price"].values.astype(float)
pred = np.clip(oof["lgb"].values, 0, PRICE_CAP)
fold = oof["fold"].values
log_pred = np.log1p(pred)
abs_err = np.abs(np.log1p(price) - log_pred)
bounds = [float(x) for x in np.quantile(pred, [.25, .5, .75])]
band = band_of(pred, bounds)
q_global, q_band = cross_conformal(abs_err, fold, band)
rows = band_rows(price, pred, band, q_global, q_band)

# EN: scale arm — comparables are counted without the listing's own fold | TR: ölçek kolu — emsal, ilanın kendi katı sayılmadan
X = listings[FEATURES].reset_index(drop=True)
comparables = CF.other_fold_counts(listings["model"].astype(str) + "|" + listings["gb_year"].astype(str), fold)
scores, s = CF.nested_scores(X, abs_err, log_pred, comparables, fold)
r_scale, p_hit = cross_scale(X, abs_err, log_pred, comparables, fold, s)
hit = abs_err <= np.log1p(CF.WITHIN)
in_global, in_scale = inside(price, pred, q_global), inside(price, pred, r_scale)
no_comparable = comparables == 0
target = 100 * LEVEL
axes_in, axes_out = input_axes(listings), unseen_axes(listings)
width = CF.width_pct(pred, r_scale)
latest = (listings["snap"] == listings["snap"].max()).values
scale = {
    "k": round(conformal_quantile(scores), 4), "floor_hits": int((s <= CF.S_FLOOR).sum()),
    "no_comparable_n": int(no_comparable.sum()),
    "arms": arm_rows(price, pred, {"global": q_global, "band": q_band, "scale": r_scale}, no_comparable),
    "by_band": [[name, int((band == b).sum()), pct(in_scale[band == b]), round(float(np.median(width[band == b])), 1)]
                for b, name in enumerate(QUARTILES)],
    "by_comparables": [[name, int(m.sum()), pct(in_global[m]), pct(inside(price, pred, q_band)[m]), pct(in_scale[m])]
                       for name, lo, hi in COMPARABLE_BUCKETS
                       for m in [(comparables >= lo) & (comparables <= (hi if hi is not None else comparables.max()))]],
    "axes": {"global": axis_summary(in_global, axes_in, axes_out, target),
             "scale": axis_summary(in_scale, axes_in, axes_out, target)},
    "width": {"min": round(float(width.min()), 1),
              "percentiles": [[q, round(float(np.quantile(width, q / 100)), 1),
                               round(float((1 - np.exp(-np.quantile(r_scale, q / 100))) * 100), 1),
                               round(float((np.exp(np.quantile(r_scale, q / 100)) - 1) * 100), 1)] for q in WIDTH_PCTS],
              "narrower_share": pct(r_scale < q_global)},
    "partial_dependence": partial_dependence(X, abs_err, log_pred, comparables),
    "profiles": profile_rows(profiles(listings, band, comparables), price, pred, q_global, r_scale, p_hit, hit),
    "latest_snapshot": {"snap": str(listings["snap"].max()), "n": int(latest.sum()),
                        "coverage_global": pct(in_global[latest]), "coverage_scale": pct(in_scale[latest])},
    "level_shift": level_shift_rows(price, pred, q_global, r_scale)}
res = {"rows": rows, "bounds": bounds, "q_served": conformal_q(np.log1p(price), pred, LEVEL),
       "all_global": round(float(in_global.mean() * 100), 2), "scale": scale, "confidence": CF.reliability(p_hit, hit)}
print(f"overall | genel: {res['all_global']}% ·", [(r[0], r[2], r[3]) for r in rows])
print("arms | kollar:", scale["arms"], "· k", scale["k"], "· hit | isabet",
      res["confidence"]["mean_predicted"], res["confidence"]["mean_observed"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
if __name__ == "__main__":
    print("written | yazıldı:", save_metrics("08_conformal_coverage", to_metrics(res), run_id=oof_info["run_id"]))
