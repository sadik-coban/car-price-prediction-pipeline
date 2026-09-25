"""
01_engine_rule.py
EN: Technical report §1 — engine power and size: from a range to one number (the owner's rule).
    On many listings the site gives power/size as a bucket (e.g. 1401–1600 cc). The model uses
    hp = mean(lower, upper) and cc = upper bound. Each candidate (lower / midpoint / upper) is compared
    with the median exact value of the same model; the chosen candidate must have the smallest median
    gap, otherwise the script stops. The (lower, upper) pairs the site gives feed figure 30.
TR: Teknik rapor §1 — motor gücü ve hacmi: aralıktan tek sayıya (kullanıcının kuralı).
    Site gücü/hacmi birçok ilanda kova olarak veriyor (ör. 1401–1600 cc). Model hp = ort(alt, üst),
    cc = üst sınır kullanır. Her aday (alt / orta / üst) aynı modelin kesin değer medyanıyla karşılaştırılır;
    seçilen aday en küçük medyan farkı vermezse betik durur. Sitenin verdiği (alt, üst) çiftleri Figür 30'u besler.
Output / Çıktı: metrics/01_engine_rule.json
"""

# %% [1] Setup | Kurulum
import numpy as np
import pandas as pd

from lib.common import ROOT, load_clean, save_metrics

CHOSEN = {"engine_cc": "ust", "power_hp": "orta"}     # the owner's rule | kullanıcının kuralı
UNITS = {"engine_cc": "cc", "power_hp": "hp"}
MIN_BUCKET_N = 100


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def assert_rule_in_code(common_src):
    """
    EN: Stops if the rule in lib/common.py (the code the model is trained with) no longer matches CHOSEN,
        so this measurement never defends an old rule.
    TR: lib/common.py'deki kural (modelin eğitildiği kod) CHOSEN ile artık uyuşmuyorsa durur; böylece bu ölçü
        eski bir kuralı savunmaz.
    """
    assert 'df["engine_cc_val"] = df["engine_cc_up"]' in common_src, "cc rule changed in common.py"
    assert 'df["power_hp_val"] = df[["power_hp_low", "power_hp_up"]].mean(axis=1)' in common_src, \
        "hp rule changed in common.py"


def bucket_frame(listings, p):
    """
    EN: Splits one measure into exact listings and bucketed listings that have a reference.
        Reference = median exact value of the same model; only buckets with both bounds known.
        p: "engine_cc" or "power_hp".
        Returns: (is_bucket mask, exact rows, bucketed rows with a "ref" column).
    TR: Bir ölçüyü kesin değerli ilanlar ve referansı olan kovalı ilanlar olarak ayırır.
        Referans = aynı modelin kesin değer medyanı; yalnız iki sınırı da bilinen kovalar.
        p: "engine_cc" ya da "power_hp".
        Döndürür: (kova maskesi, kesin satırlar, "ref" kolonlu kovalı satırlar).
    """
    lo, up = listings[f"{p}_low"], listings[f"{p}_up"]
    is_bucket = listings[f"{p}_is_range"].fillna(False).astype(bool)
    exact = listings[~is_bucket & up.notna()]
    ref = exact.groupby("model")[f"{p}_up"].median()
    bucketed = listings[is_bucket & lo.notna() & up.notna()].copy()
    bucketed["ref"] = bucketed["model"].map(ref)
    return is_bucket, exact, bucketed[bucketed["ref"].notna()]


def candidate_gaps(bucketed, p):
    """
    EN: |candidate − reference| for the three candidates. Returns: DataFrame with columns alt/orta/ust.
    TR: Üç aday için |aday − referans|. Döndürür: alt/orta/ust kolonlu DataFrame.
    """
    lo, up = bucketed[f"{p}_low"], bucketed[f"{p}_up"]
    return pd.DataFrame({"alt": (lo - bucketed["ref"]).abs(),
                         "orta": ((lo + up) / 2 - bucketed["ref"]).abs(),
                         "ust": (up - bucketed["ref"]).abs()})


def summarize_gaps(gaps):
    """
    EN: Median, mean and 5/25/50/75/95 percentiles of each candidate's gap (raw, unrounded).
    TR: Her adayın farkının medyanı, ortalaması ve 5/25/50/75/95 yüzdelikleri (ham, yuvarlanmamış).
    """
    return {k: {"medyan": float(np.median(gaps[k])), "ortalama": float(gaps[k].mean()),
                "yuzdelik": [float(q) for q in np.percentile(gaps[k], [5, 25, 50, 75, 95])]} for k in gaps}


def assert_chosen_is_best(summary, chosen, p):
    """
    EN: Stops if the chosen candidate does not have the smallest median gap (the rule contradicts the data).
    TR: Seçilen aday en küçük medyan farkı vermiyorsa durur (kural veriyle çelişiyor).
    """
    medians = {k: round(v["medyan"], 1) for k, v in summary.items()}
    assert all(medians[chosen] < medians[k] for k in medians if k != chosen), \
        f"{p}: chosen candidate '{chosen}' is not the best: {medians}"


def bucket_winners(bucketed, gaps, p, min_n):
    """
    EN: For every bucket with at least min_n listings, the candidate with the smallest median gap
        (ties joined with '+', e.g. 'alt+orta'). Returns: [[low, up, n, winner, {candidate: median}], ...].
    TR: En az min_n ilanlı her kovada medyan farkı en küçük aday (beraberlik '+' ile, ör. 'alt+orta').
        Döndürür: [[alt, üst, n, kazanan, {aday: medyan}], ...].
    """
    g = gaps.assign(lo=bucketed[f"{p}_low"].values, up=bucketed[f"{p}_up"].values)
    out = []
    for (lo, up), grp in g.groupby(["lo", "up"]):
        if len(grp) >= min_n:
            med = {k: float(grp[k].median()) for k in ("alt", "orta", "ust")}
            best = min(med.values())
            out.append([int(lo), int(up), int(len(grp)), "+".join(k for k in med if med[k] == best), med])
    return out


def position_in_bucket(bucketed, p):
    """
    EN: Where the reference sits inside its bucket (0 = lower bound, 1 = upper bound).
        Returns: Series of positions (buckets of zero width excluded).
    TR: Referansın kendi kovasının neresinde durduğu (0 = alt sınır, 1 = üst sınır).
        Döndürür: konum serisi (genişliği sıfır olan kovalar hariç).
    """
    width = bucketed[f"{p}_up"] - bucketed[f"{p}_low"]
    return ((bucketed["ref"] - bucketed[f"{p}_low"]) / width)[width > 0]


def most_common_bucket(bucketed, exact, p):
    """
    EN: The most common bucket, and the most common exact value among the models found in it.
    TR: En sık kova ve o kovada görülen modellerin en sık kesin değeri.
    """
    counts = bucketed.groupby([f"{p}_low", f"{p}_up"]).size().sort_values(ascending=False)
    lo, up = counts.index[0]
    models = set(bucketed.loc[(bucketed[f"{p}_low"] == lo) & (bucketed[f"{p}_up"] == up), "model"])
    values = exact.loc[exact["model"].isin(models), f"{p}_up"].value_counts()
    return {"alt": int(lo), "ust": int(up), "n": int(counts.iloc[0]),
            "en_sik_kesin": int(values.index[0]), "en_sik_kesin_n": int(values.iloc[0])}


def bound_pairs(listings, p):
    """
    EN: Every (lower, upper) pair the site gives for one measure, with its listing count (figure 30). An exact
        value has lower = upper; open-ended and no-value listings have no pair.
        Returns: [[low, up, is_range, n], ...], sorted by (is_range, low, up).
    TR: Sitenin bir ölçü için verdiği her (alt, üst) çifti ve ilan sayısı (Figür 30). Kesin değerde alt = üst;
        açık uçlu ve değersiz ilanların çifti yok.
        Döndürür: [[alt, üst, aralık mı, n], ...], (aralık mı, alt, üst) sırasıyla.
    """
    lo, up = listings[f"{p}_low"], listings[f"{p}_up"]
    both = lo.notna() & up.notna()
    is_range = listings[f"{p}_is_range"].fillna(False).astype(bool)
    counts = pd.DataFrame({"r": is_range[both], "lo": lo[both], "up": up[both]}).groupby(["r", "lo", "up"]).size()
    return [[int(a), int(b), bool(r), int(n)] for (r, a, b), n in counts.items()]


def measure(listings, p, chosen):
    """
    EN: Everything published for one measure (engine_cc or power_hp), raw values.
    TR: Bir ölçü (engine_cc ya da power_hp) için yayımlanan her şey, ham değerler.
    """
    is_bucket, exact, bucketed = bucket_frame(listings, p)
    gaps = candidate_gaps(bucketed, p)
    summary = summarize_gaps(gaps)
    assert_chosen_is_best(summary, chosen, p)
    lo, up = listings[f"{p}_low"], listings[f"{p}_up"]
    inside = (bucketed["ref"] >= bucketed[f"{p}_low"]) & (bucketed["ref"] <= bucketed[f"{p}_up"])
    n_exact = exact.groupby("model")[f"{p}_up"].nunique()
    position = position_in_bucket(bucketed, p)
    return {"chosen": chosen, "listings": int(len(listings)), "exact": int(len(exact)),
            "bucketed": int(is_bucket.sum()),
            "open_ended": int((is_bucket & (lo.isna() ^ up.isna())).sum()),
            "no_value": int((lo.isna() & up.isna()).sum()),
            "with_reference": int(len(bucketed)), "reference_models": int(bucketed["model"].nunique()),
            "inside_share": float(inside.mean()),
            "position_median": float(position.median()),
            "position_quartiles": [float(q) for q in position.quantile([.25, .75])],
            "candidates": summary,
            "bucket_winners": bucket_winners(bucketed, gaps, p, MIN_BUCKET_N),
            "outside_multi_value_share": (float((bucketed.loc[~inside, "model"].map(n_exact) > 1).mean())
                                          if (~inside).any() else None),
            "most_common_bucket": most_common_bucket(bucketed, exact, p),
            "pairs": bound_pairs(listings, p)}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under error_drivers.hp_cc_kurali, with the reports' key names and rounding.
    TR: error_drivers.hp_cc_kurali altında, raporların anahtar adları ve yuvarlamasıyla yayımlanır.
    """
    out = {"tanim": ("referans = ayni modelin kesin (kovasiz) degerli ilanlarinin medyani; fark = |aday − "
                     "referans|; yalniz iki siniri da bilinen kovalar ve referansi olan modeller; "
                     "yuzdelikler 5/25/50/75/95; konum = (referans − alt) / (ust − alt); ciftler = [alt, ust, "
                     "aralik mi, ilan], iki siniri da olan ilanlar (kesin degerde alt = ust)")}
    for p, m in res.items():
        out[p] = {"birim": UNITS[p], "secilen": m["chosen"], "ilan": m["listings"], "kesin": m["exact"],
                  "aralikli": m["bucketed"], "acik_uclu": m["open_ended"], "degersiz": m["no_value"],
                  "referansli": m["with_reference"], "referans_model": m["reference_models"],
                  "icinde_pct": round(100 * m["inside_share"], 1),
                  "konum_medyan": round(m["position_median"], 3),
                  "konum_ceyrek": [round(q, 3) for q in m["position_quartiles"]],
                  "adaylar": {k: {"medyan": round(v["medyan"], 1), "ortalama": round(v["ortalama"], 1),
                                  "yuzdelik": [round(q, 1) for q in v["yuzdelik"]]} for k, v in m["candidates"].items()},
                  "kova_kazanan": [[lo, up, n, w, {k: round(v, 1) for k, v in med.items()}]
                                   for lo, up, n, w, med in m["bucket_winners"]],
                  "kova_disi_coklu_pct": (round(100 * m["outside_multi_value_share"], 1)
                                          if m["outside_multi_value_share"] is not None else None),
                  "ornek_kova": m["most_common_bucket"], "ciftler": m["pairs"]}
    return {"error_drivers": {"hp_cc_kurali": out}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
common_src = (ROOT / "analysis" / "lib" / "common.py").read_text(encoding="utf-8")

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
assert_rule_in_code(common_src)
res = {p: measure(listings, p, CHOSEN[p]) for p in ("engine_cc", "power_hp")}
for p, m in res.items():
    print(p, {k: round(v["medyan"], 1) for k, v in m["candidates"].items()})

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_engine_rule", to_metrics(res)))
