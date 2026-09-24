"""
02_missingness.py
EN: Technical report §1–§2 — which raw columns are missing, which go missing together, the kb/gb twins,
    and the account of every raw column (in the model, target, or dropped for a stated reason).
    Co-missing blocks are built from the co-missing share (both missing / either missing), not from
    correlations (the owner's decision).
TR: Teknik rapor §1–§2 — hangi ham kolonlar eksik, hangileri birlikte eksik, kb/gb ikizleri ve her ham
    kolonun dökümü (modelde, hedef ya da gerekçesiyle atıldı). Birlikte-eksik bloklar korelasyondan değil,
    birlikte-eksik payından kurulur (ikisi birden eksik / en az biri eksik; kullanıcının kararı).
Output / Çıktı: metrics/02_missingness.json
"""

# %% [1] Setup | Kurulum
import duckdb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from lib.common import DB_PATH, FEATURES, NUM, load_clean, save_metrics

MISSING_TOKENS = ["", "-", "nan", "None", "NaN"]
MIN_RATE = 2          # columns above this missing % are listed | bu eksik yüzdesinin üstündekiler listelenir
BLOCK_RATE = 20       # co-missing blocks among columns above this % | bu yüzdenin üstündekiler arasında bloklar
BLOCK_DIST = 0.02     # 1 − co-missing share; 0.02 = at least 98% co-missing | en az %98 birlikte eksik
DAMAGE_FEATURES = ("roof_state", "hood_state", "trunk_state", "door_", "fender_", "bumper_")
FEATURE_SOURCES = {"vehicle_age": ["gb_year", "search_date"], "power_hp_val": ["power_hp_low", "power_hp_up"],
                   "engine_cc_val": ["engine_cc_up"], "segment": ["series", "model"]}
IDENTITY = ["ad_id", "url", "listing_date", "ad_title", "location", "scraped_at", "description_text",
            "description_clean", "eids_model"]
DERIVED = ["engine_cc_low", "engine_cc_val", "engine_cc_is_range", "power_hp_val", "power_hp_is_range",
           "count_changed", "count_painted", "count_local_painted"]
# EN: body type — the owner's decision: kb is more general (gb merges the seat count).
# TR: kasa tipi — kullanıcının kararı: kb daha genel (gb koltuk sayısıyla birleşik).
TWIN_REASON_BY_HAND = {"gb_body_type": ("kb daha genel: gb kasa tipini koltuk sayısıyla birleştiriyor",
                                        "kb is more general: gb merges body type with the seat count")}
CLASSES = [("model", "modelde (doğrudan ya da türetilerek)", "in the model (directly or derived)"),
           ("hedef", "hedef", "target"),
           ("C", "Kimlik / metin / zaman", "Identity / text / time"),
           ("F", "Türetilmiş tekrar", "Derived duplicate"),
           ("B", "kb/gb ikizi", "kb/gb twin"),
           ("A", "Yarı-sabit (en sık değer ≥ %99)", "Quasi-constant (top value ≥ 99%)"),
           ("D", "Eksik > %40", "Missing > 40%"),
           ("E", "Katalog bloğu (birlikte eksik)", "Catalogue block (co-missing)"),
           ("G", "Modele alınmadı, gerekçe kayıtlı değil", "Not in the model, no recorded reason")]


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def is_missing(col):
    """
    EN: Missing mask of one raw column; empty strings, '-' and text 'nan'/'None' count as missing.
    TR: Bir ham kolonun eksiklik maskesi; boş metin, '-' ve 'nan'/'None' yazıları da eksik sayılır.
    """
    if col.dtype != object:
        return col.isna()
    return col.isna() | col.astype(str).str.strip().isin(MISSING_TOKENS)


def missing_rates(raw, min_rate):
    """
    EN: Columns whose missing share exceeds min_rate %, as [column, % rounded to 0.1], highest first.
    TR: Eksik payı min_rate %'yi aşan kolonlar, [kolon, 0.1'e yuvarlı %], en yüksek önce.
    """
    rows = [[c, round(float(is_missing(raw[c]).mean() * 100), 1)] for c in raw.columns
            if is_missing(raw[c]).mean() > min_rate / 100]
    return sorted(rows, key=lambda x: -x[1])


def comissing_blocks(raw, rates, block_rate, max_dist):
    """
    EN: Groups the heavily missing columns into blocks that go missing together. Distance between two
        columns = 1 − (both missing / either missing); average linkage cut at max_dist. Stops if a block's
        co-missing share falls below 90% (it would be published as "missing together").
        Returns: [{"kolon_sayisi", "ort_eksik_pct", "birliktelik_pct", "ornek_kolonlar", "kolonlar"}, ...].
    TR: Çok eksik kolonları birlikte eksik olan bloklara ayırır. İki kolon arası uzaklık = 1 − (ikisi birden
        eksik / en az biri eksik); ortalama bağlantı, max_dist'te kesilir. Bir bloğun birlikte-eksik payı %90'ın
        altına düşerse durur ("birlikte eksik" diye yayımlanacağı için).
        Döndürür: [{"kolon_sayisi", "ort_eksik_pct", "birliktelik_pct", "ornek_kolonlar", "kolonlar"}, ...].
    """
    cols = [c for c, r in rates if r > block_rate]
    if len(cols) < 2:
        return []
    miss = np.array([is_missing(raw[c]).values for c in cols])
    dist = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            union = (miss[i] | miss[j]).sum()
            dist[i, j] = dist[j, i] = 1 - ((miss[i] & miss[j]).sum() / union if union else 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=max_dist, criterion="distance")
    groups = {}
    for col, g in zip(cols, labels):
        groups.setdefault(g, []).append(col)
    rate_of, blocks = dict(rates), []
    for _g, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(members) >= 2:
            sub = pd.DataFrame({c: is_missing(raw[c]) for c in members})
            both, either = sub.all(axis=1).mean(), sub.any(axis=1).mean()
            share = round(float(both / either * 100) if either > 0 else 0, 1)
            assert share >= 90, f"co-missing block is not missing together: {members} {share}%"
            blocks.append({"kolon_sayisi": len(members),
                           "ort_eksik_pct": round(float(np.mean([rate_of[c] for c in members])), 1),
                           "birliktelik_pct": share, "ornek_kolonlar": members[:8], "kolonlar": members})
    return blocks


def columns_used(raw_cols, features):
    """
    EN: The raw columns the model's features come from (directly, derived, or the 39 damage flags).
        Returns: (set of used raw columns, list of damage flag columns).
    TR: Modelin özniteliklerinin geldiği ham kolonlar (doğrudan, türetilerek ya da 39 hasar bayrağı).
        Döndürür: (kullanılan ham kolon kümesi, hasar bayrağı kolonları listesi).
    """
    flags = [c for c in raw_cols if c.endswith(("_degisen", "_boyali", "_lokal"))]
    used = set()
    for f in features:
        src = flags if f.startswith(DAMAGE_FEATURES) else FEATURE_SOURCES.get(f, [f])
        assert all(c in raw_cols for c in src), f"feature source missing in the raw table: {f} {src}"
        used.update(src)
    return used, flags


def twin_pairs(raw, used):
    """
    EN: The kb/gb twin pairs (plus kb_is_heavy_damaged ↔ is_heavy_damaged): missing %, distinct values,
        identical share, which side the model keeps, and why the other side was dropped.
        Returns: (list of pair records, {dropped column: (reason TR, reason EN)}).
    TR: kb/gb ikiz çiftleri (+ kb_is_heavy_damaged ↔ is_heavy_damaged): eksik %, tekil değer, aynı olma payı,
        modelin hangi tarafı tuttuğu ve öbür tarafın neden atıldığı.
        Döndürür: (çift kayıtları listesi, {atılan kolon: (gerekçe TR, gerekçe EN)}).
    """
    pairs = [(c, "gb_" + c[3:]) for c in raw.columns if c.startswith("kb_") and "gb_" + c[3:] in raw.columns]
    pairs.append(("kb_is_heavy_damaged", "is_heavy_damaged"))
    records, reasons = [], {}
    for kb, gb in pairs:
        same = float(((raw[kb].astype(str) == raw[gb].astype(str)) | (raw[kb].isna() & raw[gb].isna())).mean() * 100)
        records.append({"kb": kb, "gb": gb, "kb_eksik_pct": round(float(is_missing(raw[kb]).mean() * 100), 1),
                        "gb_eksik_pct": round(float(is_missing(raw[gb]).mean() * 100), 1),
                        "kb_tekil": int(raw[kb].nunique()), "gb_tekil": int(raw[gb].nunique()),
                        "ayni_pct": round(same, 1),
                        "tutulan": kb if kb in used else (gb if gb in used else None)})
        dropped = gb if kb in used or gb not in used else kb
        empty = is_missing(raw[dropped]).mean() * 100
        if dropped in TWIN_REASON_BY_HAND:
            reasons[dropped] = TWIN_REASON_BY_HAND[dropped]
        elif same >= 99.9:
            other = kb if dropped == gb else gb
            reasons[dropped] = (f"`{other}` ile birebir aynı", f"identical to `{other}`")
        elif empty > 40:
            reasons[dropped] = (f"%{empty:.0f} boş", f"{empty:.0f}% empty")
        else:
            raise SystemExit(f"twin {dropped} has no reason (identical {same:.1f}%, empty {empty:.1f}%)")
    return records, reasons


def classify_columns(raw, used, twin_reasons, blocks):
    """
    EN: Assigns every raw column to exactly one class, in priority order (in the model > target >
        identity > derived duplicate > twin > quasi-constant > missing > 40% > catalogue block > no reason).
        Returns: {class code: [columns]}.
    TR: Her ham kolonu öncelik sırasıyla tam bir sınıfa atar (modelde > hedef > kimlik > türetilmiş tekrar >
        ikiz > yarı-sabit > %40+ eksik > katalog bloğu > gerekçe kayıtlı değil).
        Döndürür: {sınıf kodu: [kolonlar]}.
    """
    catalogue = {c for b in blocks for c in b["kolonlar"]}
    rules = {"model": lambda c: c in used, "hedef": lambda c: c == "price", "C": lambda c: c in IDENTITY,
             "F": lambda c: c in DERIVED, "B": lambda c: c in twin_reasons,
             "A": lambda c: round(float(raw[c].value_counts(dropna=False, normalize=True).iloc[0]) * 100, 1) >= 99.0,
             "D": lambda c: is_missing(raw[c]).mean() > 0.40, "E": lambda c: c in catalogue, "G": lambda c: True}
    columns = [c for c in raw.columns if c not in ("id", "rn")]
    for c in IDENTITY + DERIVED:          # a typo must not fall silently into G | yazım hatası sessizce G'ye düşmesin
        assert c in columns, f"listed column not in the raw table: {c}"
    assignment = {c: next(code for code, *_ in CLASSES if rules[code](c)) for c in columns}
    by_class = {code: [c for c in columns if assignment[c] == code] for code, *_ in CLASSES}
    assert sum(len(v) for v in by_class.values()) == len(columns)
    return by_class


def feature_missing(listings, features):
    """
    EN: Missing % of every model feature after preparation (numeric NaN, or the 'missing' category).
    TR: Hazırlıktan sonra her model özniteliğinin eksik %'si (sayısalda NaN, kategorikte 'missing').
    """
    rates = {c: round(float(pd.to_numeric(listings[c], errors="coerce").isna().mean() * 100) if c in NUM
                      else float((listings[c].astype(str) == "missing").mean() * 100), 1) for c in features}
    return sorted([[k, v] for k, v in rates.items()], key=lambda x: -x[1])


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology, meta) and the report inputs (error_drivers).
    TR: Site ağacında (methodology, meta) ve rapor girdilerinde (error_drivers) yayımlanır.
    """
    by_class, names = res["classes"], {code: (tr, en) for code, tr, en in CLASSES}
    return {
        "meta": {"n_features": len(FEATURES)},
        "methodology": {
            "feature_kept": FEATURES,
            "column_missing": res["feature_missing"],
            "impute_note": ("Sayısal özniteliklerde doldurma yapılmadı: eksik değerler LightGBM/CatBoost'a "
                            "NaN olarak girer ve kütüphanenin kendi eksik-değer yönlendirmesi kullanılır. "
                            "Yalnız KMeans/PCA için genel medyanla dolduruldu. torque_nm %27.6 eksik olduğu için çıkarıldı."),
            "column_missing_all": res["rates"],
            "sistematik_missing": {"column_missing_all": res["rates"], "sistematik_gruplar": res["blocks"],
                                   "not": ("Katalog eşleştirmesi: standart modeller eşleşir, özel varyantlar eşleşmez → "
                                           "tüm spec birden boş. Gruplar birlikte-eksik payıyla kuruldu: ikisi birden "
                                           "eksik / en az biri eksik ≥ %98.")},
            "feature_drop": [[code, names[code][0], len(by_class[code]), by_class[code], names[code][1]]
                             for code, *_ in CLASSES if code not in ("model", "hedef") and by_class[code]],
            "kolon_hesabi": {"ham": sum(len(v) for v in by_class.values()), "modelde": len(by_class["model"]),
                             "hasar_bayragi": len(res["flags"]), "hedef": len(by_class["hedef"]),
                             "atilan": sum(len(by_class[k]) for k in by_class if k not in ("model", "hedef")),
                             "ikiz_gerekce": {c: list(v) for c, v in res["twin_reasons"].items()},
                             "not": ("feature_drop = [grup, gerekçe (TR), kolon sayısı, kolonlar, gerekçe (EN)]; her "
                                     "ham kolon öncelik sırasıyla tek bir sınıfa atanır (modelde > hedef > C > F > B > "
                                     "A > D > E > G).")},
            "kb_gb_ikiz": res["twins"]},
        "error_drivers": {"ham_kolon": res["table_columns"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
raw = load_clean(derived=False)
listings = load_clean()
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    db_columns = [r[0] for r in con.execute("DESCRIBE car_listings").fetchall()]

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
rates = missing_rates(raw, MIN_RATE)
blocks = comissing_blocks(raw, rates, BLOCK_RATE, BLOCK_DIST)
used, flags = columns_used(list(raw.columns), FEATURES)
twins, twin_reasons = twin_pairs(raw, used)
classes = classify_columns(raw, used, twin_reasons, blocks)
res = {"rates": rates, "blocks": blocks, "flags": flags, "twins": twins, "twin_reasons": twin_reasons,
       "classes": classes, "feature_missing": feature_missing(listings, FEATURES),
       "table_columns": {"tablo_kolon": len(db_columns), "ham_kolon": len([c for c in db_columns if c != "id"])}}
print({k: len(v) for k, v in classes.items()}, "| blocks:", [b["kolon_sayisi"] for b in blocks])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("02_missingness", to_metrics(res)))
