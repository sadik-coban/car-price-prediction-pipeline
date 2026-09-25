"""
02_missingness.py
EN: Technical report §1–§2 — which raw columns are missing, which go missing together, the kb/gb twins,
    and the account of every raw column (in the model, target, or dropped for a stated reason).
    Co-missing blocks are built from the co-missing share (both missing / either missing), not from
    correlations (the owner's decision). The columns whose NULL means "the page does not say"
    ("Belirtilmemiş": heavy damage, first owner, the panel flags) are not missing data: the model codes that
    unknown as no / original, the same rule the gold DB applies (db/gold_rules.json, read here). They are
    left out of the missing list and the blocks and reported on their own (owner's decision 2026-09-24).
TR: Teknik rapor §1–§2 — hangi ham kolonlar eksik, hangileri birlikte eksik, kb/gb ikizleri ve her ham
    kolonun dökümü (modelde, hedef ya da gerekçesiyle atıldı). Birlikte-eksik bloklar korelasyondan değil,
    birlikte-eksik payından kurulur (ikisi birden eksik / en az biri eksik; kullanıcının kararı). NULL'u
    "sayfa söylemiyor" demek olan kolonlar ("Belirtilmemiş": ağır hasar, ilk sahip, panel bayrakları) eksik
    veri değil: model bu bilinmeyeni hayır / orijinal kodlar; gold DB'nin uyguladığı kuralın aynısı
    (db/gold_rules.json, burada okunur). Eksik listesine ve bloklara girmez, ayrıca raporlanır (kullanıcı
    kararı 2026-09-24).
Output / Çıktı: metrics/02_missingness.json
"""

# %% [1] Setup | Kurulum
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from lib.common import DB_PATH, FEATURES, NUM, ROOT, load_clean, save_metrics

MIN_RATE = 2          # columns above this missing % are listed | bu eksik yüzdesinin üstündekiler listelenir
BLOCK_RATE = 20       # co-missing blocks among columns above this % | bu yüzdenin üstündekiler arasında bloklar
BLOCK_DIST = 0.02     # 1 − co-missing share; 0.02 = at least 98% co-missing | en az %98 birlikte eksik
DAMAGE_FEATURES = ("roof_state", "hood_state", "trunk_state", "door_", "fender_", "bumper_")
FEATURE_SOURCES = {"vehicle_age": ["gb_year", "search_date"], "power_hp_val": ["power_hp_low", "power_hp_up"],
                   "engine_cc_val": ["engine_cc_up"], "segment": ["series", "model"]}
IDENTITY = ["ad_id", "url", "listing_date", "ad_title", "location", "scraped_at", "description_text",
            "eids_model"]
# EN: kb_paint_change_summary is the page's own "Boya-değişen" line ("2 değişen, 3 boyalı"): a coarse summary
#     of the 39 damage flags the model already uses, so it is a derived duplicate like the damage counters.
# TR: kb_paint_change_summary sayfanın kendi "Boya-değişen" satırı ("2 değişen, 3 boyalı"): modelin zaten
#     kullandığı 39 hasar bayrağının kaba özeti; hasar sayaçları gibi türetilmiş tekrar.
DERIVED = ["engine_cc_low", "engine_cc_val", "engine_cc_is_range", "power_hp_val", "power_hp_is_range",
           "count_changed", "count_painted", "count_local_painted", "kb_paint_change_summary"]
# EN: body type — the owner's decision: kb is more general (gb merges the seat count).
# TR: kasa tipi — kullanıcının kararı: kb daha genel (gb koltuk sayısıyla birleşik).
TWIN_REASON_BY_HAND = {"gb_body_type": ("kb daha genel: gb kasa tipini koltuk sayısıyla birleştiriyor",
                                        "kb is more general: gb merges body type with the seat count")}
CLASSES = [("model", "modelde (doğrudan ya da türetilerek)", "in the model (directly or derived)"),
           ("target", "hedef", "target"),
           ("C", "Kimlik / metin / zaman", "Identity / text / time"),
           ("F", "Türetilmiş tekrar", "Derived duplicate"),
           ("B", "kb/gb ikizi", "kb/gb twin"),
           ("A", "Yarı-sabit (en sık değer ≥ %99)", "Quasi-constant (top value ≥ 99%)"),
           ("D", "Eksik > %40", "Missing > 40%"),
           ("E", "Katalog bloğu (birlikte eksik)", "Catalogue block (co-missing)"),
           ("G", "Modele alınmadı, gerekçe kayıtlı değil", "Not in the model, no recorded reason")]


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def unknown_coded(gold_rules):
    """
    EN: The columns whose NULL means "Belirtilmemiş" (not stated) and whose unknown is coded no / 0 — the
        fill columns of db/gold_rules.json, in its order. Returns: list of column names.
    TR: NULL'u "Belirtilmemiş" demek olan ve bilinmeyeni hayır / 0 kodlanan kolonlar — db/gold_rules.json'un
        doldurduğu kolonlar, onun sırasıyla. Döndürür: kolon adları listesi.
    """
    return [c for rule in gold_rules["fill"] for c in rule["columns"]]


def unspecified_summary(raw, coded):
    """
    EN: How often the page leaves a coded column unstated: heavy damage (and its kb twin), first owner, the
        panel flags' range, the damage counters' maximum, and every column's %. Stops if a column is absent.
    TR: Sayfanın kodlanan kolonu ne sıklıkla belirtmediği: ağır hasar (ve kb ikizi), ilk sahip, panel
        bayraklarının aralığı, hasar sayaçlarının en yükseği ve her kolonun %'si. Kolon yoksa durur.
    """
    assert all(c in raw.columns for c in coded), f"coded column not in the raw table: {set(coded) - set(raw.columns)}"
    pct = {c: round(float(raw[c].isna().mean() * 100), 1) for c in coded}
    panels = [c for c in coded if c.endswith(("_degisen", "_boyali", "_lokal"))]
    counts = [c for c in coded if c.startswith("count_")]
    return {"n_columns": len(coded), "heavy_damage_pct": pct["is_heavy_damaged"],
            "kb_heavy_damage_pct": pct["kb_is_heavy_damaged"], "first_owner_pct": pct["gb_is_first_owner"],
            "panel_flags": len(panels),
            "panel_min_pct": min(pct[c] for c in panels), "panel_max_pct": max(pct[c] for c in panels),
            "counter_max_pct": max(pct[c] for c in counts), "columns": [[c, pct[c]] for c in coded]}


def is_missing(col):
    """
    EN: Missing mask of one raw column: NULL. The DB writes blanks and "-" as NULL already (db/lib/process_for_db.py)
        and no text column of the DB holds "", "-", "nan", "None" or "NaN" (all counted 2026-09-25), so no text token
        is guessed at.
    TR: Bir ham kolonun eksiklik maskesi: NULL. DB boş ve "-" değerleri zaten NULL yazar (db/lib/process_for_db.py) ve
        DB'nin hiçbir metin kolonunda "", "-", "nan", "None" ya da "NaN" yok (hepsi 2026-09-25'te sayıldı); bu yüzden
        hiçbir metin jetonu tahmin edilmez.
    """
    return col.isna()


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
        Returns: [{"n_columns", "mean_missing_pct", "co_missing_pct", "sample_columns", "columns"}, ...].
    TR: Çok eksik kolonları birlikte eksik olan bloklara ayırır. İki kolon arası uzaklık = 1 − (ikisi birden
        eksik / en az biri eksik); ortalama bağlantı, max_dist'te kesilir. Bir bloğun birlikte-eksik payı %90'ın
        altına düşerse durur ("birlikte eksik" diye yayımlanacağı için).
        Döndürür: [{"n_columns", "mean_missing_pct", "co_missing_pct", "sample_columns", "columns"}, ...].
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
            blocks.append({"n_columns": len(members),
                           "mean_missing_pct": round(float(np.mean([rate_of[c] for c in members])), 1),
                           "co_missing_pct": share, "sample_columns": members[:8], "columns": members})
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
        records.append({"kb": kb, "gb": gb, "kb_missing_pct": round(float(is_missing(raw[kb]).mean() * 100), 1),
                        "gb_missing_pct": round(float(is_missing(raw[gb]).mean() * 100), 1),
                        "kb_unique": int(raw[kb].nunique()), "gb_unique": int(raw[gb].nunique()),
                        "same_pct": round(same, 1),
                        "kept": kb if kb in used else (gb if gb in used else None)})
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
    catalogue = {c for b in blocks for c in b["columns"]}
    rules = {"model": lambda c: c in used, "target": lambda c: c == "price", "C": lambda c: c in IDENTITY,
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
            "systematic_missing": {"column_missing_all": res["rates"], "systematic_groups": res["blocks"],
                                   "unspecified": res["unspecified"],
                                   "note": ("Katalog eşleştirmesi: standart modeller eşleşir, özel varyantlar eşleşmez → "
                                           "tüm spec birden boş. Gruplar birlikte-eksik payıyla kuruldu: ikisi birden "
                                           "eksik / en az biri eksik ≥ %98. 'Belirtilmemiş' kolonları (gold kurallarının "
                                           "doldurduğu kolonlar) eksik listesine ve bloklara girmez; belirtilmemis'te.")},
            "feature_drop": [[code, names[code][0], len(by_class[code]), by_class[code], names[code][1]]
                             for code, *_ in CLASSES if code not in ("model", "target") and by_class[code]],
            "column_accounting": {"raw": sum(len(v) for v in by_class.values()), "in_model": len(by_class["model"]),
                                  "damage_flags": len(res["flags"]), "target": len(by_class["target"]),
                                  "dropped": sum(len(by_class[k]) for k in by_class if k not in ("model", "target")),
                                  "twin_reasons": {c: list(v) for c, v in res["twin_reasons"].items()},
                                  "note": ("feature_drop = [grup, gerekçe (TR), kolon sayısı, kolonlar, gerekçe (EN)]; her "
                                     "ham kolon öncelik sırasıyla tek bir sınıfa atanır (modelde > hedef > C > F > B > "
                                     "A > D > E > G).")},
            "kb_gb_twins": res["twins"]},
        "error_drivers": {"raw_columns": res["table_columns"]},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
raw = load_clean(derived=False)
listings = load_clean()
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    db_columns = [r[0] for r in con.execute("DESCRIBE car_listings").fetchall()]
gold_rules = json.loads((ROOT / "db" / "gold_rules.json").read_text(encoding="utf-8"))

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
coded = unknown_coded(gold_rules)
rates = missing_rates(raw.drop(columns=coded), MIN_RATE)
blocks = comissing_blocks(raw, rates, BLOCK_RATE, BLOCK_DIST)
used, flags = columns_used(list(raw.columns), FEATURES)
twins, twin_reasons = twin_pairs(raw, used)
classes = classify_columns(raw, used, twin_reasons, blocks)
res = {"rates": rates, "blocks": blocks, "unspecified": unspecified_summary(raw, coded),
       "flags": flags, "twins": twins, "twin_reasons": twin_reasons,
       "classes": classes, "feature_missing": feature_missing(listings, FEATURES),
       "table_columns": {"table_columns": len(db_columns), "raw": len([c for c in db_columns if c != "id"])}}
print({k: len(v) for k, v in classes.items()}, "| blocks:", [b["n_columns"] for b in blocks])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("02_missingness", to_metrics(res)))
