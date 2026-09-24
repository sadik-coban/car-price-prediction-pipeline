"""
01_unspecified_panels.py
EN: Technical report §1 — "Belirtilmemiş" (unspecified) panels are counted as original (the owner's decision).
    The gold schema codes "unspecified" the same as "original" (0, 0, 0). Here the decision is measured:
    how many panels are unspecified, how many listings have all 13 unspecified, and how their price compares
    with original, lightly damaged and damaged listings (relative to the same model-and-year median).
    Read from the raw JSONL (read-only) and checked row by row against the database's damage flags.
TR: Teknik rapor §1 — "Belirtilmemiş" paneller orijinal sayılıyor (kullanıcının kararı).
    Gold şema "belirtilmemiş"i "orijinal" ile aynı kodluyor (0, 0, 0). Burada karar ölçülür: kaç panel
    belirtilmemiş, kaç ilanda 13 panelin hepsi belirtilmemiş, fiyatları orijinal, hafif hasarlı ve hasarlı
    ilanlara göre nerede (aynı model ve yılın medyanına oranla). Ham JSONL'den okunur (salt okunur) ve
    veritabanının hasar bayraklarıyla satır satır sınanır.
Output / Çıktı: metrics/01_unspecified_panels.json
"""

# %% [1] Setup | Kurulum
import json
import re

import numpy as np
import pandas as pd

from lib.common import ROOT, load_clean, save_metrics

CELL_MIN_N = 5          # (model, year) cells with at least this many listings | en az bu kadar ilanlı hücreler
GOLD_PANEL = {"roof_status": "tavan", "engine_hood_status": "kaput", "trunk_lid_status": "bagaj",
              "bumper_front_status": "bumper_front", "bumper_rear_status": "bumper_rear"}


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def damage_maps(mappings_text):
    """
    EN: The panel and status maps the db/ parser uses, from the text of db/lib/damage_mappings.json.
        Returns: (panel map {site label: column}, status map {site label: enum}).
    TR: db/ ayrıştırıcısının kullandığı parça ve durum eşlemeleri, db/lib/damage_mappings.json'un
        metninden. Döndürür: (parça eşlemesi {site etiketi: kolon}, durum eşlemesi {site etiketi: enum}).
    """
    maps = json.loads(mappings_text)
    return maps["panels"], maps["statuses"]


def parse_damage_records(raw_records, panels, statuses):
    """
    EN: One row per (ad_id, snapshot) with each panel's status. Rows seen twice in the same snapshot are dropped.
        raw_records: [(ad number text, search_date text, damage list), ...].
        Returns: (DataFrame, number of dropped duplicate rows).
    TR: (ad_id, tarama) başına bir satır, her panelin durumuyla. Aynı taramada iki kez görülen satırlar atılır.
        raw_records: [(ilan no metni, search_date metni, hasar listesi), ...].
        Döndürür: (DataFrame, atılan tekrar satır sayısı).
    """
    rows = []
    for ad_no, date, damage in raw_records:
        rec = {"ad_id": int(ad_no), "tarih": date[:10]}
        for entry in damage:
            if isinstance(entry, str) and ":" in entry:
                part, state = (s.strip() for s in entry.split(":", 1))
                if part in panels and state in statuses:
                    rec[panels[part]] = statuses[state]
        rows.append(rec)
    h = pd.DataFrame(rows)
    dup = h.duplicated(["ad_id", "tarih"], keep=False)
    return h[~dup], int(dup.sum())


def match_to_listings(listings, h, panel_cols):
    """
    EN: Joins the raw panel statuses onto the model's listings (one-to-one on ad_id + snapshot) and stops if
        any rebuilt flag differs from the database's damage flags.
        Returns: joined DataFrame (model, gb_year, price, panel statuses).
    TR: Ham panel durumlarını modelin ilanlarına bağlar (ad_id + tarama üzerinden bire bir) ve yeniden kurulan
        herhangi bir bayrak veritabanının hasar bayraklarından farklıysa durur.
        Döndürür: birleşik DataFrame (model, gb_year, price, panel durumları).
    """
    m = listings[["ad_id", "model", "gb_year", "price"]].copy()
    m["tarih"] = pd.to_datetime(listings["search_date"]).dt.strftime("%Y-%m-%d").values
    m["_i"] = np.arange(len(listings))
    m = m.merge(h, on=["ad_id", "tarih"], how="inner", validate="one_to_one")
    for c in panel_cols:
        gold = GOLD_PANEL.get(c, c.replace("_status", ""))
        for suffix, state in (("degisen", "changed"), ("boyali", "painted"), ("lokal", "local_painted")):
            db = pd.to_numeric(listings[f"{gold}_{suffix}"], errors="coerce").fillna(0).values[m["_i"].values]
            assert np.array_equal((m[c] == state).astype(int).values, db.astype(int)), f"raw/gold differ: {c} {suffix}"
    return m


def panel_structure(panel_cols, statuses):
    """
    EN: Counts the report quotes instead of typing them: panels, flags, answers per panel, panels per group.
    TR: Raporun andığı sayıları elle yazmak yerine sayar: panel, bayrak, panel başına cevap, grup başına panel.
    """
    groups = {}
    for c in panel_cols:
        g = c.split("_")[0]
        if g in ("door", "fender", "bumper"):
            groups[g] = groups.get(g, 0) + 1
    return {"panel": len(panel_cols), "bayrak": 3 * len(panel_cols), "cevap_sayisi": len(set(statuses.values())),
            "grup_panel": groups, "tek_panel": len(panel_cols) - sum(groups.values())}


def price_evidence(m, panel_cols, min_n):
    """
    EN: Median price relative to the (model, year) median, for original / unspecified / lightly damaged /
        damaged listings. Only cells with at least min_n listings.
    TR: (model, yıl) medyanına oranla medyan fiyat: orijinal / belirtilmemiş / hafif hasarlı / hasarlı ilanlar.
        Yalnız en az min_n ilanlı hücreler.
    """
    st = m[panel_cols]
    n_unspec = (st == "unspecified").sum(axis=1)
    n_damage = st.isin(["painted", "local_painted", "changed"]).sum(axis=1)
    n_orig = (st == "original").sum(axis=1)
    n_changed = (st == "changed").sum(axis=1)
    group = np.select([n_damage > 0, n_unspec > 0, n_orig == len(panel_cols)], ["hasarli", "belirtilmemis", "orijinal"], "diger")
    light = ((n_damage > 0) & (n_damage <= 2) & (n_changed == 0)).values
    cell = m.groupby(["model", "gb_year"])["price"]
    ok = (cell.transform("size") >= min_n).values
    ratio = (m["price"] / cell.transform("median")).values
    ev = {g: {"ilan": int(((group == g) & ok).sum()), "medyan_oran": float(np.median(ratio[(group == g) & ok]))}
          for g in ("orijinal", "belirtilmemis", "hasarli")}
    ev["hafif_hasarli"] = {"ilan": int((light & ok).sum()), "medyan_oran": float(np.median(ratio[light & ok])),
                           "tanim": "1-2 boyali/lokal panel, degisen yok"}
    return ev, n_unspec


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under error_drivers.belirtilmemis, with the reports' key names and rounding.
    TR: error_drivers.belirtilmemis altında, raporların anahtar adları ve yuvarlamasıyla yayımlanır.
    """
    st, n_unspec = res["statuses"], res["n_unspec"]
    known = int(st.notna().sum().sum())
    unspec = int((st == "unspecified").sum().sum())
    ev = {g: {**v, "medyan_oran": round(v["medyan_oran"], 4)} for g, v in res["evidence"].items()}
    return {"error_drivers": {"belirtilmemis": {
        "yapi": res["structure"], "eslesen_ilan": res["matched"], "ilan": res["n_listings"],
        "cift_kayit_atlanan": res["dropped"], "panel_sayisi": known, "belirtilmemis_panel": unspec,
        "belirtilmemis_pct": round(100 * unspec / known, 1),
        "hepsi_belirtilmemis": int((n_unspec == res["structure"]["panel"]).sum()),
        "en_az_bir": int((n_unspec > 0).sum()), "hucre_min_n": CELL_MIN_N, "kanit": ev}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
mappings_text = (ROOT / "db" / "lib" / "damage_mappings.json").read_text(encoding="utf-8")
raw_records = []
for path in sorted((ROOT / "data" / "raw").glob("*/*/details.jsonl")):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            ad_no = re.sub(r"\D", "", str(raw.get("KısaBilgi - İlan No") or ""))
            if ad_no and raw.get("Hasar_Listesi"):
                raw_records.append((ad_no, str(raw.get("search_date") or ""), raw["Hasar_Listesi"]))

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
panels, statuses = damage_maps(mappings_text)
panel_cols = list(panels.values())
h, dropped = parse_damage_records(raw_records, panels, statuses)
m = match_to_listings(listings, h, panel_cols)
evidence, n_unspec = price_evidence(m, panel_cols, CELL_MIN_N)
res = {"structure": panel_structure(panel_cols, statuses), "matched": len(m), "n_listings": len(listings),
       "dropped": dropped, "statuses": m[panel_cols], "n_unspec": n_unspec, "evidence": evidence}
print({g: round(v["medyan_oran"], 4) for g, v in evidence.items()})

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_unspecified_panels", to_metrics(res)))
