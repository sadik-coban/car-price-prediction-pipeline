"""
01_unspecified_panels.py
EN: Technical report §1 — "Belirtilmemiş" (unspecified) panels are counted as original (the owner's decision).
    The gold schema codes "unspecified" the same as "original" (0, 0, 0). Here the decision's reach is
    counted: how many panels are unspecified, how many listings have at least one and how many have all 13
    unspecified. (The price comparison with original / damaged listings was removed on 2026-09-25, owner's
    decision.) Read from the raw JSONL (read-only) and checked row by row against the database's damage flags.
TR: Teknik rapor §1 — "Belirtilmemiş" paneller orijinal sayılıyor (kullanıcının kararı).
    Gold şema "belirtilmemiş"i "orijinal" ile aynı kodluyor (0, 0, 0). Burada kararın kapsamı sayılır: kaç
    panel belirtilmemiş, kaç ilanda en az biri ve kaçında 13 panelin hepsi belirtilmemiş. (Orijinal / hasarlı
    ilanlarla fiyat karşılaştırması 2026-09-25'te kaldırıldı, kullanıcının kararı.) Ham JSONL'den okunur (salt
    okunur) ve veritabanının hasar bayraklarıyla satır satır sınanır.
Output / Çıktı: metrics/01_unspecified_panels.json
"""

# %% [1] Setup | Kurulum
import json
import re

import numpy as np
import pandas as pd

from lib.common import ROOT, load_clean, save_metrics

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


def unspecified_per_listing(m, panel_cols):
    """
    EN: How many of each listing's panels are "Belirtilmemiş". Returns: Series (one count per matched listing).
    TR: Her ilanın kaç paneli "Belirtilmemiş". Döndürür: Series (eşleşen ilan başına bir sayı).
    """
    return (m[panel_cols] == "unspecified").sum(axis=1)


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under error_drivers.belirtilmemis, with the reports' key names and rounding.
    TR: error_drivers.belirtilmemis altında, raporların anahtar adları ve yuvarlamasıyla yayımlanır.
    """
    st, n_unspec = res["statuses"], res["n_unspec"]
    known = int(st.notna().sum().sum())
    unspec = int((st == "unspecified").sum().sum())
    return {"error_drivers": {"belirtilmemis": {
        "yapi": res["structure"], "eslesen_ilan": res["matched"], "ilan": res["n_listings"],
        "cift_kayit_atlanan": res["dropped"], "panel_sayisi": known, "belirtilmemis_panel": unspec,
        "belirtilmemis_pct": round(100 * unspec / known, 1),
        "hepsi_belirtilmemis": int((n_unspec == res["structure"]["panel"]).sum()),
        "en_az_bir": int((n_unspec > 0).sum())}}}


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
n_unspec = unspecified_per_listing(m, panel_cols)
res = {"structure": panel_structure(panel_cols, statuses), "matched": len(m), "n_listings": len(listings),
       "dropped": dropped, "statuses": m[panel_cols], "n_unspec": n_unspec}
print({"eslesen": len(m), "en_az_bir": int((n_unspec > 0).sum()), "hepsi": int((n_unspec == len(panel_cols)).sum())})

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_unspecified_panels", to_metrics(res)))
