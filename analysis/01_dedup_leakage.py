"""
01_dedup_leakage.py
EN: Technical report §1 — from snapshot rows to listings, and what could leak between folds.
    Snapshot rows → unique listings, repeated listings and price changes, plate scope, the collection
    filters and their footprint in the data, content-based duplicates that ad_id cannot see.
TR: Teknik rapor §1 — tarama satırlarından ilanlara ve fold'lar arasına ne sızabilir.
    Tarama satırı → tekil ilan, tekrar görülen ilanlar ve fiyat değişimi, plaka kapsamı, toplama
    filtreleri ve verideki karşılıkları, ad_id'nin göremediği içerik tekrarları.
Output / Çıktı: metrics/01_dedup_leakage.json
Run / Çalıştır: python analysis/01_dedup_leakage.py   ·   VS Code: Shift+Enter cell by cell | hücre hücre
"""

# %% [1] Setup | Kurulum
import json

import duckdb
import numpy as np
import pandas as pd

from lib.common import DB_PATH, ROOT, load_clean, save_metrics

TR_PLATE = "(TR) Türkiye"
# EN: columns that must all match for two listings to count as the same car (strict) / a looser version.
# TR: iki ilanın aynı araç sayılması için hepsinin tutması gereken kolonlar (katı) / daha gevşek hâli.
STRICT_KEY = ["price", "gb_mileage", "gb_year", "brand", "series", "model", "kb_fuel",
              "is_heavy_damaged", "count_painted", "count_changed", "power_hp_up", "engine_cc_up"]
LOOSE_KEY = ["price", "gb_mileage", "gb_year", "model"]


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def count_rows(all_rows, listings):
    """
    EN: Snapshot rows versus unique listings.
        all_rows: every TR snapshot row; listings: latest row per ad_id.
        Returns: {"snapshot_rows", "listings", "repeat_rows"}.
    TR: Tarama satırları ile tekil ilanlar.
        all_rows: bütün TR tarama satırları; listings: ad_id başına son kayıt.
        Döndürür: {"snapshot_rows", "listings", "repeat_rows"}.
    """
    return {"snapshot_rows": len(all_rows), "listings": len(listings),
            "repeat_rows": len(all_rows) - len(listings)}


def dataset_summary(listings):
    """
    EN: The snapshot dates in the data and the listing count per brand.
        Returns: {"snapshots": [dates], "brands": {brand: n}}.
    TR: Verideki tarama tarihleri ve marka başına ilan sayısı.
        Döndürür: {"snapshots": [tarihler], "brands": {marka: n}}.
    """
    return {"snapshots": sorted(listings["snap"].unique().tolist()),
            "brands": {str(k): int(v) for k, v in listings["brand"].value_counts().items()}}


def price_changes(all_rows):
    """
    EN: Listings seen in more than one snapshot, and how their price moved between the first and the
        last time they were seen (cut / rise / changed but came back to the same price).
        Returns: counts {"listings", "seen_again", "changed", "cuts", "rises", "returned"}.
    TR: Birden çok taramada görülen ilanlar ve fiyatlarının ilk ve son görülme arasında nasıl değiştiği
        (indirim / zam / değişip aynı fiyata dönen).
        Döndürür: sayılar {"listings", "seen_again", "changed", "cuts", "rises", "returned"}.
    """
    rows = all_rows.sort_values(["ad_id", "search_date"])
    g = rows.groupby("ad_id")
    per = pd.DataFrame({"snapshots": g["search_date"].nunique(), "prices": g["price"].nunique(),
                        "first": g["price"].first(), "last": g["price"].last()})
    seen_again = per["snapshots"] > 1
    changed = seen_again & (per["prices"] > 1)
    cuts = int((seen_again & (per["last"] < per["first"])).sum())
    rises = int((seen_again & (per["last"] > per["first"])).sum())
    return {"listings": len(per), "seen_again": int(seen_again.sum()), "changed": int(changed.sum()),
            "cuts": cuts, "rises": rises, "returned": int(changed.sum()) - cuts - rises}


def plate_scope(plate_rows):
    """
    EN: Which plates reach the model. Foreign/blue plates never enter the database; here the split of
        what is in it: rows and listings per plate label, and listings with no TR row at all (dropped).
        plate_rows: every row with price > 0, columns ad_id and gb_plate_origin.
        Returns: {"by_plate": [[label, rows, listings], ...], "tr_listings", "dropped_listings",
                  "listings_with_both"}.
    TR: Modele hangi plakalar giriyor. Yabancı/mavi plakalar veritabanına hiç girmez; burada içindekinin
        dökümü: plaka etiketi başına satır ve ilan, hiç TR satırı olmayan (elenen) ilanlar.
        plate_rows: fiyatı > 0 bütün satırlar, kolonlar ad_id ve gb_plate_origin.
        Döndürür: {"by_plate": [[etiket, satır, ilan], ...], "tr_listings", "dropped_listings",
                   "listings_with_both"}.
    """
    label = plate_rows["gb_plate_origin"].fillna("(bilinmiyor)")
    by = (plate_rows.assign(label=label).groupby("label")
          .agg(rows=("ad_id", "size"), listings=("ad_id", "nunique"))
          .sort_values("rows", ascending=False))
    is_tr = plate_rows["gb_plate_origin"] == TR_PLATE
    per_ad = pd.DataFrame({"tr": is_tr, "other": ~is_tr, "ad_id": plate_rows["ad_id"]}).groupby("ad_id").max()
    tr_row = by.loc[TR_PLATE] if TR_PLATE in by.index else None
    return {"by_plate": [[str(i), int(r.rows), int(r.listings)] for i, r in by.iterrows()],
            "tr_listings": int(tr_row["listings"]) if tr_row is not None else 0,
            "dropped_listings": int((~per_ad["tr"]).sum()),
            "listings_with_both": int((per_ad["tr"] & per_ad["other"]).sum())}


def read_collection_filters(config_text):
    """
    EN: The collection filters from the text of scraper/collection_config.json — the file the scraper itself
        reads, so they are never typed by hand here: brands, price range, max km, earliest model year, fuel
        types, site category.
        Returns: dict of filters.
    TR: Toplama filtreleri, scraper/collection_config.json'un metninden — scraper'ın kendisinin okuduğu dosya,
        burada elle yazılmaz: markalar, fiyat aralığı, en fazla km, en eski model yılı, yakıt türleri, site
        kategorisi.
        Döndürür: filtre sözlüğü.
    """
    config = json.loads(config_text)
    brands, query = config["selected_brands"], config["listing_query"]
    ranges = [config["brands"].get(b, config["brands"]["DEFAULT"]) for b in brands]
    return {"brands": brands,
            "price_min": min(r["min"] for r in ranges),
            "price_max": max(r["max"] for r in ranges),
            "max_km": query["max_km"],
            "min_year": query["min_year"],
            "fuels": query["fuels"],
            "category": query["category"]}


def filters_in_data(filters, listings):
    """
    EN: What the collection filters look like in the data: the price cap (right-truncated), the earliest
        model year, SUVs and electric cars left out. Stops if the data breaks a filter (then either the
        filter changed or the code reading it is broken).
        Returns: measured values.
    TR: Toplama filtrelerinin verideki karşılığı: fiyat tavanı (sağdan kesik), en eski model yılı,
        dışarıda kalan SUV ve elektrikliler. Veri bir filtreyi aşıyorsa durur (ya filtre değişmiş ya da
        onu okuyan kod bozulmuş).
        Döndürür: ölçülen değerler.
    """
    price = listings["price"].astype(float).values
    km = pd.to_numeric(listings["gb_mileage"], errors="coerce").values
    year = pd.to_numeric(listings["gb_year"], errors="coerce").values
    fuel = listings["kb_fuel"].astype(str)
    assert filters["price_min"] <= price.min() and price.max() <= filters["price_max"], "price outside the filter"
    assert np.nanmax(km) <= filters["max_km"] and np.nanmin(year) >= filters["min_year"], "km/year outside the filter"
    return {"price_max": float(price.max()),
            "at_cap": int((price == filters["price_max"]).sum()),
            "within_1pct_of_cap": int((price >= 0.99 * filters["price_max"]).sum()),
            "km_max": float(np.nanmax(km)),
            "year_min": int(np.nanmin(year)),
            "listings_in_min_year": int((year == filters["min_year"]).sum()),
            "suv": int((listings["kb_body_type"].astype(str) == "SUV").sum()),
            "electric": int(fuel.str.contains("Elektrik", case=False).sum()),
            "by_fuel": {str(k): int(v) for k, v in fuel.value_counts().items()},
            "max_age": int(pd.Timestamp(listings["search_date"].max()).year - filters["min_year"])}


def excess_rows(listings, key):
    """
    EN: Rows beyond the first in every group of listings that agree on all key columns.
        Returns: (excess row count, DataFrame of groups with more than one row and their size n).
    TR: Bütün anahtar kolonlarda aynı olan ilan gruplarında ilkinin ötesindeki satırlar.
        Döndürür: (fazla satır sayısı, birden çok satırlı gruplar ve boyutları n — DataFrame).
    """
    groups = listings.groupby(key, dropna=False).size().reset_index(name="n")
    multi = groups[groups["n"] > 1]
    return int((multi["n"] - 1).sum()), multi


def content_duplicates(listings, strict_key, loose_key):
    """
    EN: Duplicates ad_id cannot see: different ad_id, same car. Three definitions: strict, loose, and
        strict without price (a re-post at a new price only shows up there). Mileage is usually typed as a
        round number, so the no-price count is also given on non-round mileage.
        Returns: counts and the five most repeated groups.
    TR: ad_id'nin göremediği tekrar: ad_id farklı, araç aynı. Üç tanım: katı, gevşek ve fiyat hariç katı
        (yeni fiyatla yeniden yayımlanan ilan yalnız orada görünür). Kilometre çoğunlukla yuvarlak
        yazıldığı için fiyatsız sayı, yuvarlak olmayan kilometrede de verilir.
        Döndürür: sayılar ve en çok tekrar eden beş grup.
    """
    strict, strict_groups = excess_rows(listings, strict_key)
    loose, _ = excess_rows(listings, loose_key)
    no_price_key = [c for c in strict_key if c != "price"]
    no_price, _ = excess_rows(listings, no_price_key)
    round_km = pd.to_numeric(listings["gb_mileage"], errors="coerce") % 1000 == 0
    no_price_non_round, _ = excess_rows(listings[~round_km], no_price_key)
    top = strict_groups.nlargest(5, "n")
    return {"n": len(listings), "strict": strict, "strict_groups": len(strict_groups), "loose": loose,
            "no_price": no_price, "no_price_non_round_km": no_price_non_round,
            "round_km_share": float(round_km.mean()), "strict_key": strict_key,
            "most_repeated": [{"model": str(r["model"])[:30], "repeats": int(r["n"]),
                               "price": float(r["price"]), "year": int(r["gb_year"])} for _, r in top.iterrows()]}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Places the raw results where they are published — the site tree (meta / methodology) and the
        report inputs (error_drivers) — under the key names the site and the reports read, with the
        reports' rounding.
    TR: Ham sonuçları yayımlandıkları yere — site ağacı (meta / methodology) ve rapor girdileri
        (error_drivers) — sitenin ve raporların okuduğu anahtar adlarıyla, raporların yuvarlamasıyla yerleştirir.
    """
    rows, pc, pl, fl, fi, dup = (res[k] for k in ("rows", "price_changes", "plates", "filters", "in_data", "duplicates"))
    n = dup["n"]
    return {
        "meta": {"n_raw": rows["snapshot_rows"], "n_dedup": rows["listings"],
                 "snapshots": res["snapshots"], "brands": res["brands"]},
        "methodology": {"icerik_duplike": {
            "kati_tanim_fazla": dup["strict"], "kati_tanim_pct": round(dup["strict"] / n * 100, 2),
            "gevsek_tanim_fazla": dup["loose"], "gevsek_tanim_pct": round(dup["loose"] / n * 100, 2),
            "fiyatsiz_tanim_fazla": dup["no_price"], "fiyatsiz_tanim_pct": round(dup["no_price"] / n * 100, 2),
            "fiyatsiz_yuvarlak_olmayan_km_fazla": dup["no_price_non_round_km"],
            "km_yuvarlak_pct": round(dup["round_km_share"] * 100, 1),
            "duplike_grup_sayisi": dup["strict_groups"], "kati_tanim_kolonlari": dup["strict_key"],
            "en_cok_tekrar": [{"model": r["model"], "n_tekrar": r["repeats"], "fiyat": r["price"], "yil": r["year"]}
                              for r in dup["most_repeated"]],
            "not": ("ad_id-dedup DIŞINDA içerik-bazlı duplike kontrolü: ad_id farklı ama tüm ayırt "
                    "edici özellikler (fiyat, km, yaş, model, hasar, motor) aynı. Düşük oran veri "
                    "toplama temizliğini doğrular. Bir kısmı gerçek tekrar ilan, bir kısmı tesadüfi "
                    "çakışma (yaygın modellerde benzer özellikli farklı araçlar).")}},
        "error_drivers": {
            "fiyat_degisimi": {"ilan": pc["listings"], "coklu": pc["seen_again"],
                               "coklu_pct": round(100 * pc["seen_again"] / pc["listings"], 1),
                               "degisen": pc["changed"], "degisen_pct": round(100 * pc["changed"] / pc["listings"], 1),
                               "indirim": pc["cuts"], "zam": pc["rises"], "donup_ayni": pc["returned"]},
            "plaka_kapsami": {
                "dagilim": [{"plaka": p, "satir": r, "ilan": i} for p, r, i in pl["by_plate"]],
                "egitime_giren_ilan": pl["tr_listings"], "elenen_ilan": pl["dropped_listings"],
                "iki_etiketli_ilan": pl["listings_with_both"],
                "not": ("Yabancı/mavi plakalı ilanlar veritabanına hiç yazılmadı (build_duckdb.py); "
                        "burada elenenler plaka bilgisi BOŞ olan ilanlardır. Eğitim, doğrulama ve "
                        "backtest'in tamamı TR plakalı ilanlar üzerindedir.")},
            "kapsam": {"markalar": fl["brands"], "fiyat_min": fl["price_min"], "fiyat_max": fl["price_max"],
                       "max_km": fl["max_km"], "min_yil": fl["min_year"], "yakit_filtresi": fl["fuels"],
                       "yol": fl["category"],
                       "olculen": {"fiyat_max": fi["price_max"], "tavanda": fi["at_cap"],
                                   "tavan_ust_yuzde1": fi["within_1pct_of_cap"], "km_max": fi["km_max"],
                                   "yil_min": fi["year_min"], "min_yilda": fi["listings_in_min_year"],
                                   "suv": fi["suv"], "elektrik": fi["electric"], "yakit": fi["by_fuel"]},
                       "yas_tavani": fi["max_age"]}},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
all_rows = load_clean(all_snapshots=True)
listings = load_clean()
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    plate_rows = con.execute("SELECT ad_id, gb_plate_origin FROM car_listings WHERE price > 0").df()
config_text = (ROOT / "scraper" / "collection_config.json").read_text(encoding="utf-8")

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
filters = read_collection_filters(config_text)
res = {"rows": count_rows(all_rows, listings),
       **dataset_summary(listings),
       "price_changes": price_changes(all_rows),
       "plates": plate_scope(plate_rows),
       "filters": filters,
       "in_data": filters_in_data(filters, listings),
       "duplicates": content_duplicates(listings, STRICT_KEY, LOOSE_KEY)}
print(res["rows"], res["price_changes"], sep="\n")

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_dedup_leakage", to_metrics(res)))
