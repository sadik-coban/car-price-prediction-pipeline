"""
01_gold_contract.py
EN: Technical report §1 — the data the API gets (gold). The analysis reads the semi-raw DB, where what the
    page does not say stays NULL; the live API expects the old contract, where an unknown reads as "no" / 0.
    db/build_gold_db.py derives the gold file from the semi-raw DB with the rules in db/gold_rules.json (read
    here too, so the rule and the report cannot drift apart). This script counts, on the whole car_listings
    table (every snapshot's every row, as the gold file holds them), how many cells each rule fills and on how
    many rows, which column is not taken and how many descriptions are empty.
TR: Teknik rapor §1 — API'ye giden veri (gold). Analiz, sayfanın söylemediği bilginin NULL kaldığı yarı ham DB'yi
    okur; canlı API ise bilinmeyenin "hayır" / 0 okunduğu eski sözleşmeyi bekler. db/build_gold_db.py gold
    dosyasını yarı ham DB'den db/gold_rules.json'daki kurallarla türetir (burada da o dosya okunur; kural ile
    rapor birbirinden kopamaz). Bu betik bütün car_listings tablosunda (gold dosyasının tuttuğu gibi her
    taramanın her satırı) her kuralın kaç hücreyi ve kaç satırı doldurduğunu, hangi kolonun alınmadığını ve kaç
    açıklamanın boş olduğunu sayar.
Output / Çıktı: metrics/01_gold_contract.json
"""

# %% [1] Setup | Kurulum
import json

import duckdb

from lib.common import DB_PATH, ROOT, save_metrics

RULES_PATH = ROOT / "db" / "gold_rules.json"
TABLE = "car_listings"


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def check_rules_fit(rules, db_columns):
    """
    EN: Stops if a rule names a column the semi-raw DB does not have (the rules and the DB drifted apart).
    TR: Bir kural yarı ham DB'de olmayan bir kolonu anıyorsa durur (kurallar ile DB birbirinden kopmuş).
    """
    named = [c for r in rules["fill"] for c in r["columns"]] + [d["column"] for d in rules["drop"]]
    missing = [c for c in named if c not in db_columns]
    assert not missing, f"gold rule columns not in the semi-raw DB | yarı ham DB'de yok: {missing}"


def fill_counts(table, rules):
    """
    EN: For each fill rule: its value, column count, the NULL cells it fills and the rows that have at least one.
        Returns: [{"name", "value", "columns", "cells", "rows"}, ...] in the rule file's order.
    TR: Her doldurma kuralı için: değeri, kolon sayısı, doldurduğu NULL hücre ve en az birini taşıyan satır.
        Döndürür: kural dosyasının sırasıyla [{"name", "value", "columns", "cells", "rows"}, ...].
    """
    out = []
    for rule in rules["fill"]:
        nulls = table[rule["columns"]].isna()
        out.append({"name": rule["name"], "value": rule["value"], "columns": len(rule["columns"]),
                    "cells": int(nulls.values.sum()), "rows": int(nulls.any(axis=1).sum())})
    return out


def gold_shape(db_columns, rules):
    """
    EN: Column counts of the semi-raw and the gold car_listings (id not counted) and the columns not taken.
    TR: Yarı ham ve gold car_listings'in kolon sayıları (id sayılmaz) ve alınmayan kolonlar.
    """
    semi = [c for c in db_columns if c != "id"]
    dropped = [d["column"] for d in rules["drop"]]
    return {"semi": len(semi), "gold": len([c for c in semi if c not in dropped]), "dropped": dropped}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under error_drivers.gold_contract, the report's key names.
    TR: error_drivers.gold_contract altında, raporun anahtar adlarıyla yayımlanır.
    """
    return {"error_drivers": {"gold_contract": {
        "table_rows": res["rows"], "semi_raw_columns": res["shape"]["semi"], "gold_columns": res["shape"]["gold"],
        "groups": [{"name": g["name"], "value": g["value"], "n_columns": g["columns"], "filled_cells": g["cells"],
                    "affected_rows": g["rows"]} for g in res["groups"]],
        "total_cells": sum(g["cells"] for g in res["groups"]),
        "dropped": res["shape"]["dropped"], "empty_description_rows": res["empty_descriptions"],
        "note": ("Sayımlar bütün car_listings tablosunda (her taramanın her satırı; gold dosyası bu satırları tutar). "
                "Kurallar db/gold_rules.json'dan; gold'u db/build_gold_db.py kurar.")}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    db_columns = [r[0] for r in con.execute(f"DESCRIBE {TABLE}").fetchall()]
    check_rules_fit(rules, db_columns)
    wanted = [c for r in rules["fill"] for c in r["columns"]] + ["description_text"]
    table = con.execute(f"SELECT {', '.join(wanted)} FROM {TABLE}").df()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
res = {"rows": len(table), "groups": fill_counts(table, rules), "shape": gold_shape(db_columns, rules),
       "empty_descriptions": int(table["description_text"].isna().sum())}
print({g["name"]: (g["cells"], g["rows"]) for g in res["groups"]}, res["shape"], res["empty_descriptions"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_gold_contract", to_metrics(res)))
