"""
01_gold_contract.py
EN: Technical report §1 — the data the API gets (gold). The analysis reads the semi-raw DB, where what the
    page does not say stays NULL; the live API expects the old contract, where an unknown reads as "no" / 0.
    db/build_gold_db.py derives the gold file from the semi-raw DB with the rules in db/gold_rules.json (read
    here too, so the rule and the report cannot drift apart). This script counts the rows gold leaves out (the
    row rule: blue plates, which the semi-raw DB keeps) and, on the rows gold takes (every snapshot's every row
    otherwise), how many cells each rule fills and on how many rows, which column is not taken and how many
    descriptions are empty.
TR: Teknik rapor §1 — API'ye giden veri (gold). Analiz, sayfanın söylemediği bilginin NULL kaldığı yarı ham DB'yi
    okur; canlı API ise bilinmeyenin "hayır" / 0 okunduğu eski sözleşmeyi bekler. db/build_gold_db.py gold
    dosyasını yarı ham DB'den db/gold_rules.json'daki kurallarla türetir (burada da o dosya okunur; kural ile
    rapor birbirinden kopamaz). Bu betik gold'un almadığı satırları (satır kuralı: yarı ham DB'nin tuttuğu mavi
    plakalar) ve gold'un aldığı satırlarda (bunun dışında her taramanın her satırı) her kuralın kaç hücreyi ve kaç
    satırı doldurduğunu, hangi kolonun alınmadığını ve kaç açıklamanın boş olduğunu sayar.
Output / Çıktı: metrics/01_gold_contract.json
"""

# %% [1] Setup | Kurulum
import json

import duckdb
import pandas as pd

from lib.common import DB_PATH, ROOT, save_metrics

RULES_PATH = ROOT / "db" / "gold_rules.json"
TABLE = "car_listings"


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def check_rules_fit(rules, db_columns):
    """
    EN: Stops if a rule names a column the semi-raw DB does not have (the rules and the DB drifted apart).
    TR: Bir kural yarı ham DB'de olmayan bir kolonu anıyorsa durur (kurallar ile DB birbirinden kopmuş).
    """
    named = ([c for r in rules["fill"] for c in r["columns"]] + [d["column"] for d in rules["drop"]]
             + [r["column"] for r in rules.get("drop_rows", [])])
    missing = [c for c in named if c not in db_columns]
    assert not missing, f"gold rule columns not in the semi-raw DB | yarı ham DB'de yok: {missing}"


def row_rule(table, rules):
    """
    EN: The rows gold leaves out: a row whose column holds one of a row rule's values (NULL never matches).
        Returns: (mask of dropped rows, [{"column", "values", "rows", "listings"}, ...]).
    TR: Gold'un almadığı satırlar: kolonunda bir satır kuralının değerlerinden birini taşıyan satır (NULL hiç
        eşleşmez). Döndürür: (düşen satır maskesi, [{"column", "values", "rows", "listings"}, ...]).
    """
    dropped = pd.Series(False, index=table.index)
    out = []
    for rule in rules.get("drop_rows", []):
        hit = table[rule["column"]].isin(rule["values"]).fillna(False).astype(bool)
        dropped |= hit
        out.append({"column": rule["column"], "values": list(rule["values"]), "rows": int(hit.sum()),
                    "listings": int(table.loc[hit, "ad_id"].nunique())})
    return dropped.to_numpy(dtype=bool), out


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
        "table_rows": res["rows"], "semi_raw_rows": res["semi_rows"], "dropped_rows": res["dropped_rows"],
        "semi_raw_columns": res["shape"]["semi"], "gold_columns": res["shape"]["gold"],
        "groups": [{"name": g["name"], "value": g["value"], "n_columns": g["columns"], "filled_cells": g["cells"],
                    "affected_rows": g["rows"]} for g in res["groups"]],
        "total_cells": sum(g["cells"] for g in res["groups"]),
        "dropped": res["shape"]["dropped"], "empty_description_rows": res["empty_descriptions"],
        "note": ("Sayımlar gold'un aldığı car_listings satırlarında (satır kuralının düşürdükleri dışında her "
                "taramanın her satırı). Kurallar db/gold_rules.json'dan; gold'u db/build_gold_db.py kurar.")}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    db_columns = [r[0] for r in con.execute(f"DESCRIBE {TABLE}").fetchall()]
    check_rules_fit(rules, db_columns)
    wanted = ([c for r in rules["fill"] for c in r["columns"]] + ["description_text", "ad_id"]
              + [r["column"] for r in rules.get("drop_rows", [])])
    table = con.execute(f"SELECT {', '.join(dict.fromkeys(wanted))} FROM {TABLE}").df()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
dropped, dropped_rows = row_rule(table, rules)
gold_rows = table[~dropped]
res = {"rows": len(gold_rows), "semi_rows": len(table), "dropped_rows": dropped_rows,
       "groups": fill_counts(gold_rows, rules), "shape": gold_shape(db_columns, rules),
       "empty_descriptions": int(gold_rows["description_text"].isna().sum())}
print({g["name"]: (g["cells"], g["rows"]) for g in res["groups"]}, res["shape"], res["empty_descriptions"], dropped_rows)

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("01_gold_contract", to_metrics(res)))
