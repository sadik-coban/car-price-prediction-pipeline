"""
Temizlik: plaka türü ve ham sayfalardaki hasar listeleri.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import json
import re

import pandas as pd

from lib import data

# Veride görülen plaka değerleri ve kısa kimlikleri; boş plaka "empty" (db/observed_values.json)
PLATE_KINDS = {data.TR_PLATE: "tr", "Mavi plakalı": "blue"}

# Durum kolonundan veritabanı bayrak önekine: çoğu "_status" atılınca çıkar, üç tek parçanın öneki Türkçe
FLAG_PREFIX = {"roof_status": "tavan", "engine_hood_status": "kaput", "trunk_lid_status": "bagaj"}
# Sitedeki cevaptan bayrak sonekine; "Orjinal" ve "Belirtilmemiş" cevaplarında üç bayrak da 0
FLAG_OF_ANSWER = {"Değişmiş": "degisen", "Boyalı": "boyali", "Lokal boyalı": "lokal"}


def plate_kind(plate):
    """Plaka değerini kısa kimliğe çevirir: tr, blue ya da empty (boş). Görülmemiş bir değerde durur."""
    kind = plate.map(PLATE_KINDS)
    unseen = plate[kind.isna() & plate.notna()].unique()
    if len(unseen):
        raise ValueError(f"görülmemiş plaka değeri: {list(unseen)}")
    return kind.fillna("empty")


def read_damage_lists(panel_map, status_map):
    """
    data/raw altındaki her details.jsonl sayfasından ilan numarası, tarama günü ve hasar listesi okur (salt okunur).
    Hasar listesi olmayan sayfalar atlanır. Döndürür: ad_id, snap ve her parça için bir durum kolonu.
    Listede "parça: cevap" biçiminde olmayan bir kayıt ya da eşlemede olmayan bir parça/cevap görülürse durur.
    """
    rows = []
    for path in sorted(data.RAW_DIR.glob("*/*/details.jsonl")):
        with open(path, encoding="utf-8") as file:
            for line in file:
                page = json.loads(line)
                ad_number = re.sub(r"\D", "", str(page.get("KısaBilgi - İlan No") or ""))
                if not (ad_number and page.get("Hasar_Listesi")):
                    continue
                row = {"ad_id": int(ad_number), "snap": str(page.get("search_date") or "")[:10]}
                for entry in page["Hasar_Listesi"]:
                    if not (isinstance(entry, str) and ":" in entry):
                        raise ValueError(f"beklenmeyen hasar kaydı: {entry!r}")
                    panel, answer = (s.strip() for s in entry.split(":", 1))
                    row[panel_map[panel]] = status_map[answer]
                rows.append(row)
    return pd.DataFrame(rows)


def flag_prefix(status_column):
    """Bir durum kolonunun veritabanındaki bayrak öneki, ör. door_fl_status -> door_fl, roof_status -> tavan."""
    return FLAG_PREFIX.get(status_column, status_column.removesuffix("_status"))


def flags_from_statuses(statuses, status_map):
    """Parça durumlarından veritabanının üç bayrağını (değişen, boyalı, lokal) 0/1 olarak yeniden kurar."""
    flags = {}
    for column in statuses.columns:
        for answer, suffix in FLAG_OF_ANSWER.items():
            flags[f"{flag_prefix(column)}_{suffix}"] = (statuses[column] == status_map[answer]).astype(int)
    return pd.DataFrame(flags, index=statuses.index)
