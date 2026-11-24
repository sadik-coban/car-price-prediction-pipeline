"""
Veri: ilanları veritabanından okur ve tahmin modelinin kullandığı kolonları türetir.
Yollar bu dosyanın konumundan bulunur. Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "cars.duckdb"
RAW_DIR = ROOT / "data" / "raw"
COLLECTION_CONFIG = ROOT / "scraper" / "collection_config.json"
DAMAGE_MAPPINGS = ROOT / "db" / "dblib" / "damage_mappings.json"
FILL_RULES = ROOT / "db" / "gold_rules.json"

TR_PLATE = "(TR) Türkiye"

# Tahmin modelinin öznitelikleri: metin, kategorik ve sayısal
TEXT = ["model", "series"]
CAT = ["brand", "kb_body_type", "kb_drivetrain", "segment", "kb_transmission", "kb_fuel",
       "roof_state", "hood_state", "trunk_state"]
NUM = ["vehicle_age", "gb_mileage", "power_hp_val", "engine_cc_val",
       "door_changed", "door_painted", "door_local", "fender_changed", "fender_painted", "fender_local",
       "bumper_changed", "bumper_painted", "bumper_local", "is_heavy_damaged"]
FEATURES = TEXT + CAT + NUM

# Hasar: tek parçaların durum kolonu ve bayrak öneki; grupların parçaları; bayrak türü ve kolon soneki
SINGLE_PANELS = {"roof_state": "tavan", "hood_state": "kaput", "trunk_state": "bagaj"}
PANEL_GROUPS = {"door": ["door_fl", "door_fr", "door_rl", "door_rr"],
                "fender": ["fender_fl", "fender_fr", "fender_rl", "fender_rr"],
                "bumper": ["bumper_front", "bumper_rear"]}
FLAG_KINDS = {"changed": "degisen", "painted": "boyali", "local": "lokal"}

# Segment kuralı. Yalnız veride görülen seriler (db/observed_values.json); görülmemiş seri segment almaz, durur.
SEGMENT_MAP = {"1 Serisi": "C", "2 Serisi": "C", "3 Serisi": "D", "4 Serisi": "D", "5 Serisi": "E",
               "6 Serisi": "E", "7 Serisi": "F", "Z Serisi": "S", "A1": "B", "A3": "C", "A4": "D", "A5": "D",
               "A6": "E", "A7": "E", "A8": "F", "TT": "S", "TTS": "S", "R8": "S"}
# Performans aileleri: araç modeli adındaki rakam temel seriyi verir (M3 -> 3 Serisi, RS 6 -> A6)
PERF_BASE = {"M Serisi": "{} Serisi", "S": "A{}", "RS": "A{}"}
# Rakam kuralının yanlış sonuç verdiği adlar, önce bakılır: i8 spor coupe, Z4 M roadster
MODEL_SEG = {"i Serisi": {"i8": "S"}, "M Serisi": {"Z4 M": "S"}}
PERF_RE = re.compile(r"^(?:M|S|RS)\s?(\d)")


def query(sql):
    """SQL sorgusunu veritabanında salt okunur koşar, sonucu DataFrame olarak döndürür."""
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(sql).df()


def load_rows(latest_only=True):
    """
    Fiyatı 0'dan büyük, TR plakalı satırlar; veritabanı kolonları olduğu gibi.
    latest_only=True her ilanın (ad_id) yalnız en son taramadaki satırını, False bütün tarama satırlarını verir.
    Sıra ad_id, sonra tarama tarihi.
    """
    where = f"price > 0 AND gb_plate_origin = '{TR_PLATE}'"
    if latest_only:
        return query(f"SELECT * FROM car_listings WHERE {where} "
                     "QUALIFY ROW_NUMBER() OVER (PARTITION BY ad_id ORDER BY search_date DESC) = 1 ORDER BY ad_id")
    return query(f"SELECT * FROM car_listings WHERE {where} ORDER BY ad_id, search_date")


def resolve_segment(series, model):
    """
    Bir ilanın segmenti ve onu çözen yol: (segment, yol). Yol 'model_table' (MODEL_SEG), 'series_map' (SEGMENT_MAP)
    ya da 'base_class' (performans ailesi: araç modeli adındaki rakamdan temel seri); çözülemezse (None, None).
    """
    series, model = str(series), str(model)
    for prefix, segment in MODEL_SEG.get(series, {}).items():
        if model.startswith(prefix):
            return segment, "model_table"
    if series in SEGMENT_MAP:
        return SEGMENT_MAP[series], "series_map"
    if series in PERF_BASE:
        digit = PERF_RE.match(model)
        base = PERF_BASE[series].format(digit.group(1)) if digit else None
        if base in SEGMENT_MAP:
            return SEGMENT_MAP[base], "base_class"
    return None, None


def segment_of(series, model):
    """Bir ilanın segmenti (B, C, D, E, F ya da S); çözülemezse None."""
    return resolve_segment(series, model)[0]


def segments(series, models):
    """Her ilanın segmenti. Çözülemeyen seri/araç modeli çifti kalırsa durur ve çiftleri listeler."""
    out = [segment_of(s, m) for s, m in zip(series, models)]
    unresolved = sorted({(str(s), str(m)) for s, m, seg in zip(series, models, out) if seg is None})
    if unresolved:
        raise ValueError(f"segmenti çözülemeyen {len(unresolved)} seri/araç modeli: {unresolved[:20]}")
    return out


def panel_state(df, part):
    """Tek bir parçanın durumu, üç bayrağından: öncelik değişen > boyalı > lokal > orijinal; boş bayrak 0 sayılır."""
    changed, painted, local = (df[f"{part}_{s}"].fillna(0).values for s in FLAG_KINDS.values())
    return np.where(changed == 1, "changed",
                    np.where(painted == 1, "painted", np.where(local == 1, "local", "original")))


def add_features(raw):
    """
    Ham satırlara tahmin modelinin kolonlarını ekler; yeni bir tablo döndürür.
    Güç = alt ve üst sınırın ortalaması, hacim = üst sınır, yaş = tarama yılı - model yılı (en az 0), boş ağır hasar
    kaydı = 0, segment, tek parçaların durumu, grupların değişen/boyalı/lokal sayısı, snap (tarama tarihi metni).
    Metin ve kategorik özniteliklerde boş değer 'missing' olur; sayısal öznitelikler sayıya çevrilir.
    """
    df = raw.copy()
    df["power_hp_val"] = df[["power_hp_low", "power_hp_up"]].mean(axis=1)
    df["engine_cc_val"] = df["engine_cc_up"]
    df["vehicle_age"] = (pd.to_datetime(df["search_date"]).dt.year - df["gb_year"]).clip(lower=0)
    df["is_heavy_damaged"] = df["is_heavy_damaged"].astype("boolean").fillna(False).astype(int)
    df["segment"] = segments(df["series"], df["model"])
    for column, part in SINGLE_PANELS.items():
        df[column] = panel_state(df, part)
    for group, panels in PANEL_GROUPS.items():
        for kind, suffix in FLAG_KINDS.items():
            df[f"{group}_{kind}"] = df[[f"{p}_{suffix}" for p in panels]].fillna(0).sum(axis=1).astype(int).values
    df["snap"] = df["search_date"].astype(str)
    for column in CAT + TEXT:
        df[column] = df[column].fillna("missing").astype(str)
    for column in NUM:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
