"""
common.py
EN: Shared entry point for every analysis script. Exactly two public functions:
    load_clean()   -> the cleaned listings (same rows and columns the model is trained on)
    save_metrics() -> writes one script's results to metrics/<name>.json
    Everything else here is a path/feature constant or a small private helper (leading underscore).
TR: Bütün analiz betiklerinin ortak girişi. Dışarıya açık tam iki fonksiyon:
    load_clean()   -> temizlenmiş ilanlar (modelin eğitildiği satır ve kolonların aynısı)
    save_metrics() -> bir betiğin sonuçlarını metrics/<name>.json'a yazar
    Geri kalan her şey yol/öznitelik sabiti ya da alt çizgili küçük bir yardımcı.
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ── Paths | Yollar ────────────────────────────────────────────────────────────────────────────
# EN: resolved from this file, so scripts work from any working directory (CLI or VS Code cells).
# TR: bu dosyadan çözülür; betikler hangi klasörden koşulursa koşulsun çalışır (komut satırı ya da VS Code).
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "cars.duckdb"
METRICS_DIR = ROOT / "metrics"
ANALYSIS_DIR = ROOT / "data" / "analysis"          # intermediate artefacts | ara artefaktlar (gitignore'lı)

from . import segment_rule as _SR                 # segment rule, single source | segment kuralı, tek kaynak

# ── Model features | Model öznitelikleri ─────────────────────────────────────────────────────────
TEXT = ["model", "series"]
CAT = ["brand", "kb_body_type", "kb_drivetrain", "segment", "kb_transmission", "kb_fuel",
       "roof_state", "hood_state", "trunk_state"]
NUM = ["vehicle_age", "gb_mileage", "power_hp_val", "engine_cc_val",
       "door_changed", "door_painted", "door_local", "fender_changed", "fender_painted", "fender_local",
       "bumper_changed", "bumper_painted", "bumper_local", "is_heavy_damaged"]
FEATURES = TEXT + CAT + NUM

_TR_PLATE = "(TR) Türkiye"
_PANEL_GROUPS = {"door": ["door_fl", "door_fr", "door_rl", "door_rr"],
                 "fender": ["fender_fl", "fender_fr", "fender_rl", "fender_rr"],
                 "bumper": ["bumper_front", "bumper_rear"]}


# ── Public | Dışarıya açık ───────────────────────────────────────────────────────────────────────
def load_clean(all_snapshots=False, derived=True):
    """
    EN: Loads the listings the model is built on and adds the derived features.
        Rows: TR-plated, price > 0. With all_snapshots=False, the latest row per ad_id (dedup);
        with True, every snapshot row. Always ordered by ad_id (KFold positions depend on it).
        Adds (derived=True): vehicle_age, power_hp_val (mean of bounds), engine_cc_val (upper bound),
        damage states and counts, segment (stops if a series/model cannot be resolved). Categorical/text
        features get 'missing' for empty values; numeric features are coerced to numbers.
        derived=False returns the database columns untouched (for questions about the raw columns,
        e.g. missingness, where filling 'missing' would hide the gaps).
        Returns: pandas DataFrame.
    TR: Modelin kurulduğu ilanları yükler ve türetilen öznitelikleri ekler.
        Satırlar: TR plakalı, fiyatı > 0. all_snapshots=False ise ad_id başına son kayıt (tekilleştirme);
        True ise bütün tarama satırları. Sıra her zaman ad_id (KFold konumları buna bağlı).
        Ekler (derived=True): vehicle_age, power_hp_val (sınırların ortalaması), engine_cc_val (üst sınır),
        hasar durumları ve sayıları, segment (seri/model çözülemezse durur). Kategorik/metin özniteliklerde
        boş değer 'missing' olur; sayısal öznitelikler sayıya çevrilir.
        derived=False veritabanı kolonlarını olduğu gibi döndürür (ham kolonlarla ilgili sorular için,
        ör. eksiklik; orada 'missing' doldurmak boşlukları gizlerdi).
        Döndürür: pandas DataFrame.
    """
    where = f"price > 0 AND gb_plate_origin = '{_TR_PLATE}'"
    if all_snapshots:
        sql = f"SELECT * FROM car_listings WHERE {where} ORDER BY ad_id, search_date"
    else:
        sql = (f"WITH r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY ad_id ORDER BY search_date DESC) rn "
               f"FROM car_listings WHERE {where}) SELECT * EXCLUDE (rn) FROM r WHERE rn = 1 ORDER BY ad_id")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        raw = con.execute(sql).df()
    finally:
        con.close()
    return _prepare(raw) if derived else raw


def save_metrics(name, values, run_id=None):
    """
    EN: Writes one script's results to metrics/<name>.json with a _meta block: script name, run time,
        last snapshot date in the data, DB fingerprint and (for model-derived metrics) run_id.
        Written to a temporary file first and then renamed, so a half-written JSON never exists.
        name: script name without .py, e.g. "01_dedup_leakage" or "shap/02_oof_shap".
        Returns: the Path written.
    TR: Bir betiğin sonuçlarını _meta bloğuyla metrics/<name>.json'a yazar: betik adı, koşum zamanı,
        verideki son tarama tarihi, DB parmak izi ve (modelden türeyen metriklerde) run_id.
        Önce geçici dosyaya yazılıp yeniden adlandırılır; yarım yazılmış bir JSON hiç oluşmaz.
        name: .py'siz betik adı, ör. "01_dedup_leakage" ya da "shap/02_oof_shap".
        Döndürür: yazılan Path.
    """
    path = METRICS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_meta": {"script": f"{name}.py",
                     "generated_at": datetime.now().isoformat(timespec="seconds"),
                     "data_until": _last_snapshot_date(),
                     "db": _db_fingerprint(),
                     "run_id": run_id},
           **values}
    text = json.dumps(_to_jsonable(doc), ensure_ascii=False, indent=1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


# ── Private helpers | Özel yardımcılar ─────────────────────────────────────────────────────────────
def _prepare(raw):
    """
    EN: Adds the derived features to raw DB rows (the same rules the model is trained with).
        Returns: a new DataFrame.
    TR: Ham DB satırlarına türetilen öznitelikleri ekler (modelin eğitildiği kuralların aynısı).
        Döndürür: yeni bir DataFrame.
    """
    df = raw.copy()
    # EN: bucket → one number (owner's rule: hp = mean of bounds, cc = upper bound)
    # TR: kova → tek sayı (kullanıcının kuralı: hp = sınırların ortalaması, cc = üst sınır)
    df["power_hp_val"] = df[["power_hp_low", "power_hp_up"]].mean(axis=1)
    df["engine_cc_val"] = df["engine_cc_up"]
    df["vehicle_age"] = (pd.to_datetime(df["search_date"]).dt.year - df["gb_year"]).clip(lower=0)
    df["is_heavy_damaged"] = df["is_heavy_damaged"].fillna(0).astype(int)
    df["segment"] = _SR.apply(df["series"], df["model"])
    for part, col in (("tavan", "roof_state"), ("kaput", "hood_state"), ("bagaj", "trunk_state")):
        df[col] = _panel_state(df, part)
    for group, panels in _PANEL_GROUPS.items():
        for kind, suffix in (("changed", "degisen"), ("painted", "boyali"), ("local", "lokal")):
            df[f"{group}_{kind}"] = df[[f"{p}_{suffix}" for p in panels]].fillna(0).sum(axis=1).astype(int).values
    df["snap"] = df["search_date"].astype(str)
    for c in CAT + TEXT:
        df[c] = df[c].fillna("missing").astype(str)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _panel_state(df, part):
    """
    EN: State of a single panel from its three flags; priority changed > painted > local > original.
        Returns: numpy array of strings.
    TR: Tek bir panelin durumu, üç bayrağından; öncelik değişen > boyalı > lokal > orijinal.
        Döndürür: metin dizisi (numpy).
    """
    changed = df[f"{part}_degisen"].fillna(0).values
    painted = df[f"{part}_boyali"].fillna(0).values
    local = df[f"{part}_lokal"].fillna(0).values
    return np.where(changed == 1, "changed",
                    np.where(painted == 1, "painted", np.where(local == 1, "local", "original")))


_FINGERPRINT = None


def _db_fingerprint():
    """
    EN: sha256 and row count of data/cars.duckdb (computed once per process). Lets the report
        builders check that every metrics file comes from the same database.
        Returns: {"path", "sha256", "rows"}.
    TR: data/cars.duckdb'nin sha256'sı ve satır sayısı (süreç başına bir kez hesaplanır). Rapor
        derleyicileri bununla bütün metrik dosyalarının aynı veritabanından geldiğini sınar.
        Döndürür: {"path", "sha256", "rows"}.
    """
    global _FINGERPRINT
    if _FINGERPRINT is None:
        h = hashlib.sha256()
        with open(DB_PATH, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            rows = con.execute("SELECT COUNT(*) FROM car_listings").fetchone()[0]
        finally:
            con.close()
        _FINGERPRINT = {"path": "data/cars.duckdb", "sha256": h.hexdigest(), "rows": int(rows)}
    return _FINGERPRINT


def _last_snapshot_date():
    """
    EN: The latest snapshot date in the database (YYYY-MM-DD).
    TR: Veritabanındaki en son tarama tarihi (YYYY-AA-GG).
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return str(con.execute("SELECT MAX(search_date) FROM car_listings").fetchone()[0])[:10]
    finally:
        con.close()


def _to_jsonable(obj):
    """
    EN: Converts numpy/pandas values to plain JSON types; NaN and ±Inf become null.
    TR: numpy/pandas değerlerini düz JSON tiplerine çevirir; NaN ve ±Inf null olur.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj
