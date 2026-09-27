"""
build_duckdb.py
EN: Builds the SEMI-RAW database data/cars.duckdb from data/raw/{audi,bmw}/<snapshot>/details.jsonl (the
    analysis reads it). Rows come from lib/process_for_db.py; this script only reads the files and writes the
    tables — every listing the site showed is kept (blue plates too): scope decisions belong to gold
    (db/gold_rules.json) and to the analysis (analysis/lib/common.py keeps TR plates). WHAT is written:
      - car_listings: every listing in every snapshot (no de-duplication), `id` (row number, 1…N) + DB_COLUMNS.
        Semi-raw: what the page does not say stays NULL — heavy damage "Belirtilmemiş" / missing, first owner
        missing, and the three flags of a panel whose status is "Belirtilmemiş". The analysis decides how to
        read them. The description is the seller's text without the page heading "Açıklama";
        kb_paint_change_summary is the raw "Boya-değişen" summary line;
      - duplicate_ad_ids (+ duplicate_ad_ids.csv next to the DB): ad_ids seen in more than one snapshot (no
        ad_id repeats inside one snapshot);
      - price_history: price and km per ad_id and snapshot, with the change from the previous snapshot.
    Nothing is carried over from the previous DB: the dashboard_cache / options_cache tables it used to copy
    forward (stale output of an archived producer that nothing read) were dropped on 2026-09-27 — nothing from
    the archive reaches the live chain.
    HOW, safely: the new DB is built in <out>.tmp and moved over the old one only when complete, so a failure
    leaves the old DB untouched. It stops before touching anything when <out>.wal exists (the old DB was not
    closed cleanly), when the old DB is open in another program, or when the raw data is incomplete (a missing
    brand folder, a snapshot folder without details.jsonl, an unreadable line).
    NO GUESSING: before anything is read into rows, the raw files are compared with the register of observed
    values (db/observed_values.json, db/lib/observed_values.py): a new field, a value or format the register has
    not seen, or an unknown damage label stops the build with the field, the value, how many records and the first
    file:line — nothing is written. The parsers (lib/process_for_db.py) raise UnknownValue on any form the data never
    showed, as a second guard. Adapt the code to the new value first, then update the register
    (python tools/observed_values.py); tests/db/test_observed_values.py checks the code against the register.
    NOTE — the API: this semi-raw file is NOT what the API gets. The live API expects unknown = false / 0;
    db/build_gold_db.py derives data/cars_gold.duckdb from this file, and publish_data_to_s3.py uploads only a
    file that keeps that gold contract (docs/database.md → "Gold adımı").
TR: data/raw/{audi,bmw}/<tarama>/details.jsonl'den YARI HAM veritabanı data/cars.duckdb'yi kurar (analiz okur).
    Satırlar lib/process_for_db.py'den gelir; bu betik yalnız dosyaları okur ve tabloları yazar — sitenin
    gösterdiği her ilan tutulur (mavi plakalar da): kapsam kararları gold'a (db/gold_rules.json) ve analize
    (analysis/lib/common.py TR plakalıları tutar) aittir. NE yazılır:
      - car_listings: her taramadaki her ilan (tekilleştirme yok), `id` (satır numarası, 1…N) + DB_COLUMNS.
        Yarı ham: sayfanın söylemediği NULL kalır — ağır hasar "Belirtilmemiş" / yok, ilk sahip yok ve durumu
        "Belirtilmemiş" olan panelin üç bayrağı. Nasıl okunacağına analiz karar verir. Açıklama, sayfa
        başlığı "Açıklama" olmadan satıcının metni; kb_paint_change_summary ham "Boya-değişen" özet satırı;
      - duplicate_ad_ids (+ DB'nin yanında duplicate_ad_ids.csv): birden çok taramada görülen ad_id'ler (tek
        tarama içinde tekrar yok);
      - price_history: ad_id ve tarama başına fiyat ve km, önceki taramaya göre değişimle.
    Eski DB'den hiçbir şey taşınmaz: eskiden aynen taşınan dashboard_cache / options_cache tabloları (arşivdeki
    bir üretecin bayat çıktısı, okuyan yoktu) 2026-09-27'de kaldırıldı — arşivden canlı zincire hiçbir şey girmez.
    NASIL, güvenle: yeni DB <out>.tmp'de kurulur ve yalnız tamamlanınca eskisinin yerine konur; düşerse eski DB
    dokunulmadan kalır. <out>.wal varsa (eski DB düzgün kapanmamış), eski DB başka programda açıksa ya da ham
    veri eksikse (marka klasörü yok, details.jsonl'suz tarama klasörü, okunamayan satır) hiçbir şeye dokunmadan
    durur.
    TAHMİN YOK: satırlar okunmadan önce ham dosyalar gözlenen değerler kaydıyla karşılaştırılır
    (db/observed_values.json, db/lib/observed_values.py): yeni bir alan, kaydın görmediği bir değer ya da biçim veya
    bilinmeyen bir hasar etiketi kurulumu alan, değer, kayıt sayısı ve ilk dosya:satır ile durdurur — hiçbir şey
    yazılmaz. Ayrıştırıcılar (lib/process_for_db.py) verinin hiç göstermediği her biçimde ikinci emniyet olarak
    UnknownValue yükseltir. Önce kodu yeni değere göre uyarlayın, sonra kaydı güncelleyin
    (python tools/observed_values.py); tests/db/test_observed_values.py kodu kayda göre sınar.
    NOT — API: bu yarı ham dosya API'ye giden dosya DEĞİL. Canlı API bilinmeyen = false / 0 bekler;
    db/build_gold_db.py bu dosyadan data/cars_gold.duckdb'yi türetir ve publish_data_to_s3.py yalnız o gold
    sözleşmesini tutan dosyayı yükler (docs/database.md → "Gold adımı").
Run / Koşum:
    python db/build_duckdb.py [--out PATH]
"""
import argparse
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import observed_values as OV  # noqa: E402
from lib.process_for_db import DAMAGE_STATUS_MAP, jsonl_to_silver_df  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent          # repo root | depo kökü
DATA_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "cars.duckdb"
BRANDS = ("audi", "bmw")
TABLE = "car_listings"
CSV_NAME = "duplicate_ad_ids.csv"

# EN: DB column prefix of a panel → silver status column; all 13 panels (the first 11 in the old S3 order,
#     the bumpers were added at the end)
# TR: panelin DB kolon öneki → silver durum kolonu; 13 parçanın hepsi (ilk 11'i eski S3 sırasıyla, tamponlar
#     sona eklendi)
DAMAGE_PREFIX = {
    "tavan": "roof_status",
    "kaput": "engine_hood_status",
    "bagaj": "trunk_lid_status",
    "door_fl": "door_fl_status",
    "door_fr": "door_fr_status",
    "door_rl": "door_rl_status",
    "door_rr": "door_rr_status",
    "fender_fl": "fender_fl_status",
    "fender_fr": "fender_fr_status",
    "fender_rl": "fender_rl_status",
    "fender_rr": "fender_rr_status",
    "bumper_front": "bumper_front_status",
    "bumper_rear": "bumper_rear_status",
}
# EN: flag suffix → the status that sets it | TR: bayrak son eki → onu 1 yapan durum
DAMAGE_SUFFIX = {"degisen": "changed", "boyali": "painted", "lokal": "local_painted"}
UNSPECIFIED = DAMAGE_STATUS_MAP["Belirtilmemiş"]
KNOWN_STATUSES = tuple(s for s in DAMAGE_STATUS_MAP.values() if s != UNSPECIFIED)


def damage_flag_columns():
    """
    EN: The 39 damage flags (13 panels × changed / painted / local), as (column, type, SQL expression):
        1 = the panel has that status, 0 = it has another known status, NULL = "Belirtilmemiş" (unknown).
    TR: 39 hasar bayrağı (13 panel × değişen / boyalı / lokal), (kolon, tip, SQL ifadesi) olarak:
        1 = panel o durumda, 0 = bilinen başka bir durumda, NULL = "Belirtilmemiş" (bilinmiyor).
    """
    known = ", ".join(f"'{s}'" for s in KNOWN_STATUSES)
    out = []
    for prefix, status_col in DAMAGE_PREFIX.items():
        for suffix, status in DAMAGE_SUFFIX.items():
            expr = f"CASE WHEN {status_col} = '{status}' THEN 1 WHEN {status_col} IN ({known}) THEN 0 END"
            out.append((f"{prefix}_{suffix}", "BIGINT", expr))
    return out


# EN: car_listings columns: (column, type, silver source or SQL expression). The legacy block keeps the names,
#     types and order of the S3 car_listings.parquet the API was built on; later columns were added after it.
# TR: car_listings kolonları: (kolon, tip, silver kaynağı ya da SQL ifadesi). Eski blok, API'nin üzerine kurulduğu
#     S3 car_listings.parquet'in ad, tip ve sırasını korur; sonraki kolonlar arkasına eklendi.
DB_COLUMNS = [
    ("ad_id", "BIGINT", "ad_id"),
    ("listing_date", "DATE", "listing_date"),
    ("ad_title", "VARCHAR", "ad_title"),
    ("brand", "VARCHAR", "brand"),
    ("series", "VARCHAR", "series"),
    ("model", "VARCHAR", "model"),
    ("location", "VARCHAR", "location"),
    ("price", "DOUBLE", "price"),
    # EN: engine volume: exact → low = up = val; bucket → low / high / midpoint (open bucket: the known bound)
    # TR: motor hacmi: kesin → alt = üst = değer; kova → alt / üst / orta nokta (açık kovada bilinen sınır)
    ("engine_cc_low", "DOUBLE", "CASE WHEN engine_cc_is_range THEN engine_cc_low ELSE engine_cc END"),
    ("engine_cc_up", "DOUBLE", "CASE WHEN engine_cc_is_range THEN engine_cc_high ELSE engine_cc END"),
    # EN: NOTE — the model's feature of the same name is different: analysis/lib/common.py takes cc from
    #     engine_cc_up (the upper bound; technical report §1)
    # TR: DİKKAT — modelin aynı adlı özniteliği farklı: analysis/lib/common.py cc'yi engine_cc_up'tan (üst sınır)
    #     alır (teknik rapor §1)
    ("engine_cc_val", "DOUBLE", "CASE WHEN engine_cc_is_range THEN COALESCE((engine_cc_low + engine_cc_high)/2.0, "
                                "engine_cc_high, engine_cc_low) ELSE engine_cc END"),
    ("engine_cc_is_range", "BOOLEAN", "engine_cc_is_range"),
    ("power_hp_low", "DOUBLE", "CASE WHEN power_hp_is_range THEN power_hp_low ELSE power_hp END"),
    ("power_hp_up", "DOUBLE", "CASE WHEN power_hp_is_range THEN power_hp_high ELSE power_hp END"),
    ("power_hp_val", "DOUBLE", "CASE WHEN power_hp_is_range THEN COALESCE((power_hp_low + power_hp_high)/2.0, "
                               "power_hp_high, power_hp_low) ELSE power_hp END"),
    ("power_hp_is_range", "BOOLEAN", "power_hp_is_range"),
    ("kb_year", "BIGINT", "kb_year"),
    ("gb_year", "BIGINT", "gb_year"),
    ("kb_mileage", "BIGINT", "kb_mileage"),
    ("gb_mileage", "BIGINT", "gb_mileage"),
    ("kb_transmission", "VARCHAR", "kb_transmission"),
    ("gb_transmission", "VARCHAR", "gb_transmission"),
    ("kb_fuel", "VARCHAR", "kb_fuel"),
    ("gb_fuel", "VARCHAR", "gb_fuel"),
    ("kb_body_type", "VARCHAR", "kb_body_type"),
    ("gb_body_type", "VARCHAR", "gb_body_type"),
    ("kb_color", "VARCHAR", "kb_color"),
    ("gb_color", "VARCHAR", "gb_color"),
    ("kb_drivetrain", "VARCHAR", "kb_drivetrain"),
    ("kb_condition", "VARCHAR", "kb_condition"),
    ("kb_is_heavy_damaged", "BOOLEAN", "is_heavy_damaged"),
    ("kb_trade_available", "BOOLEAN", "kb_trade_available"),
    ("kb_seller_type", "VARCHAR", "kb_seller_type"),
    ("kb_fuel_cons_avg", "DOUBLE", "fuel_cons_avg"),
    ("kb_fuel_tank", "DOUBLE", "fuel_tank"),
    ("gb_warranty_status", "VARCHAR", "warranty_status"),
    ("gb_usage_type", "VARCHAR", "usage_type"),
    ("gb_is_first_owner", "BOOLEAN", "is_first_owner"),
    ("gb_segment", "VARCHAR", "segment"),
    ("gb_mtv_yearly", "DOUBLE", "mtv_yearly"),
    ("torque_nm", "DOUBLE", "torque_nm"),
    ("cylinder_count", "DOUBLE", "cylinder_count"),
    ("max_speed_kmh", "DOUBLE", "max_speed_kmh"),
    ("accel_0_100", "DOUBLE", "accel_0_100"),
    ("city_fuel_cons", "DOUBLE", "city_fuel_cons"),
    ("highway_fuel_cons", "DOUBLE", "highway_fuel_cons"),
    ("length_mm", "DOUBLE", "length_mm"),
    ("width_mm", "DOUBLE", "width_mm"),
    ("height_mm", "DOUBLE", "height_mm"),
    ("weight_kg", "DOUBLE", "weight_kg"),
    ("curb_weight_kg", "DOUBLE", "curb_weight_kg"),
    ("trunk_capacity_lt", "DOUBLE", "trunk_capacity_lt"),
    ("wheelbase_mm", "DOUBLE", "wheelbase_mm"),
    ("is_heavy_damaged", "BOOLEAN", "is_heavy_damaged"),
    ("tramer_fee", "DOUBLE", "tramer_fee"),
    ("count_changed", "BIGINT", "count_changed"),
    ("count_painted", "BIGINT", "count_painted"),
    ("count_local_painted", "BIGINT", "count_local_painted"),
    ("kb_paint_change_summary", "VARCHAR", "kb_paint_change_summary"),
    *damage_flag_columns(),
    # EN: added after the legacy block | TR: eski bloğun arkasına eklenenler
    ("gb_plate_origin", "VARCHAR", "plate_origin"),
    ("gb_drivetrain", "VARCHAR", "gb_drivetrain"),
    ("gb_condition", "VARCHAR", "gb_condition"),
    ("gb_seller_type", "VARCHAR", "gb_seller_type"),
    ("gb_trade_available", "BOOLEAN", "gb_trade_available"),
    ("transmission_brand", "VARCHAR", "transmission_brand"),
    ("gb_kasko_avg", "DOUBLE", "kasko_avg"),
    ("gb_traffic_insurance_avg", "DOUBLE", "traffic_insurance_avg"),
    ("seat_count", "DOUBLE", "seat_count"),
    ("front_tire_spec", "VARCHAR", "front_tire_spec"),
    ("production_year_start", "BIGINT", "production_year_start"),
    ("production_year_end", "BIGINT", "production_year_end"),
    ("rpm_max", "DOUBLE", "rpm_max"),
    ("rpm_min", "DOUBLE", "rpm_min"),
    ("eids_model", "VARCHAR", "eids_model"),
    ("url", "VARCHAR", "url"),
    ("description_text", "VARCHAR", "description_text"),
    ("scraped_at", "TIMESTAMPTZ", "scraped_at"),
    ("search_date", "DATE", "search_date"),
]


def duckdb_safe_dtypes(df):
    """
    EN: Turns pandas 3's `str` (StringDtype) columns into object columns with None for missing. DuckDB 1.4's
        register() did not recognise that dtype; 1.5.5 does, and the result is the same either way.
    TR: pandas 3'ün `str` (StringDtype) kolonlarını, eksikleri None olan object kolonlara çevirir. DuckDB 1.4'ün
        register()'ı bu dtype'ı tanımıyordu; 1.5.5 tanıyor, sonuç iki durumda da aynı.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) in ("str", "string"):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def read_all_silver(data_dir=DATA_DIR, brands=BRANDS):
    """
    EN: Every snapshot of every brand → one silver DataFrame (no de-duplication), every plate kept.
        Stops with FileNotFoundError on a missing brand folder or a snapshot folder without details.jsonl, and
        with ValueError on an unreadable line — nothing is skipped silently. Returns an empty DataFrame when
        no record is usable.
    TR: Her markanın her taraması → tek silver DataFrame (tekilleştirme yok), her plaka tutulur.
        Marka klasörü ya da details.jsonl'suz tarama klasörü eksikse FileNotFoundError, okunamayan satırda
        ValueError ile durur — hiçbir şey sessizce atlanmaz. Kullanılabilir kayıt yoksa boş DataFrame döner.
    """
    frames = []
    for brand in brands:
        brand_dir = Path(data_dir) / brand
        if not brand_dir.is_dir():
            raise FileNotFoundError(f"brand folder missing | marka klasörü yok: {brand_dir}")
        for snapshot in sorted(p for p in brand_dir.iterdir() if p.is_dir()):
            jsonl = snapshot / "details.jsonl"
            if not jsonl.is_file():
                raise FileNotFoundError(f"snapshot folder without details.jsonl | details.jsonl'suz tarama: {snapshot}")
            silver = jsonl_to_silver_df(jsonl)
            if silver.empty:
                print(f"  ! {brand}/{snapshot.name}: no usable record | kullanılabilir kayıt yok")
                continue
            silver["brand"] = silver["brand"].fillna(brand)
            frames.append(silver)
            print(f"  + {brand}/{snapshot.name}: {len(silver)} rows | satır")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return duckdb_safe_dtypes(df)


def duplicate_report(df):
    """
    EN: ad_ids that occur in more than one row, most frequent first. Returns: DataFrame (ad_id, occurrence_count).
    TR: Birden çok satırda geçen ad_id'ler, en sık önce. Döndürür: DataFrame (ad_id, occurrence_count).
    """
    counts = df.dropna(subset=["ad_id"]).groupby("ad_id").size().reset_index(name="occurrence_count")
    dups = counts[counts["occurrence_count"] > 1].sort_values("occurrence_count", ascending=False)
    return dups.reset_index(drop=True)


def assert_not_open(path):
    """
    EN: Stops with RuntimeError if the existing DB at path cannot be opened (e.g. it is open in another program),
        before any work is done. Nothing to check when there is no DB yet.
    TR: path'teki mevcut DB açılamıyorsa (ör. başka bir programda açık), hiçbir iş yapılmadan RuntimeError ile
        durdurur. Henüz DB yoksa sınanacak bir şey yok.
    """
    path = Path(path)
    if not path.exists():
        return
    try:
        duckdb.connect(str(path), read_only=True).close()
    except duckdb.Error as e:
        raise RuntimeError(f"{path} cannot be opened — is it open in another program? | açılamıyor — "
                           f"başka bir programda açık mı? ({e})") from e


def write_duckdb(df, dups, path):
    """
    EN: Writes car_listings, duplicate_ad_ids and price_history into a NEW DuckDB file at path (TRY_CAST: a value
        that does not fit its type becomes NULL; none does in the real data). Returns: the price_history row count.
    TR: car_listings, duplicate_ad_ids ve price_history'yi path'teki YENİ bir DuckDB dosyasına yazar (TRY_CAST:
        tipine uymayan değer NULL olur; gerçek veride hiç yok). Döndürür: price_history satır sayısı.
    """
    col_defs = ",\n    ".join(f"{name} {dtype}" for name, dtype, _ in DB_COLUMNS)
    col_names = ", ".join(name for name, _, _ in DB_COLUMNS)
    select_cast = ", ".join(f"TRY_CAST({expr} AS {dtype}) AS {name}" for name, dtype, expr in DB_COLUMNS)
    con = duckdb.connect(str(path))
    try:
        con.register("silver_df", df)
        con.register("dups_df", dups)
        con.execute("CREATE SEQUENCE id_seq START 1;")
        con.execute(f"CREATE TABLE {TABLE} (\n    id BIGINT PRIMARY KEY DEFAULT nextval('id_seq'),\n    {col_defs}\n);")
        con.execute(f"INSERT INTO {TABLE} ({col_names}) SELECT {select_cast} FROM silver_df;")
        con.execute("CREATE TABLE duplicate_ad_ids AS SELECT * FROM dups_df;")
        con.execute(f"""
            CREATE TABLE price_history AS
            SELECT
                ad_id,
                search_date,
                scraped_at,
                price,
                kb_mileage,
                listing_date,
                price - lag(price) OVER w        AS price_delta,
                row_number()        OVER w        AS snapshot_idx,
                count(*)            OVER (PARTITION BY ad_id) AS snapshot_count
            FROM {TABLE}
            WINDOW w AS (PARTITION BY ad_id ORDER BY search_date, scraped_at)
            ORDER BY ad_id, search_date, scraped_at;
        """)
        ph_rows = con.execute("SELECT count(*) FROM price_history").fetchone()[0]
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return int(ph_rows)


def write_csv(dups, path):
    """
    EN: duplicate_ad_ids as CSV (UTF-8 with BOM, for Excel), written to a temp file and moved into place.
    TR: duplicate_ad_ids'i CSV olarak yazar (Excel için BOM'lu UTF-8); önce geçici dosyaya, sonra yerine taşır.
    """
    tmp = Path(f"{path}.tmp")
    dups.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def check_observed(data_dir=DATA_DIR, brands=BRANDS, register=None):
    """
    EN: Compares the raw files with the register of observed values (register: a dict, default the file
        db/observed_values.json). Raises ValueError listing every unseen field, value, format or damage label (with
        counts and the first file:line) — the build must not guess.
    TR: Ham dosyaları gözlenen değerler kaydıyla karşılaştırır (register: dict; varsayılan db/observed_values.json
        dosyası). Görülmemiş her alanı, değeri, biçimi ya da hasar etiketini (sayı ve ilk dosya:satır ile) listeleyen
        ValueError yükseltir — kurulum tahmin etmemeli.
    """
    register = OV.load() if register is None else register
    _, detail = OV.scan_raw(OV.raw_files(data_dir, brands))
    problems = OV.unseen(detail, register)
    if problems:
        raise ValueError("raw data has what the register has not seen | ham veride kaydın görmediği şeyler var:\n  "
                         + "\n  ".join(problems)
                         + "\nadapt the code, then update the register | kodu uyarlayın, sonra kaydı güncelleyin: "
                           "python tools/observed_values.py")


def build(out_path=DEFAULT_OUT, data_dir=DATA_DIR, brands=BRANDS, register=None):
    """
    EN: Builds the DB at out_path from data_dir (steps in the module header) and writes duplicate_ad_ids.csv
        next to it. First checks the raw files against the register of observed values (check_observed; register
        None = db/observed_values.json) and stops before anything is written on an unseen value.
        Returns: {"out", "csv", "rows", "columns", "duplicate_ad_ids", "price_history"}.
        Raises SystemExit when there is no data at all.
    TR: data_dir'den out_path'e DB'yi kurar (adımlar modül başlığında) ve yanına duplicate_ad_ids.csv yazar. Önce
        ham dosyaları gözlenen değerler kaydına göre sınar (check_observed; register None = db/observed_values.json)
        ve görülmemiş bir değerde hiçbir şey yazmadan durur.
        Döndürür: {"out", "csv", "rows", "columns", "duplicate_ad_ids", "price_history"}.
        Hiç veri yoksa SystemExit.
    """
    out_path = Path(out_path)
    wal = Path(f"{out_path}.wal")
    if wal.exists():
        raise RuntimeError(f"{wal} exists: the old DB was not closed cleanly; open and close it once with DuckDB "
                           f"first | eski DB düzgün kapanmamış; önce DuckDB ile bir kez açıp kapatın")
    print(f"Checking against the register | kayda göre sınanıyor: {OV.REGISTRY_PATH.name}")
    check_observed(data_dir, brands, register)
    print(f"Reading | okunuyor: {', '.join(brands)}")
    df = read_all_silver(data_dir, brands)
    if df.empty:
        raise SystemExit("no data to load | yüklenecek veri yok")
    dups = duplicate_report(df)
    assert_not_open(out_path)

    tmp = Path(f"{out_path}.tmp")
    tmp_wal = Path(f"{tmp}.wal")
    tmp.unlink(missing_ok=True)                        # a leftover of an earlier failed run | önceki düşen koşudan
    tmp_wal.unlink(missing_ok=True)
    try:
        ph_rows = write_duckdb(df, dups, tmp)
        if tmp_wal.exists():
            raise RuntimeError(f"{tmp_wal} left after closing | kapatınca geride kaldı")
        try:
            os.replace(tmp, out_path)
        except PermissionError as e:
            raise RuntimeError(f"{out_path} is open in another program | başka bir programda açık") from e
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp_wal.unlink(missing_ok=True)
        raise

    csv_path = out_path.parent / CSV_NAME
    write_csv(dups, csv_path)
    summary = {"out": out_path, "csv": csv_path, "rows": len(df), "columns": len(DB_COLUMNS),
               "duplicate_ad_ids": len(dups), "price_history": ph_rows}
    print(f"\nWritten | yazıldı: {out_path}")
    print(f"  - {TABLE}: {summary['rows']} rows | satır, {summary['columns']} columns + id | kolon + id")
    print(f"  - duplicate_ad_ids: {summary['duplicate_ad_ids']} · price_history: {ph_rows}")
    print(f"  - {csv_path}")
    return summary


def main(argv=None):
    """
    EN: Command line. Returns: exit code (0 = success, 1 = failure; the reason goes to stderr).
    TR: Komut satırı. Döndürür: çıkış kodu (0 = başarı, 1 = hata; sebep stderr'e yazılır).
    """
    ap = argparse.ArgumentParser(description="Build the semi-raw DB from data/raw | yarı ham DB'yi kur")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output DuckDB file (default: data/cars.duckdb); duplicate_ad_ids.csv goes next to it")
    args = ap.parse_args(argv)
    try:
        build(Path(args.out))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
