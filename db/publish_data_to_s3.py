"""
publish_data_to_s3.py
EN: Publishes the locally built GOLD database data/cars_gold.duckdb (db/build_gold_db.py) to S3 as
    data/cars.duckdb — the object the API's data-refresh poll reads, if that poll is enabled (the service lives
    outside this repo and is not verified here). WHAT and IN WHICH ORDER:
      1. the file is validated: a DuckDB file with a non-empty car_listings that keeps the gold contract
         (columns = id + GOLD_COLUMNS, no NULL in a gold rule column) — so the semi-raw analysis DB, or an
         old-contract file, can never reach the API;
      2. its sha256 is computed and the version is set (the manifest on S3 + 1, unless --version is given);
      3. data/cars.duckdb is uploaded FIRST, then data/manifest.json (version, sha256, build time, row count),
         so the poll never sees a new manifest pointing at a missing or old file.
    HOW to talk to S3 is in lib/s3_publish.py. --dry-run does steps 1–2 and prints the manifest without
    connecting to S3 or reading .env.
TR: Yerelde kurulan GOLD veritabanı data/cars_gold.duckdb'yi (db/build_gold_db.py) S3'e data/cars.duckdb adıyla
    yayımlar — API'nin veri tazeleme yoklamasının okuduğu nesne, o yoklama açıksa (servis bu deponun dışında,
    burada doğrulanmıyor). NE ve HANGİ SIRAYLA:
      1. dosya doğrulanır: car_listings'i boş olmayan ve gold sözleşmesini tutan bir DuckDB dosyası (kolonlar =
         id + GOLD_COLUMNS, kural kolonlarında NULL yok) — yarı ham analiz DB'si ya da eski sözleşmeli bir dosya
         API'ye asla gidemez;
      2. sha256 hesaplanır ve sürüm belirlenir (S3'teki manifest + 1; --version verilmediyse);
      3. ÖNCE data/cars.duckdb, SONRA data/manifest.json (sürüm, sha256, kurulum zamanı, satır sayısı) yüklenir;
         böylece yoklama eksik ya da eski bir dosyayı gösteren yeni bir manifest görmez.
    S3 ile NASIL konuşulacağı lib/s3_publish.py'de. --dry-run 1–2. adımları yapar ve manifesti basar; S3'e
    bağlanmaz, .env'i okumaz.
Run / Koşum:
    python db/publish_data_to_s3.py --dry-run          # check only | yalnız kontrol
    python db/publish_data_to_s3.py [--duckdb PATH] [--version N]
"""
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from build_gold_db import contract_problems
from lib import s3_publish

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUCKDB = ROOT / "data" / "cars_gold.duckdb"
# EN: tables that must exist before publishing. dashboard_cache / options_cache left the list in 2026-09: their
#     producer (build_aggregates.py) is archived and no live code reads them; keeping them would make a fresh DB
#     fail to publish.
# TR: yayından önce bulunması ZORUNLU tablolar. dashboard_cache / options_cache 2026-09'da listeden çıktı: üreticileri
#     (build_aggregates.py) arşivde ve canlı kodda okuyan yok; listede kalsalardı taze bir DB yayımlanamazdı.
REQUIRED_TABLES = ("car_listings",)
# EN: object names on S3 (paths inside the bucket, not secrets); the poll looks for exactly these
# TR: S3'teki nesne adları (bucket içindeki yollar, sır değil); yoklama tam bu adlara bakar
DATA_KEY = "data/cars.duckdb"
MANIFEST_KEY = "data/manifest.json"


def validate_duckdb(path):
    """
    EN: Checks that path is a DuckDB file with every required table, a non-empty car_listings and the gold
        contract (build_gold_db.contract_problems). Returns: the car_listings row count. Raises ValueError otherwise.
    TR: path'in, zorunlu tabloların hepsi bulunan, car_listings'i boş olmayan ve gold sözleşmesini tutan
        (build_gold_db.contract_problems) bir DuckDB dosyası olduğunu sınar. Döndürür: car_listings satır sayısı.
        Değilse ValueError.
    """
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as e:
        raise ValueError(f"not a valid DuckDB file | geçerli bir DuckDB dosyası değil ({e})") from e
    try:
        names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = [t for t in REQUIRED_TABLES if t not in names]
        if missing:
            raise ValueError(f"missing required tables | eksik zorunlu tablo: {', '.join(missing)}")
        rows = con.execute("SELECT count(*) FROM car_listings").fetchone()[0]
        if not rows:
            raise ValueError("car_listings is empty | car_listings boş")
        problems = contract_problems(con)
        if problems:
            raise ValueError("not a gold DB (python db/build_gold_db.py) | gold DB değil: " + "; ".join(problems))
        return int(rows)
    finally:
        con.close()


def file_sha256(path):
    """
    EN: sha256 of the file (read in 1 MB blocks). Returns: hex digest.
    TR: Dosyanın sha256'sı (1 MB'lık bloklarla okunur). Döndürür: onaltılık özet.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def next_version(manifest):
    """
    EN: The version after the one in manifest (the manifest currently on S3); 1 if there is none or it is broken.
    TR: manifest'teki (S3'teki güncel manifest) sürümden sonraki sürüm; yoksa ya da bozuksa 1.
    """
    if manifest and manifest.get("version") is not None:
        try:
            return int(manifest["version"]) + 1
        except (TypeError, ValueError):
            pass
    return 1


def build_manifest(version, sha256, rows, built_at):
    """
    EN: The manifest written next to the database (the fields the poll reads). built_at: ISO time string.
    TR: Veritabanının yanına yazılan manifest (yoklamanın okuduğu alanlar). built_at: ISO zaman metni.
    """
    return {"version": version, "sha256": sha256, "built_at": built_at, "car_listings_rows": rows}


def prepare(duckdb_path, version=None, now=None):
    """
    EN: Steps 1–2 without S3: validates the file, hashes it and builds the manifest. version=None leaves the
        version open (it is decided from S3 in publish). now: the build time (UTC now by default).
        Returns: {"path", "rows", "sha256", "built_at", "version", "size_mb"}.
    TR: S3'süz 1–2. adımlar: dosyayı doğrular, hash'ler ve manifesti hazırlar. version=None sürümü açık bırakır
        (publish'te S3'ten belirlenir). now: kurulum zamanı (varsayılan UTC şimdi).
        Döndürür: {"path", "rows", "sha256", "built_at", "version", "size_mb"}.
    """
    path = Path(duckdb_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — build it first | bulunamadı — önce: python db/build_duckdb.py "
                                f"&& python db/build_gold_db.py")
    rows = validate_duckdb(path)
    return {"path": path, "rows": rows, "sha256": file_sha256(path),
            "built_at": (now or datetime.now(timezone.utc)).isoformat(), "version": version,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1)}


def publish(duckdb_path, store, version=None, now=None):
    """
    EN: Publishes duckdb_path through store (an s3_publish.S3Store or a fake): validates, hashes, sets the version
        (S3 manifest + 1 unless given), uploads the database FIRST and the manifest AFTER it.
        Returns: {"key", "manifest", "size_mb"}. Nothing is uploaded if validation fails.
    TR: duckdb_path'i store üzerinden (bir s3_publish.S3Store ya da sahtesi) yayımlar: doğrular, hash'ler, sürümü
        belirler (verilmediyse S3 manifest + 1), ÖNCE veritabanını, SONRA manifesti yükler.
        Döndürür: {"key", "manifest", "size_mb"}. Doğrulama düşerse hiçbir şey yüklenmez.
    """
    p = prepare(duckdb_path, version, now)
    ver = p["version"] if p["version"] is not None else next_version(store.read_json(MANIFEST_KEY))
    store.upload_file(p["path"], DATA_KEY)
    manifest = build_manifest(ver, p["sha256"], p["rows"], p["built_at"])
    store.put_json(MANIFEST_KEY, manifest)
    return {"key": DATA_KEY, "manifest": manifest, "size_mb": p["size_mb"]}


def main(argv=None, connect=s3_publish.connect):
    """
    EN: Command line. connect builds the S3 store only when a real upload is requested (tests pass their own).
        Returns: exit code (0 = success, 1 = failure; the reason goes to stderr).
    TR: Komut satırı. connect S3 deposunu yalnız gerçek yükleme istendiğinde kurar (testler kendininkini verir).
        Döndürür: çıkış kodu (0 = başarı, 1 = hata; sebep stderr'e yazılır).
    """
    ap = argparse.ArgumentParser(description="Publish cars.duckdb local→S3 | cars.duckdb'yi S3'e yayımla")
    ap.add_argument("--duckdb", default=str(DEFAULT_DUCKDB), help="gold database (default: data/cars_gold.duckdb)")
    ap.add_argument("--version", type=int, default=None, help="manifest version (default: S3 manifest + 1)")
    ap.add_argument("--dry-run", action="store_true", help="validate and show the manifest; no S3, no .env")
    args = ap.parse_args(argv)
    try:
        if args.dry_run:
            p = prepare(args.duckdb, args.version)
            ver = p["version"] if p["version"] is not None else "S3 manifest + 1 (not read in dry-run | okunmadı)"
            print("DRY RUN — nothing uploaded | hiçbir şey yüklenmedi")
            print(f"  file: {p['path']} ({p['size_mb']} MB) → would go to {DATA_KEY}")
            print(f"  manifest: version={ver}  rows={p['rows']}  sha256={p['sha256']}  built_at={p['built_at']}")
            return 0
        store = connect()
        result = publish(args.duckdb, store, args.version)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    m = result["manifest"]
    print(f"Published {result['size_mb']} MB → s3://{store.bucket}/{result['key']}")
    print(f"  manifest version={m['version']}  rows={m['car_listings_rows']}  sha256={m['sha256'][:16]}…")
    print("The API picks it up on its next data-sync poll, if that poll is enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
