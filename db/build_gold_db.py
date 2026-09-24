"""
build_gold_db.py
EN: Builds the GOLD database data/cars_gold.duckdb (what the API gets) from the semi-raw data/cars.duckdb (what
    the analysis reads). The semi-raw DB keeps unknowns as NULL; the live API expects the old contract, where an
    unknown reads as "no". WHAT changes, and nothing else (the rules and their reasons are in gold_rules.json):
      - car_listings: 45 columns get NULL → false / 0 (heavy damage ×2 and first owner → false; the 39 panel
        flags and the 3 damage counters → 0); kb_paint_change_summary is not taken. Same rows, same order, same
        id, same types; every other cell as it is (the description stays description_text, without the page
        heading; there is no description_clean — owner's decision 2026-09-24);
      - duplicate_ad_ids, price_history, dashboard_cache, options_cache: copied unchanged.
    It first checks that the input really is the semi-raw DB (its columns are id + build_duckdb.DB_COLUMNS), so
    an old-contract or unrelated file is never turned into gold.
    HOW, safely: the input is opened read-only; the gold DB is built in <out>.tmp and moved over the old one only
    when complete. It stops before touching anything when <out>.wal exists, when the old gold DB is open in
    another program, or when input and output are the same file.
TR: Yarı ham data/cars.duckdb'den (analizin okuduğu) GOLD veritabanı data/cars_gold.duckdb'yi (API'ye giden) kurar.
    Yarı ham DB bilinmeyeni NULL tutar; canlı API ise bilinmeyenin "hayır" okunduğu eski sözleşmeyi bekler. NE
    değişir, başka hiçbir şey değil (kurallar ve gerekçeleri gold_rules.json'da):
      - car_listings: 45 kolonda NULL → false / 0 (ağır hasar ×2 ve ilk sahip → false; 39 panel bayrağı ve 3
        hasar sayacı → 0); kb_paint_change_summary alınmaz. Aynı satırlar, aynı sıra, aynı id, aynı tipler;
        öteki her hücre olduğu gibi (açıklama sayfa başlığı olmadan description_text olarak kalır;
        description_clean yok — kullanıcı kararı 2026-09-24);
      - duplicate_ad_ids, price_history, dashboard_cache, options_cache: aynen kopyalanır.
    Önce girdinin gerçekten yarı ham DB olduğunu sınar (kolonları id + build_duckdb.DB_COLUMNS); eski
    sözleşmeli ya da ilgisiz bir dosya gold'a çevrilmez.
    NASIL, güvenle: girdi salt okunur açılır; gold DB <out>.tmp'de kurulur ve yalnız tamamlanınca eskisinin
    yerine konur. <out>.wal varsa, eski gold DB başka programda açıksa ya da girdi ile çıktı aynı dosyaysa hiçbir
    şeye dokunmadan durur.
Run / Koşum:
    python db/build_gold_db.py [--in PATH] [--out PATH]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_duckdb import DB_COLUMNS, TABLE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "cars.duckdb"
DEFAULT_OUT = ROOT / "data" / "cars_gold.duckdb"
RULES_PATH = Path(__file__).resolve().parent / "gold_rules.json"
# EN: tables copied as they are (the caches are optional: a fresh DB may not have them)
# TR: olduğu gibi kopyalanan tablolar (önbellekler isteğe bağlı: taze bir DB'de olmayabilir)
COPIED_TABLES = ("duplicate_ad_ids", "price_history")
OPTIONAL_TABLES = ("dashboard_cache", "options_cache")


def load_rules(path=RULES_PATH):
    """
    EN: Reads gold_rules.json and checks it against DB_COLUMNS: every column exists, none is in two rules, a
        dropped column is not also filled. Returns: {"fill": {column: value}, "drop": [column], "groups": [...]}.
    TR: gold_rules.json'u okur ve DB_COLUMNS'a göre sınar: her kolon var, hiçbiri iki kuralda değil, alınmayan
        kolon doldurulmuyor. Döndürür: {"fill": {kolon: değer}, "drop": [kolon], "groups": [...]}.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {n for n, _, _ in DB_COLUMNS}
    fill, groups = {}, []
    for rule in raw["fill"]:
        for col in rule["columns"]:
            if col not in names:
                raise ValueError(f"gold rule column not in DB_COLUMNS | kural kolonu DB_COLUMNS'ta yok: {col}")
            if col in fill:
                raise ValueError(f"column in two gold rules | kolon iki kuralda: {col}")
            fill[col] = rule["value"]
        groups.append({"name": rule["name"], "value": rule["value"], "columns": list(rule["columns"])})
    drop = [d["column"] for d in raw["drop"]]
    bad = [c for c in drop if c not in names or c in fill]
    if bad:
        raise ValueError(f"bad dropped column | hatalı alınmayan kolon: {bad}")
    return {"fill": fill, "drop": drop, "groups": groups}


RULES = load_rules()
# EN: the gold car_listings columns (after id): DB_COLUMNS in order, without the dropped ones
# TR: gold car_listings kolonları (id'den sonra): DB_COLUMNS sırasıyla, alınmayanlar hariç
GOLD_COLUMNS = [(n, t) for n, t, _ in DB_COLUMNS if n not in RULES["drop"]]


def sql_literal(value):
    """EN: A rule value as SQL (false / 0). / TR: Kural değerinin SQL hâli (false / 0)."""
    return ("true" if value else "false") if isinstance(value, bool) else str(int(value))


def table_columns(con, table=TABLE):
    """
    EN: The column names of a table of a connected DB, in order ([] when there is no such table). table may be
        qualified with an attached catalog, e.g. "src.car_listings".
    TR: Bağlı bir DB'deki bir tablonun kolon adları, sırasıyla (tablo yoksa []). table ATTACH edilmiş bir
        katalogla nitelenebilir, ör. "src.car_listings".
    """
    try:
        return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    except duckdb.CatalogException:
        return []


def column_diff(have, want):
    """EN: A short "extra · missing · order" text for two column lists. / TR: İki kolon listesi için kısa fark metni."""
    extra, missing = sorted(set(have) - set(want)), sorted(set(want) - set(have))
    return (f"extra/fazla {extra[:5]} · missing/eksik {missing[:5]}"
            + (" · order differs | sıra farklı" if not extra and not missing else ""))


def check_input(con, table=f"src.{TABLE}"):
    """
    EN: Stops (ValueError) unless the input's car_listings has exactly id + DB_COLUMNS, in order — an
        old-contract DB (description_clean, no kb_paint_change_summary) or an unrelated file is refused.
    TR: Girdinin car_listings'i tam olarak id + DB_COLUMNS değilse (sırasıyla) durur (ValueError) — eski
        sözleşmeli DB (description_clean var, kb_paint_change_summary yok) ya da ilgisiz dosya reddedilir.
    """
    have, want = table_columns(con, table), ["id"] + [n for n, _, _ in DB_COLUMNS]
    if have != want:
        raise ValueError("input is not the semi-raw DB (rebuild it: python db/build_duckdb.py) | girdi yarı ham DB "
                         f"değil — {column_diff(have, want)}")


def contract_problems(con, table=TABLE):
    """
    EN: What breaks the gold (API) contract in a connected DB: the car_listings columns are not id + GOLD_COLUMNS,
        or a gold rule column still holds NULL. Used on the built file and by publish_data_to_s3.py.
        Returns: list of problems ([] = the contract holds).
    TR: Bağlı bir DB'de gold (API) sözleşmesini bozan şeyler: car_listings kolonları id + GOLD_COLUMNS değil ya da
        bir kural kolonunda hâlâ NULL var. Kurulan dosyada ve publish_data_to_s3.py'de kullanılır.
        Döndürür: sorun listesi ([] = sözleşme tutuyor).
    """
    have, want = table_columns(con, table), ["id"] + [n for n, _ in GOLD_COLUMNS]
    if have != want:
        return [f"columns are not the gold contract | kolonlar gold sözleşmesi değil — {column_diff(have, want)}"]
    nulls = con.execute(f"SELECT {', '.join(f'count(*) - count({c})' for c in RULES['fill'])} FROM {table}").fetchone()
    bad = [c for c, n in zip(RULES["fill"], nulls) if n]
    return [f"NULL left in {len(bad)} gold rule columns | {len(bad)} kural kolonunda NULL var: {bad[:5]}"] if bad else []


def write_gold(src_path, path):
    """
    EN: Writes the gold tables into a NEW DuckDB file at path from src_path (opened read-only).
        Returns: {"rows", "filled": {column: cells filled}, "tables": [copied table names]}.
    TR: src_path'ten (salt okunur açılır) path'teki YENİ bir DuckDB dosyasına gold tablolarını yazar.
        Döndürür: {"rows", "filled": {kolon: doldurulan hücre}, "tables": [kopyalanan tablolar]}.
    """
    con = duckdb.connect(str(path))
    try:
        con.execute(f"ATTACH '{Path(src_path).as_posix()}' AS src (READ_ONLY)")
        check_input(con)
        null_counts = ", ".join(f"count(*) - count({c})" for c in RULES["fill"])
        filled = dict(zip(RULES["fill"], con.execute(f"SELECT {null_counts} FROM src.{TABLE}").fetchone()))
        col_defs = ",\n    ".join(f"{n} {t}" for n, t in GOLD_COLUMNS)
        select = ", ".join(f"COALESCE({n}, {sql_literal(RULES['fill'][n])}) AS {n}" if n in RULES["fill"] else n
                           for n, _ in GOLD_COLUMNS)
        con.execute(f"CREATE TABLE {TABLE} (\n    id BIGINT PRIMARY KEY,\n    {col_defs}\n);")
        con.execute(f"INSERT INTO {TABLE} SELECT id, {select} FROM src.{TABLE} ORDER BY id;")
        present = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                             "WHERE table_catalog = 'src'").fetchall()}
        missing = [t for t in COPIED_TABLES if t not in present]
        if missing:
            raise ValueError(f"input lacks tables | girdide tablo yok: {missing}")
        tables = [t for t in COPIED_TABLES + OPTIONAL_TABLES if t in present]
        for t in tables:
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
        rows = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
        problems = contract_problems(con)              # the built file checks itself | kurulan dosya kendini sınar
        if problems:
            raise ValueError("; ".join(problems))
        con.execute("DETACH src")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return {"rows": int(rows), "filled": {k: int(v) for k, v in filled.items()}, "tables": tables}


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    """
    EN: Builds the gold DB at out_path from the semi-raw DB at in_path (steps in the module header).
        Returns: {"out", "rows", "columns", "filled", "tables"}.
    TR: in_path'teki yarı ham DB'den out_path'e gold DB'yi kurar (adımlar modül başlığında).
        Döndürür: {"out", "rows", "columns", "filled", "tables"}.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found — build it first: python db/build_duckdb.py | bulunamadı")
    if in_path.resolve() == out_path.resolve():
        raise ValueError("input and output are the same file | girdi ile çıktı aynı dosya")
    wal = Path(f"{out_path}.wal")
    if wal.exists():
        raise RuntimeError(f"{wal} exists: the old gold DB was not closed cleanly; open and close it once with "
                           f"DuckDB first | eski gold DB düzgün kapanmamış; önce DuckDB ile bir kez açıp kapatın")
    tmp = Path(f"{out_path}.tmp")
    tmp_wal = Path(f"{tmp}.wal")
    tmp.unlink(missing_ok=True)                        # a leftover of an earlier failed run | önceki düşen koşudan
    tmp_wal.unlink(missing_ok=True)
    try:
        result = write_gold(in_path, tmp)
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
    summary = {"out": out_path, "columns": len(GOLD_COLUMNS), **result}
    print(f"Written | yazıldı: {out_path}")
    print(f"  - {TABLE}: {summary['rows']} rows | satır, {summary['columns']} columns + id | kolon + id")
    for group in RULES["groups"]:
        cells = sum(summary["filled"][c] for c in group["columns"])
        print(f"  - {group['name']}: {cells} NULL cells → {sql_literal(group['value'])} | hücre")
    print(f"  - not taken | alınmadı: {', '.join(RULES['drop'])} · copied | kopyalandı: {', '.join(summary['tables'])}")
    return summary


def main(argv=None):
    """
    EN: Command line. Returns: exit code (0 = success, 1 = failure; the reason goes to stderr).
    TR: Komut satırı. Döndürür: çıkış kodu (0 = başarı, 1 = hata; sebep stderr'e yazılır).
    """
    ap = argparse.ArgumentParser(description="Build the gold (API) DB from the semi-raw DB | gold DB'yi kur")
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="semi-raw DB (default: data/cars.duckdb)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="gold DB (default: data/cars_gold.duckdb)")
    args = ap.parse_args(argv)
    try:
        build(Path(args.in_path), Path(args.out))
    except (RuntimeError, FileNotFoundError, ValueError, duckdb.Error) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
