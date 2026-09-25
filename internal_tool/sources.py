"""
sources.py
EN: The three data sources of the internal tool's explorer, loaded read-only into pandas:
      raw     data/raw/<brand>/<snapshot>/details.jsonl — every scraped record as it is (blue plates, missing plate
              fields and failed pages included). Each raw field becomes a column under its own name with its
              value untouched; Hasar_Listesi ("Part: State" items) opens into "Hasar - <Part>" columns; the brand
              folder and snapshot folder are added; ad_id = the digits of "KısaBilgi - İlan No". A field whose values
              are single numbers with a unit ("400.000 TL", "12,1 sn") also gets a parsed "<field> (sayı)" column so
              it can be filtered by range. The ad text (Aciklama_HTML) is left out and read only on demand.
      silver  data/cars.duckdb, table car_listings (description_text on demand, scraped_at as text).
      gold    data/cars_gold.duckdb, the same.
    DuckDB is opened read_only, used and closed at once — no connection is kept, so a DB can be rebuilt while the
    tool is open (on Windows an open handle makes os.replace fail). The raw files are only read. Pure: no streamlit.
TR: İç aracın veri gezgininin üç kaynağı, pandas'a salt okunur yüklenir:
      raw     data/raw/<marka>/<tarama>/details.jsonl — kazınan her kayıt olduğu gibi (mavi plakalar, eksik plaka
              alanları ve başarısız sayfalar dahil). Her ham alan kendi adıyla bir kolon olur, değeri dokunulmadan;
              Hasar_Listesi ("Parça: Durum" öğeleri) "Hasar - <Parça>" kolonlarına açılır; marka ve tarama klasörü
              eklenir; ad_id = "KısaBilgi - İlan No"nun rakamları. Değerleri birimli tek sayı olan alana ("400.000 TL",
              "12,1 sn") ayrıca ayrıştırılmış "<alan> (sayı)" kolonu eklenir, böylece aralıkla filtrelenebilir. İlan
              metni (Aciklama_HTML) dışarıda kalır, yalnız istenince okunur.
      silver  data/cars.duckdb, car_listings tablosu (description_text istenince, scraped_at metin olarak).
      gold    data/cars_gold.duckdb, aynısı.
    DuckDB read_only açılır, kullanılır ve hemen kapanır — bağlantı tutulmaz; böylece araç açıkken DB yeniden
    kurulabilir (Windows'ta açık handle os.replace'i bozar). Ham dosyalar yalnız okunur. Saf: streamlit yok.
"""
import html
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"raw": "Ham (JSONL)", "silver": "Silver (cars.duckdb)", "gold": "Gold (cars_gold.duckdb)"}
DB_FILES = {"silver": "cars.duckdb", "gold": "cars_gold.duckdb"}
TABLE = "car_listings"
RAW_TEXT = "Aciklama_HTML"
DB_TEXT = "description_text"
DAMAGE_PREFIX = "Hasar - "
BRAND_DIR, SNAPSHOT_DIR, RAW_REF = "_marka_klasörü", "_tarama", "_kaynak"
NUMBER_SUFFIX = " (sayı)"
# EN: one number (Turkish style: "." thousands, "," decimals) and at most one unit word
# TR: tek sayı (Türkçe biçim: "." binlik, "," ondalık) ve en çok bir birim sözcüğü
NUMBER = re.compile(r"^\s*(-?(?:\d{1,3}(?:\.\d{3})+|\d+))(?:,(\d+))?(?:\s*[^\W\d_][\w/%.]*)?\s*$")
NUMERIC_SHARE = 0.9


def data_files(source, data_dir):
    """
    EN: The files a source reads, sorted (raw: every details.jsonl; silver/gold: the DB file).
    TR: Bir kaynağın okuduğu dosyalar, sıralı (raw: her details.jsonl; silver/gold: DB dosyası).
    """
    data_dir = Path(data_dir)
    if source == "raw":
        return sorted(data_dir.glob("raw/*/*/details.jsonl"))
    if source in DB_FILES:
        return [data_dir / DB_FILES[source]] if (data_dir / DB_FILES[source]).exists() else []
    raise ValueError(f"unknown source | bilinmeyen kaynak: {source}")


def fingerprint(files):
    """
    EN: (path, mtime_ns, size) of each file: a cache key that changes when a file is rebuilt.
    TR: Her dosyanın (yol, mtime_ns, boyut) üçlüsü: dosya yeniden kurulunca değişen önbellek anahtarı.
    """
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in map(Path, files))


def parse_number(value):
    """
    EN: A single number with an optional unit as float ("400.000 TL" → 400000, "12,1 sn" → 12.1); None otherwise
        ("1401 - 1600 cm3", "205/55 R16", dates, text).
    TR: İsteğe bağlı birimli tek sayı, float olarak ("400.000 TL" → 400000, "12,1 sn" → 12.1); değilse None
        ("1401 - 1600 cm3", "205/55 R16", tarihler, metin).
    """
    if not isinstance(value, str):
        return None
    m = NUMBER.match(value)
    if not m:
        return None
    return float(m.group(1).replace(".", "") + ("." + m.group(2) if m.group(2) else ""))


def ad_id_of(value):
    """EN: The digits of the raw ad number, as int; None if none. / TR: Ham ilan no'nun rakamları, int; yoksa None."""
    digits = re.sub(r"\D", "", value) if isinstance(value, str) else ""
    return int(digits) if digits else None


def flatten(record):
    """
    EN: One raw record as a flat row: fields kept as they are, Hasar_Listesi opened into "Hasar - <Part>" columns,
        other lists/dicts as JSON text, the ad text left out.
    TR: Tek ham kayıt, düz satır olarak: alanlar olduğu gibi, Hasar_Listesi "Hasar - <Parça>" kolonlarına açılır,
        öteki liste/sözlükler JSON metni, ilan metni dışarıda.
    """
    row = {}
    for key, value in record.items():
        if key == RAW_TEXT:
            continue
        if key == "Hasar_Listesi" and isinstance(value, list):
            for item in value:
                part, sep, state = str(item).partition(":")
                if sep:
                    row[DAMAGE_PREFIX + part.strip()] = state.strip()
            continue
        row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
    return row


def read_raw_lines(files):
    """
    EN: Yields (file index, byte offset, parsed record or None for a broken line) over the raw files, read-only.
    TR: Ham dosyalar üzerinde (dosya sırası, bayt konumu, ayrıştırılmış kayıt ya da bozuk satırsa None) üretir;
        salt okunur.
    """
    for i, path in enumerate(files):
        offset = 0
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        record = None
                    yield i, offset, record if isinstance(record, dict) else None
                offset += len(line)


def add_number_columns(df, share=NUMERIC_SHARE):
    """
    EN: Adds "<field> (sayı)" (Float64) after every text column whose non-empty values parse as numbers at least
        `share` of the time. Returns: a new DataFrame.
    TR: Boş olmayan değerlerinin en az `share` oranı sayı olarak ayrışan her metin kolonunun arkasına
        "<alan> (sayı)" (Float64) ekler. Döndürür: yeni DataFrame.
    """
    out = {}
    for col in df.columns:
        out[col] = df[col]
        if col.startswith("_") or col == "ad_id" or df[col].dtype.kind in "biuf":
            continue
        values = df[col].dropna()
        values = values[values.map(lambda v: isinstance(v, str) and v.strip() != "")]
        if values.empty:
            continue
        parsed = values.map(parse_number)
        if parsed.notna().mean() >= share:
            out[col + NUMBER_SUFFIX] = df[col].map(parse_number).astype("Float64")
    return pd.DataFrame(out)


def load_raw(files):
    """
    EN: The raw records as a DataFrame (see the module docstring) and load stats {"files", "records", "broken"}.
        Column RAW_REF keeps "<file index>:<byte offset>" so one record can be read again whole (read_raw_record).
    TR: Ham kayıtlar DataFrame olarak (modül açıklamasına bakın) ve yükleme bilgisi {"files", "records", "broken"}.
        RAW_REF kolonu "<dosya sırası>:<bayt konumu>" tutar; böylece tek kayıt bütünüyle yeniden okunabilir
        (read_raw_record).
    """
    files = [Path(f) for f in files]
    rows, broken = [], 0
    for i, offset, record in read_raw_lines(files):
        if record is None:
            broken += 1
            continue
        row = {"ad_id": ad_id_of(record.get("KısaBilgi - İlan No")), BRAND_DIR: files[i].parent.parent.name,
               SNAPSHOT_DIR: files[i].parent.name, RAW_REF: f"{i}:{offset}"}
        row.update(flatten(record))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ad_id"] = df["ad_id"].astype("Int64")
        df = add_number_columns(df)
    return df, {"files": len(files), "records": len(rows), "broken": broken}


def read_raw_record(files, ref):
    """
    EN: One raw record, whole (ad text included), from its RAW_REF "<file index>:<byte offset>".
    TR: Tek ham kayıt, bütünüyle (ilan metni dahil), RAW_REF "<dosya sırası>:<bayt konumu>" değerinden.
    """
    i, offset = (int(x) for x in ref.split(":"))
    with open(files[i], "rb") as fh:
        fh.seek(offset)
        return json.loads(fh.readline())


def raw_texts(files):
    """
    EN: The ad text of every raw record as plain text, in load_raw's row order (broken lines skipped the same way).
    TR: Her ham kaydın ilan metni düz metin olarak, load_raw'ın satır sırasıyla (bozuk satırlar aynı biçimde atlanır).
    """
    return pd.Series([html_to_text(r.get(RAW_TEXT)) for _, _, r in read_raw_lines([Path(f) for f in files])
                      if r is not None], dtype="object")


def html_to_text(value):
    """EN: HTML → plain text (tags dropped, entities decoded). / TR: HTML → düz metin (etiketler atılır)."""
    if not isinstance(value, str):
        return value
    text = re.sub(r"<(br|/p|/div|/h\d|/li)\s*/?>", "\n", value, flags=re.I)
    return re.sub(r"[ \t\xa0]+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def load_db(path, exclude=(DB_TEXT,)):
    """
    EN: car_listings of a DuckDB file, ordered by id, without the excluded columns; TIMESTAMPTZ columns come as
        text (pandas would need pytz). The connection is read-only and closed before returning.
    TR: Bir DuckDB dosyasının car_listings'i, id sırasıyla, dışlanan kolonlar olmadan; TIMESTAMPTZ kolonları metin
        olarak gelir (pandas pytz isterdi). Bağlantı salt okunur ve dönmeden kapanır.
    """
    con = duckdb.connect(str(path), read_only=True)
    try:
        cols = con.execute(f"DESCRIBE {TABLE}").fetchall()
        select = ", ".join(f'CAST("{name}" AS VARCHAR) AS "{name}"' if "TIME ZONE" in dtype else f'"{name}"'
                           for name, dtype, *_ in cols if name not in exclude)
        return con.execute(f"SELECT {select} FROM {TABLE} ORDER BY id").df()
    finally:
        con.close()


def db_texts(path):
    """EN: description_text in id order (read-only). / TR: id sırasıyla description_text (salt okunur)."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT {DB_TEXT} FROM {TABLE} ORDER BY id").df()[DB_TEXT]
    finally:
        con.close()


def load(source, data_dir):
    """
    EN: A source's DataFrame and stats. Raises FileNotFoundError when its files are missing.
    TR: Bir kaynağın DataFrame'i ve bilgisi. Dosyaları yoksa FileNotFoundError yükseltir.
    """
    files = data_files(source, data_dir)
    if not files:
        raise FileNotFoundError(f"no data for | veri yok: {source} ({data_dir})")
    if source == "raw":
        return load_raw(files)
    df = load_db(files[0])
    return df, {"files": 1, "records": len(df), "broken": 0}


def texts(source, data_dir):
    """EN: The ad texts of a source, row-aligned with load(). / TR: Bir kaynağın ilan metinleri, load() ile hizalı."""
    files = data_files(source, data_dir)
    return raw_texts(files) if source == "raw" else db_texts(files[0])
