"""
observed_values.py
EN: The register of what the data actually holds (db/observed_values.json), so that no code assumes a case the data
    never showed and no unseen value slips through silently ("don't assume cases that never happened").
      raw     every raw field: how many records have it; for a field with at most MAX_VALUES distinct values (or one
              in ALWAYS_VALUES, whose values the parsers are tested on) the values with counts; for a field with more,
              its formats ("shapes": every number not glued to a letter becomes "#": "1.250.000 TL" → "# TL",
              "1401 - 1600 cm3" → "# - # cm3"); identity and free-text fields (COUNT_ONLY) only by count;
              the Hasar_Listesi part and state labels.
      silver  every car_listings column with at most MAX_VALUES distinct values, with counts, and the series →
              model pairs.
    unseen() compares a fresh scan of raw files with the register: a new field, a value or format the register has
    not seen, or an unknown damage label is a problem, reported with the first file:line (never an ad id). The DB
    build stops on any problem before writing (build_duckdb.build); updating the register is a deliberate step
    (python tools/observed_values.py), and tests/db/test_observed_values.py then checks that the code handles every
    registered value. Data is only read.
TR: Verinin gerçekte ne tuttuğunun kaydı (db/observed_values.json); böylece hiçbir kod verinin hiç göstermediği bir
    durumu varsaymaz ve görülmemiş hiçbir değer sessizce geçmez ("yaşanmamış case'leri varsayma").
      raw     her ham alan: kaç kayıtta var; en çok MAX_VALUES farklı değerli alanda (ya da ayrıştırıcıların
              değerleriyle sınandığı ALWAYS_VALUES'taki alanda) değerler ve sayıları; daha çok değerli alanda
              biçimleri ("shapes": harfe yapışık olmayan her sayı "#" olur: "1.250.000 TL" → "# TL", "1401 - 1600
              cm3" → "# - # cm3"); kimlik ve serbest metin alanları (COUNT_ONLY) yalnız sayıyla; Hasar_Listesi
              parça ve durum etiketleri.
      silver  en çok MAX_VALUES farklı değerli her car_listings kolonu, sayılarıyla, ve seri → model çiftleri.
    unseen() ham dosyaların taze taramasını kayıtla karşılaştırır: yeni bir alan, kaydın görmediği bir değer ya da
    biçim, bilinmeyen bir hasar etiketi sorundur ve ilk dosya:satır ile bildirilir (asla ilan kimliği değil). DB
    kurulumu herhangi bir sorunda yazmadan durur (build_duckdb.build); kaydı güncellemek bilinçli bir adımdır
    (python tools/observed_values.py) ve tests/db/test_observed_values.py sonra kodun kayıttaki her değeri ele
    aldığını sınar. Veri yalnız okunur.
"""
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "db" / "observed_values.json"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "cars.duckdb"
TABLE = "car_listings"
MAX_VALUES = 60
DAMAGE_FIELD = "Hasar_Listesi"
# EN: fields whose values are kept whatever their count: the parsers are tested on each of them
# TR: sayıları ne olursa olsun değerleri tutulan alanlar: ayrıştırıcılar her biriyle sınanır
ALWAYS_VALUES = ("KısaBilgi - İlan Tarihi", "KısaBilgi - Motor Hacmi", "KısaBilgi - Motor Gücü",
                 "Genel Bakış - Üretim Yılı (İlk/Son)")
# EN: identity and free text: counted only | TR: kimlik ve serbest metin: yalnız sayılır
COUNT_ONLY = ("url", "scraped_at", "KısaBilgi - İlan No", "Ilan_Basligi", "Aciklama_HTML", "Konum", "EIDS_Model",
              "KısaBilgi - Model", "error")
SILVER_COUNT_ONLY = ("id", "ad_id", "url", "ad_title", "description_text", "location", "eids_model", "scraped_at",
                     "model")
NUMBER = re.compile(r"(?<![^\W_])\d[\d.,]*")     # not after a letter or digit | harf ya da rakamdan sonra değil
ABOUT = {
    "en": "What the data actually holds, written by python tools/observed_values.py from data/raw and data/cars.duckdb "
          "(read-only). raw.fields: per raw field, records that have it and either its values ([value, count]) or, "
          "above 60 distinct values, its formats (shapes: numbers → #); identity/free-text fields only counted. "
          "raw.damage: Hasar_Listesi labels. silver.columns: car_listings columns with at most 60 distinct values; "
          "silver.series_models: series → [model, count]. The DB build stops on anything not in here "
          "(db/build_duckdb.py), and tests check that the code handles every value here and expects nothing that "
          "is not (tests/db/test_observed_values.py, tests/analysis/test_observed_values_analysis.py).",
    "tr": "Verinin gerçekte ne tuttuğu; python tools/observed_values.py data/raw ve data/cars.duckdb'den yazar (salt "
          "okunur). raw.fields: her ham alan için onu taşıyan kayıt sayısı ve ya değerleri ([değer, sayı]) ya da 60 "
          "farklı değerin üstünde biçimleri (shapes: sayılar → #); kimlik/serbest metin alanları yalnız sayılır. "
          "raw.damage: Hasar_Listesi etiketleri. silver.columns: en çok 60 farklı değerli car_listings kolonları; "
          "silver.series_models: seri → [model, sayı]. DB kurulumu burada olmayan her şeyde durur "
          "(db/build_duckdb.py); testler kodun buradaki her değeri ele aldığını ve burada olmayan hiçbir şeyi "
          "beklemediğini sınar (tests/db/test_observed_values.py, tests/analysis/test_observed_values_analysis.py)."}


def shape(value):
    """
    EN: The format of a value: numbers not glued to a letter become "#" ("365.000 km" → "# km", "cm3" and "R16"
        stay); non-text values by their JSON form ("#", "true", "null").
    TR: Bir değerin biçimi: harfe yapışık olmayan sayılar "#" olur ("365.000 km" → "# km", "cm3" ve "R16" kalır);
        metin olmayan değerler JSON biçimiyle ("#", "true", "null").
    """
    return NUMBER.sub("#", value if isinstance(value, str) else json.dumps(value))


def value_key(value):
    """EN: A hashable, JSON-safe form of a value. / TR: Değerin hash'lenebilir, JSON'a uygun biçimi."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def pairs(counter):
    """EN: [[value, count]] by count, then value. / TR: Sayıya, sonra değere göre [[değer, sayı]]."""
    return [[v, n] for v, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def raw_files(raw_dir=RAW_DIR, brands=None):
    """
    EN: The raw files, sorted: <raw_dir>/<brand>/<snapshot>/details.jsonl (only the given brands if any).
    TR: Ham dosyalar, sıralı: <raw_dir>/<marka>/<tarama>/details.jsonl (verildiyse yalnız o markalar).
    """
    files = sorted(Path(raw_dir).glob("*/*/details.jsonl"))
    return [f for f in files if brands is None or f.parent.parent.name in brands]


def where(path, line_no):
    """EN: "brand/snapshot/details.jsonl:line". / TR: "marka/tarama/details.jsonl:satır"."""
    return f"{'/'.join(Path(path).parts[-3:])}:{line_no}"


def file_records(files):
    """
    EN: Yields (file:line, record or None for a broken line) over raw files, read-only; blank lines skipped.
    TR: Ham dosyalar üzerinde (dosya:satır, kayıt ya da bozuk satırsa None) üretir; salt okunur; boş satırlar atlanır.
    """
    for path in files:
        with open(path, "rb") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    record = None
                yield where(path, line_no), record if isinstance(record, dict) else None


def scan_raw(files):
    """
    EN: Scans raw files (scan_records over file_records). Returns: (register part, detail).
    TR: Ham dosyaları tarar (file_records üzerinde scan_records). Döndürür: (kayıt parçası, ayrıntı).
    """
    part, detail = scan_records(file_records(files))
    part["files"] = len(files)
    return part, detail


def scan_records(items):
    """
    EN: Scans (where, record) pairs — raw file lines, or records made in memory (test fixtures). Returns: (register
        part, detail) — the part as written to the register; detail keeps the full value counters and each value's
        first place for unseen(). A None record counts as a broken line.
    TR: (nerede, kayıt) çiftlerini tarar — ham dosya satırları ya da bellekte kurulan kayıtlar (test fixture'ları).
        Döndürür: (kayıt parçası, ayrıntı) — parça kayda yazıldığı gibi; ayrıntı unseen() için tam değer sayaçlarını
        ve her değerin ilk yerini tutar. None kayıt bozuk satır sayılır.
    """
    present, values = Counter(), defaultdict(Counter)
    parts, states, malformed = Counter(), Counter(), Counter()
    first, records, broken = {}, 0, 0
    for place, record in items:
        if record is None:
            broken += 1
            continue
        records += 1
        for field, value in record.items():
            present[field] += 1
            first.setdefault(("field", field), place)
            if field == DAMAGE_FIELD:
                for item in value if isinstance(value, list) else [value]:
                    part, sep, state = str(item).partition(":")
                    if not sep:
                        malformed[str(item)] += 1
                        first.setdefault(("damage.malformed", str(item)), place)
                        continue
                    for kind, label, counter in (("damage.part", part.strip(), parts),
                                                 ("damage.state", state.strip(), states)):
                        counter[label] += 1
                        first.setdefault((kind, label), place)
            elif field not in COUNT_ONLY:
                key = value_key(value)
                values[field][key] += 1
                first.setdefault((field, key), place)
    fields = {}
    for field in sorted(present):
        entry = {"present": present[field]}
        if field not in COUNT_ONLY and field != DAMAGE_FIELD:
            counts = values[field]
            entry["distinct"] = len(counts)
            if len(counts) <= MAX_VALUES or field in ALWAYS_VALUES:
                entry["values"] = pairs(counts)
            if len(counts) > MAX_VALUES:
                shapes = Counter()
                for v, n in counts.items():
                    shapes[shape(v)] += n
                entry["shapes"] = pairs(shapes)
        fields[field] = entry
    part = {"files": 0, "records": records, "broken_lines": broken, "fields": fields,
            "damage": {"parts": pairs(parts), "states": pairs(states), "malformed": pairs(malformed)}}
    return part, {"values": values, "parts": parts, "states": states, "malformed": malformed, "first": first}


def plain(value):
    """EN: A DB value as JSON-safe Python (dates as ISO text). / TR: DB değeri JSON'a uygun Python (tarih ISO metin)."""
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def scan_silver(db_path=DB_PATH):
    """
    EN: The silver part: rows, the columns with at most MAX_VALUES distinct values (with counts, NULL included) and
        series → [model, count]. The DB is opened read-only and closed.
    TR: Silver parçası: satır sayısı, en çok MAX_VALUES farklı değerli kolonlar (sayılarıyla, NULL dahil) ve seri →
        [model, sayı]. DB salt okunur açılır ve kapanır.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        columns = {}
        for name, dtype, *_ in con.execute(f"DESCRIBE {TABLE}").fetchall():
            if name in SILVER_COUNT_ONLY or "TIME ZONE" in dtype:
                continue
            distinct = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM {TABLE}').fetchone()[0]
            if distinct <= MAX_VALUES:
                counts = Counter({plain(v): n for v, n in
                                  con.execute(f'SELECT "{name}", COUNT(*) FROM {TABLE} GROUP BY 1').fetchall()})
                columns[name] = {"distinct": int(distinct), "values": pairs(counts)}
        series_models = defaultdict(Counter)
        for series, model, n in con.execute(f"SELECT series, model, COUNT(*) FROM {TABLE} GROUP BY 1, 2").fetchall():
            series_models[series][model] += n
    finally:
        con.close()
    return {"rows": int(rows), "columns": columns,
            "series_models": {s: pairs(m) for s, m in sorted(series_models.items(), key=lambda kv: str(kv[0]))}}


def collect(raw_dir=RAW_DIR, db_path=DB_PATH):
    """EN: The whole register (without writing it). / TR: Kaydın tamamı (yazmadan)."""
    return {"_about": ABOUT, "_meta": {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds")},
            "raw": scan_raw(raw_files(raw_dir))[0], "silver": scan_silver(db_path)}


def load(path=REGISTRY_PATH):
    """EN: The register file. / TR: Kayıt dosyası."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(register, path=REGISTRY_PATH):
    """EN: Writes the register (LF, UTF-8, indented). / TR: Kaydı yazar (LF, UTF-8, girintili)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(register, ensure_ascii=False, indent=1) + "\n")


def differences(a, b):
    """
    EN: The top-level parts where two registers differ, the generation stamp left out.
    TR: İki kaydın farklı olduğu üst düzey parçalar; üretim damgası hariç.
    """
    out = []
    for section in ("raw", "silver"):
        for key in sorted(set(a.get(section, {})) | set(b.get(section, {}))):
            if a.get(section, {}).get(key) != b.get(section, {}).get(key):
                out.append(f"{section}.{key}")
    return out


def unseen(detail, register):
    """
    EN: What a raw scan has that the register has not seen: new fields, values of a value-listed field, formats of a
        shape-listed field, damage labels and malformed damage items. Returns: ["<field>: … · N kayıt · ilk: file:line"].
    TR: Ham taramada olup kaydın görmediği şeyler: yeni alanlar, değerleri listelenen alanda değerler, biçimleri
        listelenen alanda biçimler, hasar etiketleri ve bozuk hasar öğeleri. Döndürür:
        ["<alan>: … · N kayıt · ilk: dosya:satır"].
    """
    known = register["raw"]["fields"]
    problems = []
    for field in sorted({f for (kind, f) in detail["first"] if kind == "field"} - set(known)):
        problems.append(f"{field}: yeni alan · new field · ilk: {detail['first'][('field', field)]}")
    for field, counts in sorted(detail["values"].items()):
        entry = known.get(field)
        if entry is None:
            continue
        if "shapes" in entry:
            seen = {s for s, _ in entry["shapes"]}
            new = Counter()
            where_ = {}
            for v, n in counts.items():
                if shape(v) not in seen:
                    new[shape(v)] += n
                    where_.setdefault(shape(v), detail["first"][(field, v)])
            problems += [f"{field}: görülmemiş biçim · unseen format {s!r} · {n} kayıt · ilk: {where_[s]}"
                         for s, n in sorted(new.items())]
        else:
            seen = {value_key(v) for v, _ in entry.get("values", [])}
            problems += [f"{field}: görülmemiş değer · unseen value {v!r} · {n} kayıt · ilk: {detail['first'][(field, v)]}"
                         for v, n in sorted(counts.items(), key=lambda kv: str(kv[0])) if v not in seen]
    damage = register["raw"]["damage"]
    for kind, key in (("damage.part", "parts"), ("damage.state", "states")):
        seen = {v for v, _ in damage[key]}
        counter = detail["parts" if key == "parts" else "states"]
        problems += [f"{DAMAGE_FIELD}: görülmemiş {'parça' if key == 'parts' else 'durum'} · unseen {key[:-1]} {v!r} · "
                     f"{n} kayıt · ilk: {detail['first'][(kind, v)]}" for v, n in sorted(counter.items()) if v not in seen]
    problems += [f"{DAMAGE_FIELD}: bozuk öğe · malformed item {v!r} · {n} · ilk: {detail['first'][('damage.malformed', v)]}"
                 for v, n in sorted(detail["malformed"].items())]
    return problems
