"""
observed.py
EN: The register of observed values (db/observed_values.json) as tables for the internal tool's "Values" page: the
    raw fields (records that have them, distinct count, and their values or formats), the damage labels, the silver
    columns with few values, and series → models. Pure: no streamlit; the register is only read.
TR: Gözlenen değerler kaydı (db/observed_values.json), iç aracın "Değerler" sayfası için tablolar olarak: ham alanlar
    (onları taşıyan kayıt sayısı, farklı değer sayısı ve değerleri ya da biçimleri), hasar etiketleri, az değerli
    silver kolonları ve seri → modeller. Saf: streamlit yok; kayıt yalnız okunur.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTER_PATH = ROOT / "db" / "observed_values.json"


def load(path=REGISTER_PATH):
    """EN: The register. / TR: Kayıt."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def kind(entry):
    """
    EN: How a raw field is kept: "değerler" (values), "biçimler" (formats) or "yalnız sayı" (count only).
    TR: Bir ham alanın nasıl tutulduğu: "değerler", "biçimler" ya da "yalnız sayı".
    """
    if "shapes" in entry:
        return "biçimler"
    return "değerler" if "values" in entry else "yalnız sayı"


def fields_table(register):
    """
    EN: One row per raw field: name, records that have it, missing records, distinct values, how it is kept.
    TR: Ham alan başına bir satır: ad, onu taşıyan kayıt, eksik kayıt, farklı değer sayısı, nasıl tutulduğu.
    """
    raw = register["raw"]
    return pd.DataFrame([{"alan": name, "kayıt": e["present"], "eksik": raw["records"] - e["present"],
                          "farklı": e.get("distinct"), "tutulan": kind(e)} for name, e in raw["fields"].items()])


def pairs_table(pairs, label):
    """
    EN: [[value, count]] as a frame with a share column; values as text (None → "(boş)").
    TR: [[değer, sayı]] çiftleri, pay kolonuyla bir çerçeve; değerler metin (None → "(boş)").
    """
    total = sum(n for _, n in pairs) or 1
    return pd.DataFrame({label: ["(boş)" if v is None else str(v) for v, _ in pairs],
                         "sayı": [n for _, n in pairs], "%": [round(100 * n / total, 2) for _, n in pairs]})


def field_tables(register, field):
    """
    EN: The tables of one raw field: {"değerler": frame, "biçimler": frame} for what it keeps (empty dict when it is
        counted only).
    TR: Tek bir ham alanın tabloları: tuttukları için {"değerler": çerçeve, "biçimler": çerçeve} (yalnız sayılıyorsa boş
        sözlük).
    """
    entry = register["raw"]["fields"][field]
    out = {}
    if "values" in entry:
        out["değerler"] = pairs_table(entry["values"], "değer")
    if "shapes" in entry:
        out["biçimler"] = pairs_table(entry["shapes"], "biçim")
    return out


def silver_table(register, column):
    """EN: The values of a silver column. / TR: Bir silver kolonunun değerleri."""
    return pairs_table(register["silver"]["columns"][column]["values"], "değer")


def series_table(register):
    """EN: One row per (series, model, count). / TR: (seri, model, sayı) başına bir satır."""
    return pd.DataFrame([{"seri": s, "model": m, "sayı": n}
                         for s, models in register["silver"]["series_models"].items() for m, n in models])
