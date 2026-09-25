"""
filters.py
EN: The explorer's condition engine, on any column of any source. A column's kind (bool, number, date, category,
    text) decides which operators it offers; a condition is (column, operator, values, enabled); the enabled
    conditions are joined with AND, and several values of one column give OR through "in". Empty means NULL/NaN or a
    blank string. Text search ("contains") ignores case and Turkish letters (ı/i, ş/s, ğ/g, ü/u, ö/o, ç/c), so
    "plakalı" finds "PLAKALIDIR". Also: the row sets (all rows, latest row per ad, the analysis set, one snapshot)
    and a readable summary of the conditions. Pure: no streamlit; the frames are never modified, masks are returned.
TR: Gezginin koşul motoru, her kaynağın her kolonunda. Kolonun türü (mantıksal, sayı, tarih, kategori, metin) hangi
    operatörleri sunduğunu belirler; koşul (kolon, operatör, değerler, etkin) dörtlüsüdür; etkin koşullar VE ile
    birleşir, bir kolonun birden çok değeri "şunlardan biri" ile VEYA verir. Boş, NULL/NaN ya da boş metin demektir.
    Metin araması ("içerir") büyük/küçük harfe ve Türkçe harflere (ı/i, ş/s, ğ/g, ü/u, ö/o, ç/c) duyarsızdır; "plakalı"
    "PLAKALIDIR"ı bulur. Ayrıca: satır kümeleri (bütün satırlar, ilan başına son satır, analiz kümesi, tek tarama) ve
    koşulların okunur özeti. Saf: streamlit yok; çerçeveler değiştirilmez, maske döndürülür.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_CATEGORIES = 60
MAX_PICK = 10000
# EN: operator → label shown in the page | TR: operatör → sayfada görünen ad
OP_LABELS = {"in": "şunlardan biri", "not_in": "hiçbiri", "between": "aralıkta", "contains": "içerir",
             "not_contains": "içermez", "equals": "eşittir", "regex": "regex", "is_true": "evet", "is_false": "hayır",
             "is_null": "boş", "not_null": "dolu"}
OPS = {"bool": ("is_true", "is_false", "is_null", "not_null"),
       "number": ("between", "in", "not_in", "is_null", "not_null"),
       "date": ("between", "in", "not_in", "is_null", "not_null"),
       "category": ("in", "not_in", "is_null", "not_null"),
       "text": ("contains", "not_contains", "equals", "regex", "is_null", "not_null")}
NULL = "(boş)"
FOLD = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "â": "a", "î": "i", "û": "u"})
TR_PLATE = "(TR) Türkiye"


@dataclass
class Condition:
    """
    EN: One filter condition. values: "in"/"not_in" → the chosen values (NULL marks "empty"); "between" → (low,
        high), None for an open end; text operators → (pattern,); the others → ().
    TR: Tek filtre koşulu. values: "in"/"not_in" → seçilen değerler (NULL "boş" demek); "between" → (alt, üst),
        açık uç için None; metin operatörleri → (desen,); ötekiler → ().
    """
    column: str
    op: str
    values: tuple = field(default_factory=tuple)
    enabled: bool = True


def column_kind(series, max_categories=MAX_CATEGORIES):
    """
    EN: "bool", "number", "date", "category" (few distinct values) or "text" for a column.
    TR: Bir kolon için "bool", "number", "date", "category" (az sayıda farklı değer) ya da "text".
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "date"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "number"
    values = series.dropna()
    if values.map(lambda v: isinstance(v, bool)).all() and len(values):
        return "bool"
    return "category" if values.astype(str).nunique() <= max_categories else "text"


def ops_for(series, kind):
    """
    EN: The operators a column offers: those of its kind, plus "in"/"not_in" for a text column with at most MAX_PICK
        distinct values (pick models, locations… from a list).
    TR: Bir kolonun sunduğu operatörler: türününkiler, ayrıca en çok MAX_PICK farklı değerli metin kolonu için
        "in"/"not_in" (model, konum… listeden seçilir).
    """
    ops = OPS[kind]
    if kind == "text" and series.dropna().astype(str).nunique() <= MAX_PICK:
        ops = ops[:4] + ("in", "not_in") + ops[4:]
    return ops


def fold(text):
    """
    EN: Lower case without Turkish letters and with every run of white space (non-breaking too) as one space, for
        a letter-blind search ("YABANCIDAN  YABANCIYA" → "yabancidan yabanciya").
    TR: Türkçe harfsiz küçük harf; her boşluk dizisi (bölünmez boşluk dahil) tek boşluk; harf duyarsız arama için
        ("YABANCIDAN  YABANCIYA" → "yabancidan yabanciya").
    """
    return " ".join(text.replace("İ", "i").replace("I", "ı").lower().translate(FOLD).split())


def fold_series(series):
    """EN: fold() over a column (NaN kept). / TR: Bir kolon üzerinde fold() (NaN korunur)."""
    return series.map(lambda v: fold(str(v)) if isinstance(v, str) else v)


def is_empty(series):
    """EN: NULL/NaN or a blank string. / TR: NULL/NaN ya da boş metin."""
    empty = np.array(series.isna().to_numpy(dtype=bool, na_value=True), dtype=bool)
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        empty |= series.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(dtype=bool)
    return empty


def _bool_array(values):
    """EN: A boolean numpy array; NA counts as False. / TR: Mantıksal numpy dizisi; NA False sayılır."""
    return np.array(pd.array(values, dtype="boolean").to_numpy(dtype=bool, na_value=False), dtype=bool)


def condition_mask(df, cond, folded=None):
    """
    EN: The rows a condition keeps (numpy bool array). Raises KeyError when the column is not in df.
        folded: an optional folded copy of the column (fold_series), reused for text search.
    TR: Bir koşulun tuttuğu satırlar (numpy bool dizisi). Kolon df'te yoksa KeyError yükseltir.
        folded: kolonun isteğe bağlı fold_series kopyası; metin aramasında yeniden kullanılır.
    """
    s = df[cond.column]
    op, vals = cond.op, tuple(cond.values)
    if op == "is_null":
        return is_empty(s)
    if op == "not_null":
        return ~is_empty(s)
    if op == "is_true":
        return _bool_array(s.map(lambda v: v is True or v == 1 if not pd.isna(v) else False))
    if op == "is_false":
        return _bool_array(s.map(lambda v: v is False or v == 0 if not pd.isna(v) else False))
    if op in ("in", "not_in"):
        chosen = [v for v in vals if v != NULL]
        keep = _bool_array(s.isin(chosen)) | (is_empty(s) if NULL in vals else False)
        return keep if op == "in" else ~keep
    if op == "between":
        low, high = (list(vals) + [None, None])[:2]
        keep = np.ones(len(s), dtype=bool)
        if low is not None:
            keep &= _bool_array(s >= low)
        if high is not None:
            keep &= _bool_array(s <= high)
        return keep & ~s.isna().to_numpy(dtype=bool, na_value=True)
    pattern = vals[0] if vals else ""
    if op == "equals":
        return _bool_array(s.astype("object").map(lambda v: isinstance(v, str) and v == pattern))
    if op == "regex":
        rx = re.compile(pattern, re.I)
        return _bool_array(s.map(lambda v: isinstance(v, str) and bool(rx.search(v))))
    if op in ("contains", "not_contains"):
        target = folded if folded is not None else fold_series(s)
        needle = fold(pattern)
        hit = _bool_array(target.map(lambda v: isinstance(v, str) and needle in v))
        return hit if op == "contains" else ~hit
    raise ValueError(f"unknown operator | bilinmeyen operatör: {op}")


def apply(df, conditions, base=None, folded=None):
    """
    EN: Joins the enabled conditions with AND over an optional base mask. folded: {column: folded series} for text
        search. Returns: (mask, steps, missing) — steps = [(condition, rows left after it)], missing = enabled
        conditions whose column this source does not have (they are skipped, not applied).
    TR: Etkin koşulları isteğe bağlı bir taban maskesi üzerinde VE ile birleştirir. folded: metin araması için
        {kolon: katlanmış seri}. Döndürür: (maske, adımlar, eksik) — adımlar = [(koşul, sonrasında kalan satır)],
        eksik = kolonu bu kaynakta olmayan etkin koşullar (atlanır, uygulanmaz).
    """
    mask = np.ones(len(df), dtype=bool) if base is None else np.asarray(base, dtype=bool).copy()
    steps, missing = [], []
    for cond in conditions:
        if not cond.enabled:
            continue
        if cond.column not in df.columns:
            missing.append(cond)
            continue
        mask &= condition_mask(df, cond, (folded or {}).get(cond.column))
        steps.append((cond, int(mask.sum())))
    return mask, steps, missing


def _value_text(v):
    """EN: A value as short text. / TR: Bir değer, kısa metin olarak."""
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):,}".replace(",", ".")
    return str(v)


def describe(cond):
    """
    EN: A condition as a readable phrase, e.g. "Genel Bakış - Plaka Uyruğu ∈ {Mavi plakalı}".
    TR: Bir koşul okunur bir ifade olarak, ör. "Genel Bakış - Plaka Uyruğu ∈ {Mavi plakalı}".
    """
    vals = tuple(cond.values)
    if cond.op in ("in", "not_in"):
        sign = "∈" if cond.op == "in" else "∉"
        return f"{cond.column} {sign} {{{', '.join(_value_text(v) for v in vals)}}}"
    if cond.op == "between":
        low, high = (list(vals) + [None, None])[:2]
        return f"{cond.column} {'…' if low is None else _value_text(low)}–{'…' if high is None else _value_text(high)}"
    if cond.op in ("contains", "not_contains", "equals", "regex"):
        return f"{cond.column} {OP_LABELS[cond.op]} “{vals[0] if vals else ''}”"
    return f"{cond.column} {OP_LABELS[cond.op]}"


def summary(conditions):
    """EN: The enabled conditions joined with "VE". / TR: Etkin koşullar "VE" ile birleşik."""
    return " VE ".join(describe(c) for c in conditions if c.enabled) or "koşul yok"


def latest_per_ad(df, ad_col, order_col):
    """
    EN: Mask of the last row of each ad by order_col (ties: the later row); rows without an ad id are kept.
    TR: Her ilanın order_col'a göre son satırının maskesi (eşitlikte sonraki satır); ilan kimliği olmayan satırlar
        tutulur.
    """
    if df.empty:
        return np.zeros(0, dtype=bool)
    frame = pd.DataFrame({"ad": df[ad_col].to_numpy(), "key": df[order_col].to_numpy(), "pos": np.arange(len(df))})
    has_ad = frame["ad"].notna().to_numpy(dtype=bool)
    last = frame[has_ad].sort_values(["ad", "key", "pos"]).groupby("ad", sort=False)["pos"].last().to_numpy()
    keep = ~has_ad
    keep[last] = True
    return keep


ROW_SETS = {"all": "Bütün satırlar", "latest": "İlan başına son satır", "analysis": "Analiz kümesi",
            "snapshot": "Tek tarama"}


def row_set(df, mode, snapshot_col, snapshot=None, ad_col="ad_id"):
    """
    EN: The base mask of a row set: all, latest row per ad, the analysis set (price > 0, TR plate, then the latest
        row per ad — as analysis/lib/common.load_clean; silver/gold only) or one snapshot.
    TR: Bir satır kümesinin taban maskesi: hepsi, ilan başına son satır, analiz kümesi (fiyat > 0, TR plaka, sonra
        ilan başına son satır — analysis/lib/common.load_clean gibi; yalnız silver/gold) ya da tek tarama.
    """
    if mode == "all":
        return np.ones(len(df), dtype=bool)
    if mode == "snapshot":
        return _bool_array(df[snapshot_col].astype(str) == str(snapshot))
    if mode == "latest":
        return latest_per_ad(df, ad_col, snapshot_col)
    if mode == "analysis":
        eligible = _bool_array(df["price"] > 0) & _bool_array(df["gb_plate_origin"] == TR_PLATE)
        keep = np.zeros(len(df), dtype=bool)
        idx = np.flatnonzero(eligible)
        keep[idx[latest_per_ad(df.iloc[idx], ad_col, snapshot_col)]] = True
        return keep
    raise ValueError(f"unknown row set | bilinmeyen satır kümesi: {mode}")
