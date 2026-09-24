"""
text_flags.py
EN: Four signals read from a listing's own free text, used by 07_text_flag and 08_large_errors:
      - the text names a much higher horsepower than the form (hp conflict);
      - the text names a different M/RS model together with conversion wording (model conflict);
      - conversion wording ("swap", "dönüşüm", "replika" …);
      - modification wording ("modifiye", "stage 1", "coilover" …), negation-safe.
    Ported unchanged from the archived text pipeline (archive/analysis-history/text_analysis/steps/build_text_insights.py and
    turkish_text.py) — only these detectors, the same patterns — so no script depends on the git-ignored
    archive. Pure pattern matching; no LLM.
TR: Bir ilanın kendi serbest metninden okunan dört sinyal; 07_text_flag ve 08_large_errors kullanır:
      - metin formdan çok daha yüksek beygir söylüyor (hp çelişkisi);
      - metin dönüşüm ifadesiyle birlikte farklı bir M/RS modeli anıyor (model çelişkisi);
      - dönüşüm ifadesi ("swap", "dönüşüm", "replika" …);
      - modifiye ifadesi ("modifiye", "stage 1", "coilover" …), olumsuzlama-güvenli.
    Arşivdeki metin zincirinden (archive/analysis-history/text_analysis/steps/build_text_insights.py ve turkish_text.py) aynen
    taşındı — yalnız bu dedektörler, aynı desenler — böylece hiçbir betik git dışı arşive bağlı değil.
    Yalnız desen eşleme; LLM yok.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

_TO_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_COMBINING_DOT = "̇"
# EN: sellers write the same word in Turkish and ASCII; 1:1 fold keeps offsets | TR: 1:1 katlama ofsetleri korur
ASCII_FOLD = str.maketrans("ışğüöç", "isguoc")
NEGATION = re.compile(r"\b(?:degil\w*|yok\w*|olmayan|olmam\w*|bulunm\w*|takilm\w*|haric)\b")
# EN: a model conflict needs conversion wording; a bare "M3 brakes" is a part mention, not a conversion
# TR: model çelişkisi dönüşüm ifadesi ister; salt "M3 fren" parça anmasıdır, dönüşüm değil
M_CONVERSION = re.compile(r"cevril|donusum|donusturul|\bswap\b|kasa cevr|gorunum(?:e|une)? ")
CONVERSION_PATTERN = (r"\bswap\b|dönüşüm|donusum|proje ara[cç]|motor değiş|motor degis|kasa dönüş|"
                      r"\bçakma\b|\breplika\b|komple değiş")
MODIFICATION = re.compile(
    r"modifiye(?!siz)\w*|sonradan (?:takil|eklen|mont)\w*|aftermarket|"
    r"m sport'?a cevr\w*|\bm donus\w*|ppf kapl\w*|cam film\w*|"
    r"coilover|coiller|air ?ride|akrapovic|\bvarex\b|downpipe|catback|manifold|"
    r"stage ?[123]|\bremap\b|chip ?tun|body ?kit|bodykit|\bspacer\b|difuzor|intercooler|"
    r"spor yay\b|dusurulm\w*|yazilimli|"
    r"\bcoil\b|dusuk yay|kisa yay|eibach|spor helezon|\bh ?& ?r\b"
)
HP_GAP, HP_RATIO = 80, 1.3               # text hp must exceed the form by >80 hp and >30% | >80 hp ve >%30


def lower_tr(s):
    """
    EN: Turkish-aware lower-casing (I → ı, İ → i), stray combining dots removed, NFC.
    TR: Türkçe-duyarlı küçültme (I → ı, İ → i), artık birleşik noktalar silinir, NFC.
    """
    if s is None:
        return ""
    s = str(s).translate(_TO_LOWER).lower().replace(_COMBINING_DOT, "")
    return unicodedata.normalize("NFC", s)


def descriptions(listings):
    """
    EN: The lower-cased description of every listing (from the DB's description_clean; empty if missing).
    TR: Her ilanın küçültülmüş açıklaması (DB'nin description_clean kolonundan; yoksa boş).
    """
    return listings["description_clean"].fillna("").astype(str).map(lower_tr)


def text_hp(text):
    """
    EN: The largest 100–1500 horsepower figure written in the text ("340 hp", "340 beygir"), else NaN.
    TR: Metinde yazan en büyük 100–1500 beygir değeri ("340 hp", "340 beygir"), yoksa NaN.
    """
    vals = [int(m) for m in re.findall(r"(\d{3,4})\s*(?:hp|beygir|bg\b|ps\b)", str(text)) if 100 <= int(m) <= 1500]
    return max(vals) if vals else np.nan


def text_model(text):
    """
    EN: An M2–M8 or RS3–RS7 token in the text, else None.
    TR: Metindeki M2–M8 ya da RS3–RS7 belirteci, yoksa None.
    """
    m = re.search(r"\bm[2345678]\b", str(text))
    rs = re.search(r"\brs[34567]\b", str(text))
    return m.group() if m else (rs.group() if rs else None)


def hp_conflict(desc, listings):
    """
    EN: Text horsepower well above the form's (by >80 hp and >30%). The form value is the model's own
        hp = mean(low, up). Returns: (mask, text hp as int/None, form hp as int/None).
    TR: Metindeki beygir formdakinden belirgin yüksek (>80 hp ve >%30). Form değeri modelin kendi
        hp = ort(alt, üst) değeri. Döndürür: (maske, metin hp int/None, form hp int/None).
    """
    t_hp = pd.to_numeric(desc.map(text_hp), errors="coerce").values
    hp = listings[["power_hp_low", "power_hp_up"]].apply(pd.to_numeric, errors="coerce").mean(axis=1).values
    with np.errstate(invalid="ignore"):
        mask = (~np.isnan(t_hp)) & (~np.isnan(hp)) & (t_hp - hp > HP_GAP) & (t_hp / np.maximum(hp, 1) > HP_RATIO)
    as_int = lambda a: [None if np.isnan(v) else int(round(v)) for v in a]      # noqa: E731
    return mask, as_int(t_hp), as_int(hp)


def model_conflict(desc, listings):
    """
    EN: The text names an M/RS model that is not the listing's model/series, together with conversion
        wording. Returns: (mask, the token found per listing).
    TR: Metin, ilanın model/serisi olmayan bir M/RS modeli anıyor ve yanında dönüşüm ifadesi var.
        Döndürür: (maske, ilan başına bulunan belirteç).
    """
    tokens = desc.map(text_model).values
    folded = desc.str.translate(ASCII_FOLD).values
    model_v, series_v = listings["model"].astype(str).values, listings["series"].astype(str).values
    mask = np.array([isinstance(tokens[i], str) and tokens[i] not in (model_v[i] + " " + series_v[i]).lower()
                     and M_CONVERSION.search(folded[i]) is not None for i in range(len(desc))])
    return mask, tokens


def conversion_flag(desc):
    """
    EN: Conversion wording in the text (swap, conversion, replica, engine/body change). Returns: bool array.
    TR: Metinde dönüşüm ifadesi (swap, dönüşüm, replika, motor/kasa değişimi). Döndürür: bool dizi.
    """
    return desc.fillna("").str.contains(CONVERSION_PATTERN, regex=True, na=False).values


def _is_modified(t):
    """
    EN: A modification phrase that is not negated or followed by "original/standard/factory"; "coil" next to
        ignition words is an ignition coil, not suspension.
    TR: Olumsuzlanmamış ve ardından "orjinal/standart/fabrika" gelmeyen bir modifiye ifadesi; ateşleme
        kelimelerinin yanındaki "coil" bobindir, süspansiyon değil.
    """
    for m in MODIFICATION.finditer(t):
        after = t[m.end(): m.end() + 20]
        if NEGATION.search(after) or re.search(r"orjinal|standart|fabrika", after):
            continue
        if m.group() == "coil" and re.search(r"ates|bobin|\bbuji", t[max(0, m.start() - 14): m.end() + 14]):
            continue
        return True
    return False


def modification_flag(desc):
    """
    EN: Modification wording in the (ASCII-folded) text, negation-safe. Returns: bool array.
    TR: (ASCII'ye katlanmış) metinde modifiye ifadesi, olumsuzlama-güvenli. Döndürür: bool dizi.
    """
    return np.array([_is_modified(x) for x in desc.fillna("").str.translate(ASCII_FOLD).values], dtype=bool)
