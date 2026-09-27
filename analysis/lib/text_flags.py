"""
text_flags.py
EN: Two signals read from a listing's own free text, used by 07_text_flag (technical report §7 and §10, business
    note):
      - conversion wording ("swap", "dönüşüm", "replika" …);
      - modification wording ("modifiye", "stage 1", "coilover" …), negation-safe.
    Origin (written down under the rule that nothing from the archive reaches the live chain, 2026-09-27): the
    patterns were ported from the archived text pipeline (build_text_insights.py, turkish_text.py), and the
    modification word list was distilled from the vocabulary of an archived LLM extraction trial. They stay
    because they are applied to the current descriptions on every run and their output is used; no archived data
    is read. The hp and M/RS model detectors that used to live here fed nothing and were removed (2026-09-27).
    Pure pattern matching; no LLM.
TR: Bir ilanın kendi serbest metninden okunan iki sinyal; 07_text_flag kullanır (teknik rapor §7 ve §10, iş notu):
      - dönüşüm ifadesi ("swap", "dönüşüm", "replika" …);
      - modifiye ifadesi ("modifiye", "stage 1", "coilover" …), olumsuzlama-güvenli.
    Köken (arşivden canlı zincire hiçbir şey girmez kuralıyla yazıldı, 2026-09-27): desenler arşivdeki metin
    zincirinden (build_text_insights.py, turkish_text.py) taşındı; modifiye kelime listesi arşivlenmiş bir LLM
    çıkarım denemesinin sözcük dağarcığından damıtıldı. Her koşuda güncel açıklamalara uygulandıkları ve çıktıları
    kullanıldığı için kalıyorlar; arşivden veri okunmaz. Burada duran beygir ve M/RS model dedektörleri hiçbir yeri
    beslemiyordu, kaldırıldı (2026-09-27). Yalnız desen eşleme; LLM yok.
"""
import re
import unicodedata

import numpy as np

_TO_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_COMBINING_DOT = "̇"
# EN: sellers write the same word in Turkish and ASCII; 1:1 fold keeps offsets | TR: 1:1 katlama ofsetleri korur
ASCII_FOLD = str.maketrans("ışğüöç", "isguoc")
NEGATION = re.compile(r"\b(?:degil\w*|yok\w*|olmayan|olmam\w*|bulunm\w*|takilm\w*|haric)\b")
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
    EN: The lower-cased description of every listing (from the DB's description_text, which has no page
        heading since the semi-raw DB of 2026-09-24; empty if missing).
    TR: Her ilanın küçültülmüş açıklaması (DB'nin description_text kolonundan; 2026-09-24 yarı ham DB'sinden beri
        sayfa başlığı yok; yoksa boş).
    """
    return listings["description_text"].fillna("").astype(str).map(lower_tr)


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
