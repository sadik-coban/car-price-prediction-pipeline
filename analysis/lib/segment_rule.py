"""
segment_rule.py
EN: The segment rule — its single source. A listing's segment comes from its series; families that do not
    map to one segment (the M / S / RS performance lines, i, Z4 M) are resolved from the model name. If a
    series/model cannot be resolved, the run stops (no silent default). The raw gb_segment is not used.
    07_final_model copies these tables into data/serving/serve/encoders.pkl so serving applies the same rule.
TR: Segment kuralı — tek kaynağı. İlanın segmenti serisinden gelir; tek segmente düşmeyen aileler (M / S / RS
    performans serileri, i, Z4 M) model adından çözülür. Çözülemeyen seri/model kalırsa koşum durur (sessiz
    varsayılan yok). Ham gb_segment kullanılmaz. 07_final_model bu tabloları
    data/serving/serve/encoders.pkl'ye kopyalar; servis aynı kuralı uygular.
"""
import re

# EN: series → segment | TR: seri → segment
SEGMENT_MAP = {"1 Serisi": "C", "2 Serisi": "C", "3 Serisi": "D", "4 Serisi": "D", "5 Serisi": "E",
               "6 Serisi": "E", "7 Serisi": "F", "8 Serisi": "F", "X1": "C", "X2": "C", "X3": "D", "X4": "D",
               "X5": "E", "X6": "E", "X7": "F", "Z4": "S", "Z Serisi": "S", "A1": "B", "A3": "C", "A4": "D",
               "A5": "D", "A6": "E", "A7": "E", "A8": "F", "Q2": "C", "Q3": "C", "Q5": "D", "Q7": "F", "Q8": "F",
               "TT": "S", "TTS": "S", "R8": "S"}
# EN: performance families: the digit in the model name gives the base series (M3 → 3 Serisi, RS 6 → A6)
# TR: performans aileleri: model adındaki rakam temel seriyi verir (M3 → 3 Serisi, RS 6 → A6)
PERF_BASE = {"M Serisi": "{} Serisi", "S": "A{}", "RS": "A{}"}
# EN: names where the digit rule is wrong (checked first): i8 is a sports coupé, Z4 M a roadster
# TR: rakam kuralının yanlış sonuç verdiği adlar (önce bakılır): i8 spor coupe, Z4 M roadster
MODEL_SEG = {"i Serisi": {"i8": "S"}, "M Serisi": {"Z4 M": "S"}}
PERF_RE = re.compile(r"^(?:M|S|RS)\s?(\d)")
# EN: the path that resolved a segment (published as 03's segment_quality.paths keys)
# TR: segmenti çözen yol (03'ün segment_quality.paths anahtarları olarak yayımlanır)
PATH_MODEL_TABLE, PATH_SERIES_MAP, PATH_BASE_CLASS = "model_table", "series_map", "base_class"


def resolve(series, model):
    """
    EN: The segment of one listing and the path that resolved it.
        Returns: (segment, path); path is PATH_MODEL_TABLE (model-name table), PATH_SERIES_MAP (series map),
        PATH_BASE_CLASS (performance base series) or (None, None) if unresolved.
    TR: Bir ilanın segmenti ve onu çözen yol.
        Döndürür: (segment, yol); yol PATH_MODEL_TABLE, PATH_SERIES_MAP, PATH_BASE_CLASS ya da çözülemezse
        (None, None).
    """
    s, m = str(series), str(model)
    for prefix, seg in MODEL_SEG.get(s, {}).items():
        if m.startswith(prefix):
            return seg, PATH_MODEL_TABLE
    if s in SEGMENT_MAP:
        return SEGMENT_MAP[s], PATH_SERIES_MAP
    if s in PERF_BASE:
        g = PERF_RE.match(m)
        if g:
            seg = SEGMENT_MAP.get(PERF_BASE[s].format(g.group(1)))
            if seg is not None:
                return seg, PATH_BASE_CLASS
    return None, None


def apply(series, models):
    """
    EN: The segment of every listing; stops and lists the pairs if any series/model cannot be resolved.
    TR: Her ilanın segmenti; çözülemeyen seri/model kalırsa durur ve çiftleri listeler.
    """
    out, unresolved = [], set()
    for s, m in zip(series, models):
        seg, _ = resolve(s, m)
        if seg is None:
            unresolved.add((str(s), str(m)))
        out.append(seg)
    if unresolved:
        sample = " · ".join(f"{a!r}/{b!r}" for a, b in sorted(unresolved)[:20])
        raise SystemExit(f"SEGMENT UNRESOLVED | SEGMENT ÇÖZÜLEMEDİ ({len(unresolved)} series/model): {sample}")
    return out
