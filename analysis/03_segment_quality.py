"""
03_segment_quality.py
EN: Technical report §3 — where the segment comes from and how it compares with the raw feed.
    The segment is derived from the series (and the model name for families that do not map to one
    segment); the raw gb_segment is not used: its "G" is not a real segment (almost all MPV bodies).
TR: Teknik rapor §3 — segment nereden geliyor ve ham beslemeyle nasıl örtüşüyor.
    Segment seriden (tek segmente düşmeyen ailelerde model adından) türetiliyor; ham gb_segment kullanılmıyor:
    oradaki "G" gerçek bir segment değil (neredeyse tamamı MPV gövde).
Output / Çıktı: metrics/03_segment_quality.json
"""

# %% [1] Setup | Kurulum

from lib import segment_rule as SR
from lib.common import load_clean, save_metrics


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def derive_segments(raw):
    """
    EN: The derived segment of every listing and the path that resolved it (map / model table / base class).
        Stops if any series/model cannot be resolved.
        Returns: DataFrame with series_key, derived (segment), path, raw_segment (raw gb_segment), kb_body_type.
    TR: Her ilanın türetilen segmenti ve onu çözen yol (harita / model tablosu / temel sınıf).
        Çözülemeyen seri/model kalırsa durur.
        Döndürür: series_key, derived (segment), path, raw_segment (ham gb_segment), kb_body_type kolonlu DataFrame.
    """
    d = raw[["series", "model", "gb_segment", "kb_body_type"]].copy()
    d["series_key"] = d["series"].fillna("missing").astype(str)
    resolved = [SR.resolve(a, m) for a, m in zip(d["series_key"], d["model"])]
    d["derived"] = [r[0] for r in resolved]
    d["path"] = [r[1] for r in resolved]
    assert d["derived"].notna().all(), "unresolved segment left"
    d["raw_segment"] = d["gb_segment"].str.replace(" Segment", "", regex=False)
    return d


def segment_quality(d):
    """
    EN: The G segment's make-up, the resolution paths, families resolved from the model name, series that
        span several segments, and how often the raw segment disagrees with the derived one.
    TR: G segmentinin içeriği, çözüm yolları, model adından çözülen aileler, birden çok segmente yayılan
        seriler ve ham segmentin türetilenle ne sıklıkla uyuşmadığı.
    """
    g = d[d["raw_segment"] == "G"]
    known = d[d["raw_segment"].notna()]
    differ = known[known["raw_segment"] != known["derived"]]
    special = d[d["path"] != SR.PATH_SERIES_MAP]
    per_series = d.groupby("series_key")["derived"].nunique()
    return {"map_size": len(SR.SEGMENT_MAP),
            "g": {"n": int(len(g)), "mpv_n": int((g["kb_body_type"] == "MPV").sum()),
                  "body": [[str(k), int(v)] for k, v in g["kb_body_type"].fillna("(bos)").value_counts().items()],
                  "series": [[str(k), int(v)] for k, v in g["series_key"].value_counts().items()]},
            "paths": {str(k): int(v) for k, v in d["path"].value_counts().items()},
            # EN: ties ordered by label so the order never depends on row order | TR: beraberlik etikete göre sıralı
            "from_model_name": [[str(s), int(len(x)),
                                 [[str(k), int(v)] for k, v in sorted(x["derived"].value_counts().items(),
                                                                      key=lambda kv: (-kv[1], str(kv[0])))]]
                                for s, x in special.groupby("series_key")],
            "multi_segment_series": sorted(str(s) for s, k in per_series.items() if k > 1),
            "raw_known": int(len(known)), "raw_differs": int(len(differ)),
            "cross": [[str(a), str(b), int(n)] for (a, b), n in
                      differ.groupby(["raw_segment", "derived"]).size().sort_values(ascending=False).items()]}


def series_segment_matrix(listings):
    """
    EN: Median price and listing count per (series, segment). Returns: list of [series, segment, median, n].
    TR: (seri, segment) başına medyan fiyat ve ilan sayısı. Döndürür: [seri, segment, medyan, n] listesi.
    """
    return (listings.groupby(["series", "segment"]).agg(median=("price", "median"), n=("price", "size"))
            .reset_index().values.tolist())


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under error_drivers.segment_quality (report) and domain.series_segment_matrix (site).
    TR: error_drivers.segment_quality (rapor) ve domain.series_segment_matrix (site) altında yayımlanır.
    """
    q = res["quality"]
    return {
        "domain": {"series_segment_matrix": res["matrix"]},
        "error_drivers": {"segment_quality": {
            "map_size": q["map_size"],
            "g_segment": {"n": q["g"]["n"], "mpv_n": q["g"]["mpv_n"], "body": q["g"]["body"], "series": q["g"]["series"]},
            "paths": q["paths"], "from_model_name": q["from_model_name"], "multi_segment_series": q["multi_segment_series"],
            "mismatch": {"raw_known": q["raw_known"], "differs": q["raw_differs"],
                         "pct": round(100 * q["raw_differs"] / q["raw_known"], 2), "cross": q["cross"]},
            "note": ("Segment seri + gerekirse model adindan turetiliyor; ham gb_segment kullanilmiyor. "
                    "'G' gercek bir segment degil (govdesi MPV). Cozulemeyen seri/model kalirsa uretec durur.")}},
    }


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
raw = load_clean(derived=False)
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
res = {"quality": segment_quality(derive_segments(raw)), "matrix": series_segment_matrix(listings)}
print("raw ≠ derived | ham ≠ türetilen:", res["quality"]["raw_differs"], "/", res["quality"]["raw_known"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("03_segment_quality", to_metrics(res)))
