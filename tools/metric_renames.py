"""
metric_renames.py
EN: The reader of the metric key rename map (docs/metric-key-renames.json) and the checks the name guard
    (tests/metrics/test_metric_key_names.py) runs with it: which group owns a metrics file, which old names are
    retired, and where a done group's file or a text still uses one. The rename program itself finished on
    2026-09-25; its step-by-step proofs compared everything with a copy of the old outputs kept in archive/ and
    were removed on 2026-09-27 (nothing from the archive reaches the live chain). A metrics file no group owns is
    written after the renames: it belongs to the always-done "new" group and its names are checked too.
TR: Metrik anahtarı ad değişikliği eşlemesinin (docs/metric-key-renames.json) okuyucusu ve ad bekçisinin
    (tests/metrics/test_metric_key_names.py) onunla koştuğu denetimler: bir metrik dosyasının grubu, hangi eski adların
    emekli olduğu ve bitmiş bir grubun dosyasında ya da bir metinde hâlâ nerede geçtiği. Ad değişikliği programı
    2026-09-25'te bitti; adım adım kanıtları her şeyi archive/'de tutulan eski çıktıların bir kopyasıyla
    karşılaştırıyordu ve 2026-09-27'de kaldırıldı (arşivden canlı zincire hiçbir şey girmez). Hiçbir grubun
    sahiplenmediği metrik dosyası ad değişikliğinden sonra yazılmıştır: her zaman bitmiş sayılan "new" grubuna
    aittir, adları da sınanır.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = ROOT / "docs" / "metric-key-renames.json"
NEW_GROUP = "new"      # metrics files written after the renames | ad değişikliğinden sonra yazılan metrik dosyaları


# ── The map | Eşleme ──────────────────────────────────────────────────────────────────────────────────────────────
def load_map(path=MAP_FILE):
    """
    EN: The rename map file with its path patterns compiled.
    TR: Yol kalıpları derlenmiş ad eşlemesi dosyası.
    """
    return compile_map(json.loads(Path(path).read_text(encoding="utf-8")))


def compile_map(m):
    """
    EN: Adds the compiled forms (_overrides, _values, _prose, _lib) to a rename map dict and returns it. "prose"
        (optional): path -> {old text: new text} for prose values that quote a renamed dotted path.
    TR: Bir ad eşlemesi dict'ine derlenmiş biçimleri (_overrides, _values, _prose, _lib) ekler ve döndürür. "prose"
        (isteğe bağlı): yol -> {eski metin: yeni metin}; yeniden adlandırılmış bir noktalı yolu anan düzyazı değerler.
    """
    m["_overrides"] = [(pattern_re(p), n) for p, n in m["overrides"].items()]
    m["_values"] = [(pattern_re(p), v) for p, v in m["values"].items()]
    m["_prose"] = [(pattern_re(p), v) for p, v in m.get("prose", {}).items()]
    m["_lib"] = set(m["lib"])
    return m


def pattern_re(pattern):
    """
    EN: A map path pattern as a regex: '*' is one key, '[]' any list element, '[i]' position i; the rest literal.
    TR: Eşleme yol kalıbı regex olarak: '*' tek anahtar, '[]' her liste öğesi, '[i]' i. konum; gerisi aynen.
    """
    return re.compile("^" + re.escape(pattern).replace(r"\*", r"[^.\[\]]+") + "$")


def group_of(name, m):
    """
    EN: The group that owns a metrics file ("01_engine_rule", "shap/02_oof_shap") by its prefixes; NEW_GROUP when
        no prefix matches (a script added after the renames).
    TR: Bir metrik dosyasının sahibi grup, öneklerine göre; hiçbir önek uymazsa NEW_GROUP (ad değişikliğinden
        sonra eklenen betik).
    """
    hits = [g for g, spec in m["groups"].items() if any(name.startswith(p) for p in spec["prefixes"])]
    if len(hits) > 1:
        raise SystemExit(f"{name}: belongs to {hits} groups, not one | tek grup değil")
    return hits[0] if hits else NEW_GROUP


def done_groups(m, extra=()):
    """
    EN: The groups marked done in the map, plus the ones named on the command line, plus NEW_GROUP.
    TR: Eşlemede bitti işaretli gruplar, komut satırında anılanlar ve NEW_GROUP.
    """
    return {g for g, spec in m["groups"].items() if spec["status"] == "done"} | set(extra) | {NEW_GROUP}


def under(path, roots):
    """
    EN: True if path is one of roots or lies below one ('.' or '[' after it).
    TR: path köklerden biriyse ya da altındaysa ('.' ya da '[' ile devam ediyorsa) True.
    """
    return any(path == r or path.startswith(r + ".") or path.startswith(r + "[") for r in roots)


def translate_pattern(pattern, m):
    """
    EN: A dotted path or glob in old names with every segment renamed by the bare-key map; a list suffix such as
        '[][3]' stays (exemption and value patterns).
    TR: Eski adlı noktalı yol ya da glob; her bölüt çıplak anahtar eşlemesiyle çevrilir, '[][3]' gibi liste eki
        kalır (istisna ve değer kalıpları).
    """
    parts = [re.match(r"^([^\[]*)(.*)$", s).groups() for s in pattern.split(".")]
    return ".".join(m["keys"].get(name, name) + rest for name, rest in parts)


# ── Files | Dosyalar ──────────────────────────────────────────────────────────────────────────────────────────────
def read(path):
    """EN: A JSON file. / TR: Bir JSON dosyası."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_docs(metrics_dir):
    """
    EN: Every metrics file under metrics_dir as {name: document}.
    TR: metrics_dir altındaki her metrik dosyası {ad: belge} olarak.
    """
    return {p.relative_to(metrics_dir).with_suffix("").as_posix(): read(p) for p in sorted(Path(metrics_dir).rglob("*.json"))}


# ── Guard | Koruma (tests/metrics/test_metric_key_names.py) ───────────────────────────────────────────────────────
DOTTED = re.compile(r"\b(?:error_drivers|domain|methodology|meta|report|oof_shap|shap_final|shap_case|shap)"
                    r"(?:\.[A-Za-z0-9_]+)+")


def retired_names(m, done):
    """
    EN: Old names that must not appear any more as keys of done groups: renamed keys that are no new name anywhere;
        segment_rule's names ("lib") only once LIB is done.
    TR: Bitmiş grupların anahtarlarında artık görünmemesi gereken eski adlar: hiçbir yerde yeni ad olmayan çevrilmiş
        anahtarlar; segment_rule adları ("lib") yalnız LIB bitince.
    """
    new = set(m["keys"].values()) | set(m["overrides"].values()) | set(m["keep"])
    retired = {k for k, v in m["keys"].items() if v != k} - new
    return retired if "LIB" in done else retired - m["_lib"]


def key_violations(doc, m, retired):
    """
    EN: Paths of a current (new-name) metrics document whose key is a retired name, data-valued subtrees left out.
    TR: Güncel (yeni adlı) bir metrik belgesinde anahtarı emekli ad olan yollar; veri değerli alt ağaçlar hariç.
    """
    roots = [translate_pattern(r, m) for r in m["data_valued"]]
    out = []

    def walk(x, path):
        """EN: Collects retired keys outside data-valued roots. / TR: Veri değerli kökler dışındaki emekli anahtarları toplar."""
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else k
                if k in retired and not under(path, roots):
                    out.append(p)
                walk(v, p)
        elif isinstance(x, list):
            for v in x:
                walk(v, path + "[]")

    walk({k: v for k, v in doc.items() if k != "_meta"}, "")
    return out


def value_violations(doc, m, done):
    """
    EN: Paths of a current metrics document where a map "values" site still holds an old value.
    TR: Güncel bir metrik belgesinde eşlemenin "values" yerlerinden birinin hâlâ eski değer tuttuğu yollar.
    """
    sites = [(pattern_re(translate_pattern(p, m)), {o for o, n in v.items() if o != n and (o not in m["_lib"] or "LIB" in done)})
             for p, v in m["values"].items()]
    out = []

    def walk(x, path, indexed):
        """EN: Collects old values at value sites. / TR: Değer yerlerindeki eski değerleri toplar."""
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else k
                walk(v, p, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[]", f"{path}[{i}]")
        elif isinstance(x, str):
            out.extend(path for rx, old in sites if (rx.match(path) or rx.match(indexed)) and x in old)

    walk(doc, "", "")
    return out


def top_owners(docs, m):
    """
    EN: {"section.top_key": groups} from the current metrics, to tell which group a dotted path in text belongs to.
    TR: Güncel metriklerden {"bölüm.üst_anahtar": gruplar}; metindeki noktalı yolun hangi gruba ait olduğunu söyler.
    """
    out = {}
    for name, doc in docs.items():
        for sec, body in doc.items():
            if sec != "_meta" and isinstance(body, dict):
                for k in body:
                    out.setdefault(f"{sec}.{k}", set()).add(group_of(name, m))
    return out


def stale_refs(text, m, done, owners_by_top, retired):
    """
    EN: Dotted metric paths in a text (code, card, doc) that still use a retired name inside a done group.
    TR: Bir metinde (kod, kart, belge) bitmiş bir grubun içinde hâlâ emekli ad kullanan noktalı metrik yolları.
    """
    out = []
    for hit in DOTTED.findall(text):
        segs = hit.split(".")
        groups = owners_by_top.get(f"{segs[0]}.{segs[1]}") or owners_by_top.get(
            f"{segs[0]}.{m['keys'].get(segs[1], segs[1])}")
        if groups and groups <= done and any(s in retired for s in segs[1:]):
            out.append(hit)
    return out


# ── Commands | Komutlar ───────────────────────────────────────────────────────────────────────────────────────────
