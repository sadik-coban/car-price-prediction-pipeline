"""
metric_renames.py
EN: Applies, proves and exports the one reviewed map of metric key names (docs/metric-key-renames.json: old Turkish
    names -> English snake_case). The renaming itself happens in the analysis scripts and builders, group by group;
    this tool only reads JSON and proves each step is a pure rename against one fixed origin, the P0 copy
    (archive/backups/renames-P0-2026-09-25/p0/, taken once before the first group):
      validate        every map entry matches the P0 metrics; the fully translated metrics still merge (no clash)
      snapshot        takes the P0 copy (refuses to overwrite it)
      check-builders  runs the four builders on translated P0 metrics in a temp root: every report and figure must be
                      byte-identical to P0, site_data equal to translated P0 (proves the builder edits)
      check-metrics   the repository metrics equal translated P0 (proves the producer edits; order and type count)
      check-outputs   the repository reports, figures and site_data equal P0 (after the rebuild)
      site-map        old -> new paths and value maps of the site tree, for the portfolio site
      mark-done G     marks a group done in the map
    A path is translated only when every script that writes it belongs to a group marked done (or named with
    --groups); names from analysis/lib/segment_rule.py (map "lib") only with the LIB group.
TR: İncelenmiş tek metrik anahtarı eşlemesini (docs/metric-key-renames.json: eski Türkçe adlar -> İngilizce
    snake_case) uygular, kanıtlar ve dışa verir. Ad değişikliğinin kendisi analiz betiklerinde ve derleyicilerde,
    grup grup yapılır; bu araç yalnız JSON okur ve her adımın tek bir sabit başlangıca, P0 kopyasına
    (archive/backups/renames-P0-2026-09-25/p0/, ilk gruptan önce bir kez alınır) göre saf ad değişikliği olduğunu
    kanıtlar:
      validate        eşlemenin her girdisi P0 metrikleriyle eşleşiyor; tamamen çevrilmiş metrikler hâlâ birleşiyor
      snapshot        P0 kopyasını alır (üzerine yazmaz)
      check-builders  dört derleyiciyi geçici bir kökte çevrilmiş P0 metrikleriyle koşar: her rapor ve figür P0 ile
                      bayt bayt aynı, site_data çevrilmiş P0'a eşit olmalı (derleyici düzenlemelerini kanıtlar)
      check-metrics   depodaki metrikler çevrilmiş P0'a eşit (üretici düzenlemelerini kanıtlar; sıra ve tip sayılır)
      check-outputs   depodaki raporlar, figürler ve site_data P0'a eşit (yeniden derlemeden sonra)
      site-map        site ağacının eski -> yeni yolları ve değer eşlemeleri, portföy sitesi için
      mark-done G     bir grubu eşlemede bitti diye işaretler
    Bir yol yalnız onu yazan her betik "done" işaretli (ya da --groups ile anılan) bir gruptaysa çevrilir;
    analysis/lib/segment_rule.py'den gelen adlar (eşlemede "lib") yalnız LIB grubuyla.
Run / Koşum:
    python tools/metric_renames.py validate
    python tools/metric_renames.py check-builders --groups G01
    python tools/metric_renames.py check-metrics --groups G01
    python tools/metric_renames.py check-outputs --groups G01
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = ROOT / "docs" / "metric-key-renames.json"
P0_DIR = ROOT / "archive" / "backups" / "renames-P0-2026-09-25" / "p0"
EXEMPTIONS_FILE = ROOT / "tests" / "baselines" / "exemptions.json"
BUILDERS = ["build_site_data.py", "build_technical_report.py", "build_business_report.py", "build_shap_report.py"]
SITE_SECTIONS = ("meta", "domain", "methodology")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
IGNORED = ("_meta.generated_at", "_meta.source")        # run stamps and code hashes | koşum damgası, kod hash'i
SITE_IGNORED = ("_meta.generated_at", "meta.metrics_generated_at")


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
    EN: The group that owns a metrics file ("01_engine_rule", "shap/02_oof_shap") by its prefixes.
    TR: Bir metrik dosyasının sahibi grup, öneklerine göre.
    """
    hits = [g for g, spec in m["groups"].items() if any(name.startswith(p) for p in spec["prefixes"])]
    if len(hits) != 1:
        raise SystemExit(f"{name}: belongs to {hits} groups, not one | tek grup değil")
    return hits[0]


def done_groups(m, extra=()):
    """
    EN: The groups marked done in the map, plus the ones named on the command line.
    TR: Eşlemede bitti işaretli gruplar ile komut satırında anılanlar.
    """
    return {g for g, spec in m["groups"].items() if spec["status"] == "done"} | set(extra)


def under(path, roots):
    """
    EN: True if path is one of roots or lies below one ('.' or '[' after it).
    TR: path köklerden biriyse ya da altındaysa ('.' ya da '[' ile devam ediyorsa) True.
    """
    return any(path == r or path.startswith(r + ".") or path.startswith(r + "[") for r in roots)


def new_name(path, key, m):
    """
    EN: The new name of the key at path (old names): an override pattern first, then the bare-key map.
    TR: path'teki (eski adlar) anahtarın yeni adı: önce yol kalıbı istisnası, sonra çıplak anahtar eşlemesi.
    """
    if key in m["keep"]:
        return key
    for rx, name in m["_overrides"]:
        if rx.match(path):
            return name
    return m["keys"].get(key, key)


def translate(doc, m, ok, lib_done, path=""):
    """
    EN: A copy of doc (a metrics file or the site tree, old names) with keys and enum values renamed, key order kept.
        ok(path) says whether the owner of a path is done; data_valued subtrees keep their keys; names in "lib"
        change only when lib_done. Stops if two keys of one dict would end with the same name.
    TR: doc'un (bir metrik dosyası ya da site ağacı, eski adlar) anahtarları ve değerleri çevrilmiş kopyası, anahtar
        sırası korunur. ok(path) yolun sahibinin bitip bitmediğini söyler; data_valued alt ağaçlarının anahtarları
        kalır; "lib" adları yalnız lib_done iken değişir. Bir dict'in iki anahtarı aynı ada düşerse durur.
    """
    if isinstance(doc, dict):
        out = {}
        for k, v in doc.items():
            p = f"{path}.{k}" if path else k
            rename = ok(p) and not under(path, m["data_valued"]) and (k not in m["_lib"] or lib_done)
            name = new_name(p, k, m) if rename else k
            if name in out:
                raise SystemExit(f"{p}: renamed to '{name}', which the dict already has | ad çakışması")
            out[name] = translate(v, m, ok, lib_done, p)
        return out
    if isinstance(doc, list):
        return [_value(translate(v, m, ok, lib_done, path + "[]"), f"{path}[]", f"{path}[{i}]", m, ok, lib_done)
                for i, v in enumerate(doc)]
    return _value(doc, path, path, m, ok, lib_done) if path and not path.endswith("]") else doc


def _value(v, path, indexed, m, ok, lib_done):
    """
    EN: An enum value renamed if its path (or its row position) is a map "values" site and its owner is done; a
        prose value at a map "prose" site gets its quoted old paths replaced.
    TR: Yolu (ya da satırdaki konumu) eşlemenin "values" yerlerinden biriyse ve sahibi bittiyse çevrilmiş değer;
        "prose" yerindeki düzyazı değerde anılan eski yollar değiştirilir.
    """
    if not isinstance(v, str) or not ok(path):
        return v
    for rx, mapping in m["_values"]:
        if (rx.match(path) or rx.match(indexed)) and v in mapping and (v not in m["_lib"] or lib_done):
            return mapping[v]
    for rx, subs in m["_prose"]:
        if rx.match(path):
            for old, new in subs.items():
                v = v.replace(old, new)
    return v


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


def owners(docs, m):
    """
    EN: Which groups write each path (old names, list elements as '[]'), from the P0 metrics.
    TR: Her yolu hangi grupların yazdığı (eski adlar, liste öğeleri '[]'), P0 metriklerinden.
    """
    out = {}

    def walk(x, path, g):
        """EN: Records the owner group of every key path. / TR: Her anahtar yolunun sahibi grubu kaydeder."""
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else k
                out.setdefault(p, set()).add(g)
                walk(v, p, g)
        elif isinstance(x, list):
            for v in x:
                walk(v, path + "[]", g)

    for name, doc in docs.items():
        walk({k: v for k, v in doc.items() if k != "_meta"}, "", group_of(name, m))
    return out


def translate_file(name, doc, m, done):
    """
    EN: One metrics file translated when its group is done (with LIB names if LIB is done too).
    TR: Grubu bittiyse çevrilmiş bir metrik dosyası (LIB de bittiyse LIB adlarıyla).
    """
    group_done = group_of(name, m) in done
    return translate(doc, m, lambda p: group_done and not p.startswith("_meta"), "LIB" in done)


def translate_site(site, m, done, own):
    """
    EN: The site tree translated path by path: a path changes when every group that writes it is done.
    TR: Site ağacı yol yol çevrilir: bir yol, onu yazan her grup bittiyse değişir.
    """
    def ok(p):
        """EN: True when every group writing p is done. / TR: p'yi yazan her grup bittiyse True."""
        groups = own.get(re.sub(r"(\[\d*\])+$", "", p))    # a value in a list belongs to the list's key
        return bool(groups) and groups <= done
    return translate(site, m, ok, "LIB" in done)


def flat(doc, ignored=(), path=""):
    """
    EN: A document as an ordered list of (path, canonical JSON) leaves; lists are one leaf, key order kept.
    TR: Belge, sıralı (yol, kanonik JSON) yaprak listesi olarak; liste tek yapraktır, anahtar sırası korunur.
    """
    if isinstance(doc, dict) and doc:
        out = []
        for k, v in doc.items():
            p = f"{path}.{k}" if path else k
            if p not in ignored:
                out += flat(v, ignored, p)
        return out
    return [(path, json.dumps(doc, ensure_ascii=False))]


def exempt_patterns(m, path=EXEMPTIONS_FILE):
    """
    EN: The baseline exemptions as (file glob, path glob) in both old and new names.
    TR: Referans istisnaları, eski ve yeni adlarla (dosya glob, yol glob).
    """
    out = []
    for e in read(path)["exemptions"]:
        out += [(e["file"], e["path"]), (e["file"], translate_pattern(e["path"], m))]
    return out


def is_exempt(name, key, patterns):
    """
    EN: True if (file, leaf path) matches an exemption pattern.
    TR: (dosya, yaprak yolu) bir istisna kalıbına uyuyorsa True.
    """
    return any(fnmatch.fnmatchcase(name, f) and fnmatch.fnmatchcase(key, p) for f, p in patterns)


def leaf_diff(name, expected, actual, patterns, ignored):
    """
    EN: The leaf paths where two documents differ (key, order, type or value), exemptions left out.
    TR: İki belgenin ayrıldığı yaprak yolları (anahtar, sıra, tip ya da değer); istisnalar hariç.
    """
    a = [(k, v) for k, v in flat(expected, ignored) if not is_exempt(name, k, patterns)]
    b = [(k, v) for k, v in flat(actual, ignored) if not is_exempt(name, k, patterns)]
    if a == b:
        return []
    if [k for k, _ in a] != [k for k, _ in b]:
        da, db = dict(a), dict(b)
        return sorted(set(da) ^ set(db)) or ["(key order | anahtar sırası)"]
    return [k for (k, x), (_, y) in zip(a, b) if x != y]


def folded(path):
    """EN: File bytes with CRLF folded to LF. / TR: CRLF'i LF'e indirilmiş dosya baytları."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def compare_outputs(built, p0):
    """
    EN: Report md files and figures under built/reports that differ from P0 (md CRLF-folded, figures by bytes).
    TR: built/reports altında P0'dan farklı rapor md'leri ve figürler (md CRLF katlanmış, figür bayt bayt).
    """
    md = sorted(p.name for p in (built / "reports").glob("*.md"))
    figs = sorted(p.name for p in (built / "reports" / "figures").glob("*.png"))
    diff = [n for n in md if not (p0 / "reports" / n).exists() or folded(built / "reports" / n) != folded(p0 / "reports" / n)]
    diff += [f"figures/{n}" for n in figs if not (p0 / "reports" / "figures" / n).exists()
             or (built / "reports" / "figures" / n).read_bytes() != (p0 / "reports" / "figures" / n).read_bytes()]
    return md, figs, diff


def site_diff(built_site, m, done, p0, own, patterns):
    """
    EN: Leaf paths where a built site_data.json differs from translated P0 (stamps and exemptions left out).
    TR: Derlenmiş site_data.json'un çevrilmiş P0'dan ayrıldığı yaprak yolları (damgalar ve istisnalar hariç).
    """
    expected = translate_site(read(p0 / "data" / "site_data.json"), m, done, own)
    return leaf_diff("*", expected, read(built_site), [("*", p) for _, p in patterns], SITE_IGNORED)


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
def cmd_validate(m, p0):
    """
    EN: Every keys / overrides / values entry matches the P0 metrics, new names are snake_case, and the fully
        translated metrics merge like the builders merge them. Returns: list of problems.
    TR: Her keys / overrides / values girdisi P0 metrikleriyle eşleşiyor, yeni adlar snake_case ve tamamen
        çevrilmiş metrikler derleyicilerin birleştirdiği gibi birleşiyor. Döndürür: sorun listesi.
    """
    sys.path.insert(0, str(ROOT / "builders"))
    from report_lib import metrics_view as MV
    docs = metrics_docs(p0 / "metrics")
    own = owners(docs, m)
    paths = [p for p in own if not under(p.rsplit(".", 1)[0] if "." in p else "", m["data_valued"])]
    keys_seen = {p.rsplit(".", 1)[-1] for p in paths}
    problems = [f"keys: '{k}' not in P0" for k in m["keys"] if k not in keys_seen]
    problems += [f"overrides: '{p}' matches no P0 path" for p, _ in m["overrides"].items()
                 if not any(pattern_re(p).match(q) for q in own)]
    problems += [f"bad new name '{n}'" for n in list(m["keys"].values()) + list(m["overrides"].values())
                 if not NAME.match(n)]
    values_seen = set()
    everything = set(m["groups"])
    full = {name: translate_file(name, doc, m, everything) for name, doc in docs.items()}

    def walk(x, path):
        """EN: Notes which value sites match a P0 value. / TR: Hangi değer yerinin bir P0 değerine uyduğunu not eder."""
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                for rx, mapping in m["_values"]:
                    if (rx.match(f"{path}[]") or rx.match(f"{path}[{i}]")) and v in mapping:
                        values_seen.add(rx.pattern)
                walk(v, path + "[]")
        else:
            for rx, mapping in m["_values"]:
                if rx.match(path) and x in mapping:
                    values_seen.add(rx.pattern)

    for doc in docs.values():
        walk(doc, "")
    problems += [f"values: '{p}' matches no P0 value" for p in m["values"] if pattern_re(p).pattern not in values_seen]
    view, merge_owners = {s: {} for s in MV.SECTIONS}, {}
    for name, doc in full.items():
        merge_owners["_current"] = name
        for sec in MV.SECTIONS:
            if sec in doc:
                MV.deep_merge(view[sec], doc[sec], sec, merge_owners)
    return problems


def cmd_snapshot(p0):
    """
    EN: Copies metrics/, reports/, data/site_data.json, data/serving/ and data/analysis/ to the P0 folder; refuses if
        it exists.
    TR: metrics/, reports/, data/site_data.json, data/serving/ ve data/analysis/'i P0 klasörüne kopyalar; klasör
        varsa reddeder.
    """
    if p0.exists():
        raise SystemExit(f"{p0} exists; P0 is taken once | P0 bir kez alınır")
    for rel in ("metrics", "reports", "data/serving", "data/analysis"):
        shutil.copytree(ROOT / rel, p0 / rel, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "data" / "site_data.json", p0 / "data" / "site_data.json")
    return p0


def cmd_check_builders(m, p0, done):
    """
    EN: Runs the four builders in a temp root on translated P0 metrics and compares their outputs with P0.
        Returns: list of problems.
    TR: Dört derleyiciyi geçici bir kökte çevrilmiş P0 metrikleriyle koşar ve çıktılarını P0 ile karşılaştırır.
        Döndürür: sorun listesi.
    """
    docs = metrics_docs(p0 / "metrics")
    own = owners(docs, m)
    with tempfile.TemporaryDirectory(prefix="renames-") as tmp:
        root, out = Path(tmp) / "root", Path(tmp) / "out"
        ignore = shutil.ignore_patterns("__pycache__")
        shutil.copytree(ROOT / "builders", root / "builders", ignore=ignore)
        shutil.copytree(ROOT / "analysis" / "lib", root / "analysis" / "lib", ignore=ignore)
        shutil.copytree(p0 / "reports" / "figures", root / "reports" / "figures")
        for name, doc in docs.items():
            target = root / "metrics" / f"{name}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(translate_file(name, doc, m, done), ensure_ascii=False, indent=1),
                              encoding="utf-8")
        env = {**os.environ, "CARDATASYS_OUT": str(out), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for b in BUILDERS:
            r = subprocess.run([sys.executable, str(root / "builders" / b)], env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode:
                return [f"{b} failed | düştü:\n{r.stderr[-3000:]}"]
        md, figs, problems = compare_outputs(out, p0)
        problems += [f"site_data: {k}" for k in site_diff(out / "data" / "site_data.json", m, done, p0, own,
                                                             exempt_patterns(m))]
        labels = read(out / "data" / "serving" / "column_labels.json")
        labels_p0 = read(p0 / "data" / "serving" / "column_labels.json")
        labels.pop("_meta"), labels_p0.pop("_meta")
        if labels != labels_p0:
            problems.append("column_labels differ | farklı")
        print(f"built | derlendi: {len(md)} md · {len(figs)} figures")
        return problems


def cmd_check_metrics(m, p0, done, metrics_dir=ROOT / "metrics"):
    """
    EN: The repository metrics against translated P0, leaf by leaf (order and type count; stamps, code hashes and
        baseline exemptions left out). Returns: {file: [differing leaves]}.
    TR: Depodaki metrikler çevrilmiş P0'a karşı, yaprak yaprak (sıra ve tip sayılır; damgalar, kod hash'leri ve
        referans istisnaları hariç). Döndürür: {dosya: [ayrılan yapraklar]}.
    """
    p0_docs, now = metrics_docs(p0 / "metrics"), metrics_docs(metrics_dir)
    patterns = exempt_patterns(m)
    out = {}
    for name in sorted(set(p0_docs) | set(now)):
        if name not in p0_docs or name not in now:
            out[name] = ["(file missing on one side | dosya bir tarafta yok)"]
            continue
        d = leaf_diff(name, translate_file(name, p0_docs[name], m, done), now[name], patterns, IGNORED)
        if d:
            out[name] = d
    return out


def cmd_check_outputs(m, p0, done):
    """
    EN: The repository reports, figures and site_data against P0 (site_data translated). A difference only in the
        SHAP report is accepted as the known native CatBoost residue when the metrics differ only there.
        Returns: list of problems.
    TR: Depodaki raporlar, figürler ve site_data P0'a karşı (site_data çevrilmiş). Yalnız SHAP raporundaki fark,
        metrikler de yalnız orada ayrılıyorsa bilinen native CatBoost kalıntısı sayılır. Döndürür: sorun listesi.
    """
    own = owners(metrics_docs(p0 / "metrics"), m)
    _, _, problems = compare_outputs(ROOT, p0)
    residue = {"shap.tr.md", "shap.en.md"}
    if problems and set(problems) <= residue and _native_only(m, p0, done):
        print("known residue | bilinen kalıntı: native CatBoost SHAP flipped (docs/reproducibility.md)")
        problems = []
    problems += [f"site_data: {k}" for k in site_diff(ROOT / "data" / "site_data.json", m, done, p0, own,
                                                         exempt_patterns(m))]
    return problems


def _native_only(m, p0, done):
    """
    EN: True if the metrics differ from translated P0 (exemptions included) only in native CatBoost SHAP leaves.
    TR: Metrikler çevrilmiş P0'dan (istisnalar dahil) yalnız native CatBoost SHAP yapraklarında ayrılıyorsa True.
    """
    p0_docs, now = metrics_docs(p0 / "metrics"), metrics_docs(ROOT / "metrics")
    diffs = [k for name in p0_docs for k in leaf_diff(name, translate_file(name, p0_docs[name], m, done),
                                                      now.get(name, {}), [], IGNORED)]
    return bool(diffs) and all("catboost_native" in k or k.endswith("sure_sn") or k.endswith(".seconds")
                               for k in diffs)


def cmd_site_map(m, p0):
    """
    EN: The site tree's old -> new paths (list elements as '[]') and value maps under the full map.
        Returns: {"paths": {old: new}, "values": {new path pattern: {old: new}}}.
    TR: Tam eşlemeyle site ağacının eski -> yeni yolları (liste öğeleri '[]') ve değer eşlemeleri.
        Döndürür: {"paths": {eski: yeni}, "values": {yeni yol kalıbı: {eski: yeni}}}.
    """
    site = read(p0 / "data" / "site_data.json")
    paths = {}

    def walk(x, old, new):
        """EN: Pairs every old site path with its new path. / TR: Her eski site yolunu yeni yoluyla eşler."""
        if isinstance(x, dict):
            for k, v in x.items():
                o = f"{old}.{k}" if old else k
                n_key = k if under(old, m["data_valued"]) else new_name(o, k, m)
                n = f"{new}.{n_key}" if new else n_key
                if o != n and o.split(".")[0] in SITE_SECTIONS:
                    paths[o] = n
                walk(v, o, n)
        elif isinstance(x, list):
            for v in x:
                walk(v, old + "[]", new + "[]")

    walk(site, "", "")
    values = {translate_pattern(p, m): v for p, v in m["values"].items() if p.split(".")[0] in SITE_SECTIONS}
    return {"paths": dict(sorted(paths.items())), "values": values, "site_only": m["site_only"]}


def cmd_mark_done(group, path=MAP_FILE):
    """
    EN: Marks a group done in the map file (key order and LF line ends kept).
    TR: Eşleme dosyasında bir grubu bitti diye işaretler (anahtar sırası ve LF satır sonları korunur).
    """
    doc = read(path)
    if group not in doc["groups"]:
        raise SystemExit(f"unknown group | bilinmeyen grup: {group}")
    doc["groups"][group]["status"] = "done"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def main(argv=None):
    """
    EN: Command line (see the module docstring). Returns: 0 when the check passes, 1 otherwise.
    TR: Komut satırı (modül açıklamasına bakın). Döndürür: sınama geçerse 0, değilse 1.
    """
    ap = argparse.ArgumentParser(description="Metric key renames: apply, prove, export | anahtar adları")
    ap.add_argument("command", choices=["validate", "snapshot", "check-builders", "check-metrics", "check-outputs",
                                        "site-map", "mark-done"])
    ap.add_argument("group", nargs="?", help="mark-done: the group | işaretlenecek grup")
    ap.add_argument("--groups", default="", help="groups to treat as done besides the map's (comma separated)")
    ap.add_argument("--p0", default=str(P0_DIR), help="the P0 copy | P0 kopyası")
    args = ap.parse_args(argv)
    p0 = Path(args.p0)
    if args.command == "snapshot":
        print("P0 taken | alındı:", cmd_snapshot(p0))
        return 0
    if args.command == "mark-done":
        cmd_mark_done(args.group)
        print("done | bitti:", args.group)
        return 0
    if not p0.exists():
        raise SystemExit(f"no P0 copy at {p0} — python tools/metric_renames.py snapshot | P0 yok")
    m = load_map()
    done = done_groups(m, [g for g in args.groups.split(",") if g])
    unknown = done - set(m["groups"])
    if unknown:
        raise SystemExit(f"unknown groups | bilinmeyen gruplar: {sorted(unknown)}")
    if args.command == "site-map":
        print(json.dumps(cmd_site_map(m, p0), ensure_ascii=False, indent=1))
        return 0
    if args.command == "check-metrics":
        diff = cmd_check_metrics(m, p0, done)
        for name, keys in diff.items():
            print(f"✗ {name}: {len(keys)} leaves · {keys[:6]}")
        print(f"check-metrics [{','.join(sorted(done)) or '-'}]: {sum(map(len, diff.values()))} residual difference(s) "
              f"| artık fark")
        return 1 if diff else 0
    problems = {"validate": lambda: cmd_validate(m, p0), "check-builders": lambda: cmd_check_builders(m, p0, done),
                "check-outputs": lambda: cmd_check_outputs(m, p0, done)}[args.command]()
    for p in problems:
        print(f"✗ {p}")
    print(f"{args.command} [{','.join(sorted(done)) or '-'}]: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
