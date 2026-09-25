"""
catalog.py
EN: What the internal tool knows about the analysis scripts, read-only: the section map (which scripts feed which
    report section, which script each figure comes from; internal_tool/section_map.json), each script's card,
    metrics _meta (when it last ran, run_id, the DB it read, whether its code changed since — reusing
    tools/analysis_coverage.py), and the text of a script or a test file for display. read_source is the one place
    that reads a .py file as text: it is shown to a human, never parsed (the hygiene test lists this exception).
    Pure: no streamlit import. --stamp records the builders' current sha256 in the map after a review.
TR: İç aracın analiz betikleri hakkında bildikleri, salt okunur: bölüm eşlemesi (hangi betikler hangi rapor bölümünü
    besliyor, her figür hangi betikten geliyor; internal_tool/section_map.json), her betiğin kartı, metrik _meta'sı
    (en son ne zaman koştu, run_id, okuduğu DB, o zamandan beri kodu değişti mi — tools/analysis_coverage.py yeniden
    kullanılır) ve gösterim için bir betiğin ya da test dosyasının metni. read_source bir .py dosyasını metin olarak
    okuyan tek yerdir: insana gösterilir, asla ayrıştırılmaz (hijyen testi bu istisnayı listeler). Saf: streamlit
    import etmez. --stamp, gözden geçirmeden sonra derleyicilerin güncel sha256'sını eşlemeye yazar.
Run / Koşum:
    python internal_tool/catalog.py --stamp
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "internal_tool" / "section_map.json"
# EN: the builder files the map was read from | TR: eşlemenin okunduğu derleyici dosyaları
REVIEWED = ("builders/build_technical_report.py", "builders/build_business_report.py",
            "builders/build_shap_report.py", "builders/report_lib/report_common.py")
SOURCE_ROOTS = ("analysis", "tests")
MAX_ITEMS = 50


@functools.lru_cache(maxsize=1)
def coverage():
    """
    EN: tools/analysis_coverage.py loaded by path (standard library only; no side effects on import).
    TR: Yol üzerinden yüklenen tools/analysis_coverage.py (yalnız standart kütüphane; import anında yan etkisi yok).
    """
    spec = importlib.util.spec_from_file_location("internal_tool_coverage", ROOT / "tools" / "analysis_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def order():
    """EN: The analysis scripts in run order ("01_x.py", "shap/02_y.py"). / TR: Koşum sırasıyla analiz betikleri."""
    return coverage().order()


def load_map(path=MAP_PATH):
    """EN: The section map. / TR: Bölüm eşlemesi."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sections_of(script, m):
    """
    EN: Where a script is used. Returns: [(report, section index, title)] in report order.
    TR: Bir betiğin kullanıldığı yerler. Döndürür: [(rapor, bölüm sırası, başlık)], rapor sırasıyla.
    """
    return [(report, i, s["title"]) for report, secs in m["reports"].items()
            for i, s in enumerate(secs) if script in s["scripts"]]


def file_sha(path, fold_crlf=True):
    """
    EN: sha256 of a file's bytes; CRLF folded to LF for source files (git may check them out either way), raw bytes
        for data files (fold_crlf=False, as save_metrics hashes the DB).
    TR: Dosya baytlarının sha256'sı; kaynak dosyalarda CRLF LF'e indirilir (git iki biçimde de çıkarabilir), veri
        dosyalarında ham bayt (fold_crlf=False; save_metrics DB'yi böyle hash'ler).
    """
    h = hashlib.sha256()
    if fold_crlf:
        h.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    else:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def map_drift(m, root=ROOT):
    """
    EN: The builder files that changed since the map was reviewed (or that it was never stamped against).
    TR: Eşleme gözden geçirildikten sonra değişen (ya da hiç damgalanmamış) derleyici dosyaları.
    """
    seen = m.get("reviewed_against") or {}
    return [rel for rel in REVIEWED if seen.get(rel) != file_sha(Path(root) / rel)]


def stamp(path=MAP_PATH, root=ROOT):
    """
    EN: Records the builders' current sha256 as reviewed_against in the map file (LF, key order kept).
    TR: Derleyicilerin güncel sha256'sını eşleme dosyasına reviewed_against olarak yazar (LF, anahtar sırası korunur).
    """
    m = load_map(path)
    m["reviewed_against"] = {rel: file_sha(Path(root) / rel) for rel in REVIEWED}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    return m


def read_json(path):
    """EN: A JSON file, or None if it does not exist. / TR: Bir JSON dosyası; yoksa None."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def files_of(script):
    """
    EN: A script's files: {"stem", "card", "test", "metrics"} (paths) plus "source" and "test_source" stems
        relative to the repository, for read_source.
    TR: Bir betiğin dosyaları: {"stem", "card", "test", "metrics"} (yollar) ve read_source için depoya göreli
        "source" ile "test_source" kökleri.
    """
    f = coverage().names(script)
    return {**f, "source": f"analysis/{f['stem']}", "test_source": Path(f["test"]).relative_to(ROOT).with_suffix("")
            .as_posix()}


def read_source(stem, root=ROOT):
    """
    EN: The text of <stem>.py for display (stem like "analysis/08_residuals" or "tests/analysis/test_x"). Only
        files under analysis/ or tests/ are served. Raises ValueError otherwise, FileNotFoundError if missing.
    TR: Gösterim için <stem>.py'nin metni (stem ör. "analysis/08_residuals" ya da "tests/analysis/test_x"). Yalnız
        analysis/ ya da tests/ altındaki dosyalar verilir; değilse ValueError, dosya yoksa FileNotFoundError.
    """
    parts = Path(stem).parts
    if not parts or parts[0] not in SOURCE_ROOTS or ".." in parts or Path(stem).is_absolute():
        raise ValueError(f"not an analysis or test file | analiz ya da test dosyası değil: {stem}")
    return (Path(root) / f"{stem}.py").read_text(encoding="utf-8")


def meta_summary(metrics_doc, silver_sha=None):
    """
    EN: The run facts of a metrics file: {"generated_at", "data_until", "run_id", "db_rows", "stale", "db_matches"}.
        stale: None (no provenance), [] (code unchanged) or the changed files; db_matches: whether the DB the
        script read is the current silver (None when silver_sha is not given).
    TR: Bir metrik dosyasının koşum bilgileri: {"generated_at", "data_until", "run_id", "db_rows", "stale",
        "db_matches"}. stale: None (kaynak izi yok), [] (kod değişmedi) ya da değişen dosyalar; db_matches:
        betiğin okuduğu DB güncel silver mı (silver_sha verilmezse None).
    """
    if metrics_doc is None:
        return None
    meta = metrics_doc.get("_meta") or {}
    db = meta.get("db") or {}
    return {"generated_at": meta.get("generated_at"), "data_until": meta.get("data_until"),
            "run_id": meta.get("run_id"), "db_rows": db.get("rows"),
            "stale": coverage().stale_sources(metrics_doc),
            "db_matches": None if silver_sha is None else db.get("sha256") == silver_sha}


def shorten(obj, max_items=MAX_ITEMS):
    """
    EN: A copy of a metrics document for display: every list longer than max_items becomes "[list of N]".
    TR: Gösterim için metrik belgesinin kopyası: max_items'tan uzun her liste "[N öğeli liste]" olur.
    """
    if isinstance(obj, dict):
        return {k: shorten(v, max_items) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_items:
            return f"[{len(obj)} öğeli liste · list of {len(obj)}]"
        return [shorten(v, max_items) for v in obj]
    return obj


def main(argv=None):
    """EN: Command line (--stamp). Returns: exit code. / TR: Komut satırı (--stamp). Döndürür: çıkış kodu."""
    ap = argparse.ArgumentParser(description="Section map of the internal tool | iç aracın bölüm eşlemesi")
    ap.add_argument("--stamp", action="store_true", help="record the reviewed builders' sha256 | derleyicileri damgala")
    args = ap.parse_args(argv)
    if args.stamp:
        stamp()
        print(f"stamped | damgalandı: {', '.join(REVIEWED)}")
    else:
        drift = map_drift(load_map())
        print("map current | eşleme güncel" if not drift else f"builders changed since review | değişti: {drift}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
