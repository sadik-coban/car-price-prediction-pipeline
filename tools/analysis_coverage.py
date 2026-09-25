"""
analysis_coverage.py
EN: The analysis coverage matrix: for every script in analysis/run_all.ORDER, four cells —
      card      analysis/cards/<script>.json is complete (every field, tr + en, the four leakage types, evidence
                that exists: its test ids are collected by pytest, its report keys are in the script's metrics);
      test      tests/analysis/test_<script>.py has at least one real test (not the scaffold's `template` stub);
      baseline  metrics/<script>.json is in the metrics baseline (tests/baselines/metrics_shape.json);
      source    the metrics were written by the current code (_meta.source hashes match the files).
    A script on the legacy list (tests/analysis/legacy.json) is exempt from the card and test cells while its
    source still has the listed sha256; once it changes, it needs a card and a test. The test in
    tests/analysis/test_coverage.py fails on any empty cell. Test ids are checked with `pytest --collect-only`,
    never by reading test source. The printed table also shows each script's generated_at (when it last wrote
    its metrics), so after an edit you can see what ran when.
TR: Analiz kapsam matrisi: analysis/run_all.ORDER'daki her betik için dört hücre —
      card      analysis/cards/<betik>.json tam (her alan, tr + en, dört sızıntı tipi, var olan kanıt: test
                kimliklerini pytest topluyor, rapor anahtarları betiğin metriğinde);
      test      tests/analysis/test_<betik>.py'de en az bir gerçek test var (iskeletin `template` taslağı değil);
      baseline  metrics/<betik>.json metrik referansında (tests/baselines/metrics_shape.json);
      source    metrikleri bugünkü kod yazmış (_meta.source hash'leri dosyalarla aynı).
    Eski betik listesindeki (tests/analysis/legacy.json) bir betik, kaynağı listelenen sha256'yı taşıdıkça kart
    ve test hücrelerinden muaf; değişince kart ve test ister. tests/analysis/test_coverage.py tek boş hücrede
    düşer. Test kimlikleri `pytest --collect-only` ile sınanır, test kaynağı hiç okunmaz. Basılan tablo her
    betiğin generated_at'ını da (metriğini en son yazdığı an) gösterir; bir düzenlemeden sonra neyin ne zaman
    koştuğu görülür.
Run / Koşum:
    python tools/analysis_coverage.py [--json]
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
CARDS = ANALYSIS / "cards"
TESTS = ROOT / "tests" / "analysis"
LEGACY = TESTS / "legacy.json"
BASELINE_SHAPE = ROOT / "tests" / "baselines" / "metrics_shape.json"
SPLITS = ("none", "kfold_oof", "oof_input", "temporal")
LEAK_TYPES = ("duplicate_ad_id", "target_in_features", "fit_on_full_data", "temporal")
LEAK_STATUS = ("none", "present", "unknown", "n/a")
TEXT_FIELDS = ("question", "data", "method")
OK = ("ok", "legacy")


def order():
    """
    EN: run_all.ORDER, loaded as a module (run_all imports nothing from lib). Returns: list of "NN_x.py" paths.
    TR: run_all.ORDER, modül olarak yüklenir (run_all lib'den hiçbir şey import etmez). Döndürür: "NN_x.py" listesi.
    """
    spec = importlib.util.spec_from_file_location("run_all", ANALYSIS / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.ORDER)


def on_disk():
    """EN: Numbered analysis scripts on disk. / TR: Diskteki numaralı analiz betikleri."""
    return sorted(p.relative_to(ANALYSIS).as_posix() for p in ANALYSIS.rglob("[0-9][0-9]_*.py")
                  if "__pycache__" not in p.parts)


def source_sha(path):
    """
    EN: sha256 of a file's bytes with CRLF folded to LF (git may check files out either way).
    TR: CRLF'i LF'e indirilmiş dosya baytlarının sha256'sı (git dosyayı iki biçimde de çıkarabilir).
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def names(script):
    """
    EN: The file names that belong to a script ("shap/02_oof_shap.py" → card, test, metrics paths).
    TR: Bir betiğe ait dosya adları ("shap/02_oof_shap.py" → kart, test, metrik yolları).
    """
    stem = script[:-3]
    return {"stem": stem, "card": CARDS / f"{stem}.json", "test": TESTS / f"test_{stem.replace('/', '_')}.py",
            "metrics": ROOT / "metrics" / f"{stem}.json"}


def collected(marker_expr):
    """
    EN: Node ids pytest collects under tests/analysis for a marker expression (parameters stripped).
    TR: pytest'in bir işaret ifadesiyle tests/analysis altında topladığı düğüm kimlikleri (parametreler atılır).
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker_expr,
                        str(TESTS.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return {line.strip().split("[")[0].replace("\\", "/") for line in r.stdout.splitlines() if "::" in line}


def has_path(doc, dotted):
    """EN: True if the dotted key path exists in doc. / TR: Noktalı anahtar yolu doc'ta varsa True."""
    o = doc
    for k in dotted.split("."):
        if not isinstance(o, dict) or k not in o:
            return False
        o = o[k]
    return True


def card_problems(card, metrics_doc, real_tests):
    """
    EN: What is missing or wrong in a card (empty list = complete).
    TR: Bir kartta eksik ya da yanlış olan (boş liste = tam).
    """
    p = []
    if card.get("status") != "complete":
        p.append(f"status is {card.get('status')!r}, not 'complete'")
    for f in TEXT_FIELDS:
        v = card.get(f) or {}
        if not (isinstance(v, dict) and str(v.get("tr", "")).strip() and str(v.get("en", "")).strip()):
            p.append(f"{f}: tr and en needed")
    if card.get("split") not in SPLITS:
        p.append(f"split must be one of {SPLITS}")
    leak = card.get("leakage") or {}
    for t in LEAK_TYPES:
        e = leak.get(t) or {}
        if e.get("status") not in LEAK_STATUS:
            p.append(f"leakage.{t}: status must be one of {LEAK_STATUS}")
        elif e["status"] in ("present", "unknown") and not (e.get("note") or {}).get("tr"):
            p.append(f"leakage.{t}: a {e['status']} risk needs a note")
    ev = card.get("evidence") or {}
    if not ev.get("tests"):
        p.append("evidence.tests is empty")
    p += [f"evidence test not collected: {t}" for t in ev.get("tests", []) if t not in real_tests]
    if not ev.get("report_keys"):
        p.append("evidence.report_keys is empty")
    if metrics_doc is not None:
        p += [f"report key not in the metrics: {k}" for k in ev.get("report_keys", []) if not has_path(metrics_doc, k)]
    if "TODO" in json.dumps(card, ensure_ascii=False):
        p.append("a TODO is left")
    return p


def stale_sources(metrics_doc):
    """
    EN: The files whose hash in _meta.source no longer matches (None when there is no _meta.source).
    TR: _meta.source'taki hash'i artık tutmayan dosyalar (_meta.source yoksa None).
    """
    src = (metrics_doc.get("_meta") or {}).get("source")
    if not src:
        return None
    return [rel for rel, sha in {**src["script"], **src["lib"]}.items()
            if not (ROOT / rel).exists() or source_sha(ROOT / rel) != sha]


def matrix():
    """
    EN: One row per ORDER script: {"script", "card", "test", "baseline", "source", "generated_at", "problems"};
        generated_at = when the script last wrote its metrics (_meta.generated_at), None without metrics.
    TR: ORDER'daki her betik için bir satır: {"script", "card", "test", "baseline", "source", "generated_at",
        "problems"}; generated_at = betiğin metriğini en son yazdığı an (_meta.generated_at), metrik yoksa None.
    """
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))["scripts"] if LEGACY.exists() else {}
    baseline = json.loads(BASELINE_SHAPE.read_text(encoding="utf-8"))
    real, stubs = collected("not template"), collected("template")
    rows = []
    for script in order():
        n, problems = names(script), []
        doc = json.loads(n["metrics"].read_text(encoding="utf-8")) if n["metrics"].exists() else None
        test_rel = n["test"].relative_to(ROOT).as_posix()
        is_legacy = script in legacy
        legacy_ok = is_legacy and source_sha(ANALYSIS / script) == legacy[script]
        if legacy_ok:
            card = test = "legacy"
        else:
            if is_legacy:
                problems.append("legacy source changed: write its card and test, then drop it from legacy.json")
            if n["card"].exists():
                cp = card_problems(json.loads(n["card"].read_text(encoding="utf-8")), doc, real)
                card = "ok" if not cp else "todo"
                problems += [f"card: {m}" for m in cp]
            else:
                card = "missing"
                problems.append(f"card missing: {n['card'].relative_to(ROOT).as_posix()}")
            mine_real = {t for t in real if t.startswith(test_rel + "::")}
            mine_stub = {t for t in stubs if t.startswith(test_rel + "::")}
            test = "ok" if mine_real else ("stub" if mine_stub else "missing")
            if test != "ok":
                problems.append(f"test {test}: {test_rel} needs at least one real test")
        base = "ok" if n["stem"] in baseline else "missing"
        if base != "ok":
            problems.append("metrics not in the baseline: python tools/snapshot_metrics.py --accept \"<reason>\"")
        stale = stale_sources(doc) if doc is not None else None
        source = "none" if stale is None else ("ok" if not stale else "stale")
        if source != "ok":
            problems.append(f"source {source}: rerun python analysis/{script}"
                            + (f" (changed: {', '.join(stale)})" if stale else ""))
        rows.append({"script": script, "card": card, "test": test, "baseline": base, "source": source,
                     "generated_at": (doc or {}).get("_meta", {}).get("generated_at"), "problems": problems})
    return rows


def main(argv=None):
    """
    EN: Prints the matrix (or --json). Returns: 0 when every cell is filled, 1 otherwise.
    TR: Matrisi basar (ya da --json). Döndürür: her hücre doluysa 0, değilse 1.
    """
    ap = argparse.ArgumentParser(description="Analysis coverage matrix | analiz kapsam matrisi")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = matrix()
    extra = sorted(set(on_disk()) - set(order()))
    if args.json:
        print(json.dumps({"rows": rows, "not_in_order": extra}, ensure_ascii=False, indent=1))
    else:
        print(f"{'script':34} {'card':8} {'test':8} {'baseline':9} {'source':7} generated_at")
        for r in rows:
            print(f"{r['script']:34} {r['card']:8} {r['test']:8} {r['baseline']:9} {r['source']:7} "
                  f"{r['generated_at'] or '-'}")
        for r in rows:
            for m in r["problems"]:
                print(f"  ✗ {r['script']}: {m}")
        for s in extra:
            print(f"  ✗ {s}: on disk but not in run_all.ORDER | diskte var, ORDER'da yok")
    return 0 if not extra and not any(r["problems"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
