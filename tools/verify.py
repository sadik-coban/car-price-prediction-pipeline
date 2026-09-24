"""
verify.py
EN: The single definition of done: a change is finished when this exits 0. It only selects which tests run and
    summarises them; every check itself is a test under tests/.
      fast  (default)  unit tests, metrics vs baseline, reports rebuilt byte for byte, repository rules; seconds
      --data           + the real-data layer (data/raw, data/cars.duckdb); minutes
      --full           reruns the analysis chain first (python analysis/run_all.py, ~16 min), then everything
    --json prints a one-line summary (used by the Claude Code Stop hook).
TR: Tek "bitti" tanımı: bir değişiklik bu komut 0 ile çıkınca biter. Yalnız hangi testlerin koşacağını seçer ve
    özetler; her kontrolün kendisi tests/ altında bir testtir.
      hızlı (varsayılan)  birim testler, referansa göre metrikler, bayt bayt yeniden üretilen raporlar, depo
                          kuralları; saniyeler
      --data              + gerçek veri katmanı (data/raw, data/cars.duckdb); dakikalar
      --full              önce analiz zincirini yeniden koşar (python analysis/run_all.py, ~16 dk), sonra hepsi
    --json tek satırlık özet basar (Claude Code Stop hook'u kullanır).
Run / Koşum:
    python tools/verify.py [--data | --full] [--json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# EN: pytest marker expression per layer | TR: katman başına pytest işaret ifadesi
LAYERS = {"fast": "not data and not full", "data": "not full", "full": ""}


def failures(junit_xml):
    """
    EN: The failed or erroring tests in a JUnit XML report, as ["<test id>: <first message line>"].
    TR: JUnit XML raporundaki düşen ya da hata veren testler, ["<test kimliği>: <mesajın ilk satırı>"] olarak.
    """
    out = []
    for case in ET.parse(junit_xml).getroot().iter("testcase"):
        for bad in list(case.findall("failure")) + list(case.findall("error")):
            message = (bad.get("message") or bad.text or "").strip().splitlines()
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
            out.append(f"{test_id}: {message[0][:200] if message else ''}")
    return out


def run_pytest(layer):
    """
    EN: Runs the tests of a layer. Returns: (exit code, passed count, failure list, seconds).
    TR: Bir katmanın testlerini koşar. Döndürür: (çıkış kodu, geçen sayısı, düşen listesi, saniye).
    """
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"]
        if LAYERS[layer]:
            cmd += ["-m", LAYERS[layer]]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not report.exists():
            return r.returncode, 0, [f"pytest did not run | koşmadı: {r.stdout[-500:]}{r.stderr[-500:]}"], time.time() - start
        root = ET.parse(report).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        total, skipped = int(suite.get("tests", 0)), int(suite.get("skipped", 0))
        bad = failures(report)
    return r.returncode, total - skipped - len(bad), bad, time.time() - start


def main(argv=None):
    """
    EN: Command line. Returns: 0 when the selected layer is green, 1 otherwise.
    TR: Komut satırı. Döndürür: seçilen katman yeşilse 0, değilse 1.
    """
    ap = argparse.ArgumentParser(description="Definition of done | bitti tanımı")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--data", action="store_true", help="also the real-data layer | gerçek veri katmanı da")
    group.add_argument("--full", action="store_true", help="rerun the analysis chain first | önce zinciri koş")
    ap.add_argument("--json", action="store_true", help="one-line JSON summary | tek satır JSON özet")
    args = ap.parse_args(argv)
    layer = "full" if args.full else "data" if args.data else "fast"
    if args.full:
        chain = subprocess.run([sys.executable, str(ROOT / "analysis" / "run_all.py")], cwd=ROOT)
        if chain.returncode:
            summary = {"ok": False, "layer": layer, "failed": ["analysis/run_all.py failed | düştü"]}
            print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
            return 1
    code, passed, bad, seconds = run_pytest(layer)
    ok = code == 0 and not bad
    summary = {"ok": ok, "layer": layer, "passed": passed, "failed": bad, "seconds": round(seconds, 1)}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(f"verify [{layer}]: {'PASS' if ok else 'FAIL'} · {passed} passed · {len(bad)} failed · {seconds:.1f}s")
        for line in bad:
            print("  ✗", line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
