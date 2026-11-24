"""
build_notebooks.py
Defterlerin kaynağı notebooks/<ad>.py dosyalarıdır (jupytext, percent biçimi). Bu araç her kaynağı, çalışma klasörü
notebooks/ olan taze bir kernel'de baştan sona koşar ve çıktılarıyla notebooks/<ad>.ipynb olarak yazar.
  - Bir hücre düşerse o defter yazılmaz.
  - Defterin tanımladığı `headline` sözlüğü (başlık sayıları) tests/baselines/<ad>.json ile karşılaştırılır. Fark
    varsa defter yazılmaz. Fark bilerek yapıldıysa --accept "<gerekçe>" referansı günceller; bu yalnız kullanıcının
    onayıyla yapılır. Referans dosyaları elle düzenlenmez.
  - Yazılan deftere kaynak özeti konur (kendi .py dosyası ve import ettiği notebooks/lib modülleri). tests/notebooks,
    defterin bayat olup olmadığını bununla anlar.
Defter Jupyter'de elle düzenlendiyse önce kaynağa aktarılır: jupytext --sync notebooks/<ad>.ipynb
Koşum:
    python tools/build_notebooks.py                        bütün defterler
    python tools/build_notebooks.py 1_data_and_cleaning    yalnız adı verilenler
    python tools/build_notebooks.py --check                koşar ve karşılaştırır, hiçbir şey yazmaz
    python tools/build_notebooks.py --accept "gerekçe"     başlık sayılarındaki farkı kabul eder
"""
import argparse
import ast
import hashlib
import json
import sys
import warnings
from pathlib import Path

import jupytext
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

warnings.filterwarnings("ignore", message="Proactor event loop")    # Windows'ta zmq'nun zararsız uyarısı

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
LIB = NOTEBOOKS / "lib"
BASELINES = ROOT / "tests" / "baselines"
MARK = "HEADLINE::"
PROBE = f'import json as _json\nprint("{MARK}" + _json.dumps(headline, ensure_ascii=False))'


def sources(names=()):
    """Koşulacak kaynak dosyaları: adı verilenler ya da notebooks/ altındaki her <ad>.py. Bilinmeyen adda durur."""
    if not names:
        return sorted(NOTEBOOKS.glob("[0-9]*.py"))
    paths = [NOTEBOOKS / f"{Path(n).stem}.py" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"böyle bir defter kaynağı yok: {missing}")
    return paths


def lib_imports(path):
    """Bir .py dosyasının import ettiği lib modüllerinin adları (from lib import x, from lib.x import y, import lib.x)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module == "lib":
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lib."):
            names.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[1] for alias in node.names if alias.name.startswith("lib."))
    return names


def used_lib_files(source):
    """Bir defterin doğrudan ya da başka bir lib modülü üzerinden import ettiği lib dosyaları ve lib/__init__.py."""
    seen, waiting = set(), [source]
    while waiting:
        for name in lib_imports(waiting.pop()) - seen:
            if not (LIB / f"{name}.py").exists():
                raise SystemExit(f"{source.name}: lib/{name}.py yok")
            seen.add(name)
            waiting.append(LIB / f"{name}.py")
    return [LIB / "__init__.py"] + [LIB / f"{name}.py" for name in sorted(seen)]


def source_digest(source):
    """
    Bir defterin kaynak özeti: kendi .py dosyası ile import ettiği lib dosyalarının sha256'sı (satır sonundan
    bağımsız). Defterin kullanmadığı bir lib modülü değişince defter bayatlamaz.
    """
    h = hashlib.sha256()
    for path in [source] + used_lib_files(source):
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(f"{path.name}\n{text}\n".encode("utf-8"))
    return h.hexdigest()


def run(source):
    """Bir kaynağı taze bir kernel'de koşar. Döndürür: (çıktılı defter, headline sözlüğü)."""
    nb = jupytext.read(source)
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:03d}"          # jupytext her okuyuşta rastgele kimlik verir; fark üretmesin
    nb.cells.append(jupytext.reads(f"# %%\n{PROBE}\n", fmt="py:percent").cells[0])
    client = NotebookClient(nb, timeout=None, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(NOTEBOOKS)}})
    try:
        client.execute()
    except CellExecutionError as e:
        raise SystemExit(f"DÜŞTÜ: {source.relative_to(ROOT).as_posix()} (yazılmadı)\n{e}")
    probe = nb.cells.pop()
    printed = "".join(o.get("text", "") for o in probe.outputs if o.get("output_type") == "stream")
    lines = [line for line in printed.splitlines() if line.startswith(MARK)]
    if len(lines) != 1:
        raise SystemExit(f"{source.name}: `headline` sözlüğü okunamadı")
    return nb, json.loads(lines[0][len(MARK):])


def differences(recorded, fresh):
    """Referans ile yeni başlık sayıları arasındaki farklar, okunur satırlar olarak (fark yoksa boş liste)."""
    out = []
    for key in sorted(set(recorded) | set(fresh)):
        if key not in fresh:
            out.append(f"  - {key}: {recorded[key]!r} (artık yok)")
        elif key not in recorded:
            out.append(f"  + {key}: {fresh[key]!r} (yeni)")
        elif recorded[key] != fresh[key]:
            out.append(f"  ~ {key}: {recorded[key]!r} -> {fresh[key]!r}")
    return out


def main(argv=None):
    """Komut satırı. Döndürür: her defter koşup başlık sayıları referansla aynıysa 0, değilse 1."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Defterleri kaynaklarından derler")
    ap.add_argument("names", nargs="*", help="notebooks/ altındaki defter adları, örn. 1_data_and_cleaning")
    ap.add_argument("--check", action="store_true", help="koşar ve karşılaştırır, hiçbir şey yazmaz")
    ap.add_argument("--accept", metavar="GEREKÇE", help="başlık sayılarındaki farkı bu gerekçeyle kabul eder")
    args = ap.parse_args(argv)
    if args.check and args.accept:
        ap.error("--check ile --accept birlikte kullanılmaz")
    failed = False
    for source in sources(args.names):
        name = source.stem
        nb, headline = run(source)
        baseline = BASELINES / f"{name}.json"
        recorded = json.loads(baseline.read_text(encoding="utf-8"))["headline"] if baseline.exists() else {}
        diff = differences(recorded, headline)
        if diff and not args.accept:
            failed = True
            print(f"FARK: {name} başlık sayıları referanstan farklı (defter yazılmadı):", *diff, sep="\n", flush=True)
            continue
        if args.check:
            print("tamam:", name, flush=True)
            continue
        if diff:
            BASELINES.mkdir(parents=True, exist_ok=True)
            record = {"accepted": args.accept, "headline": headline}
            baseline.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"kabul edildi: {baseline.relative_to(ROOT).as_posix()}", *diff, sep="\n", flush=True)
        nb.metadata["source_digest"] = source_digest(source)
        jupytext.write(nb, source.with_suffix(".ipynb"), fmt="ipynb")
        print("yazıldı:", source.with_suffix(".ipynb").relative_to(ROOT).as_posix(), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
