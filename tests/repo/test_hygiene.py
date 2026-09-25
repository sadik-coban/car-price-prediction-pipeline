"""
test_hygiene.py
EN: Repository rules checked by code instead of by eye:
      - every module, function and class has a bilingual docstring (EN: and TR:);
      - every relative link in the README files, docs/ and reports/ points to a file that exists;
      - no code reads another source file as text or parses it with ast (data must come from data files, not
        from someone else's code). The one deliberate exception is listed with its reason.
TR: Göz yerine kodla denetlenen depo kuralları:
      - her modülün, fonksiyonun ve sınıfın iki dilli docstring'i var (EN: ve TR:);
      - README'lerdeki, docs/ ve reports/'taki her göreli bağlantı var olan bir dosyaya çıkıyor;
      - hiçbir kod başka bir kaynak dosyayı metin olarak okumuyor ya da ast ile ayrıştırmıyor (veri başkasının
        kodundan değil veri dosyasından gelmeli). Tek bilinçli istisna gerekçesiyle listede.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ["analysis", "builders", "db", "tools", "tests", "internal_tool"]
EXTRA_CODE = ["scraper/collection.py"]
DOC_FILES = ["README.md", "README.en.md", "db_plan.md", "docs/*.md", "reports/*.md"]
# EN: the only code allowed to read a .py file as text, with the reason | TR: .py'yi metin olarak okuyabilen tek kod
SOURCE_READ_ALLOWED = {
    "analysis/01_engine_rule.py": "checks that lib/common.py still applies CHOSEN (owner kept it, 2026-09-24) | "
                                  "lib/common.py'nin hâlâ CHOSEN'ı uyguladığını sınar (kullanıcı tuttu)",
}
SOURCE_READ = re.compile(r"""\.py["']\s*\)\s*\.read_text|open\([^)]*\.py["']|^\s*(import ast\b|from ast import)""", re.M)


def code_files():
    """EN: Every Python file under the code folders. / TR: Kod klasörlerindeki her Python dosyası."""
    files = [p for d in CODE_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files + [ROOT / f for f in EXTRA_CODE if (ROOT / f).exists()])


def rel(path):
    """EN: Repository-relative POSIX path. / TR: Depoya göreli POSIX yolu."""
    return path.relative_to(ROOT).as_posix()


@pytest.mark.parametrize("path", code_files(), ids=rel)
def test_bilingual_docstrings(path):
    """EN: The module and each function/class carry EN: and TR:. / TR: Modül ve her fonksiyon/sınıf EN: ve TR: taşır."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [("<module>", tree)] + [(n.name, n) for n in ast.walk(tree)
                                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    missing = [name for name, node in nodes if not {"EN:", "TR:"} <= set(re.findall(r"EN:|TR:", ast.get_docstring(node) or ""))]
    assert not missing, f"{rel(path)}: no bilingual docstring | iki dilli docstring yok: {missing}"


def doc_files():
    """EN: The markdown files whose links are checked. / TR: Bağlantıları denetlenen markdown dosyaları."""
    return sorted({p for pattern in DOC_FILES for p in ROOT.glob(pattern)})


@pytest.mark.parametrize("path", doc_files(), ids=rel)
def test_relative_links(path):
    """EN: Every relative link resolves (code blocks ignored). / TR: Her göreli bağlantı çözülüyor (kod blokları hariç)."""
    body = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    broken = [t for t in re.findall(r"\]\(([^)\s]+)\)", body)
              if not t.startswith(("http://", "https://", "#", "mailto:")) and not (path.parent / t.split("#")[0]).exists()]
    assert not broken, f"{rel(path)}: broken links | kırık bağlantı: {broken}"


def test_no_source_read_as_text():
    """
    EN: No code outside tests/ reads a .py file as text or imports ast, except the listed exception.
    TR: tests/ dışında hiçbir kod .py dosyasını metin olarak okumuyor ya da ast import etmiyor; listedeki istisna hariç.
    """
    found = [rel(p) for p in code_files() if not rel(p).startswith("tests/") and SOURCE_READ.search(p.read_text(encoding="utf-8"))]
    unexpected = [f for f in found if f not in SOURCE_READ_ALLOWED]
    assert not unexpected, f"reads source code as data | kaynak kodu veri gibi okuyor: {unexpected}"
    stale = [f for f in SOURCE_READ_ALLOWED if f not in found]
    assert not stale, f"exception no longer needed, remove it | istisna artık gereksiz, kaldır: {stale}"
