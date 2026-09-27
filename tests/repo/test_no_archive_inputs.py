"""
test_no_archive_inputs.py
EN: Nothing from the archive reaches the live chain (the user's rule of 2026-09-27, CLAUDE.md). The live code never
    uses the archive/ folder or analysis/frozen/ as a path: every string constant in it (docstrings left out, so
    prose may still name the archive) is checked for an "archive" or "frozen" path segment. And analysis/frozen/,
    which used to hold numbers frozen from an archived run, does not exist.
TR: Arşivden canlı zincire hiçbir şey girmez (kullanıcının 2026-09-27 kuralı, CLAUDE.md). Canlı kod archive/
    klasörünü ya da analysis/frozen/'ı yol olarak asla kullanmaz: içindeki her string sabiti (docstring'ler hariç,
    düzyazı arşivi anabilir) bir "archive" ya da "frozen" yol parçası için sınanır. Eskiden arşivlenmiş bir
    koşudan dondurulmuş sayıları tutan analysis/frozen/ da yoktur.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIRS = ["analysis", "builders", "db", "tools", "internal_tool", "scraper"]
# EN: "archive" / "frozen" as a whole path segment ("archive", "archive/x", "a\\frozen") — not "archived" in prose
# TR: bütün bir yol parçası olarak "archive" / "frozen" ("archive", "archive/x", "a\\frozen") — düzyazıda "archived" değil
SEGMENT = re.compile(r"(^|[/\\])(archive|frozen)([/\\]|$)")


def live_files():
    """EN: Every Python file of the live chain. / TR: Canlı zincirin her Python dosyası."""
    return sorted(p for d in LIVE_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)


def path_constants(source):
    """
    EN: The string constants of a module that name an archive or frozen path, docstrings left out.
        Returns: [(line, text), ...].
    TR: Bir modülün archive ya da frozen yolu anan string sabitleri, docstring'ler hariç.
        Döndürür: [(satır, metin), ...].
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return sorted((n.lineno, n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
                  and SEGMENT.search(n.value))


def test_segment_rule():
    """
    EN: The check finds a path use and leaves prose and docstrings alone.
    TR: Denetim yol kullanımını bulur, düzyazıya ve docstring'lere dokunmaz.
    """
    found = path_constants('"""reads nothing from archive/"""\nP = ROOT / "archive" / "x"\nQ = "a/frozen/b"\n'
                           'M = "the producer is archived"\n')
    assert found == [(2, "archive"), (3, "a/frozen/b")]


@pytest.mark.parametrize("path", live_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_live_code_uses_no_archive_path(path):
    """
    EN: A live file uses no archive or frozen path.
    TR: Canlı bir dosya archive ya da frozen yolu kullanmaz.
    """
    found = path_constants(path.read_text(encoding="utf-8"))
    assert not found, (f"{path.relative_to(ROOT).as_posix()} reads from the archive | arşivden okuyor: {found} — "
                       f"the live chain must produce its own inputs (CLAUDE.md)")


def test_no_frozen_folder():
    """
    EN: analysis/frozen/ does not exist: nothing frozen from an archived run feeds the analysis.
    TR: analysis/frozen/ yok: arşivlenmiş bir koşudan dondurulmuş hiçbir şey analizi beslemez.
    """
    assert not (ROOT / "analysis" / "frozen").exists()
