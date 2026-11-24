"""
test_notebooks.py
Defterler (notebooks/<ad>.ipynb) kaynaklarından (notebooks/<ad>.py) tools/build_notebooks.py ile üretilir. Bu testler:
  - her kaynağın derlenmiş bir defteri olduğunu, hücrelerinin kaynakla aynı olduğunu ve kaynak ya da lib değiştikten
    sonra yeniden derlendiğini (bayat olmadığını),
  - defterin baştan sona, sırayla ve hatasız koştuğunu,
  - defterde kontrol kodu (assert) olmadığını; kontroller tests/ altında durur,
  - her defterin kabul edilmiş başlık sayıları olduğunu
sınar. Veri katmanında (`data` işareti) defterler yeniden koşulur ve başlık sayıları referansla karşılaştırılır.
"""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jupytext
import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
TOOL = ROOT / "tools" / "build_notebooks.py"

spec = importlib.util.spec_from_file_location("build_notebooks", TOOL)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

SOURCES = build.sources()
REBUILD = "yeniden derleyin: python tools/build_notebooks.py"


def built(source):
    """Bir kaynağın derlenmiş defteri."""
    path = source.with_suffix(".ipynb")
    assert path.exists(), f"{path.name} yok; {REBUILD}"
    return nbformat.read(path, as_version=4)


def cell_list(nb):
    """Defterin hücreleri (tür, metin) çiftleri olarak."""
    return [(c.cell_type, c.source) for c in nb.cells]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_matches_its_source(source):
    nb = built(source)
    assert cell_list(nb) == cell_list(jupytext.read(source)), f"{source.stem}: defter ile kaynak farklı; {REBUILD}"
    assert nb.metadata.get("source_digest") == build.source_digest(source), \
        f"{source.stem}: kaynak ya da lib değişti, defter bayat; {REBUILD}"


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_ran_top_to_bottom_without_error(source):
    code = [c for c in built(source).cells if c.cell_type == "code" and c.source.strip()]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    errors = [o for c in code for o in c.outputs if o.output_type == "error"]
    assert not errors


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_holds_no_checks(source):
    for cell in jupytext.read(source).cells:
        if cell.cell_type != "code":
            continue
        plain = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!")))
        asserts = [node.lineno for node in ast.walk(ast.parse(plain)) if isinstance(node, ast.Assert)]
        assert not asserts, f"{source.stem}: defterde assert var; kontroller tests/ altında durur\n{cell.source}"


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_has_accepted_headline_numbers(source):
    baseline = build.BASELINES / f"{source.stem}.json"
    assert baseline.exists(), f"{baseline.name} yok"
    record = json.loads(baseline.read_text(encoding="utf-8"))
    assert record["accepted"] and record["headline"]


@pytest.mark.data
def test_notebooks_rerun_to_the_accepted_headline_numbers():
    r = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
