"""
test_notebooks.py
Defterler (notebooks/<ad>.ipynb) kaynaklarından (notebooks/<ad>.py) tools/build_notebooks.py ile üretilir. Bu testler:
  - her kaynağın derlenmiş bir defteri olduğunu, hücrelerinin kaynakla aynı olduğunu ve kaynak ya da lib değiştikten
    sonra yeniden derlendiğini (bayat olmadığını),
  - defterin baştan sona, sırayla ve hatasız koştuğunu,
  - defterde kontrol kodu (assert) olmadığını; kontroller tests/ altında durur,
  - defterin bir veri kolonunu yalnız veride görülmüş değerlerle karşılaştırdığını (db/observed_values.json),
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


def text_constants(node):
    """Bir ifadedeki metin sabitleri: tek sabit ya da sabitlerden kurulu liste, demet, küme."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else [node]
    return [item.value for item in items if isinstance(item, ast.Constant) and isinstance(item.value, str)]


def named_column(node):
    """frame["kolon"] biçimindeki bir ifadenin kolon adı; başka her ifadede None."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def column_literals(tree):
    """
    Adıyla seçilen bir kolonun metin sabitleriyle karşılaştırıldığı yerler: (satır, kolon, sabit).
    Görülen biçimler: frame["kolon"] == "değer" (ve !=, in, not in) ile frame["kolon"].isin(["değer", ...]).
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sides = [node.left] + node.comparators
            columns = [named_column(side) for side in sides if named_column(side)]
            literals = [text for side in sides if not named_column(side) for text in text_constants(side)]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "isin":
            columns = [named_column(node.func.value)] if named_column(node.func.value) else []
            literals = [text for arg in node.args for text in text_constants(arg)]
        else:
            continue
        found += [(node.lineno, column, text) for column in columns for text in literals]
    return found


def test_column_literals_finds_the_compared_values():
    tree = ast.parse('a = frame["brand"] == "bmw"\n'
                     'b = frame.loc[frame["kb_fuel"].isin(["Benzin", "Dizel"]), "price"]\n'
                     'c = "SUV" != frame["kb_body_type"]\n'
                     'd = name == "x"\n'
                     'e = frame["model"].isin(models)\n')
    assert sorted(column_literals(tree)) == [(1, "brand", "bmw"), (2, "kb_fuel", "Benzin"), (2, "kb_fuel", "Dizel"),
                                             (3, "kb_body_type", "SUV")]


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_notebook_compares_data_columns_only_with_observed_values(source):
    """
    Defterin bir veri kolonunu karşılaştırdığı her metin sabiti gözlenen değerler kaydında (db/observed_values.json)
    o kolonun bir değeri, ya da boş değerin dolgusu. Kayıtta olmayan kolonlar (kodun kurduğu kimlikler) taranmaz;
    etiket olarak yazılan değerler (.loc["bmw"], reindex) de taranmaz, çünkü hangi kolona ait oldukları koddan okunmaz.
    """
    columns = json.loads((ROOT / "db" / "observed_values.json").read_text(encoding="utf-8"))["silver"]["columns"]
    fill = "missing"                                    # data.add_features boş kategorik değere bunu yazar
    for line, column, text in column_literals(ast.parse(source.read_text(encoding="utf-8"))):
        if column not in columns:
            continue
        observed = {value for value, _ in columns[column]["values"] if value is not None}
        assert len(observed) == columns[column]["distinct"], f"{column}: kayıt bu kolonun değerlerini tam tutmuyor"
        assert text in observed | {fill}, f"{source.name}:{line}: {column} kolonunda '{text}' hiç görülmedi"


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
