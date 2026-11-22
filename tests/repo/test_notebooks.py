"""
test_notebooks.py
EN: The notebooks under notebooks/ never write and never train. These are the owner's rules of 2026-11-22: the heavy
    parts come from the record, the light analyses may run in the notebook. A light script runs through
    analysis/lib/records.py:run, which executes its cells [1]–[5]; its save cell [6] is under
    `if __name__ == "__main__"`. A heavy script (records.HEAVY) is only read from metrics/. Checked:
      - every analysis script is either in HEAVY or guards its save cell, so run() can never write a record; every
        HEAVY name is a script; run() refuses a heavy script;
      - no code cell writes a file or fits a model (save_metrics, save_oof, lgb_oof, fit_lgb_fold, json.dump,
        .fit, .savefig, .to_parquet, .to_csv, .to_json, .to_excel, .write_text, .write_bytes);
      - no code cell names an archive/ or frozen path (CLAUDE.md: nothing from the archive);
      - every relative link in a markdown cell (figures, scripts, cards, reports) points to a file that exists;
      - the saved outputs carry no error (tools/run_notebooks.py writes a notebook only after a clean run).
TR: notebooks/ altındaki defterler hiç yazmaz ve hiç eğitmez. Kullanıcının 2026-11-22 kuralları: ağır kısımlar
    kayıttan gelir, hafif analizler defterde koşabilir. Hafif bir betik analysis/lib/records.py:run ile koşar; run
    betiğin [1]–[5] hücrelerini çalıştırır, kaydet hücresi [6] `if __name__ == "__main__"` altındadır. Ağır bir betik
    (records.HEAVY) yalnız metrics/'ten okunur. Denetlenen:
      - her analiz betiği ya HEAVY'de ya da kaydet hücresi korumalı; böylece run() hiçbir zaman kayıt yazamaz; her
        HEAVY adı bir betik; run() ağır betiği reddediyor;
      - hiçbir kod hücresi dosya yazmıyor ya da model uydurmuyor (save_metrics, save_oof, lgb_oof, fit_lgb_fold,
        json.dump, .fit, .savefig, .to_parquet, .to_csv, .to_json, .to_excel, .write_text, .write_bytes);
      - hiçbir kod hücresi archive/ ya da frozen yolu anmıyor (CLAUDE.md: arşivden hiçbir şey);
      - markdown hücrelerindeki her göreli bağlantı (figür, betik, kart, rapor) var olan bir dosyaya çıkıyor;
      - kayıtlı çıktılarda hata yok (tools/run_notebooks.py bir defteri yalnız temiz koşumdan sonra yazar).
"""
import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
# EN: every question script (lib/ and the runner left out) | TR: her soru betiği (lib/ ve koşturucu hariç)
SCRIPTS = sorted(p for p in [*(ROOT / "analysis").glob("*.py"), *(ROOT / "analysis" / "shap").glob("*.py")]
                 if p.name != "run_all.py")
GUARD = 'if __name__ == "__main__":'


def load_records():
    """
    EN: analysis/lib/records.py loaded by its path (in this pytest session the name `lib` means db/lib).
    TR: Yolundan yüklenen analysis/lib/records.py (bu pytest oturumunda `lib` adı db/lib demek).
    """
    spec = importlib.util.spec_from_file_location("notebook_records", ROOT / "analysis" / "lib" / "records.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDS = load_records()


def script_name(path):
    """EN: "08_residuals" or "shap/03_what_sets_price". / TR: "08_residuals" ya da "shap/03_what_sets_price"."""
    return path.relative_to(ROOT / "analysis").with_suffix("").as_posix()


def test_heavy_names_are_scripts():
    """EN: Every HEAVY name is an existing script. / TR: Her HEAVY adı var olan bir betik."""
    names = {script_name(p) for p in SCRIPTS}
    assert set(RECORDS.HEAVY) <= names, sorted(set(RECORDS.HEAVY) - names)


@pytest.mark.parametrize("path", SCRIPTS, ids=script_name)
def test_light_script_guards_its_save_cell(path):
    """
    EN: A script outside HEAVY has one save cell [6], and everything in it sits under the __main__ guard, so running
        it from a notebook writes nothing.
    TR: HEAVY dışındaki bir betiğin tek kaydet hücresi [6] var ve içindeki her şey __main__ korumasının altında; böylece
        defterden koşmak hiçbir şey yazmaz.
    """
    if script_name(path) in RECORDS.HEAVY:
        pytest.skip("heavy: read from the record | ağır: kayıttan okunur")
    text = path.read_text(encoding="utf-8")
    assert text.count("\n# %% [6]") == 1, f"{script_name(path)}: one save cell [6] | tek kaydet hücresi"
    body = [line for line in text.split("\n# %% [6]")[1].split("\n")[1:] if line.strip()]
    assert body and body[0].startswith(GUARD), f"{script_name(path)}: save cell not guarded | kaydet hücresi korumasız"
    assert all(line.startswith((" ", "\t")) for line in body[1:]), \
        f"{script_name(path)}: a save line outside the guard | korumanın dışında kaydet satırı"


def test_run_refuses_heavy_and_unknown():
    """
    EN: run() stops on a heavy script and on a name that is no script, before loading anything.
    TR: run() ağır betikte ve betik olmayan bir adda, hiçbir şey yüklemeden durur.
    """
    with pytest.raises(ValueError, match="heavy"):
        RECORDS.run(next(iter(RECORDS.HEAVY)))
    with pytest.raises(ValueError, match="no such script"):
        RECORDS.run("99_no_such_script")
WRITES_OR_RUNS = re.compile(r"\b(?:save_metrics|save_oof|lgb_oof|fit_lgb_fold|json\.dump)\s*\(|"
                            r"\.(?:fit|savefig|to_parquet|to_csv|to_json|to_excel|write_text|write_bytes)\s*\(")
# EN: "archive" / "frozen" as a whole path segment inside a string | TR: string içinde bütün bir yol parçası olarak
ARCHIVE = re.compile(r"""(^|[/\\"'])(archive|frozen)([/\\"']|$)""", re.M)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def rel(path):
    """EN: Repository-relative POSIX path. / TR: Depoya göreli POSIX yolu."""
    return path.relative_to(ROOT).as_posix()


def cells(path, kind):
    """EN: The cells of one kind ("code" / "markdown") of a notebook. / TR: Bir defterin bir türdeki hücreleri."""
    return [c for c in nbformat.read(path, as_version=4).cells if c.cell_type == kind]


def test_there_are_notebooks():
    """EN: The folder is not empty, so the checks below run. / TR: Klasör boş değil; aşağıdaki denetimler koşuyor."""
    assert NOTEBOOKS, "no notebooks under notebooks/ | notebooks/ altında defter yok"


def test_rules_catch_what_they_should():
    """
    EN: The patterns find a write, a fit and an archive path, and leave reading code alone.
    TR: Desenler yazmayı, uydurmayı ve arşiv yolunu bulur; okuyan koda dokunmaz.
    """
    for bad in ('save_metrics("x", v)', "model.fit(X, y)", "df.to_csv(p)", "p.write_text(s)", "json.dump(d, f)"):
        assert WRITES_OR_RUNS.search(bad), bad
    for ok in ('flat(m["fit"])', 'docs["07_lofo"]', "pd.DataFrame(rows)", 'show("fit", t)'):
        assert not WRITES_OR_RUNS.search(ok), ok
    assert ARCHIVE.search('ROOT / "archive" / "x"') and ARCHIVE.search('"a/frozen/b"')
    assert not ARCHIVE.search('"archived run"')


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_reads_only(path):
    """
    EN: No code cell writes a file or fits a model. / TR: Hiçbir kod hücresi dosya yazmıyor ya da model uydurmuyor.
    """
    found = [m.group(0) for c in cells(path, "code") for m in WRITES_OR_RUNS.finditer(c.source)]
    assert not found, f"{rel(path)}: a notebook only reads the record | defter yalnız kaydı okur: {found}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_no_archive_path(path):
    """
    EN: No code cell names an archive or frozen path. / TR: Hiçbir kod hücresi arşiv ya da frozen yolu anmıyor.
    """
    found = [c.source[:80] for c in cells(path, "code") if ARCHIVE.search(c.source)]
    assert not found, f"{rel(path)}: archive path | arşiv yolu: {found}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_links_resolve(path):
    """
    EN: Every relative link in the markdown cells resolves. / TR: Markdown hücrelerindeki her göreli bağlantı çözülüyor.
    """
    broken = [t for c in cells(path, "markdown") for t in LINK.findall(c.source)
              if not t.startswith(("http://", "https://", "#", "mailto:")) and not (path.parent / t.split("#")[0]).exists()]
    assert not broken, f"{rel(path)}: broken links | kırık bağlantı: {broken}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_no_error_outputs(path):
    """
    EN: The saved outputs carry no error. / TR: Kayıtlı çıktılarda hata yok.
    """
    errors = [o.get("ename") for c in cells(path, "code") for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{rel(path)}: error outputs, run: python tools/run_notebooks.py | hata çıktısı: {errors}"
