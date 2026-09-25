"""
test_scaffold.py
EN: Tests of tools/new_analysis.py in a temp folder: it writes the script, a stub test and a card; the stub test
    really fails and is marked `template` (so the coverage matrix does not count it as a real test); the card is
    incomplete for the coverage matrix; the script template compiles and carries bilingual docstrings; nothing is
    overwritten and a bad name is refused.
TR: tools/new_analysis.py testleri, geçici klasörde: betiği, taslak testi ve kartı yazar; taslak test gerçekten
    düşer ve `template` işaretli (kapsam matrisi onu gerçek test saymaz); kart kapsam matrisi için eksik; betik
    şablonu derlenir ve iki dilli docstring taşır; hiçbir şeyin üzerine yazılmaz, hatalı ad reddedilir.
"""
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import analysis_coverage as AC  # noqa: E402
import new_analysis as NA  # noqa: E402


@pytest.fixture
def made(tmp_path):
    """EN: One scaffold in a temp root; returns (root, paths). / TR: Geçici kökte bir iskelet; (kök, yollar)."""
    assert NA.main(["99_trial", "Deneme sorusu?", "Trial question?", "--root", str(tmp_path)]) == 0
    return tmp_path, NA.targets("99_trial", tmp_path)


def test_three_files_written(made):
    """EN: Script, test and card exist. / TR: Betik, test ve kart var."""
    _root, paths = made
    assert all(p.exists() for p in paths.values())


def test_stub_test_fails_and_is_a_template(made):
    """
    EN: Running the stub gives one failure; it is collected only under -m template.
    TR: Taslak koşunca bir hata verir; yalnız -m template altında toplanır.
    """
    root, paths = made
    (root / "pytest.ini").write_text((ROOT / "pytest.ini").read_text(encoding="utf-8"), encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, "-m", "pytest", *a, str(paths["test"])], cwd=root,  # noqa: E731
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
    r = run("-q")
    assert r.returncode == 1 and "1 failed" in r.stdout
    assert "1 deselected" in run("-q", "--collect-only", "-m", "not template").stdout


def test_card_is_incomplete(made):
    """EN: The coverage matrix finds the new card incomplete. / TR: Kapsam matrisi yeni kartı eksik bulur."""
    _root, paths = made
    card = json.loads(paths["card"].read_text(encoding="utf-8"))
    problems = AC.card_problems(card, None, set())
    assert card["question"] == {"tr": "Deneme sorusu?", "en": "Trial question?"}
    assert any("status" in p for p in problems) and any("TODO" in p for p in problems)


def test_script_template_compiles_and_is_bilingual(made):
    """EN: The template is valid Python with EN:/TR: docstrings. / TR: Şablon geçerli Python, EN:/TR: docstring'li."""
    _root, paths = made
    py_compile.compile(str(paths["script"]), doraise=True)
    text = paths["script"].read_text(encoding="utf-8")
    assert text.count("EN:") >= 3 and text.count("TR:") >= 3 and 'save_metrics("99_trial"' in text


def test_nothing_overwritten(made, capsys):
    """EN: A second run for the same name writes nothing and fails. / TR: Aynı adla ikinci koşu yazmaz, düşer."""
    root, paths = made
    before = {k: p.read_bytes() for k, p in paths.items()}
    assert NA.main(["99_trial", "x", "y", "--root", str(root)]) == 1
    assert {k: p.read_bytes() for k, p in paths.items()} == before
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["price_history", "1_x", "11-price", "11_Price", "../11_x"])
def test_bad_name_refused(tmp_path, bad):
    """EN: Only NN_lower_snake names (optionally shap/). / TR: Yalnız NN_kucuk_harf adlar (isteğe bağlı shap/)."""
    assert NA.main([bad, "x", "y", "--root", str(tmp_path)]) == 1
    assert not any(tmp_path.rglob("*.py"))
