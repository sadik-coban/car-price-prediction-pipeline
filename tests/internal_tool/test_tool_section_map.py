"""
test_tool_section_map.py
EN: The internal tool's hand-written section map (internal_tool/section_map.json) still fits the generated reports:
    every report has as many sections as the map in both languages, the map's titles are the English headings,
    every script it names is in analysis/run_all.py ORDER, every ORDER script feeds at least one section, and every
    figure a section shows comes from a script listed for that section. A builder change that only moves which
    metrics a section reads is not caught here; the tool shows a notice for that (map_drift).
TR: İç aracın elle yazılmış bölüm eşlemesi (internal_tool/section_map.json) üretilmiş raporlara hâlâ uyuyor: her
    raporun iki dilde de eşlemedeki kadar bölümü var, eşlemedeki başlıklar İngilizce başlıklar, andığı her betik
    analysis/run_all.py ORDER'da, ORDER'daki her betik en az bir bölümü besliyor ve bir bölümün gösterdiği her figür o
    bölüm için listelenen bir betikten geliyor. Yalnız bir bölümün okuduğu metrikleri değiştiren derleyici
    değişikliği burada yakalanmaz; araç bunun için uyarı gösterir (map_drift).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import catalog, report_sections as rs  # noqa: E402

MAP = catalog.load_map()
ORDER = catalog.order()
CASES = [(r, lang) for r in rs.REPORTS for lang in rs.LANGS]


@pytest.mark.parametrize("report, lang", CASES)
def test_section_count(report, lang):
    """EN: Same number of sections as the map. / TR: Eşlemedeki kadar bölüm."""
    assert len(rs.read_sections(report, lang)) == len(MAP["reports"][report])


@pytest.mark.parametrize("report", rs.REPORTS)
def test_titles_are_the_english_headings(report):
    """EN: The map's titles = the EN md headings, in order. / TR: Eşleme başlıkları = EN md başlıkları, sırasıyla."""
    assert [s["title"] for s in MAP["reports"][report]] == [s["title"] for s in rs.read_sections(report, "en")]


def test_scripts_are_in_order_and_all_used():
    """EN: Map scripts ⊆ ORDER and every ORDER script is used. / TR: Eşleme betikleri ⊆ ORDER, her ORDER betiği kullanılıyor."""
    named = {s for secs in MAP["reports"].values() for sec in secs for s in sec["scripts"]}
    assert named <= set(ORDER), sorted(named - set(ORDER))
    assert set(ORDER) <= named, f"scripts in no section | hiçbir bölümde olmayan: {sorted(set(ORDER) - named)}"
    assert set(MAP["figures"].values()) <= set(ORDER)


@pytest.mark.parametrize("report, lang", CASES)
def test_figures_come_from_the_section_scripts(report, lang):
    """
    EN: Every figure in a section is in the figure table and its script is listed for that section.
    TR: Bir bölümdeki her figür figür tablosunda ve betiği o bölüm için listelenmiş.
    """
    for sec, mapped in zip(rs.read_sections(report, lang), MAP["reports"][report]):
        for fig in rs.figures(sec["text"]):
            assert fig["key"] in MAP["figures"], f"{report} {lang} §{sec['index']}: unmapped figure {fig['path']}"
            assert MAP["figures"][fig["key"]] in mapped["scripts"], (report, lang, sec["index"], fig["key"])


def test_reviewed_builders_are_the_known_files():
    """EN: The map was stamped against the four builder files. / TR: Eşleme dört derleyici dosyasına göre damgalı."""
    assert set(MAP["reviewed_against"]) == set(catalog.REVIEWED)
    assert all((ROOT / rel).exists() for rel in catalog.REVIEWED)
