"""
test_tool_app.py
EN: The internal tool's pages run end to end in Streamlit's AppTest without an exception, and a page refuses to run
    when the server is not bound to a loopback address. Needs streamlit, so it runs only from the tool's own
    environment (.venv-tool\\Scripts\\python -m pytest tests/internal_tool); the verify gate (pipeline .venv) skips it.
TR: İç aracın sayfaları Streamlit'in AppTest'inde uçtan uca istisnasız koşar ve sunucu bir loopback adresine bağlı
    değilse sayfa koşmayı reddeder. streamlit ister, bu yüzden yalnız aracın kendi ortamından koşar
    (.venv-tool\\Scripts\\python -m pytest tests/internal_tool); verify kapısı (pipeline .venv) atlar.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit import config as st_config  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "internal_tool"
sys.path.insert(0, str(ROOT))
from internal_tool import report_sections as rs  # noqa: E402

PAGES = ["views/reports.py", "views/scripts.py", "views/explorer.py"]
TIMEOUT = 60


@pytest.fixture
def loopback():
    """EN: server.address = 127.0.0.1 for the test, restored after. / TR: Test için server.address = 127.0.0.1."""
    old = st_config.get_option("server.address")
    st_config.set_option("server.address", "127.0.0.1")
    yield
    st_config.set_option("server.address", old)


def test_router_runs(loopback):
    """EN: app.py runs its default page without an exception. / TR: app.py varsayılan sayfasını istisnasız koşar."""
    at = AppTest.from_file(str(TOOL / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(loopback, page):
    """EN: Each page runs alone without an exception. / TR: Her sayfa tek başına istisnasız koşar."""
    at = AppTest.from_file(str(TOOL / page), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]


def test_every_report_section_renders(loopback):
    """
    EN: Every report × language × section draws without an exception, with markdown and one image per figure.
    TR: Her rapor × dil × bölüm istisnasız çizilir; markdown'ı ve her figür için bir görseli var.
    """
    at = AppTest.from_file(str(TOOL / "views/reports.py"), default_timeout=TIMEOUT).run()
    for report in ("technical", "business", "shap"):
        at.sidebar.radio(key="report").set_value(report).run()
        for lang in ("tr", "en"):
            at.sidebar.radio(key="report_lang").set_value(lang).run()
            for i in range(len(at.sidebar.selectbox(key=f"section_{report}").options)):
                at.sidebar.selectbox(key=f"section_{report}").select_index(i).run()
                assert not at.exception, (report, lang, i, [e.value for e in at.exception])
                assert at.markdown, (report, lang, i)
                n_figures = len(rs.figures(rs.read_sections(report, lang)[i]["text"]))
                assert len(at.get("image")) == n_figures, (report, lang, i)


def test_every_script_detail_renders(loopback):
    """EN: Every script's detail view draws without an exception. / TR: Her betiğin ayrıntısı istisnasız çizilir."""
    at = AppTest.from_file(str(TOOL / "views/scripts.py"), default_timeout=TIMEOUT).run()
    for script in at.selectbox[0].options:
        at.selectbox[0].set_value(script).run()
        assert not at.exception, (script, [e.value for e in at.exception])
        assert at.code, script


def test_page_refuses_without_loopback():
    """EN: Unbound (all interfaces): the page stops with an error. / TR: Bağsız (bütün arayüzler): sayfa hatayla durur."""
    old = st_config.get_option("server.address")
    st_config.set_option("server.address", "0.0.0.0")
    try:
        at = AppTest.from_file(str(TOOL / PAGES[0]), default_timeout=TIMEOUT).run()
    finally:
        st_config.set_option("server.address", old)
    assert at.error and "127.0.0.1" in at.error[0].value
    assert not at.title


@pytest.fixture
def explorer(loopback, tool_data, monkeypatch):
    """EN: The explorer page on the made-up data folder. / TR: Uydurma veri klasöründe veri gezgini sayfası."""
    monkeypatch.setenv("CARDATASYS_TOOL_DATA", str(tool_data))
    return AppTest.from_file(str(TOOL / "views/explorer.py"), default_timeout=TIMEOUT).run()


def rows_metric(at):
    """EN: The "Satır" KPI as int. / TR: "Satır" KPI'ı, int."""
    return int(next(m for m in at.metric if m.label == "Satır").value.replace(".", ""))


def test_explorer_raw_combined_conditions(explorer):
    """
    EN: Raw: blue plate → 1 record; AND model year 2019–2019 → 0; the ad text search finds "MA PLAKALIDIR".
    TR: Ham: mavi plaka → 1 kayıt; VE model yılı 2019–2019 → 0; ilan metni araması "MA PLAKALIDIR"ı bulur.
    """
    at = explorer
    assert not at.exception and rows_metric(at) == 5
    at.button(key="short_raw_Mavi plakalı").click().run()
    assert rows_metric(at) == 1
    at.selectbox(key="pick_raw").set_value("KısaBilgi - Yıl (sayı)").run()
    at.button[[b.label for b in at.button].index("Koşul ekle")].click().run()
    at.number_input(key="val_1_between_lo").set_value(2019).run()
    at.number_input(key="val_1_between_hi").set_value(2019).run()
    assert not at.exception and rows_metric(at) == 0
    at.checkbox(key="on_0").uncheck().run()
    assert rows_metric(at) == 1
    at.button(key="del_1").click().run()
    at.button(key="del_0").click().run()
    at.selectbox(key="pick_raw").set_value("Aciklama_HTML").run()
    at.button[[b.label for b in at.button].index("Koşul ekle")].click().run()
    at.text_input(key="val_2_contains").input("ma plakalı").run()
    assert not at.exception and rows_metric(at) == 1


def test_explorer_silver_gold(explorer):
    """
    EN: Silver: the analysis set has 2 rows; all rows + empty plate → 1, and gold gives the same; heavy damage
        unknown → 2 in silver, 0 in gold.
    TR: Silver: analiz kümesi 2 satır; bütün satırlar + boş plaka → 1, gold da aynı; ağır hasar bilinmiyor → silver'da
        2, gold'da 0.
    """
    at = explorer
    at.sidebar.radio(key="source").set_value("silver").run()
    assert not at.exception and rows_metric(at) == 2
    at.sidebar.radio(key="mode_db").set_value("all").run()
    assert rows_metric(at) == 4
    at.button(key="short_db_Plaka boş").click().run()
    assert rows_metric(at) == 1 and next(m for m in at.metric if m.label.startswith("Aynı koşullar")).value == "1"
    at.button(key="del_0").click().run()
    at.button(key="short_db_Ağır hasar bilinmiyor").click().run()
    assert rows_metric(at) == 2
    at.sidebar.radio(key="source").set_value("gold").run()
    assert not at.exception and rows_metric(at) == 0
