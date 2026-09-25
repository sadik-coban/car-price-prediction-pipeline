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
