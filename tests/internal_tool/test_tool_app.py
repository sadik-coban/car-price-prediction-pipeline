"""
test_tool_app.py
EN: The internal tool's pages run end to end in Streamlit's AppTest without an exception, and a page refuses to run
    when the server is not bound to a loopback address. Needs streamlit, so it runs only from the tool's own
    environment (.venv-tool\\Scripts\\python -m pytest tests/internal_tool); the verify gate (pipeline .venv) skips it.
TR: İç aracın sayfaları Streamlit'in AppTest'inde uçtan uca istisnasız koşar ve sunucu bir loopback adresine bağlı
    değilse sayfa koşmayı reddeder. streamlit ister, bu yüzden yalnız aracın kendi ortamından koşar
    (.venv-tool\\Scripts\\python -m pytest tests/internal_tool); verify kapısı (pipeline .venv) atlar.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit import config as st_config  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "internal_tool"
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
