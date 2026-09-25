"""
ui.py
EN: The small Streamlit helpers every page shares: the loopback guard (a page stops unless the server is bound to a
    loopback address) and the data folder (data/, or CARDATASYS_TOOL_DATA for tests). Imports streamlit, so the
    verify gate does not import it; the logic it calls lives in the pure modules.
TR: Her sayfanın paylaştığı küçük Streamlit yardımcıları: loopback koruması (sunucu bir loopback adresine bağlı
    değilse sayfa durur) ve veri klasörü (data/, testlerde CARDATASYS_TOOL_DATA). streamlit import ettiği için verify
    kapısı onu import etmez; çağırdığı mantık saf modüllerde.
"""
import os
from pathlib import Path

import streamlit as st

from internal_tool.launch import is_loopback

ROOT = Path(__file__).resolve().parents[1]


def require_loopback():
    """
    EN: Stops the page with an error unless Streamlit's server.address is a loopback address.
    TR: Streamlit'in server.address'i bir loopback adresi değilse sayfayı hatayla durdurur.
    """
    address = st.get_option("server.address") or ""
    if not is_loopback(address):
        st.error("Bu araç yalnız 127.0.0.1'de koşar: `python internal_tool/launch.py` ile başlatın. "
                 f"(server.address = {address!r})")
        st.stop()


def data_dir():
    """EN: The data folder: CARDATASYS_TOOL_DATA or data/. / TR: Veri klasörü: CARDATASYS_TOOL_DATA ya da data/."""
    return Path(os.environ.get("CARDATASYS_TOOL_DATA") or ROOT / "data")
