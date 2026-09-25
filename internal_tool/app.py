"""
app.py
EN: The internal tool's entry point (started by launch.py): page setup, the loopback guard and the pages — start
    (opens first), reports, scripts, data explorer. Each page is its own file under views/ so it can also be tested
    alone.
TR: İç aracın giriş noktası (launch.py başlatır): sayfa ayarı, loopback koruması ve sayfalar — başlangıç (ilk açılan),
    raporlar, betikler, veri gezgini. Her sayfa views/ altında kendi dosyasında, böylece tek başına da sınanabilir.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from internal_tool import ui  # noqa: E402

st.set_page_config(page_title="cardatasys · iç araç", layout="wide")
ui.require_loopback()
st.navigation([st.Page("views/home.py", title="Başlangıç", default=True),
               st.Page("views/reports.py", title="Raporlar"),
               st.Page("views/scripts.py", title="Betikler"),
               st.Page("views/explorer.py", title="Veri gezgini")]).run()
