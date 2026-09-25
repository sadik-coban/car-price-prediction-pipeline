"""
reports.py
EN: The "Raporlar" page of the internal tool (placeholder; filled in a later part of the plan).
TR: İç aracın "Raporlar" sayfası (yer tutucu; planın sonraki bir parçasında doldurulur).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from internal_tool import ui  # noqa: E402

ui.require_loopback()
st.title("Raporlar")
st.caption("Raporlar bölüm bölüm, her bölümün betikleriyle — hazırlanıyor.")
