"""
home.py
EN: The "Start" page (the tool opens here): what each of the three pages does and how to use it, with links, and
    the state of the data — when the analyses last ran and what is on disk for raw, silver and gold (file times and
    sizes; nothing is hashed or loaded, so it opens at once).
TR: "Başlangıç" sayfası (araç burada açılır): üç sayfanın her birinin ne yaptığı ve nasıl kullanıldığı, bağlantılarla,
    ve verinin durumu — analizlerin en son ne zaman koştuğu ve ham, silver ve gold için diskte ne olduğu (dosya
    zamanları ve boyutları; hiçbir şey hash'lenmez ya da yüklenmez, sayfa hemen açılır).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from internal_tool import catalog, sources, ui  # noqa: E402

PAGES = [
    ("views/reports.py", "Raporlar", "Soldan rapor, dil ve bölüm seç. Bölümün metni ve figürleri ortada; altında o "
                                     "bölümdeki sayıları üreten analiz betikleri ve en son ne zaman koştukları var."),
    ("views/scripts.py", "Betikler", "25 analiz betiği: en son ne zaman koştu, kodu o zamandan beri değişti mi. Birini "
                                     "seçince kartı, metrikleri, testleri ve kodu görünür. Hiçbir şey koşturulmaz."),
    ("views/explorer.py", "Veri gezgini", "Kaynak seç (ham / silver / gold), **+ Koşul ekle** ile istediğin kolona "
                                          "koşul koy (koşullar VE ile birleşir). Bir ilanın herhangi bir hücresine "
                                          "tıkla: tüm ayrıntısı bir pencerede açılır."),
    ("views/values.py", "Değerler", "Verinin gerçekte ne tuttuğu: her alanın değerleri ya da biçimleri. DB kurulumu "
                                    "burada olmayan bir değerde durur; kod yalnız buradakileri bekleyebilir."),
]

ui.require_loopback()
st.title("cardatasys · iç araç")
st.caption("Yalnız bu bilgisayarda (127.0.0.1) çalışır. Veri ve raporlar yalnız okunur; buradan hiçbir şey "
           "yazılmaz ya da koşturulmaz.")
for col, (page, title, text) in zip(st.columns(len(PAGES)), PAGES):
    with col.container(border=True):
        st.subheader(title)
        st.markdown(text)
        st.page_link(page, label=f"{title} sayfasına git →")

st.subheader("Verinin durumu")
stamp, script = catalog.latest_run()
st.markdown(f"**Analizlerin son koşumu:** {stamp or '—'}" + (f" (`{script}`)" if script else ""))
st.dataframe(pd.DataFrame(sources.file_state(ui.data_dir())), hide_index=True)
