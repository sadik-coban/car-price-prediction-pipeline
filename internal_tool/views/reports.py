"""
reports.py
EN: The "Reports" page: pick a report, a language and a section; the section is drawn from the generated md (its
    figures with st.image), and below it the analysis scripts that feed it (from the section map) with when each
    last ran and whether its code changed since; a button opens a script on the Scripts page.
TR: "Raporlar" sayfası: rapor, dil ve bölüm seçilir; bölüm üretilmiş md'den çizilir (figürleri st.image ile) ve
    altında onu besleyen analiz betikleri (bölüm eşlemesinden) listelenir: her birinin en son ne zaman koştuğu ve o
    zamandan beri kodunun değişip değişmediği; bir düğme betiği Betikler sayfasında açar.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from internal_tool import catalog, report_sections as rs, ui  # noqa: E402


def stale_badge(summary):
    """EN: A short freshness text for a metrics summary. / TR: Metrik özeti için kısa tazelik metni."""
    if summary is None:
        return "⚪ metrik yok"
    if summary["stale"] is None:
        return "⚪ kaynak izi yok"
    return "🟢 kod değişmedi" if not summary["stale"] else f"🟠 bayat: {', '.join(summary['stale'])}"


def open_script(script):
    """EN: Opens a script on the Scripts page. / TR: Bir betiği Betikler sayfasında açar."""
    st.session_state["script"] = script
    st.switch_page("views/scripts.py")


ui.require_loopback()
m = catalog.load_map()
with st.sidebar:
    report = st.radio("Rapor", list(rs.REPORTS), format_func=rs.REPORTS.get, key="report")
    lang = st.radio("Dil", rs.LANGS, format_func=str.upper, horizontal=True, key="report_lang")
    sections = rs.read_sections(report, lang)
    idx = st.selectbox("Bölüm", range(len(sections)), key=f"section_{report}",
                       format_func=lambda i: sections[i]["title"] if i else "Giriş (başlık ve özet)")

st.caption("Soldan rapor, dil ve bölüm seç. Bölümün metni ve figürleri burada; altında o bölümdeki sayıları üreten "
           "analiz betikleri ve en son ne zaman koştukları var.")
drift = catalog.map_drift(m)
if drift:
    st.warning("Bölüm eşlemesi şu derleyiciler değiştikten sonra gözden geçirilmedi: " + ", ".join(drift) +
               ". Gözden geçirince: `python internal_tool/catalog.py --stamp`.")

for block in rs.blocks(sections[idx]["text"]):
    if block[0] == "md":
        st.markdown(block[1])
    else:
        st.image(str(rs.REPORTS_DIR / block[2]), caption=block[1])

st.divider()
scripts = m["reports"][report][idx]["scripts"]
st.subheader(f"Bu bölümdeki sayıları üreten betikler ({len(scripts)})")
if not scripts:
    st.caption("Bu bölüm sabit metin: hiçbir betiğin metriğinden sayı okumuyor.")
for script in scripts:
    files = catalog.files_of(script)
    card = catalog.read_json(files["card"]) or {}
    summary = catalog.meta_summary(catalog.read_json(files["metrics"]))
    question = (card.get("question") or {}).get(lang, "")
    with st.expander(f"`{script}` — {question}"):
        if summary:
            st.markdown(f"**Son koşum:** {summary['generated_at']} · **run_id:** {summary['run_id'] or '—'} · "
                        f"{stale_badge(summary)}")
        else:
            st.markdown(stale_badge(summary))
        if st.button("Betikler sayfasında aç", key=f"open_{report}_{idx}_{script}"):
            open_script(script)
