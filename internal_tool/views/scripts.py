"""
scripts.py
EN: The "Scripts" page, read-only: a table of the 25 analysis scripts (when each last ran, whether its code changed
    since, run_id, whether it read the current silver DB, the report sections it feeds), then one script in detail —
    its card, its metrics (long lists shortened, the full file downloadable), its tests and its source code. Nothing
    is executed from here.
TR: "Betikler" sayfası, salt okunur: 25 analiz betiğinin tablosu (her birinin en son ne zaman koştuğu, o zamandan
    beri kodunun değişip değişmediği, run_id, güncel silver DB'yi okuyup okumadığı, beslediği rapor bölümleri), sonra
    tek bir betiğin ayrıntısı — kartı, metrikleri (uzun listeler kısaltılır, tam dosya indirilebilir), testleri ve
    kaynak kodu. Buradan hiçbir şey koşturulmaz.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from internal_tool import catalog, report_sections as rs, ui  # noqa: E402

SHORT = {"technical": "Teknik", "business": "Karar", "shap": "SHAP"}
LEAK_NAMES = {"duplicate_ad_id": "Tekrarlanan ilan", "target_in_features": "Hedef özniteliklerde",
              "fit_on_full_data": "Tüm veride kurulum", "temporal": "Zamansal"}


@st.cache_data(show_spinner="silver DB parmak izi hesaplanıyor…")
def silver_sha(path, mtime_ns, size):
    """
    EN: sha256 of the silver DB, cached by (path, mtime, size) so a rebuilt DB is hashed again.
    TR: Silver DB'nin sha256'sı; (yol, mtime, boyut) ile önbelleklenir, yeniden kurulan DB yeniden hash'lenir.
    """
    return catalog.file_sha(path, fold_crlf=False)


def current_silver_sha():
    """EN: The silver DB's sha256, or None if it is missing. / TR: Silver DB'nin sha256'sı; yoksa None."""
    path = ui.data_dir() / "cars.duckdb"
    if not path.exists():
        return None
    st_ = path.stat()
    return silver_sha(str(path), st_.st_mtime_ns, st_.st_size)


def used_in(script, m):
    """EN: "Teknik §1, §8 · SHAP §2" style text. / TR: "Teknik §1, §8 · SHAP §2" biçiminde metin."""
    by_report = {}
    for report, i, _ in catalog.sections_of(script, m):
        by_report.setdefault(report, []).append("giriş" if i == 0 else f"§{i}")
    return " · ".join(f"{SHORT[r]} {', '.join(v)}" for r, v in by_report.items())


def overview(m, sha):
    """EN: One table row per script. / TR: Betik başına bir tablo satırı."""
    rows = []
    for script in catalog.order():
        files = catalog.files_of(script)
        s = catalog.meta_summary(catalog.read_json(files["metrics"]), sha) or {}
        card = catalog.read_json(files["card"]) or {}
        stale = s.get("stale")
        rows.append({"betik": script, "soru": (card.get("question") or {}).get("tr", ""),
                     "son koşum": s.get("generated_at"),
                     "kod": "—" if stale is None else ("değişmedi" if not stale else f"bayat ({len(stale)})"),
                     "run_id": s.get("run_id"),
                     "silver ile aynı DB": {True: "evet", False: "hayır", None: "—"}[s.get("db_matches")],
                     "raporda": used_in(script, m)})
    return pd.DataFrame(rows)


def show_card(card, lang):
    """EN: Draws a card. / TR: Bir kartı çizer."""
    if not card:
        st.info("Kart yok.")
        return
    for field, label in (("question", "Soru"), ("data", "Veri"), ("method", "Yöntem")):
        st.markdown(f"**{label}:** {(card.get(field) or {}).get(lang, '')}")
    st.markdown(f"**Bölme:** `{card.get('split')}` · **Durum:** `{card.get('status')}`")
    leak = card.get("leakage") or {}
    st.dataframe(pd.DataFrame([{"sızıntı tipi": LEAK_NAMES.get(t, t), "durum": e.get("status"),
                                "not": (e.get("note") or {}).get(lang, "")} for t, e in leak.items()]),
                 hide_index=True)
    ev = card.get("evidence") or {}
    st.markdown("**Kanıt — testler:**\n" + "\n".join(f"- `{t}`" for t in ev.get("tests", [])))
    st.markdown("**Kanıt — rapor anahtarları:**\n" + "\n".join(f"- `{k}`" for k in ev.get("report_keys", [])))


ui.require_loopback()
m = catalog.load_map()
sha = current_silver_sha()
st.title("Betikler")
st.caption("Salt okunur: buradan hiçbir betik koşturulmaz. Bir betiği koşmak için `python analysis/NN_ad.py`, "
           "ardından `snapshot_metrics --diff`.")
table = overview(m, sha)
st.dataframe(table, hide_index=True, height=35 * (len(table) + 1) + 3)

scripts = list(table["betik"])
default = st.session_state.get("script")
choice = st.selectbox("Betik", scripts, index=scripts.index(default) if default in scripts else 0)
st.session_state["script"] = choice
lang = st.radio("Kart dili", rs.LANGS, format_func=str.upper, horizontal=True, key="card_lang")
files = catalog.files_of(choice)
doc = catalog.read_json(files["metrics"])
summary = catalog.meta_summary(doc, sha)

tab_card, tab_metrics, tab_tests, tab_code = st.tabs(["Kart", "Metrikler", "Testler", "Kaynak kod"])
with tab_card:
    show_card(catalog.read_json(files["card"]), lang)
with tab_metrics:
    if doc is None:
        st.info("Bu betiğin metrik dosyası yok.")
    else:
        st.markdown(f"**Son koşum:** {summary['generated_at']} · **veri sonu:** {summary['data_until']} · "
                    f"**run_id:** {summary['run_id'] or '—'} · **DB satırı:** {summary['db_rows']}")
        if summary["stale"]:
            st.warning("Metrik yazıldıktan sonra değişen kod: " + ", ".join(summary["stale"]))
        st.download_button("Tam JSON'u indir", json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"),
                           file_name=Path(files["metrics"]).name, mime="application/json")
        st.json({k: v for k, v in catalog.shorten(doc).items() if k != "_meta"}, expanded=1)
with tab_tests:
    card = catalog.read_json(files["card"]) or {}
    for t in (card.get("evidence") or {}).get("tests", []):
        st.markdown(f"- `{t}`")
    try:
        st.code(catalog.read_source(files["test_source"]), language="python")
    except FileNotFoundError:
        st.info("Test dosyası yok.")
with tab_code:
    st.code(catalog.read_source(files["source"]), language="python")
