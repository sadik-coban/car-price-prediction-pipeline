"""
values.py
EN: The "Values" page: the register of what the data actually holds (db/observed_values.json) — every raw field with
    its values or formats, the damage labels, the silver columns with few values, series → models. The DB build stops
    on anything not in here and tests check the code against it, so this page is the list of cases the code may
    expect. Read-only.
TR: "Değerler" sayfası: verinin gerçekte ne tuttuğunun kaydı (db/observed_values.json) — her ham alan değerleri ya da
    biçimleriyle, hasar etiketleri, az değerli silver kolonları, seri → modeller. DB kurulumu burada olmayan her
    şeyde durur ve testler kodu buna göre sınar; yani bu sayfa kodun bekleyebileceği durumların listesidir. Salt
    okunur.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from internal_tool import observed, ui  # noqa: E402

ui.require_loopback()
register = observed.load()
raw = register["raw"]
st.title("Değerler")
st.caption("Verinin gerçekte ne tuttuğu: `db/observed_values.json` "
           f"(üretildi {register['_meta']['generated_at']}). DB kurulumu burada olmayan bir alanda, değerde, biçimde "
           "ya da hasar etiketinde **durur**; testler kodun yalnız buradakileri beklediğini ve hepsini ele aldığını "
           "sınar. Yeniden üretmek: `python tools/observed_values.py`.")
st.markdown(f"**Ham:** {raw['records']:,} kayıt · {raw['files']} dosya · {len(raw['fields'])} alan · "
            f"{raw['broken_lines']} bozuk satır   **Silver:** {register['silver']['rows']:,} satır".replace(",", "."))

tab_raw, tab_damage, tab_silver, tab_series = st.tabs(["Ham alanlar", "Hasar etiketleri", "Silver kolonları",
                                                       "Seri → model"])
with tab_raw:
    st.dataframe(observed.fields_table(register), hide_index=True)
    field = st.selectbox("Alan", list(raw["fields"]), key="values_field")
    entry = raw["fields"][field]
    st.markdown(f"**{field}** · {entry['present']:,} kayıtta var · tutulan: {observed.kind(entry)}".replace(",", "."))
    tables = observed.field_tables(register, field)
    if not tables:
        st.caption("Kimlik ya da serbest metin: yalnız sayılır, değerleri kayda yazılmaz.")
    for name, frame in tables.items():
        st.markdown(f"_{name}_ ({len(frame)})")
        st.dataframe(frame, hide_index=True)
with tab_damage:
    left, right = st.columns(2)
    left.dataframe(observed.pairs_table(raw["damage"]["parts"], "parça"), hide_index=True)
    right.dataframe(observed.pairs_table(raw["damage"]["states"], "durum"), hide_index=True)
with tab_silver:
    column = st.selectbox("Kolon", list(register["silver"]["columns"]), key="values_column")
    st.dataframe(observed.silver_table(register, column), hide_index=True)
with tab_series:
    st.dataframe(observed.series_table(register), hide_index=True)
