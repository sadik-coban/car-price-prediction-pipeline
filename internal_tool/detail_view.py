"""
detail_view.py
EN: Draws one listing in full for the explorer's window: a header (title, price, year, km, place, snapshot and a
    button that opens the ad on the site), then tabs — every field grouped, the 13-panel damage table with heavy
    damage and tramer, the ad's rows across snapshots with the price change, the ad text, and for raw the whole
    record as written. It draws into the current container, so the page can put it in a st.dialog and tests can
    run it alone (AppTest.from_function). Data is only read.
TR: Gezginin penceresi için tek bir ilanı bütünüyle çizer: üst bilgi (başlık, fiyat, yıl, km, yer, tarama ve ilanı
    sitede açan düğme), sonra sekmeler — gruplanmış her alan, ağır hasar ve tramerle 13 parçalık hasar tablosu,
    ilanın taramalar boyunca satırları ve fiyat farkı, ilan metni ve ham kaynakta kaydın yazıldığı hâli. Geçerli
    kabın içine çizer; böylece sayfa onu st.dialog'a koyabilir, testler tek başına koşabilir
    (AppTest.from_function). Veri yalnız okunur.
"""
import pandas as pd
import streamlit as st

from internal_tool import details, sources


def record_of(source, df, idx, data_dir):
    """
    EN: (fields, ad text, raw record or None) of one row; raw rows are read back whole from their file.
    TR: Tek satırın (alanlar, ilan metni, ham kayıt ya da None) üçlüsü; ham satırlar dosyalarından bütünüyle
        yeniden okunur.
    """
    if source == "raw":
        record = sources.read_raw_record(sources.data_files("raw", data_dir), df.loc[idx, sources.RAW_REF])
        fields = {k: v for k, v in record.items() if k != sources.RAW_TEXT}
        fields.update({"ad_id": df.loc[idx, "ad_id"], sources.SNAPSHOT_DIR: df.loc[idx, sources.SNAPSHOT_DIR]})
        return fields, sources.html_to_text(record.get(sources.RAW_TEXT)), record
    fields = df.loc[idx].to_dict()
    path = sources.data_files(source, data_dir)[0]
    return fields, sources.db_text(path, fields["id"]), None


def grouped(value, suffix=""):
    """
    EN: A number with "." thousands ("284.000"), raw text kept as it is, "—" when blank.
    TR: "." binlik ayırıcılı sayı ("284.000"); ham metin olduğu gibi kalır; boşsa "—".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not pd.isna(value):
        return f"{value:,.0f}{suffix}".replace(",", ".")
    return details.display(value) or "—"


def table(pairs):
    """EN: (field, value) pairs as a two-column frame of text. / TR: (alan, değer) çiftleri, iki kolonlu metin çerçevesi."""
    return pd.DataFrame({"alan": [k for k, _ in pairs], "değer": [details.display(v) for _, v in pairs]})


def render_detail(source, df, idx, data_dir):
    """EN: Draws the listing's full detail. / TR: İlanın tüm ayrıntısını çizer."""
    fields, text, raw = record_of(source, df, idx, data_dir)
    head = details.header(fields, source)
    st.markdown(f"### {head['title'] or '(başlık yok)'}")
    cols = st.columns(5)
    cols[0].metric("Fiyat", grouped(head["price"], " ₺"))
    cols[1].metric("Yıl", details.display(head["year"]) or "—")
    cols[2].metric("Km", grouped(head["km"]))
    cols[3].metric("Tarama", (details.display(head["snapshot"]) or "—")[:10])
    cols[4].metric("ad_id", details.display(fields.get("ad_id")) or "—")
    st.caption(details.display(head["place"]))
    if head["url"]:
        st.link_button("İlanı sitede aç ↗", head["url"])
    tabs = st.tabs(["Tüm alanlar", "Hasar", "Taramalar", "İlan metni"] + (["Ham kayıt (JSON)"] if raw else []))
    with tabs[0]:
        for group, items in details.group_fields(fields, source):
            st.markdown(f"**{group}** ({len(items)})")
            st.dataframe(table(items), hide_index=True, width="stretch")
    with tabs[1]:
        st.dataframe(details.damage_table(fields, source), hide_index=True, width="stretch")
        st.dataframe(table(details.heavy_fields(fields, source)), hide_index=True, width="stretch")
    with tabs[2]:
        hist = details.history(df, source, fields.get("ad_id"))
        st.caption("Aynı ilanın bu kaynaktaki bütün taramaları; fark bir önceki taramaya göre.")
        st.dataframe(hist, hide_index=True, width="stretch")
    with tabs[3]:
        st.text(text if isinstance(text, str) and text.strip() else "(ilan metni yok)")
    if raw:
        with tabs[4]:
            st.json(raw, expanded=True)
