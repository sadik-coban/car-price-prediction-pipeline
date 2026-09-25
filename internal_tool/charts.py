"""
charts.py
EN: The explorer's summary numbers and plotly figures for the filtered rows of any source. Each source names its
    price / year / km / brand / series / snapshot / location columns in FIELDS (raw uses its parsed "(sayı)"
    columns); a figure whose columns are missing or whose rows are empty shows "Veri yok" instead of failing. The
    damage matrix reads raw "Hasar - <Part>" states, or the DB's 13 panels × changed / painted / local flags
    (NULL = unknown, which gold turns into "original"). Pure: no streamlit; the frames are not modified.
TR: Gezginin herhangi bir kaynağın filtrelenmiş satırları için özet sayıları ve plotly figürleri. Her kaynak fiyat /
    yıl / km / marka / seri / tarama / konum kolonlarını FIELDS'ta adlandırır (ham kaynak ayrıştırılmış "(sayı)"
    kolonlarını kullanır); kolonu eksik ya da satırı boş figür hata vermek yerine "Veri yok" gösterir. Hasar matrisi
    ham "Hasar - <Parça>" durumlarını ya da DB'nin 13 panel × değişen / boyalı / lokal bayraklarını okur (NULL =
    bilinmiyor; gold bunu "orijinal" yapar). Saf: streamlit yok; çerçeveler değiştirilmez.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FIELDS = {
    "raw": {"price": "Fiyat (sayı)", "year": "KısaBilgi - Yıl (sayı)", "km": "KısaBilgi - Kilometre (sayı)",
            "brand": "_marka_klasörü", "series": "KısaBilgi - Seri", "snapshot": "_tarama", "location": "Konum"},
    "db": {"price": "price", "year": "kb_year", "km": "kb_mileage", "brand": "brand", "series": "series",
           "snapshot": "search_date", "location": "location"},
}
# EN: DB panel prefix → display name | TR: DB panel öneki → görünen ad
PANELS = {"tavan": "Tavan", "kaput": "Motor kaputu", "bagaj": "Arka kaput (bagaj)", "door_fl": "Sol ön kapı",
          "door_fr": "Sağ ön kapı", "door_rl": "Sol arka kapı", "door_rr": "Sağ arka kapı",
          "fender_fl": "Sol ön çamurluk", "fender_fr": "Sağ ön çamurluk", "fender_rl": "Sol arka çamurluk",
          "fender_rr": "Sağ arka çamurluk", "bumper_front": "Ön tampon", "bumper_rear": "Arka tampon"}
STATES = {"degisen": "değişen", "boyali": "boyalı", "lokal": "lokal boyalı"}
UNKNOWN = "(boş)"


def fields_for(source):
    """EN: The chart columns of a source. / TR: Bir kaynağın grafik kolonları."""
    return FIELDS["raw" if source == "raw" else "db"]


def _has(df, *cols):
    """EN: True when df has rows and every column. / TR: df'in satırı ve her kolonu varsa True."""
    return len(df) > 0 and all(c in df.columns for c in cols)


def empty_figure(title):
    """EN: A figure that says there is no data. / TR: Veri olmadığını söyleyen figür."""
    fig = go.Figure()
    fig.add_annotation(text="Veri yok", showarrow=False, font={"size": 16})
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def snapshot_year(values):
    """
    EN: The year of a snapshot value (a date, or a raw folder name "2026-01-18_19-56").
    TR: Bir tarama değerinin yılı (tarih ya da ham klasör adı "2026-01-18_19-56").
    """
    return pd.to_numeric(values.astype(str).str[:4], errors="coerce")


def city(location):
    """EN: The city: the part after the last comma. / TR: İl: son virgülden sonraki kısım."""
    return location.split(",")[-1].strip() if isinstance(location, str) and location.strip() else UNKNOWN


def kpis(df, source):
    """
    EN: rows, distinct ads, median price / km / age (age = snapshot year − model year) of the rows.
    TR: Satırların satır sayısı, tekil ilanı, medyan fiyatı / km'si / yaşı (yaş = tarama yılı − model yılı).
    """
    f = fields_for(source)
    out = {"rows": len(df), "ads": int(df["ad_id"].nunique()) if "ad_id" in df.columns else None}
    for key in ("price", "km"):
        out[key] = float(pd.to_numeric(df[f[key]], errors="coerce").median()) if _has(df, f[key]) else None
    if _has(df, f["year"], f["snapshot"]):
        age = snapshot_year(df[f["snapshot"]]) - pd.to_numeric(df[f["year"]], errors="coerce")
        out["age"] = float(age.median())
    else:
        out["age"] = None
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in out.items()}


def price_histogram(df, source):
    """EN: Price distribution. / TR: Fiyat dağılımı."""
    f, title = fields_for(source), "Fiyat dağılımı"
    if not _has(df, f["price"]):
        return empty_figure(title)
    return px.histogram(pd.DataFrame({"fiyat": pd.to_numeric(df[f["price"]], errors="coerce")}).dropna(),
                        x="fiyat", nbins=60, title=title)


def median_price_by_year(df, source):
    """EN: Median price per model year, one line per brand. / TR: Model yılı başına medyan fiyat, marka başına çizgi."""
    f, title = fields_for(source), "Model yılına göre medyan fiyat"
    if not _has(df, f["price"], f["year"], f["brand"]):
        return empty_figure(title)
    d = pd.DataFrame({"yıl": pd.to_numeric(df[f["year"]], errors="coerce"),
                      "fiyat": pd.to_numeric(df[f["price"]], errors="coerce"), "marka": df[f["brand"]].astype(str)})
    g = d.dropna().groupby(["marka", "yıl"], as_index=False)["fiyat"].median()
    return px.line(g, x="yıl", y="fiyat", color="marka", markers=True, title=title) if len(g) else empty_figure(title)


def km_price_density(df, source):
    """EN: km × price density. / TR: km × fiyat yoğunluğu."""
    f, title = fields_for(source), "Kilometre × fiyat yoğunluğu"
    if not _has(df, f["price"], f["km"]):
        return empty_figure(title)
    d = pd.DataFrame({"km": pd.to_numeric(df[f["km"]], errors="coerce"),
                      "fiyat": pd.to_numeric(df[f["price"]], errors="coerce")}).dropna()
    return px.density_heatmap(d, x="km", y="fiyat", nbinsx=40, nbinsy=40, title=title) if len(d) else empty_figure(title)


def series_box(df, source, top=15):
    """EN: Price by series, the most frequent `top` series. / TR: Seriye göre fiyat, en sık `top` seri."""
    f, title = fields_for(source), f"Seriye göre fiyat (en sık {top})"
    if not _has(df, f["price"], f["series"]):
        return empty_figure(title)
    d = pd.DataFrame({"seri": df[f["series"]].astype(str), "fiyat": pd.to_numeric(df[f["price"]], errors="coerce")})
    d = d.dropna()
    keep = d["seri"].value_counts().head(top).index
    d = d[d["seri"].isin(keep)]
    return px.box(d, x="seri", y="fiyat", title=title, category_orders={"seri": list(keep)}) if len(d) \
        else empty_figure(title)


def rows_per_snapshot(df, source):
    """EN: Rows per snapshot. / TR: Tarama başına satır."""
    f, title = fields_for(source), "Tarama başına satır"
    if not _has(df, f["snapshot"]):
        return empty_figure(title)
    g = df[f["snapshot"]].astype(str).str[:10].value_counts().sort_index()
    return px.bar(x=g.index, y=g.values, labels={"x": "tarama", "y": "satır"}, title=title)


def top_cities(df, source, n=20):
    """EN: Rows per city, the top n. / TR: İl başına satır, ilk n."""
    f, title = fields_for(source), f"İl başına satır (ilk {n})"
    if not _has(df, f["location"]):
        return empty_figure(title)
    g = df[f["location"]].map(city).value_counts().head(n)[::-1]
    return px.bar(x=g.values, y=g.index, orientation="h", labels={"x": "satır", "y": "il"}, title=title)


def panel_states(df, prefix):
    """
    EN: The damage state of one DB panel for every row: changed > painted > local painted when its flag is 1,
        "orijinal" when all three flags are 0, UNKNOWN otherwise (a NULL flag). None if the flags are missing.
    TR: Bir DB panelinin her satırdaki hasar durumu: bayrağı 1 ise değişen > boyalı > lokal boyalı, üç bayrak da 0
        ise "orijinal", değilse UNKNOWN (NULL bayrak). Bayraklar yoksa None.
    """
    flags = [f"{prefix}_{s}" for s in STATES]
    if not all(c in df.columns for c in flags):
        return None
    state = pd.Series(UNKNOWN, index=df.index, dtype="object")
    all_zero = pd.Series(True, index=df.index)
    for col in flags:
        all_zero &= pd.to_numeric(df[col], errors="coerce").fillna(-1).eq(0)
    state[all_zero] = "orijinal"
    for col, label in reversed(list(zip(flags, STATES.values()))):
        state[pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)] = label
    return state


def damage_matrix(df, source):
    """
    EN: Rows per panel × damage state (DataFrame: panels as rows, states as columns). DB state of a panel: changed
        > painted > local painted when its flag is 1, "orijinal" when all three flags are 0, UNKNOWN otherwise.
    TR: Panel × hasar durumu başına satır (DataFrame: satırlar panel, kolonlar durum). DB'de bir panelin durumu:
        bayrağı 1 ise değişen > boyalı > lokal boyalı, üç bayrak da 0 ise "orijinal", değilse UNKNOWN.
    """
    rows = {}
    if source == "raw":
        for col in (c for c in df.columns if c.startswith("Hasar - ")):
            rows[col.removeprefix("Hasar - ")] = df[col].fillna(UNKNOWN).value_counts()
    else:
        for prefix, name in PANELS.items():
            state = panel_states(df, prefix)
            if state is not None:
                rows[name] = state.value_counts()
    return pd.DataFrame(rows).T.fillna(0).astype(int) if rows else pd.DataFrame()


def damage_heatmap(df, source):
    """EN: The damage matrix as a heatmap. / TR: Hasar matrisi, ısı haritası olarak."""
    title = "Panel × hasar durumu (satır)"
    m = damage_matrix(df, source)
    if m.empty or len(df) == 0:
        return empty_figure(title)
    return px.imshow(m, text_auto=True, aspect="auto", title=title, labels={"x": "durum", "y": "panel", "color": "satır"})


FIGURES = (price_histogram, median_price_by_year, km_price_density, series_box, rows_per_snapshot, damage_heatmap,
           top_cities)
