"""
explorer.py
EN: The "Data explorer" page. Pick a source — raw (every scraped record, fields as written), silver or gold — and a
    row set, then add conditions on ANY column (the ad text too), joined with AND: e.g. plate = blue AND year 2024
    AND seller = dealer. Each condition shows how many rows are left after it; silver and gold share their
    conditions, and the page shows what the same conditions give in the other DB. Results: KPIs, a table with ad_id
    and a clickable url (first 5,000 rows) — a click on any cell opens the listing in full in a window
    (detail_view) — and charts. Local only (ui guard); data is read, never written.
TR: "Veri gezgini" sayfası. Kaynak — ham (kazınan her kayıt, alanlar yazıldığı gibi), silver ya da gold — ve satır
    kümesi seçilir, sonra HER kolona (ilan metni dahil) VE ile birleşen koşullar eklenir: ör. plaka = mavi VE yıl 2024
    VE satıcı = galeri. Her koşul kendisinden sonra kaç satır kaldığını gösterir; silver ile gold koşullarını paylaşır
    ve sayfa aynı koşulların öbür DB'de ne verdiğini gösterir. Sonuç: KPI'lar, ad_id ve tıklanır url'li tablo (ilk
    5.000 satır) — herhangi bir hücreye tıklamak ilanı bir pencerede bütünüyle açar (detail_view) — ve grafikler.
    Yalnız yerel (ui koruması); veri okunur, asla yazılmaz.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from internal_tool import charts, detail_view, details, filters, sources, ui  # noqa: E402

TEXT_COLUMN = {"raw": sources.RAW_TEXT, "db": sources.DB_TEXT}
SNAPSHOT = {"raw": sources.SNAPSHOT_DIR, "db": "search_date"}
ROW_SETS = {"raw": ("all", "latest", "snapshot"), "db": ("analysis", "latest", "all", "snapshot")}
ROW_SET_HELP = {"all": "her tarama satırı", "latest": "her ilanın son taraması",
                "analysis": "TR plaka, fiyat > 0, ilan başına son satır (analizlerin kümesi)", "snapshot": "tek tarama"}
DEFAULT_COLUMNS = {
    "raw": ["ad_id", "url", "_marka_klasörü", "KısaBilgi - Seri", "KısaBilgi - Model", "KısaBilgi - Yıl",
            "KısaBilgi - Kilometre", "Fiyat", "Genel Bakış - Plaka Uyruğu", "KısaBilgi - Kimden", "_tarama"],
    "db": ["ad_id", "url", "brand", "series", "model", "kb_year", "kb_mileage", "price", "gb_plate_origin",
           "kb_seller_type", "search_date"]}
SHORTCUTS = {
    "raw": {"Mavi plakalı": [("Genel Bakış - Plaka Uyruğu", "in", ("Mavi plakalı",))],
            "Plaka alanı yok": [("Genel Bakış - Plaka Uyruğu", "is_null", ())],
            "Sayfa açılmamış (hata)": [("error", "not_null", ())],
            "Ağır hasarlı": [("Agir_Hasar", "is_true", ())]},
    "db": {"Mavi plakalı": [("gb_plate_origin", "in", ("Mavi plakalı",))],
           "Plaka boş": [("gb_plate_origin", "is_null", ())],
           "Ağır hasarlı": [("is_heavy_damaged", "is_true", ())],
           "Ağır hasar bilinmiyor": [("is_heavy_damaged", "is_null", ())],
           "Hiç değişen / boyalı yok": [("count_changed", "between", (0, 0)), ("count_painted", "between", (0, 0)),
                                        ("count_local_painted", "between", (0, 0))]}}
DEFAULT_OP = {"bool": "is_true", "number": "between", "date": "between", "category": "in", "text": "contains"}
TABLE_ROWS = 5000


def group_of(source):
    """EN: "raw" or "db" (silver and gold share conditions). / TR: "raw" ya da "db" (silver ile gold koşul paylaşır)."""
    return "raw" if source == "raw" else "db"


def key_of(source):
    """EN: The cache key of a source's files. / TR: Bir kaynağın dosyalarının önbellek anahtarı."""
    return sources.fingerprint(sources.data_files(source, ui.data_dir()))


@st.cache_resource(max_entries=3, show_spinner="Veri yükleniyor…")
def load_source(source, key):
    """EN: A source's frame and stats, cached by its file key. / TR: Bir kaynağın çerçevesi ve bilgisi, önbellekli."""
    return sources.load(source, ui.data_dir())


@st.cache_resource(max_entries=3, show_spinner="İlan metinleri yükleniyor…")
def load_texts(source, key):
    """EN: The ad texts and their folded copy, cached. / TR: İlan metinleri ve katlanmış kopyası, önbellekli."""
    texts = sources.texts(source, ui.data_dir())
    return texts, filters.fold_series(texts)


@st.cache_data(max_entries=8, show_spinner="Kolon türleri çıkarılıyor…")
def column_kinds(source, key):
    """EN: {column: kind} of a source (the ad text is "text"). / TR: Bir kaynağın {kolon: tür} sözlüğü."""
    df, _ = load_source(source, key)
    kinds = {c: filters.column_kind(df[c]) for c in df.columns}
    kinds[TEXT_COLUMN[group_of(source)]] = "text"
    return kinds


def column_series(source, key, column):
    """EN: A column of a source (the ad text loaded on demand). / TR: Bir kaynağın kolonu (ilan metni istenince)."""
    if column == TEXT_COLUMN[group_of(source)]:
        return load_texts(source, key)[0]
    return load_source(source, key)[0][column]


def plain(value):
    """EN: A numpy scalar as a Python value. / TR: Numpy skaleri Python değeri olarak."""
    return value.item() if hasattr(value, "item") and not isinstance(value, pd.Timestamp) else value


@st.cache_data(max_entries=512, show_spinner=False)
def pick_list(source, key, column):
    """
    EN: The values of a column for a pick list: [(value, count)] (numbers and dates sorted by value, the rest by
        count) and the number of empty rows.
    TR: Seçim listesi için bir kolonun değerleri: [(değer, sayı)] (sayı ve tarihler değere, ötekiler sayıya göre
        sıralı) ve boş satır sayısı.
    """
    series = column_series(source, key, column)
    empty = filters.is_empty(series)
    counts = series[~empty].value_counts()
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_datetime64_any_dtype(series.dtype):
        counts = counts.sort_index()
    return [(plain(v), int(n)) for v, n in counts.items()], int(empty.sum())


@st.cache_data(max_entries=512, show_spinner=False)
def column_ops(source, key, column, kind):
    """EN: The operators a column offers. / TR: Bir kolonun sunduğu operatörler."""
    return filters.ops_for(column_series(source, key, column), kind)


def add_condition(group, column, op, values=()):
    """EN: Appends a condition to a group. / TR: Bir gruba koşul ekler."""
    cid = st.session_state["next_condition"]
    st.session_state["next_condition"] += 1
    st.session_state["conditions"][group].append({"id": cid, "column": column, "op": op, "values": tuple(values),
                                                  "enabled": True})


def add_picked(group, kinds):
    """EN: Adds a condition on the picked column. / TR: Seçilen kolona koşul ekler."""
    column = st.session_state.get(f"pick_{group}")
    if column:
        add_condition(group, column, DEFAULT_OP[kinds.get(column, "text")])
        st.session_state[f"pick_{group}"] = None


def add_shortcut(group, items):
    """EN: Adds a shortcut's conditions. / TR: Bir kısayolun koşullarını ekler."""
    for column, op, values in items:
        add_condition(group, column, op, values)


def remove_condition(group, cid):
    """EN: Removes a condition. / TR: Bir koşulu siler."""
    st.session_state["conditions"][group] = [c for c in st.session_state["conditions"][group] if c["id"] != cid]


def widget_default(key, **default):
    """
    EN: default=… only on a widget's first run (Streamlit warns when a keyed widget also gets a default later).
    TR: default=… yalnız widget'ın ilk koşumunda (anahtarlı widget'a sonra default verilirse Streamlit uyarır).
    """
    return {} if key in st.session_state else default


def value_widget(cond, source, key, kind):
    """
    EN: Draws the value input of a condition and returns its values tuple.
    TR: Bir koşulun değer girişini çizer ve değer demetini döndürür.
    """
    cid, op, column = cond["id"], cond["op"], cond["column"]
    wkey = f"val_{cid}_{op}"
    if op in ("in", "not_in"):
        values, empties = pick_list(source, key, column)
        counts = dict(values)
        options = ([filters.NULL] if empties else []) + [v for v, _ in values]
        chosen = st.multiselect("değerler", options, key=wkey, label_visibility="collapsed",
                                format_func=lambda v: f"{v}  ({empties if v == filters.NULL else counts.get(v, 0):,})",
                                placeholder="değer seç…",
                                **widget_default(wkey, default=[v for v in cond["values"] if v in options]))
        return tuple(chosen)
    if op == "between":
        series = column_series(source, key, column).dropna()
        if series.empty:
            st.caption("kolon boş")
            return ()
        low, high = (list(cond["values"]) + [None, None])[:2]
        if kind == "date":
            lo_d, hi_d = series.min().date(), series.max().date()
            picked = st.date_input("aralık", key=wkey, label_visibility="collapsed", min_value=lo_d, max_value=hi_d,
                                   **widget_default(wkey, value=(low.date() if low is not None else lo_d,
                                                                 high.date() if high is not None else hi_d)))
            picked = list(picked) + [None, None]
            return tuple(pd.Timestamp(d) if d is not None else None for d in picked[:2])
        integer = pd.api.types.is_integer_dtype(series.dtype) or bool((series % 1 == 0).all())
        cast = int if integer else float
        lo_v, hi_v = cast(series.min()), cast(series.max())
        lo_in = st.number_input("en az", key=wkey + "_lo", step=cast(1),
                                **widget_default(wkey + "_lo", value=cast(low) if low is not None else lo_v))
        hi_in = st.number_input("en çok", key=wkey + "_hi", step=cast(1),
                                **widget_default(wkey + "_hi", value=cast(high) if high is not None else hi_v))
        return (lo_in, hi_in)
    if op in ("contains", "not_contains", "equals", "regex"):
        text = st.text_input("metin", key=wkey, label_visibility="collapsed", placeholder="aranacak metin…",
                             **widget_default(wkey, value=cond["values"][0] if cond["values"] else ""))
        return (text,) if text else ()
    st.caption("—")
    return ()


def incomplete(cond):
    """
    EN: Why a condition cannot be applied yet (no values chosen, no text, a bad regex), or None.
    TR: Bir koşulun neden henüz uygulanamadığı (değer seçilmedi, metin yok, hatalı regex) ya da None.
    """
    op, vals = cond["op"], cond["values"]
    if op in ("in", "not_in") and not vals:
        return "değer seçilmedi"
    if op in ("contains", "not_contains", "equals", "regex") and not vals:
        return "metin yazılmadı"
    if op == "regex":
        try:
            re.compile(vals[0])
        except re.error as exc:
            return f"regex hatalı: {exc}"
    return None


def draw_conditions(group, source, key, kinds):
    """
    EN: Draws the condition list (enable, operator, values, delete) and returns the applicable conditions.
    TR: Koşul listesini çizer (aç/kapa, operatör, değerler, sil) ve uygulanabilir koşulları döndürür.
    """
    ready = []
    for cond in st.session_state["conditions"][group]:
        cid, column = cond["id"], cond["column"]
        c_on, c_col, c_op, c_val, c_del = st.columns([0.5, 3, 1.6, 4.5, 0.5], vertical_alignment="center")
        c_del.button("✕", key=f"del_{cid}", on_click=remove_condition, args=(group, cid), help="koşulu sil")
        if column not in kinds:
            c_col.markdown(f"**{column}**")
            c_val.caption("bu kaynakta böyle bir kolon yok — uygulanmıyor")
            continue
        kind = kinds[column]
        cond["enabled"] = c_on.checkbox("açık", key=f"on_{cid}", label_visibility="collapsed",
                                        **widget_default(f"on_{cid}", value=cond["enabled"]))
        c_col.markdown(f"**{column}**  \n<small>{kind}</small>", unsafe_allow_html=True)
        ops = column_ops(source, key, column, kind)
        op = c_op.selectbox("operatör", ops, key=f"op_{cid}", format_func=filters.OP_LABELS.get,
                            label_visibility="collapsed",
                            **widget_default(f"op_{cid}", index=ops.index(cond["op"]) if cond["op"] in ops else 0))
        if op != cond["op"]:
            cond["op"], cond["values"] = op, ()
        with c_val:
            cond["values"] = value_widget(cond, source, key, kind)
            problem = incomplete(cond)
            if problem and cond["enabled"]:
                st.caption(f"⚠️ {problem} — uygulanmıyor")
        if cond["enabled"] and not problem:
            ready.append(filters.Condition(column, cond["op"], cond["values"]))
    return ready


def filtered(source, mode, snapshot, conds):
    """
    EN: Applies a row set and conditions to a source. Returns: (frame, mask, steps, missing, base count).
    TR: Bir kaynağa satır kümesi ve koşulları uygular. Döndürür: (çerçeve, maske, adımlar, eksik, taban sayısı).
    """
    group, key = group_of(source), key_of(source)
    df, _ = load_source(source, key)
    base = filters.row_set(df, mode, SNAPSHOT[group], snapshot)
    text_col = TEXT_COLUMN[group]
    frame, folded = df, {}
    if any(c.column == text_col for c in conds):
        texts, folded_texts = load_texts(source, key)
        frame = df.assign(**{text_col: texts.to_numpy()})
        folded = {text_col: folded_texts}
    mask, steps, missing = filters.apply(frame, conds, base=base, folded=folded)
    return df, mask, steps, missing, int(base.sum())


def number(v, suffix=""):
    """EN: A KPI value as text ("—" when missing). / TR: KPI değeri metin olarak (yoksa "—")."""
    return "—" if v is None else f"{v:,.0f}".replace(",", ".") + suffix


def gold_rules_note():
    """EN: The gold rules as a short list. / TR: Gold kuralları kısa liste olarak."""
    rules = json.loads((ROOT / "db" / "gold_rules.json").read_text(encoding="utf-8"))
    lines = [f"- **{len(r['columns'])} kolon** NULL → `{str(r['value']).lower()}`: {r['reason']['tr']}"
             for r in rules["fill"]]
    lines += [f"- `{d['column']}` gold'da yok: {d['reason']['tr']}" for d in rules["drop"]]
    return "\n".join(lines)


@st.dialog("İlan ayrıntısı", width="large")
def detail_dialog(source, df, idx):
    """EN: The listing in full, in a window. / TR: İlan bütünüyle, bir pencerede."""
    detail_view.render_detail(source, df, idx, ui.data_dir())


def table_key(source, mode, snapshot, conds):
    """
    EN: The table's widget key: changes with the source, the row set and the conditions, so a selection never
        survives a filter change (it would point at another row).
    TR: Tablonun widget anahtarı: kaynak, satır kümesi ve koşullarla değişir; böylece seçim bir filtre değişikliğinden
        sağ çıkmaz (başka bir satırı gösterirdi).
    """
    state = f"{source}|{mode}|{snapshot}|{filters.summary(conds)}"
    return f"table_{source}_{hashlib.sha1(state.encode('utf-8')).hexdigest()[:10]}"


ui.require_loopback()
st.session_state.setdefault("conditions", {"raw": [], "db": []})
st.session_state.setdefault("next_condition", 0)

with st.sidebar:
    source = st.radio("Kaynak", list(sources.SOURCES), format_func=sources.SOURCES.get, key="source")
    group = group_of(source)
    if not sources.data_files(source, ui.data_dir()):
        st.error(f"Veri yok: {sources.SOURCES[source]} ({ui.data_dir()})")
        st.stop()
    key = key_of(source)
    df, stats = load_source(source, key)
    mode = st.radio("Satır kümesi", ROW_SETS[group], format_func=filters.ROW_SETS.get, key=f"mode_{group}",
                    captions=[ROW_SET_HELP[m] for m in ROW_SETS[group]])
    snapshot = None
    if mode == "snapshot":
        snapshot = st.selectbox("Tarama", sorted(df[SNAPSHOT[group]].astype(str).unique()), key=f"snap_{group}")
    if st.button("Veriyi yeniden oku", help="dosyalar değiştiyse önbelleği boşaltır"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

st.title("Veri gezgini")
st.caption(f"{sources.SOURCES[source]} · {stats['records']:,} kayıt · {stats['files']} dosya".replace(",", ".")
           + (f" · {stats['broken']} bozuk satır atlandı" if stats["broken"] else "") + " · salt okunur")
if source == "gold":
    with st.expander("Gold, silver'dan nasıl ayrılır"):
        st.markdown(gold_rules_note())

kinds = column_kinds(source, key)
st.subheader("Koşullar")
shortcuts = {name: items for name, items in SHORTCUTS[group].items() if all(c in kinds for c, _, _ in items)}
if shortcuts:
    for col, (name, items) in zip(st.columns(len(shortcuts)), shortcuts.items()):
        col.button(f"+ {name}", key=f"short_{group}_{name}", on_click=add_shortcut, args=(group, items),
                   width="stretch")
pick_col, add_col = st.columns([5, 1], vertical_alignment="bottom")
pick_col.selectbox("Kolon", sorted(kinds, key=str.lower), index=None, key=f"pick_{group}",
                   placeholder="koşul eklemek için bir kolon seç (yazarak ara)…",
                   format_func=lambda c: f"{c}  ·  {kinds[c]}")
add_col.button("Koşul ekle", on_click=add_picked, args=(group, kinds), width="stretch")
conds = draw_conditions(group, source, key, kinds)

view_df, mask, steps, missing, n_base = filtered(source, mode, snapshot, conds)
view = view_df[mask]
st.info(f"**{filters.ROW_SETS[mode]}** ({n_base:,} satır)".replace(",", ".") + " → " +
        (" → ".join(f"{filters.describe(c)} **[{n:,}]**".replace(",", ".") for c, n in steps) or "koşul yok"))
if missing:
    st.warning("Bu kaynakta kolonu olmayan koşullar uygulanmadı: " + ", ".join(filters.describe(c) for c in missing))

k = charts.kpis(view, source)
metrics = st.columns(6 if group == "db" else 5)
metrics[0].metric("Satır", number(k["rows"]))
metrics[1].metric("Tekil ilan", number(k["ads"]))
metrics[2].metric("Medyan fiyat", number(k["price"], " ₺"))
metrics[3].metric("Medyan km", number(k["km"]))
metrics[4].metric("Medyan yaş", "—" if k["age"] is None else f"{k['age']:.0f}")
if group == "db":
    other = "gold" if source == "silver" else "silver"
    if sources.data_files(other, ui.data_dir()):
        _, other_mask, _, other_missing, _ = filtered(other, mode, snapshot, conds)
        metrics[5].metric(f"Aynı koşullar {other}'da", number(int(other_mask.sum())),
                          delta=int(other_mask.sum()) - k["rows"], delta_color="off")
        if other_missing:
            st.caption(f"{other}'da olmayan kolonlar yüzünden o tarafta uygulanmayan koşullar: "
                       + ", ".join(filters.describe(c) for c in other_missing))

tab_table, tab_charts = st.tabs(["Tablo", "Grafikler"])
with tab_table:
    columns = st.multiselect("Gösterilen kolonlar", list(df.columns), key=f"cols_{group}",
                             **widget_default(f"cols_{group}",
                                              default=[c for c in DEFAULT_COLUMNS[group] if c in df.columns]))
    shown = view[columns or list(df.columns)].head(TABLE_ROWS)
    config = {"url": st.column_config.LinkColumn("url")} if "url" in shown.columns else None
    tkey = table_key(source, mode, snapshot, conds)
    st.caption("👉 Bir ilanın **herhangi bir hücresine** tıkla: ilanın tüm ayrıntısı bir pencerede açılır.")
    event = st.dataframe(shown, hide_index=True, column_config=config, on_select="rerun",
                         selection_mode=["single-row", "single-cell"], key=tkey)
    st.caption(f"{len(view):,} satırdan ilk {min(len(view), TABLE_ROWS):,} gösteriliyor.".replace(",", "."))
    picked = (event.selection.rows, event.selection.cells) if event is not None else ([], [])
    pos = details.picked_position(*picked)
    if pos is None or pos >= len(shown):
        # EN: nothing selected: the next click opens the window again | TR: seçim yok: sonraki tıklama yeniden açar
        st.session_state["opened_detail"] = None
    else:
        idx = shown.index[pos]
        token = (tkey, repr(picked))
        if st.session_state.get("opened_detail") != token:
            st.session_state["opened_detail"] = token
            detail_dialog(source, df, idx)
        elif st.button("Seçili ilanı yeniden aç"):
            detail_dialog(source, df, idx)
with tab_charts:
    grid = st.columns(2)
    for i, build in enumerate(charts.FIGURES):
        with grid[i % 2]:
            st.plotly_chart(build(view, source), key=f"chart_{i}")
