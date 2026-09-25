"""
details.py
EN: What the explorer's listing window shows, as plain data: a header (title, price, year, km, place, snapshot,
    url), every field grouped (raw by its prefix — KısaBilgi, Genel Bakış, …; the DB by kind — identity and price,
    kb_, gb_, engine and specs, damage), the 13-panel damage table (raw from Hasar_Listesi, the DB from its flags,
    NULL = unknown), the ad's rows across snapshots with the price change, and which row a table click picked. Pure:
    no streamlit.
TR: Gezginin ilan penceresinin gösterdikleri, düz veri olarak: üst bilgi (başlık, fiyat, yıl, km, yer, tarama, url),
    gruplanmış her alan (ham kaynak ön ekine göre — KısaBilgi, Genel Bakış, …; DB türüne göre — kimlik ve fiyat, kb_,
    gb_, motor ve teknik, hasar), 13 parçalık hasar tablosu (ham kaynakta Hasar_Listesi'nden, DB'de bayraklarından,
    NULL = bilinmiyor), ilanın taramalar boyunca satırları ve fiyat farkı, ve tablodaki tıklamanın hangi satırı
    seçtiği. Saf: streamlit yok.
"""
import pandas as pd

from internal_tool import charts

RAW_PREFIXES = ("KısaBilgi", "Genel Bakış", "Motor ve Performans", "Yakıt Tüketimi", "Boyut ve Kapasite")
RAW_GROUP_ORDER = ("Kayıt",) + RAW_PREFIXES + ("Diğer",)
DB_IDENTITY = ("id", "ad_id", "listing_date", "ad_title", "brand", "series", "model", "eids_model", "location", "price",
               "url", "search_date", "scraped_at")
DB_DAMAGE = ("is_heavy_damaged", "kb_is_heavy_damaged", "tramer_fee", "count_changed", "count_painted",
             "count_local_painted", "kb_paint_change_summary")
DB_SPECS = ("engine_", "power_", "torque_nm", "cylinder_count", "max_speed_kmh", "accel_0_100", "city_fuel_cons",
            "highway_fuel_cons", "length_mm", "width_mm", "height_mm", "weight_kg", "curb_weight_kg",
            "trunk_capacity_lt", "wheelbase_mm", "transmission_brand", "seat_count", "front_tire_spec",
            "production_year_", "rpm_")
DB_GROUP_ORDER = ("Kimlik ve fiyat", "Kısa bilgi (kb_)", "Genel bakış (gb_)", "Motor ve teknik", "Hasar", "Diğer")
HEADER = {"raw": {"title": "Ilan_Basligi", "price": "Fiyat", "year": "KısaBilgi - Yıl", "km": "KısaBilgi - Kilometre",
                  "place": "Konum", "snapshot": "search_date", "url": "url"},
          "db": {"title": "ad_title", "price": "price", "year": "kb_year", "km": "kb_mileage", "place": "location",
                 "snapshot": "search_date", "url": "url"}}
HEAVY = {"raw": ("Agir_Hasar", "KısaBilgi - Ağır Hasarlı", "Tramer_Tutari", "Degisen_Parca_Sayisi",
                 "Boyali_Parca_Sayisi", "Lokal_Boyali_Parca_Sayisi", "KısaBilgi - Boya-değişen"),
         "db": ("is_heavy_damaged", "tramer_fee", "count_changed", "count_painted", "count_local_painted",
                "kb_paint_change_summary")}


def kind_of(source):
    """EN: "raw" or "db". / TR: "raw" ya da "db"."""
    return "raw" if source == "raw" else "db"


def is_blank(value):
    """EN: None, NaN/NA or a blank string. / TR: None, NaN/NA ya da boş metin."""
    if isinstance(value, (list, dict)):
        return False
    return (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def raw_group(field):
    """EN: The group of a raw field. / TR: Bir ham alanın grubu."""
    prefix, sep, _ = field.partition(" - ")
    if not sep:
        return "Kayıt"
    return prefix if prefix in RAW_PREFIXES else "Diğer"


def db_group(field):
    """EN: The group of a DB column. / TR: Bir DB kolonunun grubu."""
    if field in DB_IDENTITY:
        return "Kimlik ve fiyat"
    if field in DB_DAMAGE or any(field.startswith(p + "_") for p in charts.PANELS):
        return "Hasar"
    if field.startswith("kb_"):
        return "Kısa bilgi (kb_)"
    if field.startswith("gb_"):
        return "Genel bakış (gb_)"
    if field.startswith(DB_SPECS):
        return "Motor ve teknik"
    return "Diğer"


def group_fields(fields, source):
    """
    EN: Every field once, grouped. Returns: [(group, [(field, value)])] in a fixed group order, empty groups left out.
    TR: Her alan bir kez, gruplanmış. Döndürür: [(grup, [(alan, değer)])], sabit grup sırasıyla, boş gruplar hariç.
    """
    raw = kind_of(source) == "raw"
    order, pick = (RAW_GROUP_ORDER, raw_group) if raw else (DB_GROUP_ORDER, db_group)
    groups = {g: [] for g in order}
    for field, value in fields.items():
        groups[pick(field)].append((field, value))
    return [(g, items) for g, items in groups.items() if items]


def display(value):
    """EN: A value as text for a table ("" when blank). / TR: Tablo için değer metni (boşsa "")."""
    if isinstance(value, (list, dict)):
        return ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return "" if is_blank(value) else str(value)


def damage_table(fields, source):
    """
    EN: The panels and their damage state: DataFrame ["parça", "durum"]. Raw: from Hasar_Listesi ("Part: State") or
        "Hasar - <Part>" fields; DB: from the three flags of each panel (NULL → unknown).
    TR: Paneller ve hasar durumları: DataFrame ["parça", "durum"]. Ham: Hasar_Listesi'nden ("Parça: Durum") ya da
        "Hasar - <Parça>" alanlarından; DB: her panelin üç bayrağından (NULL → bilinmiyor).
    """
    rows = []
    if kind_of(source) == "raw":
        items = fields.get("Hasar_Listesi")
        if isinstance(items, list):
            for item in items:
                part, sep, state = str(item).partition(":")
                if sep:
                    rows.append((part.strip(), state.strip()))
        else:
            rows = [(k.removeprefix("Hasar - "), display(v) or charts.UNKNOWN) for k, v in fields.items()
                    if k.startswith("Hasar - ")]
    else:
        frame = pd.DataFrame([fields])
        for prefix, name in charts.PANELS.items():
            state = charts.panel_states(frame, prefix)
            if state is not None:
                rows.append((name, state.iloc[0]))
    return pd.DataFrame(rows, columns=["parça", "durum"])


def heavy_fields(fields, source):
    """EN: Heavy damage, tramer and counters of a row. / TR: Bir satırın ağır hasar, tramer ve sayaçları."""
    return [(k, fields[k]) for k in HEAVY[kind_of(source)] if k in fields]


def header(fields, source):
    """EN: The window's header values (missing → None). / TR: Pencerenin üst bilgi değerleri (yoksa None)."""
    return {k: (None if col not in fields or is_blank(fields[col]) else fields[col])
            for k, col in HEADER[kind_of(source)].items()}


def history(df, source, ad_id):
    """
    EN: The ad's rows across the source's snapshots: DataFrame ["tarama", "fiyat", "km", "fiyat farkı"], sorted by
        snapshot; the change is against the previous snapshot. Empty without an ad id.
    TR: İlanın kaynaktaki taramalar boyunca satırları: DataFrame ["tarama", "fiyat", "km", "fiyat farkı"], taramaya
        göre sıralı; fark bir önceki taramaya göredir. İlan kimliği yoksa boş.
    """
    columns = ["tarama", "fiyat", "km", "fiyat farkı"]
    if ad_id is None or is_blank(ad_id) or "ad_id" not in df.columns:
        return pd.DataFrame(columns=columns)
    f = charts.fields_for(source)
    rows = df[df["ad_id"].eq(ad_id).fillna(False).to_numpy(dtype=bool)]
    out = pd.DataFrame({"tarama": rows[f["snapshot"]].astype(str).str[:10].to_numpy(),
                        "fiyat": pd.to_numeric(rows[f["price"]], errors="coerce").to_numpy(),
                        "km": pd.to_numeric(rows[f["km"]], errors="coerce").to_numpy()})
    out = out.sort_values("tarama", kind="stable").reset_index(drop=True)
    out["fiyat farkı"] = out["fiyat"].diff()
    return out[columns]


def picked_position(rows, cells):
    """
    EN: The table position a click picked: the selected row first, else the row of the selected cell; None if none.
    TR: Tıklamanın seçtiği tablo konumu: önce seçilen satır, yoksa seçilen hücrenin satırı; hiçbiri yoksa None.
    """
    if rows:
        return int(rows[0])
    if cells:
        cell = cells[0]
        return int(cell[0] if isinstance(cell, (list, tuple)) else cell["row"])
    return None
