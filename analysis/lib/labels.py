"""
labels.py
EN: Display names for the SHAP figures and the SHAP report (TR, EN) — one source for the shap scripts, which
    draw the figures, and builders/build_shap_report.py, which writes the text. Presentation only; nothing is
    computed here.
TR: SHAP figürlerinin ve SHAP raporunun görünen adları (TR, EN) — figürleri çizen shap betikleri ile metni yazan
    builders/build_shap_report.py için tek kaynak. Yalnız sunum; burada hesap yok.
"""
import re

LBL = {"vehicle_age": ("Yaş (yıl)", "Age (years)"), "gb_mileage": ("Kilometre", "Mileage"),
       "power_hp_val": ("Motor gücü (hp)", "Engine power (hp)"),
       "engine_cc_val": ("Motor hacmi (cc)", "Displacement (cc)"),
       "segment": ("Segment", "Segment"), "brand": ("Marka", "Brand"),
       "kb_body_type": ("Kasa tipi", "Body type"), "kb_fuel": ("Yakıt", "Fuel"),
       "kb_transmission": ("Vites", "Transmission"), "kb_drivetrain": ("Çekiş", "Drivetrain"),
       "MODEL_SERIES (text)": ("Model/seri adı", "Model/series name"),
       "ENGINE": ("Motor (hp + cc)", "Engine (hp + cc)"), "DAMAGE": ("Hasar (panel + ağır hasar)", "Damage (panels + heavy damage)"),
       "is_heavy_damaged": ("Ağır hasar", "Heavy damage"), "hood_state": ("Kaput", "Hood"),
       "roof_state": ("Tavan", "Roof"), "trunk_state": ("Bagaj", "Trunk"),
       "door_painted": ("Kapı boyalı", "Door painted"), "door_changed": ("Kapı değişen", "Door changed"),
       "door_local": ("Kapı lokal", "Door local"), "fender_painted": ("Çamurluk boyalı", "Fender painted"),
       "fender_changed": ("Çamurluk değişen", "Fender changed"), "fender_local": ("Çamurluk lokal", "Fender local"),
       "bumper_painted": ("Tampon boyalı", "Bumper painted"), "bumper_changed": ("Tampon değişen", "Bumper changed"),
       "bumper_local": ("Tampon lokal", "Bumper local")}
# EN: mileage prints like 100000 and the ticks collide, so it is shown in thousands | TR: km binlik gösterilir
KM_SUF = {"tr": " (bin km)", "en": " (000 km)"}
# EN: English names of category values (EN labels only; model names stay) | TR: kategori değerlerinin İngilizcesi
CAT_EN = {"Otomatik": "Automatic", "Yarı Otomatik": "Semi-automatic", "Düz": "Manual",
          "Arkadan İtiş": "Rear-wheel drive", "Önden Çekiş": "Front-wheel drive",
          "4WD (Sürekli)": "4WD (permanent)", "AWD (Elektronik)": "AWD (electronic)",
          "Benzin": "Petrol", "Dizel": "Diesel", "LPG & Benzin": "LPG & Petrol", "Hibrit": "Hybrid",
          "missing": "unknown"}
STATE_LBL = {"changed": ("değişen", "changed"), "painted": ("boyalı", "painted"),
             "local": ("lokal", "local"), "original": ("orijinal", "original")}
CAT_COLS = ("brand", "kb_body_type", "kb_drivetrain", "segment", "kb_transmission", "kb_fuel",
            "roof_state", "hood_state", "trunk_state")


def lb(key, lang):
    """
    EN: Display name of a feature or group in lang ("tr"/"en"); unknown keys are returned as they are.
    TR: Bir özniteliğin ya da grubun lang ("tr"/"en") dilindeki adı; bilinmeyen anahtar olduğu gibi döner.
    """
    t = LBL.get(key)
    return (t[0] if lang == "tr" else t[1]) if t else key


def value_label(row, key, lang):
    """
    EN: The value text of one feature of one listing, as the waterfall and the report print it: category
        names instead of codes, panel counts, units, TR/EN number format. row: the listing's record.
    TR: Bir ilanın bir özniteliğinin değer metni, waterfall ve rapor nasıl basıyorsa: kod yerine kategori
        adı, panel sayısı, birim, TR/EN sayı biçimi. row: ilanın kaydı.
    """
    if key == "MODEL_SERIES (text)":
        return str(row["model"])
    if key.endswith("_state"):
        v = str(row[key])
        assert v in STATE_LBL, f"unknown damage state | bilinmeyen hasar durumu: {v}"
        return STATE_LBL[v][0 if lang == "tr" else 1]
    if key == "is_heavy_damaged":
        return ("var" if lang == "tr" else "yes") if int(row[key]) else ("yok" if lang == "tr" else "no")
    if key.startswith(("door_", "fender_", "bumper_")):
        n = int(float(row[key]))
        return f"{n} parça" if lang == "tr" else f"{n} panel" + ("" if n == 1 else "s")
    if key in CAT_COLS:
        v_ = str(row[key])
        return v_ if lang == "tr" else CAT_EN.get(v_, v_)
    try:
        v = float(row[key])
    except (TypeError, ValueError):
        return "—"
    if v != v or v in (float("inf"), float("-inf")):
        return "—"
    if key == "gb_mileage":
        return f"{int(v):,}".replace(",", "." if lang == "tr" else ",") + " km"
    unit = {"vehicle_age": (" yıl", " yr"), "power_hp_val": (" hp", " hp"), "engine_cc_val": (" cc", " cc")}.get(key)
    s = f"{v:.0f}" if float(v).is_integer() or abs(v) >= 10 else f"{v:.1f}"
    return s + (unit[0] if lang == "tr" else unit[1]) if unit else s


def other_features_tr(text):
    """
    EN: shap's "Sum of N other features" / "N other features" row text in Turkish.
    TR: shap'in "Sum of N other features" / "N other features" satır metninin Türkçesi.
    """
    text = re.sub(r"^Sum of (\d+) other features$", r"\1 diğer özniteliğin toplamı", text)
    return re.sub(r"^(\d+) other features$", r"\1 diğer öznitelik", text)
