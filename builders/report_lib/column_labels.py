"""
column_labels.py
EN: Display labels of raw columns ({tr, en, tab}) for the site and the reports. gb_/kb_ come from the listing's
    "Genel Bakış" / "KısaBilgi" tabs (`tab`); the plain name is shown, with the tab added where two columns
    would otherwise share a name. Static presentation data — build_site_data.py writes it into site_data.json
    and data/serving/column_labels.json; the report builders read it from here (builders/report_lib/).
TR: Ham kolonların görünen etiketleri ({tr, en, tab}); site ve raporlar için. gb_/kb_ ilanın "Genel Bakış" /
    "KısaBilgi" sekmelerinden gelir (`tab`); sade ad gösterilir, iki kolon aynı adı alacaksa sekme eklenir.
    Durağan sunum verisi — build_site_data.py bunu site_data.json'a ve data/serving/column_labels.json'a
    yazar; rapor derleyicileri buradan okur (builders/report_lib/).
"""

COLUMN_LABELS = {
    # sayısal (NUM)
    "vehicle_age":        {"tr": "Yaş (yıl)",            "en": "Age (years)",         "tab": None},
    "gb_mileage":         {"tr": "Kilometre",            "en": "Mileage",             "tab": "gb"},
    "power_hp_val":       {"tr": "Motor Gücü (hp)",      "en": "Power (hp)",          "tab": None},
    "engine_cc_val":      {"tr": "Motor Hacmi (cc)",     "en": "Engine (cc)",         "tab": None},
    "door_changed":       {"tr": "Kapı Değişen",         "en": "Door Changed",        "tab": None},
    "door_painted":       {"tr": "Kapı Boyalı",          "en": "Door Painted",        "tab": None},
    "door_local":         {"tr": "Kapı Lokal Boya",      "en": "Door Local Paint",    "tab": None},
    "fender_changed":     {"tr": "Çamurluk Değişen",     "en": "Fender Changed",      "tab": None},
    "fender_painted":     {"tr": "Çamurluk Boyalı",      "en": "Fender Painted",      "tab": None},
    "fender_local":       {"tr": "Çamurluk Lokal Boya",  "en": "Fender Local Paint",  "tab": None},
    "bumper_changed":     {"tr": "Tampon Değişen",       "en": "Bumper Changed",      "tab": None},
    "bumper_painted":     {"tr": "Tampon Boyalı",        "en": "Bumper Painted",      "tab": None},
    "bumper_local":       {"tr": "Tampon Lokal Boya",    "en": "Bumper Local Paint",  "tab": None},
    "is_heavy_damaged":   {"tr": "Ağır Hasarlı",         "en": "Heavy Damaged",       "tab": None},
    # kategorik (CAT)
    "brand":              {"tr": "Marka",                "en": "Brand",               "tab": None},
    "kb_body_type":       {"tr": "Kasa Tipi",            "en": "Body Type",           "tab": "kb"},
    "kb_drivetrain":      {"tr": "Çekiş",                "en": "Drivetrain",          "tab": "kb"},
    "segment":            {"tr": "Segment",              "en": "Segment",             "tab": None},
    "kb_transmission":    {"tr": "Vites Tipi",           "en": "Transmission",        "tab": "kb"},
    "kb_fuel":            {"tr": "Yakıt Tipi",           "en": "Fuel Type",           "tab": "kb"},
    "roof_state":         {"tr": "Tavan Durumu",         "en": "Roof State",          "tab": None},
    "hood_state":         {"tr": "Kaput Durumu",         "en": "Hood State",          "tab": None},
    "trunk_state":        {"tr": "Bagaj Durumu",         "en": "Trunk State",         "tab": None},
    # metin / kimlik (TEXT) — Cramér/Theil'e eklendi
    "model":              {"tr": "Model",                "en": "Model",               "tab": None},
    "series":             {"tr": "Seri",                 "en": "Series",              "tab": None},
    # site_data'da geçen diğer gb/kb + alanlar (özellikle Veri Kalitesi / audit)
    "gb_year":            {"tr": "Yıl",                  "en": "Year",                "tab": "gb"},
    "gb_segment":         {"tr": "Segment (ham)",        "en": "Segment (raw)",       "tab": "gb"},
    "gb_warranty_status": {"tr": "Garanti Durumu",       "en": "Warranty Status",     "tab": "gb"},
    "gb_kasko_avg":       {"tr": "Ortalama Kasko",       "en": "Avg. Casco Insurance","tab": "gb"},
    "gb_mtv_yearly":      {"tr": "Yıllık MTV",           "en": "Annual Vehicle Tax",  "tab": "gb"},
    "gb_traffic_insurance_avg": {"tr": "Ort. Trafik Sigortası", "en": "Avg. Traffic Insurance", "tab": "gb"},
    "kb_fuel_cons_avg":   {"tr": "Ort. Yakıt Tüketimi",  "en": "Avg. Fuel Consumption","tab": "kb"},
    "kb_fuel_tank":       {"tr": "Yakıt Deposu",         "en": "Fuel Tank",           "tab": "kb"},
    "gb_drivetrain":      {"tr": "Çekiş",                "en": "Drivetrain",          "tab": "gb"},
    "price":              {"tr": "Fiyat (₺)",            "en": "Price (₺)",           "tab": None},
    "count_changed":      {"tr": "Değişen Parça",        "en": "Changed Parts",       "tab": None},
    "count_painted":      {"tr": "Boyalı Parça",         "en": "Painted Parts",       "tab": None},
    "count_local_painted":{"tr": "Lokal Boyalı Parça",   "en": "Local-Painted Parts", "tab": None},
    "tramer_fee":         {"tr": "Tramer Tutarı (₺)",    "en": "Tramer (Damage Record) ₺", "tab": None},
    # tab kodu açıklaması (frontend için)
    "_tab_meaning":       {"gb": "Genel Bakış", "kb": "KısaBilgi"},
}
