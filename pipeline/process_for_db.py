import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

df = pd.read_json("data\\audi\\2026-01-27_02-10\\details.jsonl", lines=True)
df = df[df["Fiyat"].notna()].copy()
df = df[df["Fiyat"] != ""].copy()


# data\bmw\2026-01-18_19-56\details.jsonl
# data\audi\2026-01-18_19-56\details.jsonl

# data\bmw\2026-01-27_02-10\details.jsonl
# data\audi\2026-01-27_02-10\details.jsonl


def parse_turkish_date(date_str):
    """
    '26 Kasım 2025' formatındaki metni '2025-11-26' (datetime.date) formatına çevirir.
    """
    if not date_str or pd.isna(date_str):
        return None

    # Gereksiz boşlukları temizle
    date_str = str(date_str).strip()

    tr_months = {
        "Ocak": "01",
        "Şubat": "02",
        "Mart": "03",
        "Nisan": "04",
        "Mayıs": "05",
        "Haziran": "06",
        "Temmuz": "07",
        "Ağustos": "08",
        "Eylül": "09",
        "Ekim": "10",
        "Kasım": "11",
        "Aralık": "12",
    }

    try:
        # Örn: "26 Kasım 2025" -> ["26", "Kasım", "2025"]
        parts = date_str.split()
        if len(parts) == 3:
            day, month_txt, year = parts
            month_num = tr_months.get(month_txt)
            if month_num:
                # YYYY-MM-DD formatında string veya datetime objesi döndür
                return datetime.strptime(f"{year}-{month_num}-{day}", "%Y-%m-%d").date()
    except:
        return None
    return None


def process_to_silver_full(raw_item):
    """
    Tüm kolonları kapsayan, prefixli ve teknik 4'lü yapıya sahip Silver işlemci.
    """

    # --- YARDIMCI ARAÇLAR ---
    def clean_num(val):
        if val is None or pd.isna(val):
            return None
        # Sayı dışındaki her şeyi at (nokta ve virgül hariç)
        # 1.620 -> 1620 | 11,8 -> 11.8
        cleaned = re.sub(r"[^\d,]", "", str(val)).replace(",", ".")
        return float(cleaned) if cleaned else None

    def get_interval_stats(val):
        """Low, Up, Val (Mean), Is_Range döner."""
        if not val or pd.isna(val):
            return None, None, None, False
        nums = re.findall(r"(\d+)", str(val).replace(".", ""))
        if not nums:
            return None, None, None, False
        nums = [float(n) for n in nums]
        if len(nums) >= 2:
            low, up = min(nums), max(nums)
            return low, up, (low + up) / 2, True
        else:
            v = nums[0]
            return v, v, v, False

    def analyze_damage_locations(damage_list):
        """Hasar listesini parçalara ve durumlara (Degisen/Boyali/Lokal) böler."""
        text = str(damage_list).lower()
        parts_map = {
            "tavan": "tavan",
            "kaput": "kaput",
            "bagaj": "bagaj",
            "door_fl": "sol ön kapı",
            "door_fr": "sağ ön kapı",
            "door_rl": "sol arka kapı",
            "door_rr": "sağ arka kapı",
            "fender_fl": "sol ön çamurluk",
            "fender_fr": "sağ ön çamurluk",
            "fender_rl": "sol arka çamurluk",
            "fender_rr": "sağ arka çamurluk",
        }
        damage_results = {}
        for key, tr_name in parts_map.items():
            # Değişen (1), Lokal (1), Boyalı (1) - Bilgi kaybı olmaması için her durum ayrı kolon
            deg = (
                1
                if f"{tr_name}: değişen" in text or f"{tr_name}: değişmiş" in text
                else 0
            )
            lok = 1 if f"{tr_name}: lokal" in text and deg == 0 else 0
            boy = 1 if f"{tr_name}: boyalı" in text and deg == 0 and lok == 0 else 0

            damage_results[f"{key}_degisen"] = deg
            damage_results[f"{key}_boyali"] = boy
            damage_results[f"{key}_lokal"] = lok
        return damage_results

    # 1. TEMEL METADATA
    # İlan Tarihi Dönüşümü
    # JSON'da key genellikle "KısaBilgi - İlan Tarihi" olarak gelir, verini kontrol et.
    listing_date = parse_turkish_date(raw_item.get("KısaBilgi - İlan Tarihi"))
    ad_id_raw = raw_item.get("KısaBilgi - İlan No", "")
    ad_id = (
        int(re.search(r"(\d+)", ad_id_raw).group(1))
        if re.search(r"(\d+)", ad_id_raw)
        else None
    )

    # Saat yuvarlama (Dakika/Saniye -> 00)
    scraped_at = pd.to_datetime(raw_item.get("scraped_at"))

    # search_date (Örn: 2026-01-18_01-05) -> SQL Timestamp formatına
    search_raw = str(raw_item.get("search_date", ""))
    try:
        # Alt tireli formatı standart zaman formatına çeviriyoruz
        search_ts = datetime.strptime(search_raw, "%Y-%m-%d_%H-%M")
    except:
        search_ts = None

    # Açıklama Temizliği
    desc_html = raw_item.get("Aciklama_HTML", "")
    desc_text = (
        BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True).lower()
        if desc_html
        else ""
    )

    # CC ve HP için 4'lü Yapı
    cc_low, cc_up, cc_val, cc_is_range = get_interval_stats(
        raw_item.get("KısaBilgi - Motor Hacmi")
    )
    hp_low, hp_up, hp_val, hp_is_range = get_interval_stats(
        raw_item.get("KısaBilgi - Motor Gücü")
    )

    # Detaylı Hasar Analizi
    loc_damage = analyze_damage_locations(raw_item.get("Hasar_Listesi", []))

    # 2. SILVER SÖZLÜĞÜ OLUŞTURMA
    silver_data = {
        # Kimlik ve Başlık
        "ad_id": ad_id,
        "listing_date": listing_date,
        "ad_title": raw_item.get("Ilan_Basligi"),
        # "ilan tarihini buraya ekle"
        "brand": raw_item.get("brand"),
        "series": raw_item.get("KısaBilgi - Seri"),
        "model": raw_item.get("KısaBilgi - Model"),
        "location": raw_item.get("Konum"),
        "price": clean_num(raw_item.get("Fiyat")),
        # --- KISA BİLGİ (kb_) KOLONLARI ---
        "kb_year": int(raw_item.get("KısaBilgi - Yıl", 0)),
        "kb_mileage": clean_num(raw_item.get("KısaBilgi - Kilometre")),
        "kb_transmission": raw_item.get("KısaBilgi - Vites Tipi"),
        "kb_fuel": raw_item.get("KısaBilgi - Yakıt Tipi"),
        "kb_body_type": raw_item.get("KısaBilgi - Kasa Tipi"),
        "kb_color": raw_item.get("KısaBilgi - Renk"),
        "kb_drivetrain": raw_item.get("KısaBilgi - Çekiş"),
        "kb_condition": raw_item.get("KısaBilgi - Araç Durumu"),
        "kb_is_heavy_damaged": raw_item.get("KısaBilgi - Ağır Hasarlı") == "Evet",
        "kb_trade_available": raw_item.get("KısaBilgi - Takasa Uygun")
        == "Takasa Uygun",
        "kb_seller_type": raw_item.get("KısaBilgi - Kimden"),
        "kb_fuel_cons_avg": clean_num(raw_item.get("KısaBilgi - Ort. Yakıt Tüketimi")),
        "kb_fuel_tank": clean_num(raw_item.get("KısaBilgi - Yakıt Deposu")),
        # --- GENEL BAKIŞ (gb_) KOLONLARI ---
        "gb_year": int(raw_item.get("Genel Bakış - Yıl", 0)),
        "gb_mileage": clean_num(raw_item.get("Genel Bakış - Kilometre")),
        "gb_fuel": raw_item.get("Genel Bakış - Yakıt Tipi"),
        "gb_transmission": raw_item.get("Genel Bakış - Vites Tipi"),
        "gb_color": raw_item.get("Genel Bakış - Renk"),
        "gb_warranty_status": raw_item.get("Genel Bakış - Garanti Durumu"),
        "gb_usage_type": raw_item.get("Genel Bakış - Araç Türü"),
        "gb_is_first_owner": "İlk Sahibi"
        in str(raw_item.get("Genel Bakış - Aracın ilk sahibiyim")),
        "gb_segment": raw_item.get("Genel Bakış - Sınıfı"),
        "gb_body_type": raw_item.get("Genel Bakış - Kasa Tipi"),
        "gb_mtv_yearly": clean_num(raw_item.get("Genel Bakış - Yıllık MTV")),
        # --- TEKNİK ÖZELLİKLER (4'lü Yapı ve Tekiller) ---
        "engine_cc_low": cc_low,
        "engine_cc_up": cc_up,
        "engine_cc_val": cc_val,
        "engine_cc_is_range": cc_is_range,
        "power_hp_low": hp_low,
        "power_hp_up": hp_up,
        "power_hp_val": hp_val,
        "power_hp_is_range": hp_is_range,
        "torque_nm": clean_num(raw_item.get("Motor ve Performans - Tork")),
        "cylinder_count": clean_num(
            raw_item.get("Motor ve Performans - Silindir Sayısı")
        ),
        "max_speed_kmh": clean_num(raw_item.get("Motor ve Performans - Maksimum Hız")),
        "accel_0_100": clean_num(
            raw_item.get("Motor ve Performans - Hızlanma (0-100)")
        ),
        # Yakıt Detayları
        "city_fuel_cons": clean_num(
            raw_item.get("Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi")
        ),
        "highway_fuel_cons": clean_num(
            raw_item.get("Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi")
        ),
        # Boyut ve Kapasite
        "length_mm": clean_num(raw_item.get("Boyut ve Kapasite - Uzunluk")),
        "width_mm": clean_num(raw_item.get("Boyut ve Kapasite - Genişlik")),
        "height_mm": clean_num(raw_item.get("Boyut ve Kapasite - Yükseklik")),
        "weight_kg": clean_num(raw_item.get("Boyut ve Kapasite - Ağırlık")),
        "curb_weight_kg": clean_num(raw_item.get("Boyut ve Kapasite - Boş Ağırlığı")),
        "trunk_capacity_lt": clean_num(raw_item.get("Boyut ve Kapasite - Bagaj Hacmi")),
        "wheelbase_mm": clean_num(raw_item.get("Boyut ve Kapasite - Aks Aralığı")),
        # Hasar Durumu
        "is_heavy_damaged": bool(raw_item.get("Agir_Hasar", False)),
        "tramer_fee": raw_item.get("Tramer_Tutari", 0),
        "count_changed": raw_item.get("Degisen_Parca_Sayisi", 0),
        "count_painted": raw_item.get("Boyali_Parca_Sayisi", 0),
        "count_local_painted": raw_item.get("Lokal_Boyali_Parca_Sayisi", 0),
        **loc_damage,
        # Zamanlama ve Açıklama
        "description_text": desc_text,
        "scraped_at": scraped_at,
        "search_date": search_ts,
    }
    print(silver_data["ad_id"])
    return silver_data


silver_df = df.apply(
    lambda row: pd.Series(process_to_silver_full(row.to_dict())), axis=1
)


print(silver_df.head())

print(silver_df["kb_mileage"], silver_df["gb_mileage"], silver_df["engine_cc_val"])

load_dotenv()

connection_string = os.getenv("DATABASE_URL")

engine = create_engine(connection_string)

# 1. Veritabanındaki mevcut ID'leri çek
existing_ids = pd.read_sql("SELECT ad_id FROM test.car_listings", engine)[
    "ad_id"
].tolist()

# 2. df içinden veritabanında OLMAYANLARI filtrele
new_df = silver_df[~silver_df["ad_id"].isin(existing_ids)]

# 3. Sadece yenileri gönder
if not new_df.empty:
    new_df.to_sql(
        name="car_listings",
        con=engine,
        schema="test",
        if_exists="append",
        index=False,
        chunksize=500,
    )
    print(f"🚀 {len(new_df)} yeni ilan eklendi.")
else:
    print("✨ Tüm ilanlar zaten güncel, yeni veri yok.")
