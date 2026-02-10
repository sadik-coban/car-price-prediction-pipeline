-- 1. Şemayı Oluştur
CREATE SCHEMA IF NOT EXISTS test;

-- 2. Tabloyu Oluştur
CREATE TABLE IF NOT EXISTS test.car_listings (
    -- Kimlik Bilgileri
    ad_id BIGINT PRIMARY KEY,
    listing_date DATE, -- YENİ EKLENEN ALAN (26 Kasım 2025 gibi tarihler için)
    ad_title TEXT,
    brand VARCHAR(50),
    series VARCHAR(100),
    model VARCHAR(200),
    location TEXT,
    price NUMERIC(15, 2), -- Para birimi hassas olmalı, NUMERIC kalmalı.

    -- Teknik 4'lüler (Motor hacmi ondalıklı olabilir ama genelde tam sayıdır, INT yeterli)
    engine_cc_low INTEGER, engine_cc_up INTEGER, engine_cc_val INTEGER, engine_cc_is_range BOOLEAN,
    power_hp_low INTEGER, power_hp_up INTEGER, power_hp_val INTEGER, power_hp_is_range BOOLEAN,

    -- KB vs GB (Yıl ve Kilometre Optimizasyonu)
    kb_year SMALLINT, gb_year SMALLINT, -- Yıl 32.000'i geçmez (SMALLINT)
    kb_mileage INTEGER, gb_mileage INTEGER, -- KM 2 milyarı geçmez (INTEGER)
    
    -- Kategorik Veriler (VARCHAR kalabilir)
    kb_transmission VARCHAR(50), gb_transmission VARCHAR(50),
    kb_fuel VARCHAR(50), gb_fuel VARCHAR(50),
    kb_body_type VARCHAR(50), gb_body_type VARCHAR(50),
    kb_color VARCHAR(50), gb_color VARCHAR(50),
    kb_drivetrain VARCHAR(50),
    kb_condition VARCHAR(50),
    
    -- Bayraklar (Boolean en hafifi - 1 bit/byte)
    kb_is_heavy_damaged BOOLEAN,
    kb_trade_available BOOLEAN,
    kb_seller_type VARCHAR(50),
    
    -- Yakıt Tüketimi (Ondalıklı veriler)
    kb_fuel_cons_avg NUMERIC(4, 1), -- Örn: 5.4 (Daha az yer kaplar)
    kb_fuel_tank SMALLINT, -- Depo 32.000 litreyi geçmez

    -- Genel Bakış Ekleri
    gb_warranty_status VARCHAR(100),
    gb_usage_type VARCHAR(50),
    gb_is_first_owner BOOLEAN,
    gb_segment VARCHAR(50),
    gb_mtv_yearly NUMERIC(10, 2),

    -- Teknikler (Fiziksel Özellikler - SMALLINT/INTEGER Optimizasyonu)
    torque_nm SMALLINT, 
    cylinder_count SMALLINT, 
    max_speed_kmh SMALLINT,
    accel_0_100 NUMERIC(4, 1), -- 0-100 hızlanma örn: 8.2 sn
    city_fuel_cons NUMERIC(4, 1), 
    highway_fuel_cons NUMERIC(4, 1),
    length_mm SMALLINT, -- Araba uzunluğu 32m'yi geçmez
    width_mm SMALLINT, 
    height_mm SMALLINT,
    weight_kg INTEGER, -- Ağırlık INTEGER (Tır olabilir)
    curb_weight_kg INTEGER, 
    trunk_capacity_lt SMALLINT, 
    wheelbase_mm SMALLINT,

    -- Hasar Bilgileri (Sütunlar)
    is_heavy_damaged BOOLEAN, 
    tramer_fee INTEGER, -- Tramer genelde tam sayıdır
    count_changed SMALLINT, -- Değişen sayısı 32.000 olamaz
    count_painted SMALLINT, 
    count_local_painted SMALLINT,
    
    -- 33 Sütunlu Hasar Lokasyonları (Hepsi SMALLINT yapıldı - Çok yer kazandırır)
    tavan_degisen SMALLINT, tavan_boyali SMALLINT, tavan_lokal SMALLINT,
    kaput_degisen SMALLINT, kaput_boyali SMALLINT, kaput_lokal SMALLINT,
    bagaj_degisen SMALLINT, bagaj_boyali SMALLINT, bagaj_lokal SMALLINT,
    door_fl_degisen SMALLINT, door_fl_boyali SMALLINT, door_fl_lokal SMALLINT,
    door_fr_degisen SMALLINT, door_fr_boyali SMALLINT, door_fr_lokal SMALLINT,
    door_rl_degisen SMALLINT, door_rl_boyali SMALLINT, door_rl_lokal SMALLINT,
    door_rr_degisen SMALLINT, door_rr_boyali SMALLINT, door_rr_lokal SMALLINT,
    fender_fl_degisen SMALLINT, fender_fl_boyali SMALLINT, fender_fl_lokal SMALLINT,
    fender_fr_degisen SMALLINT, fender_fr_boyali SMALLINT, fender_fr_lokal SMALLINT,
    fender_rl_degisen SMALLINT, fender_rl_boyali SMALLINT, fender_rl_lokal SMALLINT,
    fender_rr_degisen SMALLINT, fender_rr_boyali SMALLINT, fender_rr_lokal SMALLINT,

    description_text TEXT,
    scraped_at TIMESTAMPTZ DEFAULT NOW(), -- Otomatik kayıt zamanı
    search_date DATE -- Arama yapılan gün (filtreleme için)
);

-- Performans İçin İndeksler (Sorguların Uçması İçin Şart)
CREATE INDEX IF NOT EXISTS idx_listing_date ON test.car_listings(listing_date);
CREATE INDEX IF NOT EXISTS idx_brand_model ON test.car_listings(brand, model);
CREATE INDEX IF NOT EXISTS idx_price ON test.car_listings(price);