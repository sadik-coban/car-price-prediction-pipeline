import os, json, time, random, jsonlines
import pandas as pd
from datetime import datetime, timezone
from getdetails import get_car_details
from getlistofcars import scrape_links_price_blocks

# --- [1] KONTROL PANELİ / YAPILANDIRMA ---
# Her marka için özel fiyat aralıkları. Listede olup burada olmayanlar için DEFAULT kullanılır.
BRAND_CONFIGS = {
    "audi": {"min": 300000, "max": 6500000, "step": 100000},
    "bmw": {"min": 300000, "max": 6500000, "step": 100000},
    "mercedes": {"min": 800000, "max": 5000000, "step": 300000},
    "fiat": {"min": 200000, "max": 900000, "step": 100000},
    "ford": {"min": 300000, "max": 1500000, "step": 150000},
    "renault": {"min": 300000, "max": 1200000, "step": 100000},
    "volkswagen": {"min": 400000, "max": 2000000, "step": 150000},
    "DEFAULT": {"min": 200000, "max": 1000000, "step": 200000},
}

SELECTED_BRANDS = [
    "audi",
    "bmw",
]  # , "mercedes", "volkswagen", "fiat", "ford", "renault"

# Hız Ayarı
SLEEP_RANGE = (1.5, 3.0)


def run_pipeline():
    # Bugünün tarihini al (Klasörleme için)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M")
    root_dir = "data"

    for brand in SELECTED_BRANDS:
        # Markaya özel konfigürasyonu seç
        cfg = BRAND_CONFIGS.get(brand, BRAND_CONFIGS["DEFAULT"])

        # Klasör Yapısı: data / marka / yyyy-mm-dd
        brand_date_dir = f"{root_dir}/{brand}/{today}"
        os.makedirs(brand_date_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"🚗 MARKA: {brand.upper()} | TARİH: {today}")
        print(f"💰 ARALIK: {cfg['min']} - {cfg['max']} TL")
        print(f"{'='*50}")

        links_file = f"{brand_date_dir}/links.csv"
        data_file = f"{brand_date_dir}/details.jsonl"
        progress_file = f"{brand_date_dir}/progress.json"

        # --- [2] LİNK TOPLAMA ---
        if not os.path.exists(links_file):
            print(f"🔗 Linkler toplanıyor: {brand}")
            scrape_links_price_blocks(
                cfg["min"], cfg["max"], cfg["step"], brand_date_dir, brand
            )

        # Linkleri oku ve tekilleştir
        if os.path.exists(links_file):
            df_links = pd.read_csv(links_file)
            unique_links = df_links["url"].drop_duplicates().tolist()
            print(f"💎 Tekil Link: {len(unique_links)}")
        else:
            print(f"⚠ {brand} için link bulunamadı, geçiliyor.")
            continue

        # --- [3] İLERLEME KONTROLÜ ---
        start_i = 0
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                start_i = json.load(f).get("index", 0)

        if start_i >= len(unique_links):
            print(f"✅ {brand} ({today}) verisi zaten tamamlanmış.")
            continue

        # --- [4] DETAY ÇEKME ---
        print(f"🚀 {start_i}. indexten devam ediliyor...")

        with jsonlines.open(data_file, mode="a") as writer:
            for i in range(start_i, len(unique_links)):
                url = unique_links[i]
                print(f"[{brand.upper()} - {today}] {i+1}/{len(unique_links)}")

                try:
                    car_data = get_car_details(url)
                    car_data.update(
                        {
                            "brand": brand,
                            "url": url,
                            "scraped_at": datetime.now(timezone.utc).isoformat(),
                            "search_date": today,  # Analiz yaparken filtrelemek için
                        }
                    )

                    writer.write(car_data)

                    with open(progress_file, "w") as f:
                        json.dump({"index": i + 1, "last_url": url}, f)

                except Exception as e:
                    print(f"❌ Hata: {url} -> {e}")
                    writer.write(
                        {"url": url, "brand": brand, "error": str(e), "date": today}
                    )

                # time.sleep(random.uniform(*SLEEP_RANGE))

        print(f"✨ {brand.upper()} bitti. Markalar arası kısa mola...")
        time.sleep(1)


if __name__ == "__main__":
    run_pipeline()
