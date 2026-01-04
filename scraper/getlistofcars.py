from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import json


def scrape_links_price_blocks(min_price, max_price, step, output_dir, brand):
    os.makedirs(output_dir, exist_ok=True)

    OUTPUT_FILE = f"{output_dir}/links.csv"
    PROGRESS_FILE = f"{output_dir}/progress.json"

    def save_progress(current_min):
        with open(PROGRESS_FILE, "w") as f:
            json.dump(
                {
                    "last_min": current_min,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )

    def load_progress():
        if os.path.exists(PROGRESS_FILE):
            return json.load(open(PROGRESS_FILE))["last_min"]
        return None

    def append_links_to_csv(links):
        write_header = not os.path.exists(OUTPUT_FILE)
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["url"])
            for link in links:
                writer.writerow([link])

    def safe_request(url, retries=3, delay=0.5):
        for _ in range(retries):
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla"}, timeout=10)
                if r.status_code == 200:
                    return r.text
            except:
                time.sleep(delay)
        return None

    last_min = load_progress()
    start = last_min if last_min else min_price

    for block_min in range(start, max_price, step):

        block_max = block_min + step
        print(f"\n🔥 BLOK: {block_min} – {block_max}")

        block_links = []

        base = (
            f"https://www.arabam.com/ikinci-el/otomobil/{brand}"
            f"?currency=TL&fuel=Benzin&fuel=Dizel&fuel=Hibrit&fuel=LPG"
            f"&maxPrice={block_max}&maxkm=700000&minPrice={block_min}"
            "&minYear=2005&page="
        )

        try:
            for page in range(1, 51):
                url = base + str(page)
                print(f"   ▶ Sayfa {page}: {url}")

                html = safe_request(url)
                if html is None:
                    print("   ⚠ Sayfa alınamadı, atlandı.")
                    continue

                soup = BeautifulSoup(html, "html.parser")
                rows = soup.find_all("tr", class_="listing-list-item")
                if not rows:
                    print("   ❗ Bu blokta sayfalar bitti.")
                    break

                for r in rows:
                    a = r.find("a")
                    if a and a.get("href"):
                        block_links.append("https://www.arabam.com" + a["href"])

                time.sleep(0.5)

            append_links_to_csv(block_links)
            print(f"   💾 {len(block_links)} link kaydedildi.")
            save_progress(block_min)

        except Exception as e:
            print(f"❌ Blok hata verdi: {e} — devam ediyor...")
            continue

    print("\n🎉 TÜM BLOKLAR TAMAMLANDI!")
