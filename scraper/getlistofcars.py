import os, time, json, csv, requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("BASE_URL")

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def scrape_links_price_blocks(
    min_price, max_price, step, output_dir, brand, minYear=2005
):
    os.makedirs(output_dir, exist_ok=True)
    links_path = os.path.join(output_dir, "links.csv")
    prog_path = os.path.join(output_dir, "progress.json")

    # Kaldığı yeri yükle
    start = min_price
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            start = json.load(f).get("last_min", min_price)

    for b_min in range(start, max_price, step):
        b_max = b_min + step
        print(f"\n🔥 BLOK: {b_min} – {b_max}")

        block_links = []
        base = f"{base_url}/ikinci-el/otomobil/{brand}?currency=TL&fuel=Benzin&fuel=Dizel&fuel=Hibrit&fuel=LPG&maxPrice={b_max}&maxkm=700000&minPrice={b_min}&minYear={minYear}&page="

        for page in range(1, 51):
            try:
                print(f"   ▶ Sayfa {page}: {base}{page}")
                r = session.get(base + str(page), timeout=10)
                if r.status_code != 200:
                    break

                soup = BeautifulSoup(r.text, "lxml")  # 'lxml' daha hızlıdır
                rows = soup.find_all("tr", class_="listing-list-item")

                if not rows:
                    print("   ❗ Blok bitti.")
                    break

                for row in rows:
                    if a := row.find("a", href=True):
                        block_links.append(base_url + a["href"])

                # time.sleep(0.4)  # Bot koruması için kısa mola
            except Exception as e:
                print(f"   ⚠ Hata: {e}")
                continue

        # Linkleri Kaydet (CSV Append)
        write_header = not os.path.exists(links_path)
        with open(links_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["url"])
            for link in set(block_links):
                writer.writerow([link])  # set() ile mükerrerleri engelle

        # İlerlemeyi Kaydet
        with open(prog_path, "w") as f:
            json.dump(
                {
                    "last_min": b_min,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )

        print(f"   💾 {len(block_links)} link eklendi.")

    print("\n🎉 TÜM BLOKLAR TAMAMLANDI!")
