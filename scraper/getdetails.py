import requests
from bs4 import BeautifulSoup

# Global session: Tüm istekler bu üzerinden geçecek (Hız ve Güvenlik için)
session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
)


def get_agir_hasar_info(soup):
    tramer_info = soup.find("div", class_="tramer-info")
    if tramer_info and (p := tramer_info.find("p")):
        text = p.get_text(strip=True).lower()
        return "ağır hasar" in text or "agir hasar" in text
    return False


def get_location(soup):
    loc = soup.find(attrs={"class": "product-location"})
    if loc:
        spans = loc.find_all("span")
        return spans[-1].text.strip() if spans else loc.get_text(strip=True)
    return None


def get_damage_and_tramer(soup):
    result = {
        "Tramer_Tutari": None,
        "Degisen_Parca_Sayisi": 0,
        "Boyali_Parca_Sayisi": 0,
        "Lokal_Boyali_Parca_Sayisi": 0,
        "Hasar_Listesi": [],
    }

    # Tramer Tutarı
    tramer_div = soup.find("div", class_="tramer-info")
    if tramer_div and (val_span := tramer_div.find("span", class_="property-value")):
        raw = "".join(filter(str.isdigit, val_span.text))
        if raw:
            result["Tramer_Tutari"] = int(raw)

    # Boya / Değişen Bilgisi
    if container := soup.find("div", class_="car-damage-info"):
        for item in container.find_all("div", class_="car-damage-info-item"):
            category = item.p.text.strip() if item.p else ""
            parts = [
                li.text.strip() for li in item.find_all("li") if li.text.strip() != "-"
            ]

            if "Değişmiş" in category:
                result["Degisen_Parca_Sayisi"] = len(parts)
            elif "Lokal" in category:
                result["Lokal_Boyali_Parca_Sayisi"] = len(parts)
            elif "Boyalı" in category:
                result["Boyali_Parca_Sayisi"] = len(parts)

            result["Hasar_Listesi"].extend([f"{p}: {category}" for p in parts])
    return result


def get_car_details(url):
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "lxml")  # lxml daha hızlıdır
    except Exception as e:
        return {"url": url, "error": str(e)}

    data = {}

    # 1) Temel Bilgiler
    data["Fiyat"] = getattr(
        soup.select_one(".desktop-information-price"), "text", ""
    ).strip()
    data["Ilan_Basligi"] = getattr(
        soup.select_one(".product-name-container"), "text", ""
    ).strip()
    data["Konum"] = get_location(soup)

    # 2) EIDS Model
    eids_box = soup.find("div", class_="eids-vehicle-model")
    data["EIDS_Model"] = (
        eids_box.span.text.strip() if (eids_box and eids_box.span) else None
    )

    # 3) Kısa Özet (KısaBilgi section)
    if overview := soup.find("div", class_="product-properties-details"):
        for item in overview.find_all("div", class_="property-item"):
            key = item.find("div", class_="property-key").text.strip()
            val = item.find("div", class_="property-value").text.strip()
            data[f"KısaBilgi - {key}"] = val

    # 4) Teknik Detaylar (Section bazlı)
    if info_tab := soup.find("div", id="tab-car-information"):
        for c in info_tab.find_all(
            "div", class_="tab-content-car-information-container"
        ):
            section_name = c.find("h3").text.strip()
            for li in c.find_all("li"):
                k = li.find("span", class_="property-key").text.strip()
                v = li.find("span", class_="property-value").text.strip()
                data[f"{section_name} - {k}"] = v

    # 5) Açıklama ve Hasar
    desc = soup.find("div", id="tab-description")
    data["Aciklama_HTML"] = desc.decode_contents() if desc else None
    data["Agir_Hasar"] = get_agir_hasar_info(soup)
    data.update(get_damage_and_tramer(soup))

    return data
