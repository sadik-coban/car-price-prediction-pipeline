def get_title(soup):
    div = soup.find("div", class_="product-name-container")
    return div.text.strip() if div else None


def get_agir_hasar_info(soup):
    """
    Arabam.com ilanında 'Bu araç ağır hasar kayıtlıdır.' ifadesi varsa True döner.
    Tramer-info içinde <p> olarak yer alır.
    """
    tramer_info = soup.find("div", class_="tramer-info")
    if not tramer_info:
        return False

    p = tramer_info.find("p")
    if not p:
        return False

    text = p.get_text(strip=True).lower()
    return "ağır hasar" in text or "agir hasar" in text


def get_location(soup):
    """
    product-location alanını güvenli şekilde yakalar.
    """
    loc = soup.find(attrs={"class": "product-location"})
    if not loc:
        return None

    spans = loc.find_all("span")
    if spans:
        return spans[-1].text.strip()

    return loc.get_text(strip=True)


def get_damage_and_tramer(soup):
    result = {
        "Tramer_Tutari": None,
        "Degisen_Parca_Sayisi": 0,
        "Boyali_Parca_Sayisi": 0,
        "Lokal_Boyali_Parca_Sayisi": 0,
        "Hasar_Listesi": [],
    }

    # ---> TRAMER TUTARI
    tramer_div = soup.find("div", class_="tramer-info")
    if tramer_div:
        val_span = tramer_div.find("span", class_="property-value")
        if val_span:
            raw = val_span.text.replace("TL", "").replace(".", "").strip()
            if raw.isdigit():
                result["Tramer_Tutari"] = int(raw)

    # ---> BOYA / DEĞİŞEN
    damage_container = soup.find("div", class_="car-damage-info")
    if damage_container:
        items = damage_container.find_all("div", class_="car-damage-info-item")

        for item in items:
            category = item.find("p").text.strip() if item.find("p") else ""
            ul_tag = item.find("ul")
            parts = (
                [
                    li.text.strip()
                    for li in ul_tag.find_all("li")
                    if li.text.strip() != "-"
                ]
                if ul_tag
                else []
            )

            if "Değişmiş" in category:
                result["Degisen_Parca_Sayisi"] = len(parts)
                for part in parts:
                    result["Hasar_Listesi"].append(f"{part}: Değişen")

            elif "Lokal" in category:
                result["Lokal_Boyali_Parca_Sayisi"] = len(parts)
                for part in parts:
                    result["Hasar_Listesi"].append(f"{part}: Lokal Boya")

            elif "Boyalı" in category:
                result["Boyali_Parca_Sayisi"] = len(parts)
                for part in parts:
                    result["Hasar_Listesi"].append(f"{part}: Boyalı")

    return result


def get_car_details(url):
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser")

    data = {}

    # =============================
    # 1) FİYAT
    # =============================
    price = soup.find("div", class_="desktop-information-price")
    data["Fiyat"] = price.text.strip() if price else None

    # =============================
    # 2) BAŞLIK
    # =============================
    title_div = soup.find("div", class_="product-name-container")
    data["Ilan_Basligi"] = title_div.text.strip() if title_div else None

    # =============================
    # 3) KONUM
    # =============================
    data["Konum"] = get_location(soup)

    # =============================
    # 5) EIDS
    # =============================
    eids_box = soup.find("div", class_="eids-vehicle-model")
    if eids_box:
        model_tag = eids_box.find("span")
        data["EIDS_Model"] = model_tag.text.strip() if model_tag else None
    else:
        data["EIDS_Model"] = None

    # =============================
    # 6) PROPERTY-ITEM (Kısa özet)
    # =============================
    overview = soup.find("div", class_="product-properties-details")
    if overview:
        items = overview.find_all("div", class_="property-item")
        for item in items:
            key = item.find("div", class_="property-key").text.strip()
            val = item.find("div", class_="property-value").text.strip()
            data[f"KısaBilgi - {key}"] = val

    # =============================
    # 7) TEKNİK DETAYLAR
    # =============================
    info_tab = soup.find("div", id="tab-car-information")
    if info_tab:
        containers = info_tab.find_all(
            "div", class_="tab-content-car-information-container"
        )
        for c in containers:
            section = c.find("h3").text.strip()
            for li in c.find_all("li"):
                key = li.find("span", class_="property-key").text.strip()
                val = li.find("span", class_="property-value").text.strip()
                data[f"{section} - {key}"] = val

    # =============================
    # 8) AÇIKLAMA HTML
    # =============================
    desc = soup.find("div", id="tab-description")
    data["Aciklama_HTML"] = desc.decode_contents() if desc else None

    # =============================
    # 9) HASAR / TRAMER
    # =============================
    data["Agir_Hasar"] = get_agir_hasar_info(soup)
    data.update(get_damage_and_tramer(soup))

    return data
