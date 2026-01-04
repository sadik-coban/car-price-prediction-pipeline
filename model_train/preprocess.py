from bs4 import BeautifulSoup
import pandas as pd
from difflib import SequenceMatcher
import numpy as np

# df = pd.read_json("arabam_details_bmw.jsonl", lines=True)


def preprocess(df):
    df = df.dropna(subset=["Fiyat"])
    pVdf = df[df["KısaBilgi - Yakıt Tipi"] != "Elektrik"].copy()

    empty_colspV = pVdf.columns[pVdf.isna().all()]
    pVdf = pVdf.drop(columns=empty_colspV)

    def na_relationship(df, col1, col2):
        a = df[col1]
        b = df[col2]

        # Kolon2 NaN olup Kolon1 DOLU olanlar (problemli kısım)
        col2_nan_col1_full = ((b.isna()) & (a.notna())).sum()

        # Kolon2 NaN olup Kolon1 de NaN olanlar (ortak boşluk)
        col2_nan_col1_nan = ((b.isna()) & (a.isna())).sum()

        # Kolon1 NaN olup Kolon2 dolu olanlar (ters problem)
        col1_nan_col2_full = ((a.isna()) & (b.notna())).sum()

        print("==== ASİMETRİK NaN ANALİZİ ====")
        print(f"Kolon1: {col1}")
        print(f"Kolon2: {col2}")
        print("-----------------------------------")
        print(f"{col2} NaN & {col1} FULL : {col2_nan_col1_full}")
        print(f"{col2} NaN & {col1} NaN  : {col2_nan_col1_nan}")
        print(f"{col1} NaN & {col2} FULL : {col1_nan_col2_full}")
        print("===================================")

        return {
            "Col2_Na_Col1_Full": col2_nan_col1_full,
            "Col2_Na_Col1_Na": col2_nan_col1_nan,
            "Col1_Na_Col2_Full": col1_nan_col2_full,
        }

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    results = []

    cols = pVdf.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            c1, c2 = cols[i], cols[j]

            # 1) Kolon adı benzerliği
            name_sim = round(similar(c1, c2) * 100, 2)

            # 2) Veri tipi karşılaştırma
            dtype1 = str(pVdf[c1].dtype)
            dtype2 = str(pVdf[c2].dtype)

            # 3) NaN oranları
            nan1 = round(pVdf[c1].isna().mean() * 100, 2)
            nan2 = round(pVdf[c2].isna().mean() * 100, 2)

            # 4) Karşılaştırılabilir veri maskesi (NaN olmayanlar)
            mask = pVdf[c1].notna() & pVdf[c2].notna()
            count_valid = mask.sum()

            # 5) Sadece geçerli satırlarda eşleşme oranı
            if count_valid > 0:
                equal_ratio = (pVdf.loc[mask, c1] == pVdf.loc[mask, c2]).mean() * 100
                equal_ratio = round(equal_ratio, 2)
                status = "OK"
            else:
                equal_ratio = None
                status = "NO_DATA"

            results.append(
                {
                    "Kolon1": c1,
                    "Kolon2": c2,
                    "AdBenzerligi(%)": name_sim,
                    "Tip1": dtype1,
                    "Tip2": dtype2,
                    "NaN(1)%": nan1,
                    "NaN(2)%": nan2,
                    "KarsilastirilabilirKayit": count_valid,
                    "DegerEslesmeOrani(%)": equal_ratio,
                    "Durum": status,
                }
            )

    compare_all = pd.DataFrame(results)

    # Daha okunur sıralama
    compare_all = compare_all.sort_values(["DegerEslesmeOrani(%)"], ascending=False)

    canDeleted = compare_all[compare_all["DegerEslesmeOrani(%)"] == 100]
    canDeleted

    for idx, row in canDeleted.iterrows():
        na_relationship(pVdf, row["Kolon1"], row["Kolon2"])

    pVdf = pVdf.drop(columns=canDeleted["Kolon2"].tolist()).copy()

    pVdf["KısaBilgi - Ağır Hasarlı"] = pVdf["KısaBilgi - Ağır Hasarlı"].fillna("Hayır")

    pVdf = pVdf.drop(columns=["KısaBilgi - Ağır Hasarlı"])

    pVdf["Fiyat"] = (
        pVdf["Fiyat"]
        .str.replace("TL", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    pVdf["KısaBilgi - Kilometre"] = (
        pVdf["KısaBilgi - Kilometre"]
        .str.replace("km", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    numbers = (
        pVdf["KısaBilgi - Motor Hacmi"].astype(str).str.extractall(r"(\d+)").unstack()
    )
    pVdf["KısaBilgi - Motor Hacmi"] = numbers.astype(float).mean(axis=1)

    numbers = (
        pVdf["KısaBilgi - Motor Gücü"].astype(str).str.extractall(r"(\d+)").unstack()
    )
    pVdf["KısaBilgi - Motor Gücü"] = numbers.astype(float).mean(axis=1)

    pVdf["KısaBilgi - Yakıt Deposu"] = (
        pVdf["KısaBilgi - Yakıt Deposu"]
        .str.replace("lt", "", regex=False)
        .astype(float)
    )

    pVdf["Genel Bakış - Yıllık MTV"] = (
        pVdf["Genel Bakış - Yıllık MTV"]
        .str.replace("TL", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    pVdf["Motor ve Performans - Tork"] = (
        pVdf["Motor ve Performans - Tork"]
        .str.replace("nm", "", regex=False)
        .astype(float)
    )

    pVdf["Motor ve Performans - Maksimum Güç"] = (
        pVdf["Motor ve Performans - Maksimum Güç"]
        .str.replace("rpm", "", regex=False)
        .astype(float)
    )
    pVdf["Motor ve Performans - Minimum Güç"] = (
        pVdf["Motor ve Performans - Minimum Güç"]
        .str.replace("rpm", "", regex=False)
        .astype(float)
    )

    pVdf["Motor ve Performans - Maksimum Hız"] = (
        pVdf["Motor ve Performans - Maksimum Hız"]
        .str.replace("km/s", "", regex=False)
        .astype(float)
    )

    pVdf["Boyut ve Kapasite - Uzunluk"] = (
        pVdf["Boyut ve Kapasite - Uzunluk"]
        .str.replace("mm", "", regex=False)
        .astype(float)
    )
    pVdf["Boyut ve Kapasite - Genişlik"] = (
        pVdf["Boyut ve Kapasite - Genişlik"]
        .str.replace("mm", "", regex=False)
        .astype(float)
    )
    pVdf["Boyut ve Kapasite - Yükseklik"] = (
        pVdf["Boyut ve Kapasite - Yükseklik"]
        .str.replace("mm", "", regex=False)
        .astype(float)
    )

    pVdf["Boyut ve Kapasite - Boş Ağırlığı"] = (
        pVdf["Boyut ve Kapasite - Boş Ağırlığı"]
        .str.replace("kg", "", regex=False)
        .astype(float)
    )

    pVdf["Boyut ve Kapasite - Aks Aralığı"] = (
        pVdf["Boyut ve Kapasite - Aks Aralığı"]
        .str.replace("mm", "", regex=False)
        .astype(float)
    )

    pVdf["KısaBilgi - Ort. Yakıt Tüketimi"] = (
        pVdf["KısaBilgi - Ort. Yakıt Tüketimi"]
        .str.replace("lt", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    pVdf["Genel Bakış - Ortalama Trafik Sigortası"] = (
        pVdf["Genel Bakış - Ortalama Trafik Sigortası"]
        .str.replace("TL", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )
    pVdf["Motor ve Performans - Hızlanma (0-100)"] = (
        pVdf["Motor ve Performans - Hızlanma (0-100)"]
        .str.replace("sn", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    pVdf["Boyut ve Kapasite - Ağırlık"] = (
        pVdf["Boyut ve Kapasite - Ağırlık"]
        .str.replace("kg", "", regex=False)
        .astype(float)
    )
    pVdf["Boyut ve Kapasite - Bagaj Hacmi"] = (
        pVdf["Boyut ve Kapasite - Bagaj Hacmi"]
        .str.replace("lt", "", regex=False)
        .astype(float)
    )
    pVdf["Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi"] = (
        pVdf["Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi"]
        .str.replace("lt", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    pVdf["Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi"] = (
        pVdf["Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi"]
        .str.replace("lt", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    pVdf["Genel Bakış - Ortalama Kasko"] = (
        pVdf["Genel Bakış - Ortalama Kasko"]
        .str.replace("TL", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    pVdf["KısaBilgi - İlan No"] = (
        pVdf["KısaBilgi - İlan No"].str.extract(r"(\d+)").astype(int)
    )

    pVdf.groupby("KısaBilgi - Kasa Tipi")["Fiyat"].agg(
        count="count", mean="mean", median="median", min="min", max="max"
    ).sort_values("median", ascending=False)

    pVdf.loc[:, "is_body_type_filled"] = pVdf["KısaBilgi - Kasa Tipi"] == "-"

    def detect_body_type_cabrio_priority(row):

        current = str(row["KısaBilgi - Kasa Tipi"])
        if current != "-" and current != "nan":
            return current
        text = (
            str(row["Ilan_Basligi"])
            + " "
            + str(row["KısaBilgi - Marka"])
            + " "
            + str(row["KısaBilgi - Model"])
        ).lower()

        if (
            "cabrio" in text
            or "kabrio" in text
            or "roadster" in text
            or "spider" in text
            or "üstü açık" in text
        ):
            return "Cabrio"

        if "coupe" in text or "tek kapı" in text:
            return "Coupe"
        if "station" in text or "sw" in text or "touring" in text:
            return "Station wagon"
        if "suv" in text or "jeep" in text or "4x4" in text:
            return "SUV"
        hb_list = [
            "clio",
            "golf",
            "polo",
            "fiesta",
            "corsa",
            "i20",
            "leon",
            "ibiza",
            "cooper",
            "auris",
            "yaris",
            "micra",
        ]
        if (
            any(model in text for model in hb_list)
            or "hatchback" in text
            or "hb" in text
        ):
            return "Hatchback"

        if "sedan" in text:
            return "Sedan"

        return "Sedan"

    mask = pVdf["KısaBilgi - Kasa Tipi"] == "-"
    pVdf.loc[mask, "KısaBilgi - Kasa Tipi"] = pVdf[mask].apply(
        detect_body_type_cabrio_priority, axis=1
    )

    def html_to_text(html):
        if pd.isna(html):
            return ""
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()

    pVdf["Tramer_Tutari"] = pVdf["Tramer_Tutari"].fillna(0)

    pVdf["Aciklama_Text"] = pVdf["Aciklama_HTML"].apply(html_to_text)

    pVdf.drop(columns=["Aciklama_HTML"], inplace=True)

    #####

    # --- DETAYLI HASAR ANALİZİ (Lokal Boya Ayrımı Dahil) ---
    def profesyonel_hasar_analizi_lokal(row):
        text = str(row["Hasar_Listesi"]).lower()

        if text == "nan" or text == "[]" or text == "":
            return pd.Series([0] * 17)  # Returning 17 features now

        # 1. HAYATİ ORGANLAR (Binary: Değişen / Tam Boya / Lokal Boya)

        # TAVAN
        tavan_degisen = 1 if "tavan: değişen" in text else 0
        tavan_lokal = 1 if ("tavan: lokal" in text) and tavan_degisen == 0 else 0
        tavan_tam_boya = (
            1
            if ("tavan: boyalı" in text) and tavan_degisen == 0 and tavan_lokal == 0
            else 0
        )

        # KAPUT
        kaput_degisen = (
            1 if "kaput: değişen" in text or "motor kaputu: değişen" in text else 0
        )
        kaput_lokal = (
            1
            if ("kaput: lokal" in text or "motor kaputu: lokal" in text)
            and kaput_degisen == 0
            else 0
        )
        kaput_tam_boya = (
            1
            if ("kaput: boyalı" in text or "motor kaputu: boyalı" in text)
            and kaput_degisen == 0
            and kaput_lokal == 0
            else 0
        )

        # BAGAJ
        bagaj_degisen = (
            1 if "bagaj: değişen" in text or "arka kaput: değişen" in text else 0
        )
        bagaj_lokal = (
            1
            if ("bagaj: lokal" in text or "arka kaput: lokal" in text)
            and bagaj_degisen == 0
            else 0
        )
        bagaj_tam_boya = (
            1
            if ("bagaj: boyalı" in text or "arka kaput: boyalı" in text)
            and bagaj_degisen == 0
            and bagaj_lokal == 0
            else 0
        )

        # 2. YAN PARÇALAR (Toplam Adet Sayısı - Lokal Ayrımı)

        # KAPILAR
        kapi_degisen_sayisi = text.count("kapı: değişen")
        kapi_lokal_sayisi = text.count("kapı: lokal")
        # "kapı: boyalı" ifadesi bazen lokal'i de kapsayabilir mi?
        # String "Sağ Kapı: Lokal Boya" içeriyorsa "Kapı: Boyalı" count'u artar mı?
        # "Kapı: Boyalı" stringi "Kapı: Lokal Boya" stringinin içinde GEÇMEZ (arada Lokal var).
        # Bu yüzden ayrı ayrı saymak güvenlidir.
        kapi_tam_boya_sayisi = text.count("kapı: boyalı")

        # ÇAMURLUKLAR
        camurluk_degisen_sayisi = text.count("çamurluk: değişen")
        camurluk_lokal_sayisi = text.count("çamurluk: lokal")
        camurluk_tam_boya_sayisi = text.count("çamurluk: boyalı")

        # TAMPONLAR
        tampon_islem_sayisi = text.count("tampon: değişen") + text.count(
            "tampon: boyalı"
        )

        # 3. GENEL SKOR
        toplam_degisen = (
            tavan_degisen
            + kaput_degisen
            + bagaj_degisen
            + kapi_degisen_sayisi
            + camurluk_degisen_sayisi
        )

        return pd.Series(
            [
                tavan_degisen,
                tavan_tam_boya,
                tavan_lokal,
                kaput_degisen,
                kaput_tam_boya,
                kaput_lokal,
                bagaj_degisen,
                bagaj_tam_boya,
                bagaj_lokal,
                kapi_degisen_sayisi,
                kapi_tam_boya_sayisi,
                kapi_lokal_sayisi,
                camurluk_degisen_sayisi,
                camurluk_tam_boya_sayisi,
                camurluk_lokal_sayisi,
                tampon_islem_sayisi,
                toplam_degisen,
            ]
        )

    cols_lokal = [
        "Tavan_Degisen",
        "Tavan_Tam_Boyali",
        "Tavan_Lokal_Boyali",
        "Kaput_Degisen",
        "Kaput_Tam_Boyali",
        "Kaput_Lokal_Boyali",
        "Bagaj_Degisen",
        "Bagaj_Tam_Boyali",
        "Bagaj_Lokal_Boyali",
        "Kapi_Degisen_Adet",
        "Kapi_Tam_Boyali_Adet",
        "Kapi_Lokal_Boyali_Adet",
        "Camurluk_Degisen_Adet",
        "Camurluk_Tam_Boyali_Adet",
        "Camurluk_Lokal_Boyali_Adet",
        "Tampon_Islem_Adet",
        "Genel_Toplam_Degisen",
    ]

    # Uygula
    pVdf[cols_lokal] = pVdf.apply(profesyonel_hasar_analizi_lokal, axis=1)

    # Kontrol: Lokal Boyalı Araç Örnekleri
    # Hem Lokal Boya hem Tam Boya içeren bir örnek bulalım
    mask_mix = (pVdf["Kapi_Lokal_Boyali_Adet"] > 0) & (pVdf["Kapi_Tam_Boyali_Adet"] > 0)
    print("Hem Lokal Hem Tam Boyalı Kapı İçeren Örnek:")
    print(
        pVdf[mask_mix][
            ["Hasar_Listesi", "Kapi_Lokal_Boyali_Adet", "Kapi_Tam_Boyali_Adet"]
        ].head(3)
    )

    # Lokal boya istatistikleri
    print("\n--- Lokal Boya İstatistikleri ---")
    print(f"Lokal Boyalı Kapı Toplamı: {pVdf['Kapi_Lokal_Boyali_Adet'].sum()}")
    print(f"Tam Boyalı Kapı Toplamı: {pVdf['Kapi_Tam_Boyali_Adet'].sum()}")
    print(f"Tavanı Lokal Boyalı Araç Sayısı: {pVdf['Tavan_Lokal_Boyali'].sum()}")

    pVdf["Agir_Hasar"] = pVdf["Agir_Hasar"].astype(int)

    pVdf["Genel Bakış - Garanti Durumu"] = pVdf["Genel Bakış - Garanti Durumu"].replace(
        "-", np.nan
    )

    pVdf["KısaBilgi - Takasa Uygun"] = pVdf["KısaBilgi - Takasa Uygun"].replace(
        "-", np.nan
    )

    # 1. Her Marka_Seri grubu için 'En Sık Geçen Segmenti' (Mode) bul
    segment_map = (
        pVdf.groupby(["KısaBilgi - Marka", "KısaBilgi - Seri"])["Genel Bakış - Sınıfı"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .to_dict()
    )

    # 2. Fonksiyon: Haritadan bakıp doldur
    def impute_segment(row):
        if pd.notnull(row["Genel Bakış - Sınıfı"]):
            return row["Genel Bakış - Sınıfı"]

        key = (row["KısaBilgi - Marka"], row["KısaBilgi - Seri"])
        return segment_map.get(key, np.nan)

    # 3. Uygula
    pVdf["Segment_Clean"] = pVdf.apply(impute_segment, axis=1)

    # 4. Hala boş kalanlar (Hiç verisi olmayan eski arabalar)
    # Tofaş Şahin vb. genelde C segmentidir (Dönemine göre) veya B.
    # Basitçe 'C Segment' (En yaygın) atayabiliriz veya boş bırakıp 'Bilinmiyor' diyebiliriz.
    pVdf["Segment_Clean"] = pVdf["Segment_Clean"].fillna("C Segment")

    # 5. Label Encoding (Ordinal)
    # Segmentler arasında büyüklük ilişkisi vardır (A < B < C < D < E < F)
    seg_map = {
        "A Segment": 1,
        "B Segment": 2,
        "C Segment": 3,
        "D Segment": 4,
        "E Segment": 5,
        "F Segment": 6,
        "G Segment": 7,
        "H Segment": 8,
        "K Segment": 9,
    }
    pVdf["Segment_Encoded"] = (
        pVdf["Segment_Clean"].map(seg_map).fillna(3)
    )  # Bulamazsa C (3)

    # print("Segmentler dolduruldu ve sayısallaştırıldı.")
    print(pVdf["Segment_Clean"].value_counts())

    pVdf.drop(columns=["Motor ve Performans - Çekiş"], inplace=True)

    ################### ML preprocess

    mlpvDf = pVdf[
        [
            "Fiyat",
            "Segment_Encoded",
            "Segment_Clean",
            #  "KısaBilgi - Marka"
            "KısaBilgi - Seri",
            "KısaBilgi - Yıl",
            "KısaBilgi - Kilometre",
            "KısaBilgi - Vites Tipi",
            "KısaBilgi - Yakıt Tipi",
            "KısaBilgi - Kasa Tipi",
            "KısaBilgi - Motor Hacmi",
            "KısaBilgi - Motor Gücü",
            "KısaBilgi - Çekiş",
            "KısaBilgi - Yakıt Deposu",
            "Genel Bakış - Yıllık MTV",
            "Motor ve Performans - Silindir Sayısı",
            "Motor ve Performans - Tork",
            "Motor ve Performans - Maksimum Güç",
            "Motor ve Performans - Minimum Güç",
            "Motor ve Performans - Maksimum Hız",
            "Boyut ve Kapasite - Uzunluk",
            "Boyut ve Kapasite - Boş Ağırlığı",
            "Boyut ve Kapasite - Koltuk Sayısı",
            "Agir_Hasar",
            "Tramer_Tutari",
            "Genel Bakış - Garanti Durumu",
            "Degisen_Parca_Sayisi",
            "Boyali_Parca_Sayisi",
            "Lokal_Boyali_Parca_Sayisi",
            "KısaBilgi - Ort. Yakıt Tüketimi",
            "Genel Bakış - Ortalama Trafik Sigortası",
            "Motor ve Performans - Hızlanma (0-100)",
            "Boyut ve Kapasite - Bagaj Hacmi",
            "Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi",
            "Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi",
            "Tavan_Degisen",
            "Tavan_Tam_Boyali",
            "Tavan_Lokal_Boyali",
            "Kaput_Degisen",
            "Kaput_Tam_Boyali",
            "Kaput_Lokal_Boyali",
            "Bagaj_Degisen",
            "Bagaj_Tam_Boyali",
            "Bagaj_Lokal_Boyali",
            "Kapi_Degisen_Adet",
            "Kapi_Tam_Boyali_Adet",
            "Kapi_Lokal_Boyali_Adet",
            "Camurluk_Degisen_Adet",
            "Camurluk_Tam_Boyali_Adet",
            "Camurluk_Lokal_Boyali_Adet",
            "Tampon_Islem_Adet",
            "Genel_Toplam_Degisen",
            "KısaBilgi - Takasa Uygun",
            "is_body_type_filled",
            "Aciklama_Text",
        ]
    ].copy()

    bool_cols = mlpvDf.select_dtypes(include="bool").columns
    for col in bool_cols:
        mlpvDf[col] = mlpvDf[col].astype(int)

    def ultimate_ekspertiz_skoru_ozel(row):
        skor = 0

        # ---------------------------------------------------------
        # 1. KIRMIZI ÇİZGİLER (Tavan ve Kaput - Yapısal Riskler)
        # ---------------------------------------------------------
        # Tavan değişimi en büyük risktir (Takla/Ağır Kaza)
        if row["Tavan_Degisen"] == 1:
            skor += 150
        elif row["Tavan_Tam_Boyali"] == 1:
            skor += 75  # Tavan boyası şüphe çeker
        elif row["Tavan_Lokal_Boyali"] == 1:
            skor += 40  # Lokal bile olsa tavanda sevilmez

        # Kaput (Önden Kaza Riski)
        if row["Kaput_Degisen"] == 1:
            skor += 60
        elif row["Kaput_Tam_Boyali"] == 1:
            skor += 30
        elif row["Kaput_Lokal_Boyali"] == 1:
            skor += 15

        # ---------------------------------------------------------
        # 2. ARKA VE YAN HASARLAR (Değer Kaybı)
        # ---------------------------------------------------------
        # Bagaj (Arkadan Çarpma)
        if row["Bagaj_Degisen"] == 1:
            skor += 40
        elif row["Bagaj_Tam_Boyali"] == 1:
            skor += 20
        elif row["Bagaj_Lokal_Boyali"] == 1:
            skor += 10

        # Yan Parçalar (Kapı - Adet Bazlı)
        skor += row["Kapi_Degisen_Adet"] * 10  # Değişen kapı
        skor += row["Kapi_Tam_Boyali_Adet"] * 5  # Tam boyalı kapı
        skor += row["Kapi_Lokal_Boyali_Adet"] * 2  # Lokal boyalı kapı (Çok düşük etki)

        # Yan Parçalar (Çamurluk - Adet Bazlı)
        skor += row["Camurluk_Degisen_Adet"] * 8
        skor += row["Camurluk_Tam_Boyali_Adet"] * 4
        skor += row["Camurluk_Lokal_Boyali_Adet"] * 2

        # ---------------------------------------------------------
        # 3. KOZMETİK (Tampon)
        # ---------------------------------------------------------
        # Plastik aksamdır, fiyata etkisi çok azdır.
        skor += row["Tampon_Islem_Adet"] * 1

        return skor

    # 1. Yeni Skoru Oluştur
    print("🧮 Veri setine özel Ekspertiz Skoru hesaplanıyor...")
    # Dataframe adını kendi koduna göre güncelle (mlpvDf veya mlpvDf_final)
    mlpvDf["Ekspertiz_Risk_Skoru"] = mlpvDf.apply(ultimate_ekspertiz_skoru_ozel, axis=1)

    # 2. TEMİZLİK LİSTESİ (Drop List)
    # Artık görevini tamamlayan ve skorun içinde eriyen o kalabalık sütunları atıyoruz.
    drop_list_final = [
        # Toplam Sayılar (Modelin kafasını karıştırır, skor daha iyidir)
        "Degisen_Parca_Sayisi",
        "Boyali_Parca_Sayisi",
        "Lokal_Boyali_Parca_Sayisi",
        "Genel_Toplam_Degisen",
        # Detay Boya/Değişen Sütunları (Risk Skoru bunları kapsadı)
        "Tavan_Degisen",
        "Tavan_Tam_Boyali",
        "Tavan_Lokal_Boyali",
        "Kaput_Degisen",
        "Kaput_Tam_Boyali",
        "Kaput_Lokal_Boyali",
        "Bagaj_Degisen",
        "Bagaj_Tam_Boyali",
        "Bagaj_Lokal_Boyali",
        "Kapi_Degisen_Adet",
        "Kapi_Tam_Boyali_Adet",
        "Kapi_Lokal_Boyali_Adet",
        "Camurluk_Degisen_Adet",
        "Camurluk_Tam_Boyali_Adet",
        "Camurluk_Lokal_Boyali_Adet",
        "Tampon_Islem_Adet",
    ]

    # Güvenli silme (Sadece listede olanları siler)
    cols_to_drop = [col for col in drop_list_final if col in mlpvDf.columns]
    mlpvDf = mlpvDf.drop(columns=cols_to_drop)

    print(f"✅ Temizlik Tamam! {len(cols_to_drop)} adet detay sütun atıldı.")
    print(
        "🚀 Model artık sadece 'Ekspertiz_Risk_Skoru', 'Tramer' ve 'Ağır Hasar' üçlüsüne bakacak."
    )

    drop_list = [
        # 1. ESKİ VE YETERSİZ "DEĞİŞEN" BİLGİLERİ
        # Artık elimizde akıllı 'Ekspertiz_Risk_Skoru' var, bunlara gerek kalmadı.
        "Degisen_Parca_Sayisi",
        "Genel_Toplam_Degisen",
        "Ekspertiz_Skoru",  # Risk Skoru varken bu zayıf kalır
        "Sehir",
        # 2. MOTOR GÜCÜ KARMAŞASI (RPM ve Gürültü)
        # 'KısaBilgi - Motor Gücü' (HP) ana veri. Diğerleri devir sayısı veya gürültü.
        #'Motor ve Performans - Maksimum Güç',
        "Motor ve Performans - Minimum Güç",
        # 3. FİYATLA DOLAYLI İLİŞKİLİ / KİŞİSEL VERİLER
        "Genel Bakış - Yıllık MTV",  # Motor Hacmi ve Yıl zaten bunu kapsıyor
        "Genel Bakış - Ortalama Trafik Sigortası",  # Sürücüye göre değişir, arabanın değeri değildir
        "KısaBilgi - Takasa Uygun",  # Satıcının niyeti, arabanın değeri değil
        # 4. FABRİKA VERİLERİ (Gereksiz Detay)
        # Motor Hacmi, Yakıt Tipi ve Kasa Tipi varken bunlara model bakmaz.
        "Boyut ve Kapasite - Bagaj Hacmi",
        "Boyut ve Kapasite - Koltuk Sayısı",  # %99'u 5 koltuktur
        "Boyut ve Kapasite - Uzunluk",  # Segment (C, D) zaten bunu kapsar
        # 5. YAKIT TÜKETİMİ (SHAP'ta etkisi yoktu)
        "KısaBilgi - Ort. Yakıt Tüketimi",
        "Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi",
        "Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi",
        # 6. TEKNİK ÇÖPLER
        "is_body_type_filled",
        "Genel Bakış - Garanti Durumu",
        "Tampon_Islem_Adet",
        "Tavan_Tam_Boyali",
        "Bagaj_Lokal_Boyali",
        "Genel Bakış - Araç Türü",
        "Aciklama_Text",
    ]

    # Güvenli silme işlemi
    mlpvDf_final = mlpvDf.drop(
        columns=[col for col in drop_list if col in mlpvDf.columns]
    )

    print(f"🧹 Temizlik Tamamlandı! {len(drop_list)} sütun atıldı.")
    print(f"📉 Yeni Sütun Sayısı: {mlpvDf.shape[1]}")

    # 1. Kategorik (Object) olan tüm sütunları bul
    cat_cols = mlpvDf_final.select_dtypes(include=["object"]).columns

    # 2. Bu sütunlardaki NaN (boş) değerleri "Bilinmiyor" stringi ile doldur
    # Böylece CatBoost hata vermez, bunu ayrı bir kategori olarak öğrenir.
    for col in cat_cols:
        mlpvDf_final[col] = mlpvDf_final[col].fillna("Bilinmiyor").astype(str)

    # 3. KONTROL: Hala NaN kaldı mı? (0 çıkması lazım)
    print(
        "Kategorik sütunlarda kalan NaN sayısı:",
        mlpvDf_final[cat_cols].isnull().sum().sum(),
    )

    for col in cat_cols:
        mlpvDf_final[col] = mlpvDf_final[col].astype("category")
    return mlpvDf_final
