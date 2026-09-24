# Veritabanı (`cars.duckdb`) notları

> 🇬🇧 English: [database.en.md](database.en.md) · ← [README](../README.md)

Bu dosya analizin girdisi; bugüne kadar `publish_data_to_s3.py` ile S3'e de gitti (API okuyor).

- **Yarı ham (2026-09-24, kullanıcının ilkesi: "dbdeki her şey yarı raw olmalı").** Sayfanın söylemediği
  değer tahmin edilmez, `NULL` kalır; "Belirtilmemiş"in nasıl okunacağına analiz karar verir. Bir sonraki
  kurulumdan itibaren geçerli (bugünkü dosya 2026-09-23'te eski kodla kuruldu):
  - `is_heavy_damaged` / `kb_is_heavy_damaged` "KısaBilgi - Ağır Hasarlı"dan: Evet / Hayır / `NULL`
    (Belirtilmemiş ya da alan yok; 29.641 satır `False` → `NULL`). Scraper'ın `Agir_Hasar` bayrağı sayfa bir
    şey demediğinde de `False` olduğu için bırakıldı;
  - durumu "Belirtilmemiş" olan panelin üç bayrağı `NULL` (97.996 panel, 293.988 hücre; bilinen durumda 1/0);
  - `gb_is_first_owner` alan yoksa `NULL` (120 satır); hasar sayaçları boşsa `NULL` (bugün 0 satır);
  - açıklama: `description_text`'ten yalnız sayfanın bölüm başlığı "Açıklama" (ve arkasındaki tek boşluk)
    silinir; satıcının "-" / ":" işareti kalır (778 satırda eski `description_clean`'den farklı).
    `description_clean` kolonu kalktı;
  - yeni ham kolon `kb_paint_change_summary`: kısa bilgi kutusundaki "Boya-değişen" satırı ("2 değişen,
    3 boyalı" gibi), olduğu gibi. Hasar şemasının kaba özeti.

  Yeniden kurulunca analizde yapılacaklar: `analysis/lib/text_flags.descriptions` `description_text`'i
  okur; `02_missingness`'in `IDENTITY` listesinden `description_clean` çıkar, `kb_paint_change_summary`'ye
  bir sınıf verilir (§2 eksiklik tablosu değişir). Modele giren değerler değişmez: `load_clean` bu
  `NULL`'ları zaten 0 sayıyor.
- **Güvenli kurulum.** Yeni DB `<out>.tmp`'de kurulur, bitince eskisinin yerine konur; düşerse eski DB
  dokunulmadan kalır. `<out>.wal` varsa, DB başka programda açıksa ya da ham veri eksikse (marka klasörü,
  `details.jsonl`'suz tarama, okunamayan satır) hiçbir şeye dokunmadan durur. `duplicate_ad_ids.csv` DB'nin
  yanına yazılır.
- **`engine_cc_val` iki yerde iki anlamda.** DB'de kovanın orta noktası (açık uçlu kovada bilinen sınır);
  modelin aynı adlı özniteliği `engine_cc_up` (üst sınır, gerekçe teknik rapor §1). DB kolonunun adı,
  onu okuyan bir API kırılmasın diye değiştirilmedi.
- **`dashboard_cache` / `options_cache`** bu depodaki hiçbir üretecin tablosu değil (arşivdeki
  `build_aggregates.py`); içerikleri bayat (2026-07-14, TR plaka filtresi yok). `build_duckdb.py` yeniden
  kurulumda onları eski DB'den aynen taşır: ne silinir ne güncellenir.
- **`gb_body_type`** satırların bir kısmında kasa tipini koltuk sayısıyla birleşik taşır; model
  `kb_body_type`'ı kullanır.
- **`duplicate_ad_ids`** birden çok satırı olan `ad_id`'ler, yani sonraki taramalarda yeniden görülen
  ilanlar (tek bir taramanın içinde tekrar yok).
- **2026-09-23 ayrıştırıcı düzeltmeleri:** takas alanında "Takasa Uygun Değil" artık `False` (eskiden
  "uygun" kelimesi yüzünden `True`), alan yoksa `NULL`; ilk sahip Türkçe büyük İ yüzünden hep `False`'tu;
  "… HP'ye kadar" / "… ve üzeri" güç kovaları artık açık uçlu kova; açık uçlu hacimde DB'nin orta noktası
  `NULL`'du; `kb_fuel_cons_avg` "Yakıt Tüketimi" sekmesinden de dolar; yalnız "Açıklama" yazan metin
  `description_clean`'de `NULL` (`description_text` ham hâliyle kalır). Modele giren hiçbir değer değişmedi (eski ve yeni DB kolon kolon karşılaştırıldı). DB yerelde
  yeniden kuruldu; **S3'e yüklemek ayrı bir adım** (önce `python db/publish_data_to_s3.py --dry-run`, sonra
  `--dry-run`'sız).

## Sonraki iş: gold adımı (API) — henüz yazılmadı

Kullanıcı kararı (2026-09-24): not alındı, şimdi yapılmıyor. Analiz yarı ham DB'yi okur; canlı API (bu
deponun dışında) eski sözleşmeyi bekler: bilinmeyen ağır hasar / ilk sahip `false`, "Belirtilmemiş" panel `0`,
118 kolon, `description_clean` dahil. Tek dosya ikisine birden hizmet edemez.

> ⚠️ **Sıra:** yarı ham DB gerçekten yeniden kurulmadan önce gold yazılmalı. Kurulduktan sonra gold olmadan
> `publish_data_to_s3.py` koşulursa API'ye `NULL`'lar gider ve `description_clean` kolonu kaybolur. Bugünkü
> `data/cars.duckdb` eski sözleşmeli; S3 bugün güvende.

```
data/raw ──build_duckdb──► data/cars.duckdb (yarı ham, analiz) ──build_gold_db──► data/cars_gold.duckdb (API)
                                                                                   └─publish_data_to_s3─► S3 "data/cars.duckdb"
```

- **`db/build_gold_db.py`** yarı ham DB'yi yalnız okur, `data/cars_gold.duckdb`'yi güvenli yazar
  (`.tmp` + yer değiştirme, `.wal` kontrolü). Kurallar tek listede (`GOLD_RULES`), gerekçeleriyle:
  - `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner` → `COALESCE(x, false)`;
  - 39 panel bayrağı ve 3 hasar sayacı → `COALESCE(x, 0)`;
  - `kb_paint_change_summary` alınmaz; `description_clean` = `description_text`, `description_text`'in hemen
    arkasına. Sonuç: bugünkü 118 kolon, aynı ad, tip ve sırayla;
  - öteki tablolar (`duplicate_ad_ids`, `price_history`, `dashboard_cache`, `options_cache`) aynen.
- **`publish_data_to_s3.py`**: varsayılan girdi `data/cars_gold.duckdb`; S3 nesne adı aynı kalır; "dosya yok"
  ipucu iki adımı söyler.
- **Koşum sırası:** `build_duckdb.py` → `build_gold_db.py` → (kullanıcı) `publish_data_to_s3.py`.
- **Testler** (`tests/db/test_build_gold_db.py`): kural kolonlarında `NULL` → false/0 ve `NULL`
  olmayana dokunulmaması; kural dışı kolonlar ve dört tablo hücre hücre aynı; şema bugünküyle aynı; güvenli
  yazma; yayının yeni varsayılanı.
- **Kanıt:** gold, eski kodun ham veriden ürettiği DB ile hücre hücre karşılaştırılır. Eski kod
  `archive/obsolete/pipeline-yedek-2026-09-24/`'te; bugünkü DB'yi birebir ürettiği doğrulandı (klasörün README'si
  nasıl koşulacağını anlatır). Beklenen farklar yalnız açıklamada: `description_text` her satırda (başlık yok;
  yalnız başlıktan ibaret 114 satır `NULL`), `description_clean` tam 778 satırda. Başka hiçbir hücre
  değişmemeli.
