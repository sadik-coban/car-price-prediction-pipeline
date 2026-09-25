# Veritabanı (`cars.duckdb`) notları

> 🇬🇧 English: [database.en.md](database.en.md) · ← [README](../README.md)

Bu dosya analizin girdisi. API'ye doğrudan gitmez: yayın, ondan türetilen gold dosyayı
(`data/cars_gold.duckdb`, aşağıda) yükler. 2026-09-24'e kadar bu dosyanın kendisi yayımlanıyordu.

- **Yarı ham (2026-09-24, kullanıcının ilkesi: "dbdeki her şey yarı raw olmalı").** Sayfanın söylemediği
  değer tahmin edilmez, `NULL` kalır; "Belirtilmemiş"in nasıl okunacağına analiz karar verir. Gerçek dosya
  2026-09-24'te bu kuralla yeniden kuruldu (eski sözleşmeli önceki dosya
  `archive/backups/cars-duckdb-eski-sozlesme-2026-09-24/`'te):
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

  Yeniden kurulumla analizde yapılanlar (2026-09-24):
  - `analysis/lib/text_flags.descriptions` `description_text`'i okuyor; metin bayraklarının hiçbiri
    değişmedi;
  - `02_missingness`: `description_clean` `IDENTITY`'den çıktı, `kb_paint_change_summary` "türetilmiş tekrar"
    (F) sınıfında. NULL'u "Belirtilmemiş" demek olan 45 kolon (gold kurallarının doldurduğu kolonlar,
    `db/gold_rules.json`'dan okunur) eksik veri listesine ve bloklara girmiyor, ayrıca raporlanıyor: ağır hasar
    kaydı ilanların %68,2'sinde, panel bayrakları %12,2–%19,1'inde belirtilmemiş (teknik rapor §1–§2);
  - `load_clean`: `is_heavy_damaged` artık `NULL` taşıdığı için pandas kolonu nullable `boolean` okuyor ve
    eski `fillna(0)` hata veriyordu; bilinmeyen `False` (ağır hasarsız) sayılıyor, sonuç eskisiyle aynı.
  Modele giren değerler ve bütün model sayıları değişmedi (metrik referansıyla doğrulandı:
  `tests/baselines/accept_log.jsonl`).
- **Güvenli kurulum.** Yeni DB `<out>.tmp`'de kurulur, bitince eskisinin yerine konur; düşerse eski DB
  dokunulmadan kalır. `<out>.wal` varsa, DB başka programda açıksa ya da ham veri eksikse (marka klasörü,
  `details.jsonl`'suz tarama, okunamayan satır) hiçbir şeye dokunmadan durur. `duplicate_ad_ids.csv` DB'nin
  yanına yazılır.
- **Tahmin yok: gözlenen değerler kaydı (2026-09-25, kullanıcının ilkesi: "yaşanmamış case'leri varsayma").**
  `db/observed_values.json` verinin gerçekte ne tuttuğunu yazar:
  - her ham alanın değerleri (60 farklı değere kadar) ya da biçimleri (`"# TL"`, `"# - # cm3"`);
  - hasar etiketleri;
  - az değerli silver kolonları;
  - seri → modeller.

  Dosya `python tools/observed_values.py` ile üretilir (`--check` veriyle karşılaştırır). Kurulum bu kayda üç yerden
  bağlanır:
  - `build_duckdb.py` satırları okumadan önce ham veriyi kayıtla karşılaştırır. Yeni bir alan, görülmemiş bir değer
    ya da biçim veya bilinmeyen bir hasar etiketi kurulumu **hiçbir şey yazmadan** durdurur; mesaj alanı, değeri,
    kaç kayıtta olduğunu ve ilk dosya:satır'ı söyler.
  - Ayrıştırıcılar (`db/lib/process_for_db.py`) verinin hiç göstermediği biçimde tahmin etmek yerine
    `UnknownValue` yükseltir. Eskiden tek yıllı üretim yılı "ilk = son" oluyor, bilinmeyen ilk sahip metni `False`
    sayılıyor, bilinmeyen hasar etiketi atlanıyordu.
  - Testler iki yönde sınar: kodun andığı her değer veride görülmüş, verideki her değer kodda bilerek ele alınmış
    (`tests/db/test_observed_values.py`, `tests/analysis/test_observed_values_analysis.py`).

  Bu sırada veride hiç görülmeyen üç varsayım kaldırıldı: plaka `"Yabancı plakalı"`, segment kuralındaki 14 seri
  (`8 Serisi`, `X1`–`X7`, `Z4`, `Q2`–`Q8`) ve `02_missingness`'in `"nan"`/`"None"` gibi eksik jetonları.
  Gerçek ham veri geçici bir yola yeniden kurulunca `data/cars.duckdb` ile satır satır aynı çıkıyor (veri testi).

  Yeni bir değer görülürse önce kod o değere göre uyarlanır, sonra kayıt güncellenir.
- **`engine_cc_val` iki yerde iki anlamda.** DB'de kovanın orta noktası (açık uçlu kovada bilinen sınır);
  modelin aynı adlı özniteliği `engine_cc_up` (üst sınır, gerekçe teknik rapor §1). DB kolonunun adı,
  onu okuyan bir API kırılmasın diye değiştirilmedi. API'nin drift ekranı bu kolonu okuyor, yani orta noktayı
  gösteriyor; model üst sınırı kullanıyor (kovalı 7.827 ilanda fark, medyan 99,5 cc; 2026-09-25 ölçümü). Drift iki
  taramayı aynı tanımla kıyasladığı için sonuç bozulmuyor; kullanıcı kararıyla böyle bırakıldı.
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

## Gold adımı (API) — yazıldı 2026-09-24

Analiz yarı ham DB'yi okur; canlı API (bu deponun dışında) eski sözleşmeyi bekler: bilinmeyen ağır hasar / ilk
sahip `false`, "Belirtilmemiş" panel `0`. Tek dosya ikisine birden hizmet edemediği için API'ye ayrı bir gold
dosyası gider.

```
data/raw ──build_duckdb──► data/cars.duckdb (yarı ham, analiz) ──build_gold_db──► data/cars_gold.duckdb (API)
                                                                                   └─publish_data_to_s3─► S3 "data/cars.duckdb"
```

- **`db/build_gold_db.py`** yarı ham DB'yi yalnız okur, `data/cars_gold.duckdb`'yi güvenli yazar (`.tmp` + yer
  değiştirme, `.wal` kontrolü). Önce girdinin gerçekten yarı ham DB olduğunu sınar (kolonlar `id` +
  `DB_COLUMNS`); eski sözleşmeli ya da gold bir dosya reddedilir. Kurallar gerekçeleriyle
  **`db/gold_rules.json`**'da (teknik raporun gold bölümü de aynı dosyayı okuyacak):
  - `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner`: `NULL` → `false`;
  - 39 panel bayrağı ve 3 hasar sayacı: `NULL` → `0` ("Belirtilmemiş" panel, analizdeki gibi orijinal sayılır);
  - `kb_paint_change_summary` alınmaz;
  - açıklama yarı ham DB'deki gibi: yalnız başlıksız `description_text`, **`description_clean` yok** (kullanıcı
    kararı 2026-09-24: "açıklama metni aynı kalsın, description_text olsun; API yapısına dokunuyorsa API'de fix
    atarım");
  - sonuç 116 kolon + `id`; satırlar, sıra, id'ler, tipler ve öteki her hücre aynı; öteki dört tablo aynen.
- **`publish_data_to_s3.py`**: varsayılan girdi `data/cars_gold.duckdb`, S3 nesne adı aynı (`data/cars.duckdb`).
  Yüklemeden önce gold sözleşmesini sınar (kolonlar = `id` + gold kolonları, kural kolonlarında `NULL` yok);
  yarı ham ya da eski sözleşmeli bir dosya API'ye gidemez.
- **Koşum sırası:** `build_duckdb.py` → `build_gold_db.py` → (kullanıcı) `publish_data_to_s3.py`.
- **Testler** (`tests/db/test_build_gold_db.py` + yayın testleri): kural dosyası DB kolonlarıyla uyumlu; kural
  kolonlarında `NULL` → false/0, bilinen değer değişmiyor; kural dışı kolonlar ve dört tablo hücre hücre aynı;
  şema ve tipler; girdi denetimi; güvenli yazma; yayının gold olmayan dosyayı ve kural kolonundaki `NULL`'u
  reddetmesi.
- **Kanıt (2026-09-24, gerçek ham veriyle, geçici klasörde):** yeni kodla kurulan yarı ham DB → gold, bugünkü
  `data/cars.duckdb` ile (eski kodun ürünü, eski sözleşme) hücre hücre karşılaştırıldı:
  - kolonlar: eskisi eksi `description_clean` = gold (117; ad, sıra ve tip aynı); 45.277 satır, id'ler aynı;
  - farklı tek kolon `description_text` (her satırda sayfa başlığı yok). Eski `description_clean` ile yalnız 778
    satırda ayrışıyor (satıcının baştaki "-" / ":" işareti); yalnız başlıktan ibaret 114 satır `NULL`;
  - `duplicate_ad_ids`, `price_history` ve iki önbellek birebir aynı;
  - gold'un doldurduğu hücre: 59.402 (ağır hasar ×2 + ilk sahip), 293.988 (panel bayrakları), 0 (sayaçlar);
  - `publish --dry-run`: gold geçti; yarı ham DB ve bugünkü eski DB reddedildi.
- **API'nin göreceği fark: yok** (2026-09-25'te API kodundan ve API'nin kendi fonksiyonlarıyla doğrulandı,
  `sadik-portfolio/api`). API yalnız `car_listings`'ten 55 kolon okuyor; `description_clean`, `description_text`,
  `kb_paint_change_summary`, `kb_is_heavy_damaged`, `gb_is_first_owner` ve öteki dört tablo hiç okunmuyor. API'de
  fix gerekmiyor. API yarı ham (silver) dosyayla da birebir aynı çıktıyı veriyor (pano satırları, tarama listesi,
  drift). NULL'ları kodunda örtük olarak 0 sayıyor. Yine de yayına gold gidiyor: API'nin bu örtük davranışına
  güvenmemek için açık sözleşme (kullanıcı kararı).
- **Gerçek dosyalar (2026-09-24):** `data/cars.duckdb` yarı ham olarak yeniden kuruldu, `data/cars_gold.duckdb`
  ondan türetildi. Kanıt gerçek dosyalarla tekrarlandı: gold, yedeklenen eski DB ile açıklama dışında hücre hücre
  aynı (778 / 114 satır, öteki tablolar birebir). S3'e hiçbir şey yüklenmedi; yayın kullanıcının kararı.
- **Raporda (2026-09-24):** teknik rapor §1 → "API'ye giden veri (gold)". Sayıları `analysis/01_gold_contract.py`
  sayar (bütün tarama satırları üzerinden, kurallar `db/gold_rules.json`'dan): 353.390 hücre doldurulur
  (59.402 ağır hasar + ilk sahip, 293.988 panel bayrağı), `kb_paint_change_summary` gitmez, 114 açıklama boş.
