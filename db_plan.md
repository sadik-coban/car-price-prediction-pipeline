# Gold DB (API) adımı — plan

> **Durum (2026-09-24): üç parça da yapıldı.** Parça 3: `analysis/01_gold_contract.py` → teknik rapor §1 "API'ye
> giden veri (gold)" (tablo + açıklama, tr/en); testi `tests/analysis/test_01_gold_contract.py`. Kalan tek iş
> kullanıcının: gold'u S3'e yayımlamak (`python db/publish_data_to_s3.py --dry-run`, sonra `--dry-run`'sız) ve
> API'nin yeni dosyayı alması (kullanıcıda). API açıklama kolonlarını hiç okumuyor, API'de fix gerekmiyor
> (2026-09-25 incelemesi, `docs/database.md` → "Gold adımı").
>
> Parça 1 ve Parça 2 yapıldı. Parça 2: gerçek DB yarı ham olarak yeniden kuruldu (eski
> dosya `archive/backups/cars-duckdb-eski-sozlesme-2026-09-24/`), gold türetildi, kanıt gerçek dosyalarla tuttu;
> analizde üç uyarlama + `load_clean` düzeltmesi; §2'de "belirtilmemiş" ayrı paragraf, §1'de ağır hasar cümlesi;
> metrik referansı kullanıcı onayıyla güncellendi. **Parça 1:** — `db/build_gold_db.py`, `db/gold_rules.json`, yayında sözleşme
> denetimi, testler ve gerçek ham veriyle kanıt (sonuçlar `docs/database.md` → "Gold adımı"; kanıt tuttu: tek
> fark açıklamada, 778 / 114 satır). Aşağısı onaylanan ilk plan.

## Context
Analiz yarı ham `data/cars.duckdb`'yi okuyor (bilinmeyen = NULL, 2026-09-24). Canlı API (depo dışında) ise eski
sözleşmeyi bekliyor: bilinmeyen = false/0, 118 kolon, `description_clean` dahil (açıklamada kullanıcı artık
`description_text`'e geçiyor, aşağıda). Tanım `docs/database.md`
"Sonraki iş: gold adımı" bölümünde (kullanıcı onaylı). Gold, yarı ham DB gerçekten yeniden kurulmadan ÖNCE
yazılmalı. Bu iş gold'u yazar ve kanıtlar; gerçek DB'yi yeniden kurmaz, S3'e yüklemez, commit etmez.

## Silver → gold, eksiksiz (koddan sayıldı)
Satırlar: aynı 45.277 satır, aynı `id`, aynı sıra. Süzme, tekilleştirme, yeniden ayrıştırma ve tip değişikliği yok.

| silver'da (117 kolon + id) | gold'da | adet | gerçek veride etkisi |
|---|---|---:|---|
| `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner` | NULL → `false` | 3 | 29.641 + 29.641 + 120 hücre |
| 39 panel bayrağı: `{tavan, kaput, bagaj, door_fl/fr/rl/rr, fender_fl/fr/rl/rr, bumper_front/rear}_{degisen, boyali, lokal}` | NULL → `0` | 39 | 293.988 hücre |
| `count_changed`, `count_painted`, `count_local_painted` | NULL → `0` | 3 | 0 hücre (önlem) |
| `kb_paint_change_summary` | alınmaz | 1 | — |
| öteki 71 kolon (`description_text` dahil) | aynen | 71 | — |

- Gold: 116 kolon + `id`.
- Öteki dört tablo (`duplicate_ad_ids`, `price_history`, `dashboard_cache`, `options_cache`) aynen kopyalanır.
- Kural dışı kolonlardaki NULL'lara dokunulmaz (ör. takas, teknik özellikler). Bunlar bugünkü API verisinde de
  NULL; eski sözleşme yalnız bu 45 kolonu dolduruyordu.

## Yapılacaklar
1. **`db/build_gold_db.py`** (yeni):
   - Yarı ham DB'yi salt okur; `data/cars_gold.duckdb`'yi `build_duckdb`'deki yöntemle güvenli yazar (`.tmp` +
     `os.replace`, `.wal` / kilit kontrolü).
   - **Girdi denetimi (ek):** girdinin kolon listesi `build_duckdb.DB_COLUMNS` + `id` değilse durur. Böylece
     bugünkü eski sözleşmeli DB ya da yanlış bir dosya gold diye işlenmez.
   - `GOLD_RULES`, tek listede ve iki dilli gerekçeyle:
     - `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner` → `COALESCE(x, false)`;
     - 39 panel bayrağı + 3 sayaç → `COALESCE(x, 0)`.
   - Şema: yarı ham DB'nin kolonları aynı sırayla, yalnız `kb_paint_change_summary` alınmaz. Açıklama yarı ham
     DB'deki gibi kalır: yalnız `description_text` (başlıksız); `description_clean` geri eklenmez. Kullanıcı
     (2026-09-24): "açıklama metni yine aynı kalsın, description_text olsun; API yapısına dokunuyorsa API'de fix
     atarım". Sonuç 116 kolon + `id` = 117; bugünkü 118'den tek fark `description_clean`'in olmaması
     (`GOLD_COLUMNS`, `DB_COLUMNS`'tan türetilir).
   - `duplicate_ad_ids`, `price_history`, `dashboard_cache`, `options_cache` aynen kopyalanır.
2. **`db/publish_data_to_s3.py`:**
   - varsayılan `--duckdb` → `data/cars_gold.duckdb`; S3 nesne adı aynı (`data/cars.duckdb`);
   - "dosya yok" ipucu iki adımı söyler;
   - **Sözleşme denetimi (ek):** yüklemeden önce `car_listings` kolonları `GOLD_COLUMNS` ile birebir mi ve 45
     kural kolonunda NULL var mı bakılır; tutmazsa yüklemez. Yarı ham DB yanlışlıkla API'ye gidemez. Bugünkü
     `data/cars.duckdb` da artık geçmez (`description_clean` taşıyor); yeniden yayımlanması zaten gerekmiyor.
   - Testlerin sahte DB fabrikası (`make_duckdb`) gold biçimli tabloya güncellenir.
3. **Testler** (`tests/db/test_build_gold_db.py`):
   - kural kolonlarında NULL → false/0, NULL olmayana dokunulmaz;
   - kural dışı kolonlar ve dört tablo hücre hücre aynı;
   - şema 117 kolon (bugünkü sıra, `description_clean` yok, `kb_paint_change_summary` yok);
   - girdi denetimi eski sözleşmeli DB'yi reddeder;
   - güvenli yazma (yarıda düşünce eski gold sağlam, `.wal` durdurur);
   - yayının yeni varsayılanı ve sözleşme denetimi.
4. **Belgeler:**
   - `docs/database.{md,en.md}`: "henüz yazılmadı" → yazıldı; `description_clean` kararının yeni hâli ve
     API'ye düşen not;
   - README'nin Zincir ve Nasıl koşulur bölümleri;
   - `db/build_duckdb.py` başlığındaki uyarı;
   - hafıza notu (API 118 kolon değil 117, `description_clean` yok).

## API'nin göreceği fark (bugünkü S3 dosyasına göre)
Yalnız açıklama değişir; geri kalan her hücre, kolon, tip ve sıra aynıdır (kanıt aşağıda):
- `description_text`: başındaki sayfa başlığı "Açıklama " yok (45.163 satır); yalnız başlıktan ibaret 114 satır
  NULL;
- `description_clean` kolonu yok. (2026-09-25: API bu kolonu hiç okumuyor; geçiş gerekmiyor.) Yeni
  `description_text`, eski `description_clean` ile 778 satır dışında aynı; o satırlarda satıcının baştaki
  "-" / ":" işareti görünür.

## Kanıt (gerçek veriyle, geçici klasörde)
- Yeni kodla kurulan yarı ham DB → `build_gold_db` → gold.
- Gold, yedekteki eski kodun ham veriden ürettiği DB ile (`archive/obsolete/pipeline-yedek-2026-09-24/`) hücre
  hücre karşılaştırılır. Beklenen farklar yalnız açıklamada:
  - kolon listesi = eskisi eksi `description_clean` (117, sıra aynı);
  - `description_text` her satırda (başlık yok; 114 satır NULL) ve eski `description_clean` ile 778 satır
    dışında aynı.
- Başka tek bir hücre bile farklıysa iş bitmiş sayılmaz.
- Ayrıca `publish_data_to_s3.py --dry-run --duckdb <geçici gold>` sözleşme denetiminden geçmeli; yarı ham DB ile
  denendiğinde reddetmeli.
- `python -m pytest tests`, pyflakes, iki dilli docstring taraması.

## Sonra (kullanıcının kararı, bu işte yok)
Gerçek DB'nin yeniden kurulma sırası:
1. `build_duckdb` → `build_gold_db` → kanıt karşılaştırması gerçek dosyalarla.
2. Analizde üç iş: `text_flags` → `description_text`, `02_missingness` listesi, `kb_paint_change_summary`
   için bir sınıf.
3. `run_all.py` (§2 eksiklik tablosu değişir).
4. Kullanıcı yayınlar.

**Süre tahmini:** ~30–45 dk (kod + testler + kanıt; makine süresi ~3 dk).
